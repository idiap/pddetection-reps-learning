# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
# Defining Runner module for training/evaluating the downstream model

# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Parvaneh Janbakhshi <parvaneh.janbakhshi@idiap.ch>

# This file is part of pddetection-reps-learning
#
# pddetection-reps-learning is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# pddetection-reps-learning is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pddetection-reps-learning. If not, see <http://www.gnu.org/licenses/>.
"""*********************************************************************************************"""

import os
import random
import torch
import torch.nn as nn
import numpy as np
import glob
from torch.utils.data import DataLoader
from upstream_auxiliary.pretrain_ups_expert import FeatureExtractionPretrained
from .train_downs_expert import DownStreamTrain
from audio.audio_dataset import OnlineAcousticDataset, OfflineAcousticDataset
from audio.audio_utils import get_config_args, create_transform
from torch.autograd import Variable
from collections import defaultdict
import warnings


def seed_torch(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class DownsRunner:
    """
    Runner for downstream training
    creating train/val/test data loaders, training/evaluation
    loops, checkpoint saving and loading (resume training)
    """

    def __init__(self, args):
        """Initializing the runner for training
        Args:
            args (argparse.Namespace): arguments for training
            including paths of config files, training params, fold, seed, ...
        """
        self.args = args
        self.ups_config = self.args.ups_config
        self.ds_config = self.args.ds_config
        self.ups_config_file = get_config_args(self.ups_config)
        self.ds_config_file = get_config_args(self.ds_config)
        self.ups_init_ckpt_path = (
            f"results/{self.args.expname}/ups-{self.args.expname}/outputs/saved_model/"
        )
        os.makedirs(self.ups_init_ckpt_path, exist_ok=True)
        # getting number of folds from upstream model
        # if we have only one upstream, we use that for all folds of downstream data
        # please check lack of overlap between test and train data for upstream
        # and downstream model training

        ups_model_num = len(glob.glob1(self.ups_init_ckpt_path, "UPs_NN_fold*.ckpt"))
        print(f"\n{ups_model_num} upstream pre-trained models are found.\n")
        ups_fold_num = len(
            glob.glob1(
                self.ups_config_file["dataloader"]["data_path"], "test_fold*_online.csv"
            )
        )
        print(f"\n{ups_fold_num} data folds are found for upstream training.\n")
        if ups_model_num != ups_fold_num:
            warnings.warn(
                "mismatch between number of data folds for upstream training "
                "and number of pre-trained upstream models\n If upstream models"
                " for some folds are not found, random initialization is used"
                " for upstream encoder"
            )

        if self.args.fold > ups_fold_num:
            self.ups_fold = 1
        else:
            self.ups_fold = self.args.fold

        self.ups_init_ckpt_path += f"UPs_NN_fold{self.ups_fold}.ckpt"
        self.ds_init_ckpt_path = self.args.ds_init_ckpt_path
        os.makedirs(self.ds_init_ckpt_path, exist_ok=True)
        self.ds_init_ckpt_path += f"Ds_NN_fold{self.args.fold}.ckpt"

        self.selected_encoded_layers = self.ds_config_file["SelectedEncodedLayers"]
        self.feat_config = self.args.audio_config

        self.ds_train_loader = self._get_dataloader(set="train", shuffle=True)

        if self.args.valmonitor:
            self.ds_val_loader = self._get_dataloader(set="val", shuffle=False)
            print("\n Monitoring training based on validation set.")
        else:
            self.ds_val_loader = self._get_dataloader(set="train", shuffle=False)
            print(
                "\n Training is not monitored; validation set = non-shuffledtrain set."
            )

        self.ds_test_loader = self._get_dataloader(set="test", shuffle=False)

        self.freqlen, self.seqlen = self.ds_train_loader.dataset.getDimension()

        self.upstream = None

        self._get_upstream()
        # get dimension of encoded features from upstream based on selected_encoded_layers
        self.encoded_feat_size = self.upstream.get_feature_dimension(
            selected_encoded_layers=self.selected_encoded_layers
        )

        self.downstream = None
        self.optimizer = None
        self._get_downstream()  # initialize downstream model
        self._get_optimizer()  # initialize optimizer
        # initialization for Earlystoping method if used
        self.Estop_counter = 0
        self.Estop_best_score = None
        self.Estop = False
        self.Estop_min_loss = np.Inf
        self.Saved_counter = 0

        loss = {"CE": nn.CrossEntropyLoss(), "MSE": nn.MSELoss()}
        self.loss = loss[self.ds_config_file["runner"]["optimizer"]["loss"]]

    def _get_upstream(self):
        """get upstream model (initialized or loaded from previously saved model)
        based on upstream config
        if downstream model is saved before then load upstream from previously
        tuned encoder otherwise
        load it from pretrained upstream model (which is not fine-tuned according to
        downstream task)
        """
        init_ckpt_ds = (
            torch.load(self.ds_init_ckpt_path, map_location="cpu")
            if os.path.exists(self.ds_init_ckpt_path)
            else {}
        )
        init_ckpt_ups = (
            torch.load(self.ups_init_ckpt_path, map_location="cpu")
            if os.path.exists(self.ups_init_ckpt_path)
            else {}
        )

        init_upstream_ds = init_ckpt_ds.get("UpsConfig")
        init_upstream_ups = init_ckpt_ups.get("UpsConfig")

        if init_upstream_ds:
            network_config = init_upstream_ds
        elif init_upstream_ups:
            network_config = init_upstream_ups
        else:
            network_config = self.ups_config_file
        if self.upstream is None:  # initialize for the first time
            self.upstream = FeatureExtractionPretrained(
                self.freqlen, self.seqlen, network_config
            ).to(self.args.device)
        if init_upstream_ds:
            print(
                "[FeatureExtractionPretrained] - Loading upstream model weights from the init ckpt (previous downstream saved model): ",
                f"results/{self.args.expname}/ds-{self.args.expname}/outputs/saved_model/Ds_NN_fold{self.args.fold}.ckpt",
            )
            self.upstream.load_model(init_ckpt_ds)
        elif init_upstream_ups:
            print(
                "[FeatureExtractionPretrained] - Loading upstream model weights from the init ckpt from: ",
                f"results/{self.args.expname}/ups-{self.args.expname}/outputs/saved_model/UPs_NN_fold{self.ups_fold}.ckpt",
            )
            self.upstream.load_model(init_ckpt_ups)
        print(self.upstream, flush=True)

    def _get_downstream(self):
        """get downstream model (initialized or loaded from previously saved model)
        based on downstream config
        """
        init_ckpt = (
            torch.load(self.ds_init_ckpt_path, map_location="cpu")
            if os.path.exists(self.ds_init_ckpt_path)
            else {}
        )
        init_dstream = init_ckpt.get("DsConfig")
        if init_dstream:
            network_config = init_dstream
        else:
            network_config = self.ds_config_file
        if self.downstream is None:  # initialize for the first time
            self.downstream = DownStreamTrain(
                self.encoded_feat_size, network_config
            ).to(self.args.device)
        if init_dstream:
            print(
                "[DownStreamTrain] - Loading downstream model weights from the init ckpt from: ",
                f"results/{self.args.expname}/ds-{self.args.expname}/outputs/saved_model/Ds_NN_fold{self.args.fold}.ckpt",
            )
            self.downstream.load_model(init_ckpt)

    def _init_fn(self, worker_id):
        """Setting randomness seed for multi-process data loading
        Args:
            worker_id
        """
        seed_torch(self.args.seed)

    def _get_dataloader(self, set="train", shuffle=True):
        """Get dataloader (depending on config we get online or offline dataloader)
        Args:
            set (str, optional): data splits (train/test/validation). Defaults to 'train'.
            shuffle (bool, optional): shuffling data segments. Defaults to True.
        Returns:
            (Dataloader)
        """
        feat_config = get_config_args(self.feat_config)
        feat_type = feat_config["feat_type"]
        ds_config = get_config_args(self.ds_config)
        dataloader_config = ds_config.get("dataloader")
        print(f"\n\n{set} Data...")
        if dataloader_config.get("online"):
            file_path = os.path.join(
                dataloader_config.get("data_path"),
                f"{set}_fold{self.args.fold}_online.csv",
            )
            transforms = create_transform(feat_config, dataloader_config.get("fs"))
            print(f"\n{set} file reading fold {self.args.fold} [online]...")
            dataset = OnlineAcousticDataset(
                transforms, ds_config, file_path, feat_config
            )
        else:
            file_path = os.path.join(
                dataloader_config.get("data_path"),
                f"{set}_fold{self.args.fold}_{feat_type}_offline.csv",
            )

            print(f"\n{set} file reading fold {self.args.fold} [offline]...")
            dataset = OfflineAcousticDataset(None, ds_config, file_path, feat_config)
        net = self.ups_config_file["SelectedAENetwork"]
        arch = self.ups_config_file[net]
        drop_last = arch.get("batchnorm", False)
        return DataLoader(
            dataset,
            batch_size=dataloader_config.get("batch_size"),
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=dataloader_config.get("num_workers"),
            worker_init_fn=self._init_fn,
        )

    def _get_optimizer(self):
        """Set optimizer models for downstream model
        (initialized or loaded from previously saved model)
        """
        if self.optimizer is None:  # initialize for the first time
            optimizer_config = self.ds_config_file.get("runner").get("optimizer")
            optimizer_name = optimizer_config.get("type")
            assert self.downstream is not None, (
                "define downstream model" " first before defining the optimizer."
            )
            self.lr = float(optimizer_config["lr"])
            self.upslr_ratio = float(optimizer_config["upslr_ratio"])

            model_params_ds = list(self.downstream.parameters())
            param_list = [
                {"params": model_params_ds, "lr": self.lr, "name": "Ds-model"}
            ]
            if self.args.upstream_trainable:
                model_params_ups = list(self.upstream.parameters())
                param_list += [
                    {
                        "params": model_params_ups,
                        "lr": self.lr * self.upslr_ratio,
                        "name": "UPs-model",
                    }
                ]

            if optimizer_name == "SGD":
                self.optimizer = eval(f"torch.optim.{optimizer_name}")(
                    param_list, momentum=optimizer_config.get("momentum")
                )
            else:
                self.optimizer = eval(f"torch.optim.{optimizer_name}")(param_list)

            self.max_epoch = self.ds_config_file["runner"]["Max_epoch"]
            self.minlr = float(optimizer_config["minlr"])
        init_ckpt = (
            torch.load(self.ds_init_ckpt_path, map_location="cpu")
            if os.path.exists(self.ds_init_ckpt_path)
            else {}
        )
        init_optimizer = init_ckpt.get("Optimizer")
        if init_optimizer:
            print(
                "\n[ds-Runner Optimizer] - Loading optimizer weights from the init ckpt from: ",
                f"results/{self.args.expname}/ds-{self.args.expname}/outputs/saved_model/Ds_NN_fold{self.args.fold}.ckpt",
            )
            self.optimizer.load_state_dict(init_optimizer)

    def _save_ckpt(self):
        """save models and optimizers params into the checkpoint"""
        if os.path.exists(self.ds_init_ckpt_path):
            init_ckpt = torch.load(self.ds_init_ckpt_path, map_location="cpu")
        else:
            init_ckpt = {
                "Optimizer": " ",
                "UpsConfig": self.ups_config_file,
                "DsConfig": self.ds_config_file,
            }
        init_ckpt = self.downstream.add_state_to_save(init_ckpt)
        init_ckpt = self.upstream.add_state_to_save(init_ckpt)
        init_ckpt["Optimizer"] = self.optimizer.state_dict()
        torch.save(init_ckpt, self.ds_init_ckpt_path)

    def _load_ckpt(self):
        """load models and optimizers params from the checkpoint"""
        init_ckpt = torch.load(self.ds_init_ckpt_path, map_location="cpu")
        self.downstream.load_model(init_ckpt)
        self.upstream.load_model(init_ckpt)
        init_optimizer = init_ckpt.get("Optimizer")
        self.optimizer.load_state_dict(init_optimizer)

    def _earlystopping(
        self, val_loss, patience=5, verbose=True, delta=0, lr_factor=0.5
    ):
        """Early stops the training if validation loss doesn't improve after a given patience.
        it saves best model based on validation on checkpoint path
        Adapted from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
        Args:
            val_loss (float): validation loss
            patience (int, optional): How long to wait after last time validation loss
            improved. Defaults to 5.
            verbose (bool, optional): If True, prints a message for each validation loss
            improvement. Defaults to True.
            delta (int, optional): Minimum change in the monitored quantity to qualify
            as an improvement. Defaults to 0.
            lr_factor (float): after the patience epochs, multiple learning by lr_factor.
            Defaults to 0.5.
        """
        self.Estop_delta = delta
        score = -val_loss
        if self.Estop_best_score is None:
            self.Estop_best_score = score
            if verbose:
                print(
                    f"Validation loss decreased ({self.Estop_min_loss:.6f} -->"
                    f" {val_loss:.6f}).  Saving model ...",
                    flush=True,
                )
            # save model and optimizer
            self.Estop_min_loss = val_loss
            self._save_ckpt()

        elif score < self.Estop_best_score + self.Estop_delta:
            self.Estop_counter += 1
            if verbose:
                print(f"EarlyStopping counter: {self.Estop_counter} out of {patience}")
            if self.Estop_counter >= patience:
                # load previously saved models and optimizer while multiplying the leanring rate by lr_factor
                print("No improvement since last best model", flush=True)
                self._load_ckpt()  # start from the last best model
                self.lr = self.lr * lr_factor
                for g in self.optimizer.param_groups:  # decreasing the learning rate
                    if g["name"] == "UPs-model":
                        g["lr"] = self.lr * self.upslr_ratio
                    else:
                        g["lr"] = self.lr
                print(f"Learning rate decreased, new lr: ({self.lr:.2e})", flush=True)
                if self.lr < self.minlr:
                    print(
                        "Early STOP, after saving model for ",
                        self.Saved_counter,
                        " times",
                        flush=True,
                    )
                    self.Estop = True
                self.Estop_counter = 0
        else:
            if verbose:
                print(
                    f"Validation loss decreased ({self.Estop_min_loss:.6f} -->"
                    " {val_loss:.6f}).  Saving model ...",
                    flush=True,
                )
            self.Estop_best_score = score
            self.Estop_min_loss = val_loss
            self._save_ckpt()
            self.Saved_counter += 1
            self.Estop_counter = 0

    def _train_epoch(self, train_loader):
        """training epoch
        Args:
            train_loader (DataLoader): downstream train DataLoader
        Returns:
            (float): loss value
        """
        self.downstream.train()
        self.upstream.eval()
        if self.args.upstream_trainable:
            self.upstream.train()

        batch_loss = []
        for batch_idx, data_batch in enumerate(train_loader):
            data, spk_ID, target = data_batch
            data, target = Variable(data.to(self.args.device)), Variable(
                target.to(self.args.device)
            )
            self.optimizer.zero_grad()

            if self.args.upstream_trainable:
                features = self.upstream(
                    data, selected_encoded_layers=self.selected_encoded_layers
                )
            else:
                with torch.no_grad():
                    features = self.upstream(
                        data, selected_encoded_layers=self.selected_encoded_layers
                    )

            predicted_output = self.downstream(features)
            loss = self.loss(predicted_output, target.long())
            loss.backward()
            self.optimizer.step()
            batch_loss.append(loss.item())
        return np.mean(batch_loss)

    @torch.no_grad()
    def _test_epoch(self, test_loader, get_spk_target=False):
        """testing epoch
        Args:
            test_loader (DataLoader): upstream test DataLoader
            get_spk_target (bool, optional): If True returns speaker-level targets.
            Defaults to False.

        Returns:
            (tuple): a tuple containing:
                - (numpy.ndarray): loss
                - (numpy.ndarray): chunk-level accuracy
                - (numpy.ndarray): predicted speaker-level scores [N, num of classes]
        """
        self.downstream.eval()
        self.upstream.eval()
        batch_loss = []
        sum_acc = 0
        Spk_level_scores = defaultdict(list)
        Spk_target = defaultdict(list)
        for batch_idx, data_batch in enumerate(test_loader):
            data, spk_ID, target = data_batch
            data, target = data.to(self.args.device), target.to(self.args.device)
            features = self.upstream(
                data, selected_encoded_layers=self.selected_encoded_layers
            )
            predicted_output = self.downstream(features)
            loss = self.loss(predicted_output, target.long())
            batch_loss.append(loss.item())
            predicted_scores = nn.functional.softmax(
                predicted_output, dim=1
            )  # B X (number of classes)
            _, predicted_labels = torch.max(predicted_output, 1)
            sum_acc = sum_acc + torch.sum(predicted_labels.data == target.long())
            for b_ind in range(len(target)):
                Spk_level_scores[f"SPK_{int(spk_ID[b_ind])}"].append(
                    predicted_scores[b_ind, :].cpu().detach().numpy()
                )
                if get_spk_target:
                    Spk_target[f"SPK_{int(spk_ID[b_ind])}"].append(
                        target[b_ind].cpu().detach().numpy().item()
                    )
        test_chunk_acc = (
            (sum_acc.cpu().numpy().item()) / (test_loader.dataset.__len__())
        ) * 100
        test_loss = np.mean(batch_loss)
        spk_index_sorted = [
            int(i.split("SPK_")[1]) for i in list(Spk_level_scores.keys())
        ]
        predicted_score_spk_level = np.zeros(
            (len(spk_index_sorted), predicted_output.shape[1])
        )
        spk_target = np.zeros(len(spk_index_sorted))
        for idx, spk_indx in enumerate(spk_index_sorted):
            predicted_score_spk_level[idx, :] = np.mean(
                Spk_level_scores["SPK_{:d}".format(spk_indx)], axis=0
            )
            if get_spk_target:
                assert (
                    len(set(Spk_target["SPK_{:d}".format(spk_indx)])) == 1
                ), f" targets of spk index {spk_indx} are not in-agreement with utterance targets"
                spk_target[idx] = Spk_target["SPK_{:d}".format(spk_indx)][0]

        if get_spk_target:
            return (
                np.mean(batch_loss),
                test_chunk_acc,
                predicted_score_spk_level,
                spk_target,
            )
        else:
            return np.mean(batch_loss), test_chunk_acc, predicted_score_spk_level

    def train(self):
        """Main training for all epochs, and save the final model"""
        print(" - Loss Function: ", self.loss, "\n")
        if self.args.upstream_trainable:
            print(
                " - UPstream model will be fine-tuned along with downstream model training"
            )
        epoch_len = len(str(self.max_epoch))
        for epoch in range(self.max_epoch):
            train_loss = self._train_epoch(self.ds_train_loader)
            val_loss, val_chunk_acc, val_spk_scores = self._test_epoch(
                self.ds_val_loader
            )
            print(
                f"\nEpoch [{epoch:>{epoch_len}}/{self.max_epoch:>{epoch_len}}]  train_loss: {train_loss:.5f} "
                + f"  val_loss: {val_loss:.5f}, val chunk acc: {val_chunk_acc:.3f}",
                flush=True,
            )
            if self.args.valmonitor:
                self._earlystopping(
                    val_loss,
                    patience=5,
                    verbose=self.args.verbose,
                    delta=0,
                    lr_factor=0.5,
                )
                if self.Estop:
                    print(
                        f"---Early STOP, after saving model for {self.Saved_counter} times",
                        flush=True,
                    )
                    break
        if not self.args.valmonitor:
            self._save_ckpt()
        self._load_ckpt()

        (
            last_val_loss,
            last_val_chunk_acc,
            last_val_spk_scores,
            val_spk_target,
        ) = self._test_epoch(self.ds_val_loader, get_spk_target=True)
        min_val_loss_estop = self.Estop_min_loss
        print(
            f"\nFinal Model -->  val loss: {last_val_loss:.5f} val chunk acc: {last_val_chunk_acc:.3f} "
            + f"val loss Estop: {min_val_loss_estop:.5f}\n",
            flush=True,
        )
        self.val_chunk_acc = last_val_chunk_acc
        self.val_spk_scores = last_val_spk_scores
        self.val_spk_target = val_spk_target
        self._save_ckpt()

    def evaluation(self, set="test"):
        self._load_ckpt()
        test_loss, test_chunk_acc, test_spk_scores, test_spk_target = self._test_epoch(
            eval(f"self.ds_{set}_loader"), get_spk_target=True
        )
        print(
            f"\nEvaluation Final Model --> {set} loss: {test_loss:.5f} {set} chunk acc: {test_chunk_acc:.3f}\n",
            flush=True,
        )

        self.test_chunk_acc = test_chunk_acc
        self.test_spk_scores = test_spk_scores
        self.test_spk_target = test_spk_target
        return test_loss, test_chunk_acc, test_spk_scores, test_spk_target
