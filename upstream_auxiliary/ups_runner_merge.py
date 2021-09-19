# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
# defining Runner module for training/evaluating the upstream model; we can have
# two auxiliary tasks opertaing on latent (encoded) represnetation from upstream
# model during the training

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

import sys
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch
import random
import os
from collections import defaultdict
from .pretrain_ups_expert import RepresentationPretrain
from .train_aux_expert import AuxiliaryTrain

from audio.audio_utils import get_config_args, create_transform
from audio.audio_dataset import OnlineAcousticDataset, OfflineAcousticDataset
from downstream.ds_runner import seed_torch


class UPsRunner:
    """
    Runner for upstream training
    creating train/val/test data loaders, training/evaluation loops, checkpoint saving
    and loading (resume training)
    """

    def __init__(self, args):
        """Initializing the runner for training
        Args:
            args (argparse.Namespace): arguments for training including paths of config
            files, training params, fold, seed, ...
        """
        self.args = args
        self.config_path = self.args.config  # main upstream config
        self.auxconfig_path1 = self.args.auxconfig1  # 1st aux config
        self.auxconfig_path2 = self.args.auxconfig2  # 2nd aux config
        self.init_ckpt_path = (
            f"results/{self.args.expname}/ups-{self.args.expname}/outputs/saved_model/"
        )
        if not os.path.exists(self.init_ckpt_path):
            os.makedirs(self.init_ckpt_path, exist_ok=True)
        self.init_ckpt_path += f"UPs_NN_fold{self.args.fold}.ckpt"
        self.config_file = get_config_args(self.config_path)
        self.auxconfig_file1 = get_config_args(self.auxconfig_path1)
        self.auxconfig_file2 = get_config_args(self.auxconfig_path2)
        self.feat_config_path = self.args.audio_config

        self.train_auxil = self.args.auxiltr  # auxiliary task flag
        self.aux_loss_w1 = self.args.auxlossw1  # weight of 1st auxilary loss
        self.aux_loss_w2 = self.args.auxlossw2  # weight of 2nd auxilary loss
        if self.train_auxil:
            assert not (self.aux_loss_w1 == 0 and self.aux_loss_w2 == 0), (
                "the auxiliary training is set to True at least one of aux1 or aux2 "
                "loss weights must be non-zero"
            )
            if self.aux_loss_w1 != 0:
                print(f"AUX1 added to training with loss weight {self.aux_loss_w1}")
            if self.aux_loss_w2 != 0:
                print(f"AUX2 added to training with loss weight {self.aux_loss_w2}")
        else:
            self.aux_loss_w1 = 0
            self.aux_loss_w2 = 0
        data_loaders_keys = ["train_loader", "val_loader", "test_loader"]
        data_loaders_keys += [f"aux_train_loader{d}" for d in [1, 2]]
        data_loaders_keys += [f"aux_val_loader{d}" for d in [1, 2]]
        data_loaders_keys += [f"aux_test_loader{d}" for d in [1, 2]]

        self.loader_dict = dict.fromkeys(data_loaders_keys)

        print(f"\n----UPSTREAM DATA")

        self.loader_dict["train_loader"] = self._get_dataloader(
            self.config_path, set="train", shuffle=True
        )
        if self.args.valmonitor:
            self.loader_dict["val_loader"] = self._get_dataloader(
                self.config_path, set="val", shuffle=False
            )
            print("\n Monitoring the training based on validation set.")
        else:
            self.loader_dict["val_loader"] = self._get_dataloader(
                self.config_path, set="train", shuffle=False
            )
            print(
                "\n Training is not monitored; validation set = non-shuffled train set."
            )

        self.loader_dict["test_loader"] = self._get_dataloader(
            self.config_path, set="test", shuffle=False
        )

        self.freqlen, self.seqlen = self.loader_dict[
            "train_loader"
        ].dataset.getDimension()

        if self.train_auxil:
            for aux_num in [1, 2]:
                print(f"\n----AUX{aux_num} DATA")
                if getattr(self, f"aux_loss_w{aux_num}") != 0:
                    config_path_tmp = getattr(self, f"auxconfig_path{aux_num}")
                    self.loader_dict[
                        f"aux_train_loader{aux_num}"
                    ] = self._get_dataloader(config_path_tmp, set="train", shuffle=True)
                    # get dimension of aux data
                    aux_freqlen, auxseqlen = self.loader_dict[
                        f"aux_train_loader{aux_num}"
                    ].dataset.getDimension()

                if self.args.valmonitor:
                    if getattr(self, f"aux_loss_w{aux_num}") != 0:
                        config_path_tmp = getattr(self, f"auxconfig_path{aux_num}")
                        self.loader_dict[
                            f"aux_val_loader{aux_num}"
                        ] = self._get_dataloader(
                            config_path_tmp, set="val", shuffle=False
                        )
                        print(
                            f"\n Monitoring training aux{aux_num} based on"
                            " validation set."
                        )
                else:
                    for aux_num in [1, 2]:
                        if getattr(self, f"aux_loss_w{aux_num}") != 0:
                            config_path_tmp = getattr(self, f"auxconfig_path{aux_num}")
                            self.loader_dict[
                                f"aux_val_loader{aux_num}"
                            ] = self._get_dataloader(
                                config_path_tmp, set="train", shuffle=False
                            )
                            print(
                                f"\n Training aux{aux_num} is not monitored; "
                                "validation set=train set."
                            )

            for aux_num in [1, 2]:
                if getattr(self, f"aux_loss_w{aux_num}") != 0:
                    config_path_tmp = getattr(self, f"auxconfig_path{aux_num}")
                    self.loader_dict[
                        f"aux_test_loader{aux_num}"
                    ] = self._get_dataloader(config_path_tmp, set="test", shuffle=False)

            assert (aux_freqlen, auxseqlen) == (self.freqlen, self.seqlen,), (
                "the data dimension for upstream "
                "and auxiliary networks should be the same "
                f"{(aux_freqlen, auxseqlen)} != {(self.freqlen, self.seqlen)}"
            )

        # Using the same encoder layers for both auxiliaries (dominated by aux2 settings)
        self.selected_encoded_layers = (
            -1
        )  # if no auxiliary task is used, default is set to -1
        if self.aux_loss_w1 != 0:
            self.selected_encoded_layers = self.auxconfig_file1["SelectedEncodedLayers"]
        if self.aux_loss_w2 != 0:
            self.selected_encoded_layers = self.auxconfig_file2["SelectedEncodedLayers"]

        self.upstream = None
        for aux_num in [1, 2]:
            setattr(self, f"auxiliary{aux_num}", None)
        self.optimizer_AE = None
        self.optimizer_aux1 = None
        self.optimizer_aux2 = None

        self._get_upstream()  # initialize upstream model
        self.encoded_feat_size = self.upstream.get_feature_dimension(
            selected_encoded_layers=self.selected_encoded_layers
        )
        if self.train_auxil:
            if self.aux_loss_w1 != 0:
                # if False self.auxiliary1 = None
                self._get_auxiliary(aux_num=1)
            if self.aux_loss_w2 != 0:
                # if False self.auxiliary2 = None
                self._get_auxiliary(aux_num=2)

        self._get_optimizer()  # initialize optimizer
        # initializations for Earlystoping method if used
        self.Estop_counter = 0
        self.Estop_best_score = None
        self.Estop = False
        self.Estop_min_loss = np.Inf
        self.Saved_counter = 1

        # defining upstream and aux loss functions
        ups_loss = {"L1": nn.L1Loss(), "MSE": nn.MSELoss()}
        aux_loss = {"CE": nn.CrossEntropyLoss(), "MSE": nn.MSELoss()}
        self.ups_loss = ups_loss[self.config_file["runner"]["optimizer"]["loss"]]

        for aux_num in [1, 2]:
            attribute_tmp = getattr(self, f"auxconfig_file{aux_num}")
            setattr(
                self,
                f"aux_loss{aux_num}",
                aux_loss[attribute_tmp["runner"]["optimizer"]["loss"]],
            )

    def _get_upstream(self):
        """Set upstream model (initialized or loaded from previously saved model)
        based on upstream config
        """
        init_ckpt = (
            torch.load(self.init_ckpt_path, map_location="cpu")
            if os.path.exists(self.init_ckpt_path)
            else {}
        )
        init_upstream = init_ckpt.get("UpsConfig")
        if init_upstream:
            network_config = init_upstream
        else:
            network_config = self.config_file
        if self.upstream is None:  # initialize for the first time
            self.upstream = RepresentationPretrain(
                self.freqlen, self.seqlen, network_config
            ).to(self.args.device)
        if init_ckpt:  # loading from a saved model
            print(
                "[RepresentationPretrain] - Loading model weights from "
                "the init ckpt from: ",
                f"/results/{self.args.expname}/ups-{self.args.expname}"
                f"/outputs/saved_model/UPs_NN_fold{self.args.fold}.ckpt",
            )
            self.upstream.load_model(init_ckpt)

    def _get_auxiliary(self, aux_num=1):
        """Set auxiliary model (initialized or loaded from previously saved model)
        based on auxiliary config
        Args:
            aux_num (int, optional): number of the auxiliary task. Defaults to 1.
        """
        init_ckpt = (
            torch.load(self.init_ckpt_path, map_location="cpu")
            if os.path.exists(self.init_ckpt_path)
            else {}
        )
        init_aux = init_ckpt.get(f"AuxConfig{aux_num}")
        if init_aux:
            network_config = init_aux
        else:
            network_config = getattr(self, f"auxconfig_file{aux_num}")
        # initialize for the first time
        if getattr(self, f"auxiliary{aux_num}") is None:
            print(f"AUX{aux_num}")
            setattr(
                self,
                f"auxiliary{aux_num}",
                AuxiliaryTrain(self.encoded_feat_size, network_config).to(
                    self.args.device
                ),
            )
        if init_aux:  # loading from a saved model
            print(
                f"[AuxiliaryTrain{aux_num}] - Loading auxiliary model weights from "
                "the init ckpt from: ",
                f"results/{self.args.expname}/ups-{self.args.expname}"
                f"/outputs/saved_model/UPs_NN_fold{self.args.fold}.ckpt",
            )
            getattr(self, f"auxiliary{aux_num}").load_model(init_ckpt, aux_num)

    def _init_fn(self, worker_id):
        """Setting randomness seed for multi-process data loading
        Args:
            worker_id
        """
        seed_torch(self.args.seed)

    def _get_dataloader(self, config_path, set="train", shuffle=True):
        """Get dataloader (depending on config we get online or offline dataloader)
        Args:
            config_path (str): path of config file
            set (str, optional): data splits (train/test/validation). Defaults to 'train'.
            shuffle (bool, optional): shuffling data segments. Defaults to True.
        Returns:
            Dataloader (torch.utils.data.dataloader.DataLoader) object
        """
        feat_config = get_config_args(self.feat_config_path)
        feat_type = feat_config["feat_type"]
        ups_config = get_config_args(config_path)
        dataloader_config = ups_config.get("dataloader")
        print(f"\n\n{set} Data...")
        if dataloader_config.get("online"):
            file_path = os.path.join(
                dataloader_config.get("data_path"),
                f"{set}_fold{self.args.fold}_online.csv",
            )
            transforms = create_transform(feat_config, dataloader_config.get("fs"))
            print(f"\n{set} file reading fold {self.args.fold} [online]...")
            dataset = OnlineAcousticDataset(
                transforms, ups_config, file_path, feat_config
            )
        else:
            file_path = os.path.join(
                dataloader_config.get("data_path"),
                f"{set}_fold{self.args.fold}_{feat_type}_offline.csv",
            )
            print(f"\n{set} file reading fold {self.args.fold} [offline]...")
            dataset = OfflineAcousticDataset(None, ups_config, file_path, feat_config)

        Nettype = (
            "SelectedAENetwork"
            if "SelectedAENetwork" in ups_config.keys()
            else "SelectedAuxiliaryNetwork"
        )
        net = ups_config[Nettype]
        drop_last = ups_config[net].get("batchnorm", False)
        print("batchsize", dataloader_config.get("batch_size"))
        return DataLoader(
            dataset,
            batch_size=dataloader_config.get("batch_size"),
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=dataloader_config.get("num_workers"),
            worker_init_fn=self._init_fn,
        )

    def _get_optimizer(self):
        """Set optimizer models for upstream/aux1/aux2
        (initialized or loaded from previously saved model)
        """
        if self.optimizer_AE is None:  # initialize for the first time
            ups_optimizer_config = self.config_file.get("runner").get("optimizer")
            aux_optimizer_config1 = self.auxconfig_file1.get("runner").get("optimizer")
            aux_optimizer_config2 = self.auxconfig_file2.get("runner").get("optimizer")
            optimizer_name = ups_optimizer_config.get("type")
            # main learning rate (from upstream params)
            self.lr = float(ups_optimizer_config["lr"])
            self.auxlr_ratio1 = float(
                aux_optimizer_config1["auxlr_ratio"]
            )  # lr ratio for aux1
            self.auxlr_ratio2 = float(
                aux_optimizer_config2["auxlr_ratio"]
            )  # lr ratio for aux1

            model_params_ups = list(self.upstream.parameters())
            param_list = [
                {"params": model_params_ups, "lr": self.lr, "name": "UPs-model"}
            ]

            if optimizer_name == "SGD":
                self.optimizer_AE = eval(f"torch.optim.{optimizer_name}")(
                    param_list, momentum=ups_optimizer_config.get("momentum")
                )
            else:
                self.optimizer_AE = eval(f"torch.optim.{optimizer_name}")(param_list)

            if self.train_auxil:
                if self.aux_loss_w1 != 0:
                    assert (
                        self.auxiliary1 is not None
                    ), "define auxiliary model 1 first before defining the optimizer."
                    model_params_aux1 = list(self.auxiliary1.parameters())
                    param_list1 = [
                        {
                            "params": model_params_aux1,
                            "lr": self.lr * self.auxlr_ratio1,
                            "name": "AUX1-model",
                        }
                    ]
                    if optimizer_name == "SGD":
                        self.optimizer_aux1 = eval(f"torch.optim.{optimizer_name}")(
                            param_list1, momentum=ups_optimizer_config.get("momentum")
                        )
                    else:
                        self.optimizer_aux1 = eval(f"torch.optim.{optimizer_name}")(
                            param_list1
                        )
                if self.aux_loss_w2 != 0:
                    assert (
                        self.auxiliary2 is not None
                    ), "define auxiliary model 2 first before defining the optimizer."
                    model_params_aux2 = list(self.auxiliary2.parameters())
                    param_list2 = [
                        {
                            "params": model_params_aux2,
                            "lr": self.lr * self.auxlr_ratio2,
                            "name": "AUX2-model",
                        }
                    ]
                    if optimizer_name == "SGD":
                        self.optimizer_aux2 = eval(f"torch.optim.{optimizer_name}")(
                            param_list2, momentum=ups_optimizer_config.get("momentum")
                        )
                    else:
                        self.optimizer_aux2 = eval(f"torch.optim.{optimizer_name}")(
                            param_list2
                        )

            self.max_epoch = self.config_file["runner"]["Max_epoch"]
            self.minlr = float(ups_optimizer_config["minlr"])
        init_ckpt = (
            torch.load(self.init_ckpt_path, map_location="cpu")
            if os.path.exists(self.init_ckpt_path)
            else {}
        )
        init_optimizer_ae = init_ckpt.get("Optimizer_AE")
        if init_optimizer_ae:
            print(
                "\n[Runner Optimizer] - Loading upstream optimizer weights from the"
                "init ckpt from: ",
                f"results/{self.args.expname}/ups-{self.args.expname}"
                f"/outputs/saved_model/UPs_NN_fold{self.args.fold}.ckpt",
            )
            self.optimizer_AE.load_state_dict(init_optimizer_ae)
        init_optimizer_aux1 = init_ckpt.get("Optimizer_AUX1")
        init_optimizer_aux2 = init_ckpt.get("Optimizer_AUX2")
        if (init_optimizer_aux1 is not None) & (self.train_auxil):
            self.optimizer_aux1.load_state_dict(init_optimizer_aux1)
            print(
                "\n[Runner Optimizer] - Loading auxiliary1 optimizer"
                " weights from the init ckpt..."
            )
        if (init_optimizer_aux2 is not None) & (self.train_auxil):
            print(
                "\n[Runner Optimizer] - Loading auxiliary2 optimizer"
                " weights from the init ckpt..."
            )
            self.optimizer_aux2.load_state_dict(init_optimizer_aux2)

    def _save_ckpt(self):
        """save models and optimizers params into the checkpoint"""
        if os.path.exists(self.init_ckpt_path):
            init_ckpt = torch.load(self.init_ckpt_path, map_location="cpu")
        else:
            init_ckpt = {
                "Optimizer_AE": None,
                "Optimizer_AUX1": None,
                "Optimizer_AUX2": None,
                "UpsConfig": self.config_file,
                "AuxConfig1": self.auxconfig_file1,
                "AuxConfig2": self.auxconfig_file2,
            }
        init_ckpt = self.upstream.add_state_to_save(init_ckpt)
        if self.aux_loss_w1 != 0:
            init_ckpt = self.auxiliary1.add_state_to_save(init_ckpt, 1)
        if self.aux_loss_w2 != 0:
            init_ckpt = self.auxiliary2.add_state_to_save(init_ckpt, 2)
        init_ckpt["Optimizer_AE"] = self.optimizer_AE.state_dict()
        if self.optimizer_aux1 is not None:
            init_ckpt["Optimizer_AUX1"] = self.optimizer_aux1.state_dict()
        if self.optimizer_aux2 is not None:
            init_ckpt["Optimizer_AUX2"] = self.optimizer_aux2.state_dict()
        torch.save(init_ckpt, self.init_ckpt_path)

    def _load_ckpt(self):
        """load models and optimizers params from the checkpoint"""
        init_ckpt = torch.load(self.init_ckpt_path, map_location="cpu")
        self.upstream.load_model(init_ckpt)
        if self.auxiliary1 is not None:
            self.auxiliary1.load_model(init_ckpt, 1)
        if self.auxiliary2 is not None:
            self.auxiliary2.load_model(init_ckpt, 2)
        init_optimizer = init_ckpt.get("Optimizer_AE")
        self.optimizer_AE.load_state_dict(init_optimizer)
        if self.optimizer_aux1 is not None:
            init_optimizer = init_ckpt.get("Optimizer_AUX1")
            self.optimizer_aux1.load_state_dict(init_optimizer)
        if self.optimizer_aux2 is not None:
            init_optimizer = init_ckpt.get("Optimizer_AUX2")
            self.optimizer_aux2.load_state_dict(init_optimizer)

    def _earlystopping(
        self, val_loss, patience=5, verbose=True, delta=0, lr_factor=0.5
    ):
        """Early stops the training if validation loss doesn't improve after a given
        patience.
        it saves best model based on validation on checkpoint path
        Adapted from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
        Args:
            val_loss (float): validation loss
            patience (int, optional): How long to wait after last time validation
            loss improved. Defaults to 5.
            verbose (bool, optional): If True, prints a message for each validation
            loss improvement. Defaults to True.
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
                    f" {val_loss:.6f}). Saving model ...",
                    flush=True,
                )
            self.Estop_min_loss = val_loss
            self._save_ckpt()
            self.Saved_counter += 1

        elif score < self.Estop_best_score + self.Estop_delta:
            self.Estop_counter += 1
            if verbose:
                print(f"EarlyStopping counter: {self.Estop_counter} out of {patience}")
            if self.Estop_counter >= patience:
                """load previously saved models and optimizer while multiplying the
                leanring rate by lr_factor"""
                print("No improvement since last best model", flush=True)
                self._load_ckpt()
                self.lr = self.lr * lr_factor
                for g in self.optimizer_AE.param_groups:  # decreasing the learning rate
                    g["lr"] = self.lr
                if self.train_auxil:
                    if self.aux_loss_w1 != 0:
                        for (
                            g
                        ) in (
                            self.optimizer_aux1.param_groups
                        ):  # decreasing the learning rate
                            g["lr"] = (self.lr) * self.auxlr_ratio1
                    if self.aux_loss_w2 != 0:
                        for (
                            g
                        ) in (
                            self.optimizer_aux2.param_groups
                        ):  # decreasing the learning rate
                            g["lr"] = (self.lr) * self.auxlr_ratio2
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
                    f" {val_loss:.6f}). Saving model ...",
                    flush=True,
                )
            self.Estop_best_score = score
            self.Estop_min_loss = val_loss
            # save model and optimizer
            self._save_ckpt()
            self.Saved_counter += 1
            self.Estop_counter = 0

    def _get_datasets_cycled_batch(self, upsloader, aux1loader, aux2loader):
        """generator for getting upstream data and aux1 & aux2 data (if they exist)
        since they can be of different size, the smaller dataloaders will be
        circulated while longest dataloader finishes iterating through batches of its
        data
        Args:
            upsloader (DataLoader): upstream dataloader object
            aux1 laoder (DataLoader): auxiliary 1 dataloader object
            aux2 loader (DataLoader): auxiliary 2 dataloader object
        Yields:
            (tuple): a tuple containing:
                - batch_idx (int): batch index (considering the longest dataloader)
                - ups_data_batch (tensor): batch of upstream data
                - aux1_data_batch (tensor): batch of auxiliary 1 data
                - aux2_data_batch (tensor): batch of auxiliary 2 data
        """
        if aux1loader is None:
            aux1loader = [(None, None, None)]
        if aux2loader is None:
            aux2loader = [(None, None, None)]
        if self.train_auxil:
            len_list = [len(upsloader), len(aux1loader), len(aux2loader)]
            dataloader_list = np.array(
                [upsloader, aux1loader, aux2loader], dtype=object
            )
            dataloader_len = list(map(lambda x: len(x), dataloader_list))
            # find longest
            loader_order = np.argsort(np.argsort(dataloader_len)[::-1])
            sorted_dataloader_list = dataloader_list[np.argsort(dataloader_len)]
            dataloader_iterator2 = iter(sorted_dataloader_list[0])  # short one
            dataloader_iterator1 = iter(sorted_dataloader_list[1])  # medium
            for batch_idx, batch_longloader in enumerate(sorted_dataloader_list[2]):
                try:
                    batch_medloader = next(dataloader_iterator1)
                except StopIteration:
                    dataloader_iterator1 = iter(sorted_dataloader_list[1])
                    batch_medloader = next(dataloader_iterator1)
                try:
                    batch_shortloader = next(dataloader_iterator2)
                except StopIteration:
                    dataloader_iterator2 = iter(sorted_dataloader_list[0])
                    batch_shortloader = next(dataloader_iterator2)
                returned_batch = np.array(
                    [batch_longloader, batch_medloader, batch_shortloader], dtype=object
                )
                returned_batch = returned_batch[loader_order]
                yield batch_idx, returned_batch[0], returned_batch[1], returned_batch[2]
        else:  # only upstream data is available
            for batch_idx, batch_longloader in enumerate(upsloader):
                yield batch_idx, batch_longloader

    def _train_epoch(self, train_loader, aux1_train_loader, aux2_train_loader):
        """training epoch
        Args:
            train_loader (DataLoader): upstream train DataLoader
            aux1_train_loader (DataLoader): auxiliary 1 train DataLoader
            aux2_train_loader (DataLoader): auxiliary 2 train DataLoader

        Returns:
            Average of losses (dict): 3 types of loss
        """
        batch_loss = defaultdict(list)
        self.upstream.train()
        if self.train_auxil:
            if self.aux_loss_w1 != 0:
                self.auxiliary1.train()
            if self.aux_loss_w2 != 0:
                self.auxiliary2.train()

            data_generator = self._get_datasets_cycled_batch(
                train_loader, aux1_train_loader, aux2_train_loader
            )
            for (
                batch_idx,
                data_batch,
                aux1_data_batch,
                aux2_data_batch,
            ) in data_generator:
                data, data_spk_ID, data_target = data_batch
                # if not given should be None, None, None
                aux1_data, aux1_data_spk_ID, aux1_data_target = aux1_data_batch
                aux2_data, aux2_data_spk_ID, aux2_data_target = aux2_data_batch

                data, data_target = Variable(data.to(self.args.device)), Variable(
                    data_target.to(self.args.device)
                )
                aux1_data, aux1_data_target = (
                    (
                        Variable(aux1_data.to(self.args.device)),
                        Variable(aux1_data_target.to(self.args.device)),
                    )
                    if self.aux_loss_w1 != 0
                    else (None, None)
                )
                aux2_data, aux2_data_target = (
                    (
                        Variable(aux2_data.to(self.args.device)),
                        Variable(aux2_data_target.to(self.args.device)),
                    )
                    if self.aux_loss_w2 != 0
                    else (None, None)
                )
                # ---------------- For adversarial training self.aux_loss_w1 < 0 ---------------- #
                self.optimizer_AE.zero_grad()
                predicted_AE_output, encoded = self.upstream(
                    data, selected_encoded_layers=self.selected_encoded_layers
                )
                ups_loss = self.ups_loss(data, predicted_AE_output)

                total_loss = (
                    1 - np.abs(self.aux_loss_w1) - np.abs(self.aux_loss_w2)
                ) * ups_loss
                aux_AE_loss_vals = 0
                if self.aux_loss_w1 != 0:
                    self.optimizer_aux1.zero_grad()
                    predicted_AE_aux1_output, aux1_encoded = self.upstream(
                        aux1_data, selected_encoded_layers=self.selected_encoded_layers
                    )
                    predicted_aux1_output = self.auxiliary1(aux1_encoded)
                    aux_loss1 = self.aux_loss1(
                        predicted_aux1_output, aux1_data_target.long()
                    )
                    aux1_AE_loss = self.ups_loss(aux1_data, predicted_AE_aux1_output)
                    total_loss += self.aux_loss_w1 * aux_loss1
                    aux_AE_loss_vals = 0.5 * aux1_AE_loss.item()

                if self.aux_loss_w2 != 0:
                    self.optimizer_aux2.zero_grad()
                    predicted_AE_aux2_output, aux2_encoded = self.upstream(
                        aux2_data, selected_encoded_layers=self.selected_encoded_layers
                    )
                    predicted_aux2_output = self.auxiliary2(aux2_encoded)
                    aux_loss2 = self.aux_loss2(
                        predicted_aux2_output, aux2_data_target.long()
                    )
                    aux2_AE_loss = self.ups_loss(aux2_data, predicted_AE_aux2_output)
                    total_loss += self.aux_loss_w2 * aux_loss2
                    aux_AE_loss_vals += 0.5 * aux2_AE_loss.item()

                # --------------------------------- train AE --------------------------------- #
                total_loss.backward()
                self.optimizer_AE.step()
                # --------------------------------- train auxiliary -------------------------- #
                self.optimizer_AE.zero_grad()
                if self.aux_loss_w1 != 0:
                    self.optimizer_aux1.zero_grad()
                    predicted_AE_aux1_output, aux1_encoded = self.upstream(
                        aux1_data, selected_encoded_layers=self.selected_encoded_layers
                    )
                    predicted_aux1_output = self.auxiliary1(aux1_encoded)
                    aux1_loss_new = self.aux_loss1(
                        predicted_aux1_output, aux1_data_target.long()
                    )
                    aux1_loss_new.backward()
                    self.optimizer_aux1.step()
                    self.optimizer_aux1.zero_grad()
                    batch_loss["aux1"].append(aux1_loss_new.item())

                if self.aux_loss_w2 != 0:
                    self.optimizer_aux2.zero_grad()
                    predicted_AE_aux2_output, aux2_encoded = self.upstream(
                        aux2_data, selected_encoded_layers=self.selected_encoded_layers
                    )
                    predicted_aux2_output = self.auxiliary2(aux2_encoded)
                    aux2_loss_new = self.aux_loss2(
                        predicted_aux2_output, aux2_data_target.long()
                    )
                    aux2_loss_new.backward()
                    self.optimizer_aux2.step()
                    self.optimizer_aux2.zero_grad()
                    batch_loss["aux2"].append(aux2_loss_new.item())

                batch_loss["total"].append(total_loss.item())
                batch_loss["ups_AE"].append(ups_loss.item())
                batch_loss["aux_AE"].append(aux_AE_loss_vals)
            avg_loss = {k: np.mean(batch_loss[k]) for k in batch_loss}
        else:
            data_generator = self._get_datasets_cycled_batch(train_loader, None, None)
            for batch_idx, data_batch in data_generator:
                data, data_spk_ID, data_target = data_batch
                data, data_target = Variable(data.to(self.args.device)), Variable(
                    data_target.to(self.args.device)
                )

                self.optimizer_AE.zero_grad()
                predicted_AE_output, encoded = self.upstream(
                    data, selected_encoded_layers=self.selected_encoded_layers
                )
                ups_loss = self.ups_loss(data, predicted_AE_output)
                ups_loss.backward()
                self.optimizer_AE.step()
                batch_loss["ups_AE"].append(ups_loss.item())
                batch_loss["total"].append(ups_loss.item())
            avg_loss = {k: np.mean(batch_loss[k]) for k in batch_loss}
        return avg_loss

    @torch.no_grad()
    def _test_epoch(
        self, test_loader, aux1_test_loader, aux2_test_loader, get_spk_level_pred=False
    ):
        """testing epoch

        Args:
            test_loader (DataLoader): upstream test DataLoader
            aux1_test_loader (DataLoader): auxiliary 1 test DataLoader
            aux2_test_loader (DataLoader): auxiliary 2 test DataLoader
            get_spk_level_pred (bool, optional): If True computes speaker-level
            predictions. Defaults to False.

        Returns:
            (tuple): a tuple containing:
                - (dict): avg of losses
                - (dict): chunk-level accuracy for aux1 & aux2
                - (dict): speaker-level predicted scores for aux1 & aux2
                - (dict): speaker targets/labels for aux1 & aux2

        """
        batch_loss = defaultdict(list)
        self.upstream.eval()
        for batch_idx, data_batch in enumerate(test_loader):
            data, spk_ID, data_target = data_batch
            data, data_target = data.to(self.args.device), data_target.to(
                self.args.device
            )
            predicted_output, encoded = self.upstream(data)
            loss = self.ups_loss(data, predicted_output)
            batch_loss["ups_AE"].append(loss.item())
            batch_loss["total"].append(loss.item())
        test_chunk_acc1, predicted_score_spk_level1, spk_target1 = (
            {"1": None, "2": None},
            {"1": None, "2": None},
            {"1": None, "2": None},
        )
        if self.train_auxil:
            for num in [1, 2]:
                sum_acc = 0
                Spk_level_scores = defaultdict(list)
                Spk_target = defaultdict(list)
                if getattr(self, f"aux_loss_w{num}") != 0:
                    getattr(self, f"auxiliary{num}").eval()
                    for batch_idx, aux_data_batch in enumerate(
                        eval(f"aux{num}_test_loader")
                    ):
                        aux_data, spk_ID, aux_data_target = aux_data_batch
                        aux_data, aux_data_target = aux_data.to(
                            self.args.device
                        ), aux_data_target.to(self.args.device)
                        predicted_AE_aux_output, aux_encoded = self.upstream(
                            aux_data,
                            selected_encoded_layers=self.selected_encoded_layers,
                        )
                        predicted_aux_output = getattr(self, f"auxiliary{num}")(
                            aux_encoded
                        )
                        aux_loss = getattr(self, f"aux_loss{num}")(
                            predicted_aux_output, aux_data_target.long()
                        )
                        aux_AE_loss = self.ups_loss(aux_data, predicted_AE_aux_output)
                        batch_loss["aux_AE"].append(aux_AE_loss.item())
                        batch_loss[f"aux{num}"].append(aux_loss.item())
                        # compute accuracy of auxiliary classifier
                        predicted_scores = nn.functional.softmax(
                            predicted_aux_output, dim=1
                        )  # B X (number of classes)
                        _, predicted_labels = torch.max(predicted_aux_output, 1)
                        sum_acc = sum_acc + torch.sum(
                            predicted_labels.data == aux_data_target.long()
                        )
                        if get_spk_level_pred:
                            for b_ind in range(len(aux_data_target)):
                                Spk_level_scores[f"SPK_{int(spk_ID[b_ind])}"].append(
                                    predicted_scores[b_ind, :].cpu().detach().numpy()
                                )
                                Spk_target[f"SPK_{int(spk_ID[b_ind])}"].append(
                                    aux_data_target[b_ind].cpu().detach().numpy().item()
                                )

                    test_chunk_acc1[f"{num}"] = (
                        (sum_acc.cpu().numpy().item())
                        / (eval(f"aux{num}_test_loader").dataset.__len__())
                    ) * 100
                    if get_spk_level_pred:
                        spk_index_sorted = [
                            int(i.split("SPK_")[1])
                            for i in list(Spk_level_scores.keys())
                        ]
                        predicted_score_spk_level1[f"{num}"] = np.zeros(
                            (len(spk_index_sorted), predicted_aux_output.shape[1])
                        )
                        spk_target1[f"{num}"] = np.zeros(len(spk_index_sorted))
                        for idx, spk_indx in enumerate(spk_index_sorted):
                            predicted_score_spk_level1[f"{num}"][idx, :] = np.mean(
                                Spk_level_scores["SPK_{:d}".format(spk_indx)], axis=0
                            )
                            assert (
                                len(set(Spk_target["SPK_{:d}".format(spk_indx)])) == 1
                            ), (
                                f" targets of spk index {spk_indx} are not in-agreement"
                                " with utterance targets"
                            )
                            spk_target1[f"{num}"][idx] = Spk_target[
                                "SPK_{:d}".format(spk_indx)
                            ][0]
            total_aux = 0
            for num in [1, 2]:
                if getattr(self, f"aux_loss_w{num}") != 0:
                    total_aux += getattr(self, f"aux_loss_w{num}") * np.mean(
                        batch_loss[f"aux{num}"]
                    )
            batch_loss["total"] = [
                (1 - np.abs(self.aux_loss_w1) - np.abs(self.aux_loss_w2))
                * np.mean(batch_loss["ups_AE"])
                + total_aux
            ]

        return (
            {k: np.mean(batch_loss[k]) for k in batch_loss},
            test_chunk_acc1,
            predicted_score_spk_level1,
            spk_target1,
        )

    def train(self):
        """Main training for all epochs, and save the final model"""
        print(" - AE Loss Function: ", self.ups_loss, "\n")
        if self.train_auxil:
            if self.aux_loss_w1 != 0:
                print(" - AUX1 Loss Function: ", self.aux_loss1, "\n")
            if self.aux_loss_w2 != 0:
                print(" - AUX2 Loss Function: ", self.aux_loss2, "\n")

        epoch_len = len(str(self.max_epoch))
        for epoch in range(self.max_epoch):
            train_loss = self._train_epoch(
                self.loader_dict["train_loader"],
                self.loader_dict["aux_train_loader1"],
                self.loader_dict["aux_train_loader2"],
            )
            val_loss, aux_chunk_acc, _, _ = self._test_epoch(
                self.loader_dict["val_loader"],
                self.loader_dict["aux_val_loader1"],
                self.loader_dict["aux_val_loader2"],
            )
            train_print = (
                "-TRAIN: "
                + ", ".join(
                    [
                        f" {key}: {train_loss[key]:.5f}"
                        for key in sorted(train_loss.keys())
                    ]
                )
                + ",  "
            )
            val_print = (
                "-VAL: "
                + ", ".join(
                    [f"{key}: {val_loss[key]:.5f}" for key in sorted(val_loss.keys())]
                )
                + ",  "
            )
            if self.train_auxil:
                if self.aux_loss_w1 != 0:
                    val_print += (
                        f'val aux1 chunk accuracy {aux_chunk_acc["1"]},' + " " * 2
                    )
                if self.aux_loss_w2 != 0:
                    val_print += (
                        f'val aux2 chunk accuracy {aux_chunk_acc["2"]},' + " " * 2
                    )
            print(
                f" Epoch [{epoch:>{epoch_len}}/{self.max_epoch:>{epoch_len}}]"
                + train_print
                + val_print,
                flush=True,
            )
            if self.args.valmonitor:
                self._earlystopping(
                    val_loss["total"],
                    patience=5,
                    verbose=self.args.verbose,
                    delta=0,
                    lr_factor=0.5,
                )
                if self.Estop:
                    print(
                        f"---Early STOP, after saving model for {self.Saved_counter}"
                        " times",
                        flush=True,
                    )
                    break
        if not self.args.valmonitor:  # save model at the last epoch
            self._save_ckpt()
        self._load_ckpt()
        last_val_loss, last_val_acc, _, _ = self._test_epoch(
            self.loader_dict["val_loader"],
            self.loader_dict["aux_val_loader1"],
            self.loader_dict["aux_val_loader2"],
        )
        min_val_loss_estop = self.Estop_min_loss
        val_print = (
            ", ".join(
                [
                    f"{key}: {last_val_loss[key]:.5f}"
                    for key in sorted(last_val_loss.keys())
                ]
            )
            + ",  "
        )
        if self.train_auxil:
            if self.aux_loss_w1 != 0:
                val_print += f', val aux1 chunk accuracy {last_val_acc["1"]}, '
            if self.aux_loss_w2 != 0:
                val_print += f'val aux2 chunk accuracy {last_val_acc["2"]}, '
        print(
            f"\nFinal Model VAL: -->  "
            + val_print
            + f"val_loss_Estop: {min_val_loss_estop:.5f}\n",
            flush=True,
        )
        self._save_ckpt()

    def evaluation(self, set="test", get_some_reconstruction=False, which_batch_idx=0):
        """evaluation final (saved) model

        Args:
            set (str, optional): data splits (train/test/validation). Defaults to 'test'.
            get_some_reconstruction (bool, optional): If True, samples of reconstructed
            input from the decoders will be returned. Defaults to False.
            which_batch_idx (int, optional): the batch index for computing a sample of
            reconstructed input. Defaults to 0.

        Returns:
            (tuple): a tuple containing:
                - (tuple): ((dict): losses, (numpy.ndarray): input data, (numpy.ndarray):
                reconstructed inputs, (numpy.ndarray): encoded representation)
                - (dict): chunk-level accuracy for aux1 & aux2
                - (dict): speaker-level predicted scores for aux1 & aux2
                - (dict): speaker targets/labels for aux1 & aux2

        """
        self._load_ckpt()
        loss, aux_acc, aux_spk_scores, aux_spk_target = self._test_epoch(
            self.loader_dict[f"{set}_loader"],
            self.loader_dict[f"aux_{set}_loader1"],
            self.loader_dict[f"aux_{set}_loader2"],
            get_spk_level_pred=True,
        )
        loss_print = (
            ", ".join([f"{key}: {loss[key]:.5f}" for key in loss.keys()])
            + f', val aux1 chunk accuracy {aux_acc["1"]}, '
            + f'val aux2 chunk accuracy {aux_acc["2"]}, '
        )
        print(f"\nFinal Model --> {set} " + loss_print + "\n", flush=True)
        if get_some_reconstruction:
            self.upstream.eval()
            # choosing 10th sample in the batch
            samples_num = 10 if 10 < self.loader_dict[f"{set}_loader"].batch_size else 1
            for batch_idx, data_batch in enumerate(self.loader_dict[f"{set}_loader"]):
                if batch_idx == which_batch_idx:
                    data, spk_ID, target = data_batch
                    data, target = data.to(self.args.device), target
                    predicted_output, encoded = self.upstream(data[:samples_num, :, :])
                    break
            return (
                (
                    loss,
                    data[:samples_num, :, :].cpu().detach().numpy(),
                    predicted_output.cpu().detach().numpy(),
                    encoded.cpu().detach().numpy(),
                ),
                aux_acc,
                aux_spk_scores,
                aux_spk_target,
            )
        else:
            return loss, aux_acc, aux_spk_scores, aux_spk_target
