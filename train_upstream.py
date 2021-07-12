# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
# scripts for pre-training of upstream model (AE), we can have two auxiliary tasks performing on latent
# represnetations

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
import argparse
import shutil
import torch
import numpy as np
from upstream_auxiliary.ups_runner_merge import UPsRunner
from downstream.ds_runner import seed_torch


def remove_files(dir_path):
    if not os.path.isdir(dir_path):
        os.remove(dir_path)
    else:
        for filename in os.listdir(dir_path):
            filepath = os.path.join(dir_path, filename)
            try:
                shutil.rmtree(filepath)
            except (OSError):
                try:
                    os.remove(filepath)
                except:
                    pass


def copy_file(scr, dest):
    try:
        shutil.copy(scr, dest)
    except shutil.SameFileError:
        pass


def get_pretrain_args():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)
    # Additional info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0), flush=True)

    parser = argparse.ArgumentParser(
        description="arguments for upstream training experiments"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config/upstream_config.yaml",
        help="The path of upstream config yaml file for configuring the whole upstream training",
    )
    parser.add_argument(
        "-ac1",
        "--auxconfig1",
        default="config/upstream_auxiliary1_config.yaml",
        help="The path of 1st auxiliary config yaml file for configuring the 1st auxiliary training task",
    )
    parser.add_argument(
        "-ac2",
        "--auxconfig2",
        default="config/upstream_auxiliary2_config.yaml",
        help="The path of 2nd auxiliary config yaml file for configuring the 2nd auxiliary training task",
    )
    parser.add_argument(
        "-a",
        "--audio_config",
        default="config/audio_config.yaml",
        help="The path of config yaml file for acoustic feature extraction parameters",
    )
    parser.add_argument("-s", "--seed", type=int, default=0, help="seed number")
    parser.add_argument(
        "-d", "--device", type=str, default=device.type, help="cpu or cuda:0"
    )
    parser.add_argument(
        "-n",
        "--expname",
        default="test",
        help="Save experiment at /results/ups-{expname}/outputs/saved_model/",
    )
    parser.add_argument(
        "-auxtr",
        "--auxiltr",
        action="store_true",
        default=False,
        help="if True, train with the auxiliary tasks performing on latent features",
    )
    parser.add_argument(
        "-auxw1",
        "--auxlossw1",
        type=float,
        default=0,
        help=" + or -, weight of 1st auxiliary loss in total loss in training, \
                            for adversarial training the weight < 0",
    )
    parser.add_argument(
        "-auxw2",
        "--auxlossw2",
        type=float,
        default=0,
        help=" + or -, weight of 2nd auxiliary loss in total loss in training, \
                            for adversarial training the weight < 0",
    )
    parser.add_argument(
        "-vm",
        "--valmonitor",
        action="store_true",
        default=False,
        help="if True, validation set is used to monitor/early-stoping the training",
    )
    parser.add_argument(
        "--newinit",
        action="store_true",
        default=False,
        help="remove previous upstream experiment, otherwise initializes the\
             upstream model based on the last saved experiment",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="verbose for early stoping\
             (If True prints a message for each validation loss improvement)",
    )
    parser.add_argument("--fold", type=str, default="1", help="data fold number")

    if any("SPYDER" in name for name in os.environ) | any(
        "VSCODE_AGENT_FOLDER" in name for name in os.environ
    ):
        args = parser.parse_args(args=[])
    else:
        args, unknown = parser.parse_known_args()

    main_dir = f"results/{args.expname}/ups-{args.expname}/"
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    result_ups = f"results/{args.expname}/ups-{args.expname}/outputs/saved_model/UPs_NN_fold{args.fold}.ckpt"
    # remove ups files from previous experiments
    if args.newinit:
        if os.path.exists(result_ups):
            print("...removing previous upstream experiment", flush=True)
            # remove_files(result_ups)
            remove_files(main_dir)
        copy_file(args.config, main_dir + "/ups_config.yaml")
        copy_file(args.auxconfig1, main_dir + "/aux1_config.yaml")
        copy_file(args.auxconfig2, main_dir + "/aux2_config.yaml")
        copy_file(args.audio_config, main_dir + "/feat_config.yaml")
    args.config = main_dir + "/ups_config.yaml"
    args.auxconfig1 = main_dir + "/aux1_config.yaml"
    args.auxconfig2 = main_dir + "/aux2_config.yaml"
    args.audio_config = main_dir + "/feat_config.yaml"

    return args


def visualization(mat1, mat2, save_path, set_name="test"):
    """visualization of input matrices
    Args:
        mat1 (numpy.ndarray): input array 1
        mat2 (numpy.ndarray): input array 2
        save_path (str): path where figures are saved
        set_name (str, optional): name of data splits from which the sample is extracted. Defaults to 'test'.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f, ax = plt.subplots(2, 1)
    ax[0].set_title("original")
    im1 = ax[0].imshow(mat1, cmap="jet", aspect="auto")
    f.colorbar(im1, ax=ax[0], orientation="vertical")
    ax[1].set_title("reconstructed")
    im2 = ax[1].imshow(mat2, cmap="jet", aspect="auto")
    ax[1].set_xlabel(set_name)
    f.colorbar(im2, ax=ax[1], orientation="vertical")
    f.tight_layout()
    plt.subplots_adjust(top=0.85)
    f.savefig(save_path, bbox_inches="tight")


def main():
    args = get_pretrain_args()
    print("---args: ", args)
    seed_torch(args.seed)

    upsrunner = UPsRunner(args)
    upsrunner.train()
    Data_sets = ["test", "val", "train"]
    for sets in Data_sets:
        loss, aux_acc, aux_spk_scores, aux_spk_target = upsrunner.evaluation(
            set=sets, get_some_reconstruction=False
        )
        message = ""
        message += f"\n---------{sets} data---------- \n\n"
        loss_print = (
            ", ".join([f"{key}: {loss[key]:.5f}" for key in sorted(loss.keys())])
            + ", "
            + " " * 2
        )
        message += f"\tLOSS: {loss_print}\n"
        for num in [1, 2]:
            message += f'\tchunk-level aux{num} accuracy: {aux_acc[f"{num}"]} \n'
            message += f"\t----- speaker level evaluation \n"
            if args.auxiltr:
                if getattr(args, f"auxlossw{num}"):
                    predicted_label = np.argmax(aux_spk_scores[f"{num}"], axis=1)
                    spk_acc = (
                        100
                        * (predicted_label == aux_spk_target[f"{num}"]).sum()
                        / len(aux_spk_target[f"{num}"])
                    )
                    spk_sen = (
                        100
                        * (
                            (predicted_label == 1) * (aux_spk_target[f"{num}"] == 1)
                        ).sum()
                        / (aux_spk_target[f"{num}"] == 1).sum()
                    )
                    spk_spe = (
                        100
                        * (
                            (predicted_label == 0) * (aux_spk_target[f"{num}"] == 0)
                        ).sum()
                        / (aux_spk_target[f"{num}"] == 0).sum()
                    )
                    AUC = None  # np.round(roc_auc_score(aux_spk_target[f"{num}"], 1-spk_scores[:, 0]), decimals=3)
                    message += f"\tAcc: {spk_acc}\n"
                    message += f"\tSen: {spk_sen}\n"
                    message += f"\tSpe: {spk_spe}\n\t"
                    message += (
                        r"[Spe & Sen & Spe: {:.3f} & {:.3f} & {:.3f}]".format(
                            spk_acc, spk_sen, spk_spe
                        )
                        + " \n"
                    )
                    message += f"\tAUC: {AUC}\n"
                    message += "\tAvg scores of patients (or labels==1) (▼): {:f} +- {:f}\n".format(
                        np.mean(
                            aux_spk_scores[f"{num}"][(aux_spk_target[f"{num}"] == 1), 0]
                        ),
                        np.std(
                            aux_spk_scores[f"{num}"][(aux_spk_target[f"{num}"] == 1), 0]
                        ),
                    )
                    message += "\tAvg scores of controls (or labels==0) (▲): {:f} +- {:f}\n".format(
                        np.mean(
                            aux_spk_scores[f"{num}"][(aux_spk_target[f"{num}"] == 0), 0]
                        ),
                        np.std(
                            aux_spk_scores[f"{num}"][(aux_spk_target[f"{num}"] == 0), 0]
                        ),
                    )

        print(message, flush=True)

    for sets in Data_sets:
        saved_path = (
            f"results/{args.expname}/ups-{args.expname}/outputs/{sets}_{args.fold}.png"
        )
        (
            (loss, data, reconstructed, encoded),
            aux_acc,
            aux_spk_scores,
            aux_spk_target,
        ) = upsrunner.evaluation(
            set=sets, get_some_reconstruction=True, which_batch_idx=0
        )
        idx = 0  # selecting a sample
        visualization(
            data[idx, :, :], reconstructed[idx, :, :], saved_path, set_name=sets
        )
        # Note: sampling from the train set is random and changes across different training loops (if the training is stopped earlier than Max epoch), i.e., original train samples would be different)


if __name__ == "__main__":
    main()
