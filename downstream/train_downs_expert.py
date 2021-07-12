# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
# Defining downstream model wrapper for training

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


import torch
import torch.nn as nn
import importlib


class DownStreamTrain(nn.Module):
    """
    Defining the downstream module for training (has saving and loading methods (for the checkpoint))
    Based on model configuration, initializes the model (selecting the model and its architecture)
    """

    def __init__(self, inputsize, downstream_config_file, **kwargs):
        super(DownStreamTrain, self).__init__()
        self.Network_config = downstream_config_file
        print("\n[downstream classifier] - Initializing model...")
        self.selected_model = self.Network_config["Selecteddownstream"]
        model_config = self.Network_config[self.selected_model]
        Dstream = getattr(
            importlib.import_module("downstream.ds_model"), self.selected_model
        )
        # print(Dstream, model_config)
        self.model = Dstream(inputsize, **model_config)
        print(
            "DownStreamTrain: "
            + self.selected_model
            + " - Number of parameters: "
            + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
            + "\n\n"
        )

    # Interface
    def load_model(self, all_states):
        """loading model from saved states
        Args:
            all_states (dict): dictionary of the states
        """
        self.model.classifier.load_state_dict(all_states["ClassifierNet"])

    # Interface
    def add_state_to_save(self, all_states):
        """Saving the states to the "all_states"
        Args:
            all_states (dict): dictionary of the states
        """
        all_states["ClassifierNet"] = self.model.classifier.state_dict()
        all_states["DSNetworkType"] = self.selected_model
        all_states["DSNetConfig"] = self.Network_config
        return all_states

    def forward(self, data, **kwargs):
        """
        Args:
            data:
                (tensor) [B X H]: latent features as input
        Return:
            (tensor) [B X num of classes]: predicted output classes
        """
        return self.model(data)


if __name__ == "__main__":
    # ------------------------------ get rid of this ----------------------------- #
    from pathlib import Path
    import sys

    file = Path(__file__).resolve()
    parent, root = file.parent, file.parents[1]
    sys.path.append(str(root))
    # Additionally remove the current file's directory from sys.path
    try:
        sys.path.remove(str(parent))
    except ValueError:  # Already removed
        pass
    # ---------------------------------------------------------------------------- #
    from audio.audio_utils import get_config_args, create_transform
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from audio.audio_dataset import OnlineAcousticDataset, OfflineAcousticDataset
    from Upstream.pretrain_ups_expert import FeatureExtractionPretrained

    cpath = "../config/upstream_config.yaml"
    upstream_config = get_config_args(cpath)
    cpath = "../config/downstream_config.yaml"
    downstream_config = get_config_args(cpath)

    feat_path = "../config/audiofeat_config.yaml"
    feat_config = get_config_args(feat_path)
    file_path = "../preprocess/GITA/folds/test_fold1_fbank_offline.csv"
    transforms = create_transform(feat_config, 16000)

    # online testing
    dataset = OfflineAcousticDataset(None, upstream_config, file_path, feat_config)
    # dataset = OnlineAcousticDataset(
    #     transforms, upstream_config, file_path, feat_config)
    freqdim, seqlen = dataset.getDimension()

    upstream_model = FeatureExtractionPretrained(freqdim, seqlen, upstream_config)
    selected_encoded_layers = -1
    encoded_feat_size = upstream_model.get_feature_dimension(
        selected_encoded_layers=selected_encoded_layers
    )

    downs_model = DownStreamTrain(encoded_feat_size, downstream_config)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=10, shuffle=True, num_workers=2
    )
    for batch_idx, data_batch in enumerate(data_loader):
        print("batch index: ", batch_idx)
        data, ID, targets = data_batch
        latent = upstream_model(data, selected_encoded_layers=selected_encoded_layers)
        output = downs_model(latent)
        print("output/latent size: ", output.shape, latent.shape, ID, targets)
        break
