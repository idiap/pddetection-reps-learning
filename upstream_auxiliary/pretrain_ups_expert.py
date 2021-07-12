# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
# Defining UPSTREAM wrapper for training, and latent feature extraction module from upstream
# (to be used for auxiliary and downstream tasks)

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


import yaml
import torch
import torch.nn as nn
import importlib


class RepresentationPretrain(nn.Module):
    """
    Defining the upstream module for training (has saving and loading methods
    (for the checkpoint)), based on model configuration, initializes the upstream
    model (selecting the model and its architecture)
    """

    def __init__(self, freqlen, seqlen, upstream_config_file, **kwargs):
        """initializing the RepresentationPretrain module
        Args:
            freqlen (int): freq dim (number of features)
            seqlen (int): temporal dim (number of frames)
            upstream_config_file (dict): configuration for network and training params
        """
        super(RepresentationPretrain, self).__init__()
        self.freqlen = freqlen
        self.seqlen = seqlen
        self.Network_config = upstream_config_file
        print("\n[RepresentationPretrain] - Initializing model...")
        self.selected_model = self.Network_config["SelectedAENetwork"]
        model_config = self.Network_config[self.selected_model]
        Upstream = getattr(
            importlib.import_module("upstream_auxiliary.ups_model"), self.selected_model
        )
        self.model = Upstream(freqlen, seqlen, **model_config)
        print(
            "RepresentationPretrain: "
            + self.selected_model
            + " - Number of parameters: "
            + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
            + "\n\n"
        )
        print(self.model, flush=True)

    # Interface
    def load_model(self, all_states):
        """loading model from saved states
        Args:
            all_states (dict): dictionary of the states
        """
        self.model.EncoderNet.load_state_dict(all_states["EncoderNet"])
        self.model.DecoderNet.load_state_dict(all_states["DecoderNet"])

    # Interface
    def add_state_to_save(self, all_states):
        """Saving the states to the "all_states", saving encoder and decoder separately.
        Args:
            all_states (dict): dictionary of the states
        """
        all_states["EncoderNet"] = self.model.EncoderNet.state_dict()
        all_states["DecoderNet"] = self.model.DecoderNet.state_dict()
        all_states["AENetworkType"] = self.selected_model
        all_states["NetConfig"] = self.Network_config
        return all_states

    def forward(self, data, **kwargs):
        """
        Args:
            data:
                [B X D X T] input tensor
        Return:
            predicted output, encoded features
        """
        return self.model(data)

    def get_feature_dimension(self, selected_encoded_layers=-1):
        """getting the dimension of the upstream outputs
        Args:
            selected_encoded_layers (int or list, optional): Defaults to -1.
            if list: list of indices of the
            encoder layers (to be used later as coded information in downstream or
            auxiliary task), -1: index of last layer of encoder
            {for the indices, we also consider the index of nonlinearity modules,
            e.g., indexing all modules}
        Returns:
             (int): size of latent dimension
        """
        dummy_input = torch.rand(10, self.freqlen, self.seqlen)  # B X D X T
        if torch.cuda.is_available():  # a better way to do this?!
            dummy_input = dummy_input.to("cuda:0")
            model = self.model.to("cuda:0")
        else:
            model = self.model
        return model(dummy_input, selected_encoded_layers=selected_encoded_layers)[
            1
        ].shape[1]


class FeatureExtractionPretrained(nn.Module):
    """
    Defining the uspstream model encoder for feature extraction
    (for downstream usage, where the encoder part can also be fine-tuned).
    (has saving and loading methods (for the encoder checkpoint))
    Based on model configuration, initializes the model (selecting
    the model and its architecture).
    """

    def __init__(self, freqlen, seqlen, upstream_config_file):
        """initializing the FeatureExtractionPretrained module
        Args:
            freqlen (int): freq dim (number of features)
            seqlen (int): temporal dim (number of frames)
            upstream_config_file (dict): configuration for network and training params
        """
        super(FeatureExtractionPretrained, self).__init__()
        self.freqlen = freqlen
        self.seqlen = seqlen
        self.Network_config = upstream_config_file
        print("\n[FeatureExtractionPretrained] - Initializing model...")
        self.selected_model = self.Network_config["SelectedAENetwork"]
        model_config = self.Network_config[self.selected_model]
        UpstreamEncoder = getattr(
            importlib.import_module("upstream_auxiliary.ups_model"),
            f"{self.selected_model}Encoder",
        )
        self.model = UpstreamEncoder(self.freqlen, self.seqlen, **model_config)
        print(
            "Encoder Network: "
            + self.selected_model
            + " - Number of parameters: "
            + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
            + "\n\n"
        )

    # Interface
    def load_model(self, all_states):
        """loading encoder model from saved states
        Args:
            all_states (dict): dictionary of the states
        """
        self.model.EncoderNet.load_state_dict(all_states["EncoderNet"])

    # Interface
    def add_state_to_save(self, all_states):
        """Saving the states to the "all_states", saving only encoder
        (feature extractors).
        Args:
            all_states (dict): dictionary of the states
        """
        all_states["EncoderNet"] = self.model.EncoderNet.state_dict()
        all_states["AENetworkType"] = self.selected_model
        all_states["UpsNetConfig"] = self.Network_config
        return all_states

    def forward(self, data, selected_encoded_layers=-1):
        """
        Args:
            input (tensor): [B X D X T] a 3d-tensor representing the input features.
            selected_encoded_layers (list or -1): if list: list of indices
            of the encoder layers (to be used later as coded information in
            downstream or auxiliary task), -1: index of last layer of encoder
            {for the indices, we also consider mlp and nonlinearity modules}

        Return:
            hidden representation (tensor): [B X H] the latent representation
        """
        return self.model(data, selected_encoded_layers=selected_encoded_layers)

    def get_feature_dimension(self, selected_encoded_layers=-1):
        """getting the dimension of the latent features
        Args:
            selected_encoded_layers (int or list, optional):
            Defaults to -1. If list: list of indices of the encoder layers
            (to be used later as coded information in downstream or auxiliary task), -1:
            index of last layer of encoder {for the indices, we also consider the index
            of nonlinearity modules,e.g., indexing all modules}
        Returns:
             (int): size of the latent dimension
        """
        dummy_input = torch.rand(10, self.freqlen, self.seqlen)  # B X D X T
        if torch.cuda.is_available():  # a better way to do this?!
            dummy_input = dummy_input.to("cuda:0")
            model = self.model.to("cuda:0")
        else:
            model = self.model
        return model(
            dummy_input, selected_encoded_layers=selected_encoded_layers
        ).shape[1]


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
    from torch.utils.data import DataLoader, TensorDataset
    from audio.audio_dataset import OnlineAcousticDataset, OfflineAcousticDataset
    from itertools import cycle

    cpath = "../config/upstream_config.yaml"
    upstream_config = get_config_args(cpath)
    feat_path = "../config/audiofeat_config.yaml"
    feat_config = get_config_args(feat_path)
    file_path = "../preprocess/GITA/folds/test_fold1_fbank_offline.csv"
    transforms = create_transform(feat_config, 16000)

    # online testing
    dataset = OfflineAcousticDataset(None, upstream_config, file_path, feat_config)
    freqdim, seqlen = dataset.getDimension()
    ups_config_file = get_config_args(cpath)
    model_class = RepresentationPretrain(freqdim, seqlen, ups_config_file)
    model_encoder = FeatureExtractionPretrained(freqdim, seqlen, ups_config_file)

    selected_encoded_layers = [1, 2]
    # dummy_input = torch.rand(freqdim, seqlen)
    # out = model_encoder(dummy_input, selected_encoded_layers=[0,1]).shape[1]
    print(
        "get dim of feature encoded ",
        model_encoder.get_feature_dimension(
            selected_encoded_layers=selected_encoded_layers
        ),
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=10, shuffle=True, num_workers=2
    )
    for batch_idx, data_batch in enumerate(data_loader):
        print("batch index: ", batch_idx)
        data, ID, targets = data_batch
        output = model_class(data, selected_encoded_layers=selected_encoded_layers)
        encoded = model_encoder(data, selected_encoded_layers=selected_encoded_layers)
        print("output/latent size: ", output[0].shape, output[1].shape, ID, targets)
        print("first encoded size: ", encoded[0].shape)
        break

    all_states = {
        "Optimizer": " ",
        "Step": 100,
        "UpsConfig": get_config_args(cpath),
        "Runner": " ",
    }
    all_states = model_class.add_state_to_save(all_states)
    model_class.load_model(all_states)
    name = f"testAE.ckpt"
    Experiments_dir = "../test_experiment_files/"  # should go to database directory
    save_path = os.path.join(Experiments_dir, name)
    if not os.path.exists(Experiments_dir):
        os.makedirs(Experiments_dir)
    torch.save(all_states, save_path)

    # testing merging two different dataloaders
    dataset1 = TensorDataset(-1 * torch.arange(100), torch.zeros(100, 1))
    dataset2 = TensorDataset(torch.arange(200), torch.ones(200, 1))

    dataloaders1 = DataLoader(dataset1, batch_size=5, shuffle=True)
    dataloaders2 = DataLoader(dataset2, batch_size=5, shuffle=True)
    num_epochs = 1

    for epoch in range(num_epochs):
        dataloader_iterator = iter(dataloaders1)

        for i, data1 in enumerate(dataloaders2):
            try:
                data2 = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(dataloaders1)
                data2 = next(dataloader_iterator)
            print(i, data1[0], data2[0])

        zip_list = (
            zip(dataloaders1, cycle(dataloaders2))
            if len(dataloaders1) > len(dataloaders2)
            else zip(cycle(dataloaders1), dataloaders2)
        )

        for i, (data1, data2) in enumerate(zip_list):
            print("***\n", i, data1[0], data2[0])
