# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
# downstream models

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


class NonlinRem(nn.Module):
    """Removing nonlinearity module
    Inheritance:
        nn.Module:
    """

    def __init__(self):
        super(NonlinRem, self).__init__()

    def forward(self, data):
        """passing input as output
        Args:
            data (tensor): input
        Returns:
            (tensor): passing input data as output
        """
        return data


ACT2FN = {"relu": nn.ReLU(), "leaky-relu": nn.LeakyReLU(), " ": NonlinRem()}


class MLP(nn.Module):
    def __init__(self, inputsize, outputdim, hidden_units, nonlinearity, dropout_prob):
        """initializing the parameters of the MLP classifier model (downstream)
        Args:
            inputsize (int): size of input (latent features from upstream).
            outputdim (int): the output size.
            hidden_units (list): a list indicating the number of hidden units
            of linear layers before output layer. If list is empty means we have
            input > output (1 layer only)
            nonlinearity (str): str indicating nonlinearity in between layers
            dropout_prob (float): the dropout rate.
        """

        super(MLP, self).__init__()
        self.outputdim = outputdim
        self.hidden_units = hidden_units
        self.nonlinearity = ACT2FN[nonlinearity]
        self.dropout = nn.Dropout(dropout_prob)
        self.inputsize = inputsize
        self.all_layers_uints = self.hidden_units + [self.outputdim]

        layers = []
        for layers_num in range(len(self.all_layers_uints)):
            in_dim = (
                inputsize if layers_num == 0 else self.all_layers_uints[layers_num - 1]
            )
            out_dim = self.all_layers_uints[layers_num]
            if layers_num != len(self.all_layers_uints) - 1:
                layers += [nn.Linear(in_dim, out_dim), self.nonlinearity]
            else:
                layers += [nn.Linear(in_dim, out_dim)]
        layers = [self.dropout] + layers
        self.classifier = nn.Sequential(*layers)

    def forward(self, input):
        """
        Input:
            input [B X H]: a 2d-tensor representing the input features.
        Return:
            predicted_output [B X num of classes]: The predicted outputs.
        """
        return self.classifier(input)


if __name__ == "__main__":
    # ------------------------------ path change ----------------------------- #
    import sys
    from pathlib import Path

    file = Path(__file__).resolve()
    parent, root, subroot = file.parent, file.parents[1], file.parents[2]
    sys.path.append(str(subroot))
    sys.path.append(str(root))
    os.chdir(root)
    # ---------------------------------------------------------------------------- #
    from torch.utils.data import DataLoader
    from audio.audio_utils import get_config_args, create_transform
    from audio.audio_dataset import OfflineAcousticDataset
    import matplotlib.pyplot as plt
    from upstream_auxiliary.pretrain_ups_expert import FeatureExtractionPretrained

    cpath = "config/downstream_config.yaml"
    ds_config = get_config_args(cpath)
    cpath = "config/upstream_config.yaml"
    ups_config = get_config_args(cpath)
    feat_path = "config/audio_config.yaml"
    feat_config = get_config_args(feat_path)
    file_path = "preprocess/dummy_database/folds/test_fold1_fbank_offline.csv"
    transforms = create_transform(feat_config, 16000)

    # offline testing
    dataset = OfflineAcousticDataset(None, ds_config, file_path, feat_config)
    freqdim, seqlen = dataset.getDimension()
    Network_config = ds_config
    selected_ds_model = Network_config["Selecteddownstream"]
    ds_model_config = Network_config[selected_ds_model]

    upstream_model = FeatureExtractionPretrained(freqdim, seqlen, ups_config)

    encoded_feat_size = upstream_model.get_feature_dimension(
        selected_encoded_layers=[1, 2]
    )

    ds_model = eval(f"{selected_ds_model}")(encoded_feat_size, **ds_model_config)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=30, shuffle=True, num_workers=0
    )
    for batch_idx, data_batch in enumerate(data_loader):
        print("Batch index: ", batch_idx)
        data, ID, targets = data_batch
        print("Input size,IDs, Targets ", data.shape, ID, targets)
        # data, target = Variable(data), Variable(target)
        hidden = upstream_model(data, selected_encoded_layers=[1, 2])
        output = ds_model(hidden)
        print("output/hidden: ", output.shape, hidden.shape)
        break
