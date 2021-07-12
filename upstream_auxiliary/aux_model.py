# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
# Defining AUXILIARY models

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
    def __init__(self):
        super(NonlinRem, self).__init__()

    def forward(self, data):
        """passing input as output
        Args:
            data (tensor): input
        Returns:
            (tensor): passing input data as output
        return data
        """


ACT2FN = {"relu": nn.ReLU(), "leaky-relu": nn.LeakyReLU(), " ": NonlinRem()}


class MLP(nn.Module):
    def __init__(self, inputsize, outputdim, hidden_units, nonlinearity, dropout_prob):
        """initializing the parameters of the MLP classifier model
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
        for layers_num in range(
            len(self.all_layers_uints)
        ):  # one layer would be added by default
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
            input (tensor): [B X H] a 2d-tensor representing the input features.
        Return:
            predicted_output (tensor): [B X num of classes] the predicted outputs.
        """
        return self.classifier(input)
