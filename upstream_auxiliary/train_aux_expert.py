# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
# Defining auxiliary classifier wrapper for training

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


class AuxiliaryTrain(nn.Module):
    """
    Defining the auxiliary module for training (has saving and loading methods
    (for the checkpoint)). Based on model configuration, initializes the model
    (selecting the model and its architecture)
    """

    def __init__(self, inputsize, auxconfig_file, **kwargs):
        """initializing the AuxiliaryTrain module

        Args:
            inputsize (int): size of input (latent features from upstream).
            auxconfig_file (dict): configuration for network and training params
        """
        super(AuxiliaryTrain, self).__init__()
        self.Network_config = auxconfig_file
        print("\n[Auxiliary Classifier] - Initializing model...")
        self.selected_model = self.Network_config["SelectedAuxiliaryNetwork"]
        model_config = self.Network_config[self.selected_model]
        Auxstream = getattr(
            importlib.import_module("upstream_auxiliary.aux_model"), self.selected_model
        )
        self.model = Auxstream(inputsize, **model_config)
        print(
            "AuxiliaryTrain: "
            + self.selected_model
            + " - Number of parameters: "
            + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
            + "\n\n"
        )

    # Interface
    def load_model(self, all_states, num):
        """loading model from saved states
        Args:
            all_states (dict): dictionary of the states
            num (int): number of the auxiliary task, e.g., 1 or 2
        """
        self.model.classifier.load_state_dict(all_states[f"ClassifierNet{num}"])

    # Interface
    def add_state_to_save(self, all_states, num):
        """Saving the states to the "all_states"
        Args:
            all_states (dict): dictionary of the states
            num (int): number of the auxiliary task, e.g., 1 or 2
        """
        all_states[f"ClassifierNet{num}"] = self.model.classifier.state_dict()
        all_states[f"AUXNetworkType{num}"] = self.selected_model
        all_states[f"AUXNetConfig{num}"] = self.Network_config
        return all_states

    def forward(self, data, **kwargs):
        """
        Args:
            data (tensor): [B X H] latent features as input
        Return:
            (tensor): [B X num of classes] predicted output classes
        """
        return self.model(data)
