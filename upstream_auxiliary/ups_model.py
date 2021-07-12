"""*********************************************************************************************"""
# Defining UPSTREAM models

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
import numpy as np


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
            data {tensor}: input
        Returns:
            {tensor}: passing input data as output
        """
        return data


ACT2FN = {"relu": nn.ReLU(), "leaky-relu": nn.LeakyReLU(), " ": NonlinRem()}


class AEMLPNet(nn.Module):
    def __init__(
        self, freqlen, seqlen, hidden_size, hidden_units, nonlinearity, dropout_prob
    ):
        """initializing the parameters of the MLP AE model; decoder and encoder
        (symmetric stucture)
        Args:
            freqlen (int): freq dim (number of features)
            seqlen (int): temporal dim (number of frames)
            hidden_size (int): the latent feature size (coded dimension)
            hidden_units (list): a list indicating the number of  hidden units
            of linear layers before/after latent feature. if list is empty means we have
            input > output (1 layer only)
            nonlinearity (str): str indicating nonlinearity in between layers
            dropout_prob (float): the dropout rate.
        """
        super(AEMLPNet, self).__init__()
        self.code_dim = hidden_size
        self.hidden_units = hidden_units
        self.nonlinearity = ACT2FN[nonlinearity]
        self.dropout = nn.Dropout(dropout_prob)
        self.seqlen = seqlen
        self.freqlen = freqlen
        self.all_layers_uints = self.hidden_units + [
            self.code_dim
        ]  # one layer would be added by default

        layers = []
        for layers_num in range(len(self.all_layers_uints)):
            in_dim = (
                seqlen * freqlen
                if layers_num == 0
                else self.all_layers_uints[layers_num - 1]
            )
            out_dim = self.all_layers_uints[layers_num]
            if layers_num != len(self.all_layers_uints) - 1:
                layers += [nn.Linear(in_dim, out_dim), self.nonlinearity]
            else:
                layers += [nn.Linear(in_dim, out_dim)]

        layers = [self.dropout] + layers  # add dropout in the input
        self.EncoderNet = nn.Sequential(*layers)

        decode_layers = []
        self.all_decoded_layers_uints = self.hidden_units[::-1] + [seqlen * freqlen]
        for layers_num in range(len(self.all_decoded_layers_uints)):
            in_dim = (
                self.code_dim
                if layers_num == 0
                else self.all_decoded_layers_uints[layers_num - 1]
            )
            out_dim = self.all_decoded_layers_uints[layers_num]
            if layers_num != len(self.all_decoded_layers_uints) - 1:
                decode_layers += [nn.Linear(in_dim, out_dim), self.nonlinearity]
            else:
                decode_layers += [nn.Linear(in_dim, out_dim)]
        self.DecoderNet = nn.Sequential(*decode_layers)

    def forward(self, input, selected_encoded_layers=-1):
        """
        Input:
            input (tensor): [B X D X T] a 3d-tensor representing the input features.
            selected_encoded_layers (list or -1): if list: list of indices of the
            encoder layers (to be used later as coded information in downstream or
            auxiliary task)
            -1: index of last layer of encoder
            {for the indices, we also consider the index of nonlinearity modules,
            e.g., indexing all modules}
        Return:
            (tuple): a tuple containing:
                - (tenosr): [B X D X T] the predicted output (reconstructed input).
                - (tensor): hiddens (latent) features [B X H or B X (H1+H2+ ...)]: Concatenation of
                list of features in encoder layers H1, H2, where indices are specified by
                "selected_encoded_layers"
        """
        hidden_out = self.EncoderNet(input.view(-1, self.seqlen * self.freqlen))
        predicted_out = self.DecoderNet(hidden_out).view(-1, self.freqlen, self.seqlen)
        encoder_layers = hidden_out

        if isinstance(selected_encoded_layers, list):
            assert all(
                [
                    layer_num < len(self.EncoderNet)
                    for layer_num in selected_encoded_layers
                ]
            ), (
                "selected layer for encoder feature extraction is not valid"
                " (more than number of encoder layers)"
            )
            encoder_layers = []
            encoded = input.view(-1, self.seqlen * self.freqlen)
            for i, layer in enumerate(self.EncoderNet):
                if i <= max(selected_encoded_layers):
                    encoded = layer(encoded)
                    encoder_layers.append(encoded)
            encoder_layers = torch.cat(
                list(map(encoder_layers.__getitem__, selected_encoded_layers)), dim=1
            )
        return predicted_out, encoder_layers


class AEMLPNetEncoder(AEMLPNet):
    """
    Encoder part of MLP AE model for downsrteam usage
    """

    def __init__(
        self, freqlen, seqlen, hidden_size, hidden_units, nonlinearity, dropout_prob
    ):
        super(AEMLPNetEncoder, self).__init__(
            freqlen, seqlen, hidden_size, hidden_units, nonlinearity, dropout_prob
        )
        self.DecoderNet = None

    def forward(self, input, selected_encoded_layers=-1):
        """
        Input:
            input B X D X T (tensor): a 3d-tensor representing the input features.
            selected_encoded_layers (list or -1): if list: list of indices of the
            encoder layers (to be used later as coded information in downstream
            or auxiliary task)
            -1: index of last layer of encoder
            {for the indices, we also consider the index of nonlinearity modules,
            e.g., indexing all modules}
        Return:
            (tensor): hiddens (latent) features [B X H or B X (H1+H2+ ...)]: Concatenation of list of features in encoder layers H1, H2, where indices are specified by "selected_encoded_layers"
        """
        if isinstance(selected_encoded_layers, list):
            assert all(
                [
                    layer_num < len(self.EncoderNet)
                    for layer_num in selected_encoded_layers
                ]
            ), "selected layer for encoder feature extraction is not valid (more than number of encoder layers)"
            encoder_layers = []
            encoded = input.view(-1, self.seqlen * self.freqlen)
            for i, layer in enumerate(self.EncoderNet):
                if i <= max(selected_encoded_layers):
                    encoded = layer(encoded)
                    encoder_layers.append(encoded)
            encoder_layers = torch.cat(
                list(map(encoder_layers.__getitem__, selected_encoded_layers)), dim=1
            )

        if selected_encoded_layers == -1:
            encoder_layers = self.EncoderNet(
                input.view(-1, self.seqlen * self.freqlen)
            )  # last layer
        return encoder_layers


class Interpolate(nn.Module):
    """
    Interpolating module for CNN
    """

    def __init__(
        self,
        size=None,
        scale_factor=None,
        mode="nearest",
        align_corners=None,
        recompute_scale_factor=None,
    ):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, x):
        x = self.interp(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
        )
        return x


class AECNNNetEnc(nn.Module):
    def __init__(
        self,
        freqlen,
        seqlen,
        hidden_size,
        hidden_units,
        kernelsize,
        poolingsize,
        convchannels,
        nonlinearity,
        dropout_prob,
        batchnorm,
    ):
        """initializing the parameters of the CNN Encoder model
        Args:
            freqlen (int): freq dim (number of features)
            seqlen (int): temporal dim (number of frames)
            hidden_size (int): the latent feature size (coded dimension)
            hidden_units (list): a list indicating the number of hidden units of
            linear layers between flattened CNN outputs and latent feature. if list
            is empty means we have
            input > output (1 layer only)
            kernelsize((int or tuple)): kernel size in conv layers.
            poolingsize (int or tuple): pooling size in conv layers.
            convchannels (list): channels of convs [1, outchannel1, outchannel2, ...]
            also indicates the number of convs in encoder and decoder.
            nonlinearity (str): str indicating nonlinearity in between layers
            dropout_prob (float): the dropout rate.
            batchnorm (bool): If True btach norm is applied for conv layers
        """
        super(AECNNNetEnc, self).__init__()
        self.code_dim = hidden_size
        self.hidden_units = hidden_units
        self.nonlinearity = ACT2FN[nonlinearity]
        self.dropout = nn.Dropout(dropout_prob)
        self.maxpool = nn.MaxPool2d(poolingsize)
        self.seqlen = seqlen
        self.freqlen = freqlen
        self.all_layers_uints = self.hidden_units + [self.code_dim]
        # for one-channel spectrograms
        self.convchannels = [1] + convchannels
        self.psize = poolingsize
        self.ksize = kernelsize
        conv_layers = []
        tsize = seqlen
        fsize = freqlen
        size_before_pooling = []
        for layers_num in range(len(self.convchannels) - 1):
            conv = nn.Conv2d(
                self.convchannels[layers_num],
                self.convchannels[layers_num + 1],
                kernel_size=kernelsize,
            )
            conv_bn = nn.BatchNorm2d(self.convchannels[layers_num + 1])
            tsize = int(np.floor(tsize - kernelsize + 1))
            tsize_before_pooling = tsize
            tsize = int(np.floor(((tsize - (poolingsize - 1) - 1) / (poolingsize)) + 1))

            fsize = int(np.floor(fsize - kernelsize + 1))
            fsize_before_pooling = fsize
            # collecting sizes before pooling to reverse the order it in decoder
            size_before_pooling.append((fsize_before_pooling, tsize_before_pooling))
            fsize = int(np.floor(((fsize - (poolingsize - 1) - 1) / (poolingsize)) + 1))
            if batchnorm:
                conv_layers += [conv, self.maxpool, conv_bn, self.nonlinearity]
            else:
                conv_layers += [conv, self.maxpool, self.nonlinearity]

        conv_layers = [self.dropout] + conv_layers  # add dropout to the input (only)
        self.EncoderCNN = nn.Sequential(*conv_layers)
        self.lsize = fsize * tsize * self.convchannels[layers_num + 1]
        self.CnnOutputShape = [
            (self.convchannels[layers_num + 1], fsize, tsize),
            size_before_pooling,
        ]
        layers = []
        for layers_num in range(len(self.all_layers_uints)):
            in_dim = (
                self.lsize if layers_num == 0 else self.all_layers_uints[layers_num - 1]
            )
            out_dim = self.all_layers_uints[layers_num]
            if layers_num != len(self.all_layers_uints) - 1:
                layers += [nn.Linear(in_dim, out_dim), self.nonlinearity]
            else:
                layers += [nn.Linear(in_dim, out_dim)]

        self.EncoderMLP = nn.Sequential(*layers)

    def forward(self, input):
        """
        Input:
            input B X D X T (tensor): a 3d-tensor representing the input features.
        Return:
            hidden representation (tensor) B X H: the latent representation
        """
        input = torch.unsqueeze(input, dim=1)  # add 1-channel > B X 1 X D X T
        cnn_encout = self.EncoderCNN(input)
        hidden_out = self.EncoderMLP(cnn_encout.view(-1, self.lsize))
        return hidden_out


class AECNNNetDec(nn.Module):
    def __init__(
        self,
        freqlen,
        seqlen,
        hidden_size,
        hidden_units,
        kernelsize,
        poolingsize,
        convchannels,
        nonlinearity,
        dropout_prob,
        batchnorm,
        CnnOutputShape,
    ):
        """initializing the parameters of the CNN Decoder (reverse of encoder)
        Args:
            freqlen (int): freq dim (number of features)
            seqlen (int): temporal dim (number of frames)
            hidden_size (int): the latent feature size (coded dimension)
            hidden_units (list): a list indicating the number of hidden units of
            linear layers between flattened CNN outputs and latent feature. if
            list is empty means we have
            input > output (1 layer only)
            kernelsize((int or tuple)): kernel size in conv layers.
            poolingsize (int or tuple): pooling size in conv layers.
            convchannels (list): channels of convs [1, outchannel1, outchannel2, ...]
            also indicates the number of convs in encoder and decoder.
            nonlinearity (str): str indicating nonlinearity in between layers
            dropout_prob (float): the dropout rate.
            batchnorm (bool): If True btach norm is applied for conv layers
            CnnOutputShape (list): [shape of final CNN encoder output, list of sizes
            of the output of all cnn layers before pooling]
        """
        super(AECNNNetDec, self).__init__()
        self.code_dim = hidden_size
        self.hidden_units = hidden_units
        self.nonlinearity = ACT2FN[nonlinearity]
        self.dropout = nn.Dropout(dropout_prob)
        self.maxpool = nn.MaxPool2d(poolingsize)
        self.seqlen = seqlen
        self.freqlen = freqlen
        self.all_layers_uints = self.hidden_units + [self.code_dim]
        self.convchannels = [1] + convchannels  # for one-channel spectrograms
        self.psize = poolingsize
        self.ksize = kernelsize
        self.CnnOutputShape = CnnOutputShape[
            0
        ]  # get final output size of the cnn encoder
        # get size of the output of cnn layers before pooling (needed for interpolation)
        self.pooled_correction = CnnOutputShape[1]
        self.lsize = 1
        for ele in self.CnnOutputShape:
            self.lsize *= ele
        decode_layers = []
        self.all_decoded_layers_uints = self.hidden_units[::-1] + [self.lsize]
        for layers_num in range(len(self.all_decoded_layers_uints)):
            in_dim = (
                self.code_dim
                if layers_num == 0
                else self.all_decoded_layers_uints[layers_num - 1]
            )
            out_dim = self.all_decoded_layers_uints[layers_num]
            if layers_num != len(self.all_decoded_layers_uints) - 1:
                decode_layers += [nn.Linear(in_dim, out_dim), self.nonlinearity]
            else:
                decode_layers += [nn.Linear(in_dim, out_dim)]
        self.DecoderMLP = nn.Sequential(*decode_layers)
        convt_layers = []
        self.interpolate = Interpolate
        for layers_num in range(len(self.convchannels) - 1, 0, -1):
            convt = nn.ConvTranspose2d(
                self.convchannels[layers_num],
                self.convchannels[layers_num - 1],
                kernel_size=kernelsize,
                stride=1,
            )
            convt_bn = nn.BatchNorm2d(self.convchannels[layers_num - 1])
            if layers_num != 1:
                if batchnorm:
                    convt_layers += [
                        self.interpolate(size=self.pooled_correction[layers_num - 1]),
                        convt,
                        convt_bn,
                        self.nonlinearity,
                    ]
                else:
                    convt_layers += [
                        self.interpolate(size=self.pooled_correction[layers_num - 1]),
                        convt,
                        self.nonlinearity,
                    ]
            else:
                convt_layers += [
                    self.interpolate(size=self.pooled_correction[layers_num - 1]),
                    convt,
                    nn.Sigmoid(),
                ]
        self.DecoderCNN = nn.Sequential(*convt_layers)

    def forward(self, input):
        """
        Input:
            input (tensor): hidden representation [B X H] or the latent representation
        Return:
            predicted_output (tensor): [B X D X T] the predicted output.
        """
        mlp_decout = self.DecoderMLP(input)
        predicted_out = self.DecoderCNN(mlp_decout.view(-1, *self.CnnOutputShape))
        return predicted_out.squeeze(dim=1)


class AECNNNet(nn.Module):
    def __init__(self, freqlen, seqlen, **kwargs):
        super(AECNNNet, self).__init__()
        self.EncoderNet = AECNNNetEnc(freqlen, seqlen, **kwargs)
        self.DecoderNet = AECNNNetDec(
            freqlen, seqlen, **kwargs, CnnOutputShape=self.EncoderNet.CnnOutputShape
        )

    def forward(self, input, selected_encoded_layers=-1):
        """
        Input:
            input (tensor): [B X D X T] a 3d-tensor representing the input features.
            selected_encoded_layers (list or -1): if list: list of indices
            of the encoder layers (to be used later as coded information in downstream or
            auxiliary task), -1: index of last layer of encoder
            {for the indices, we consider mlp and nonlinearity modules}

        Return:
            (tuple): a tuple containing:
                - predicted_output (tensor): [B X D X T] the predicted output
                (reconstructed input).
                - hidden representation (tensor): [B X H] the latent representation
        """
        hidden_out = self.EncoderNet(input)
        predicted_out = self.DecoderNet(hidden_out)

        if isinstance(selected_encoded_layers, list):
            input = torch.unsqueeze(input, dim=1)  # adding 1-channel > B X 1 X D X T
            cnn_encout = self.EncoderNet.EncoderCNN(input)
            encoded = cnn_encout.view(-1, self.EncoderNet.lsize)
            assert all(
                [
                    layer_num < len(self.EncoderNet.EncoderMLP)
                    for layer_num in selected_encoded_layers
                ]
            ), (
                "selected layer for encoder feature extraction is not"
                " valid (more than number of encoder MLP layers)"
            )
            hidden_out = []
            for i, layer in enumerate(self.EncoderNet.EncoderMLP):
                if i <= max(selected_encoded_layers):
                    encoded = layer(encoded)
                    hidden_out.append(encoded)
            hidden_out = torch.cat(
                list(map(hidden_out.__getitem__, selected_encoded_layers)), dim=1
            )

        elif selected_encoded_layers == -1:
            pass  # last layer as in hidden out
        return predicted_out, hidden_out


class AECNNNetEncoder(AECNNNet):
    """
    Encoder part of CNN AE model for downsrteam usage
    """

    def __init__(self, freqlen, seqlen, **kwargs):
        super(AECNNNetEncoder, self).__init__(freqlen, seqlen, **kwargs)

    def forward(self, input, selected_encoded_layers=-1):
        """
        Input:
            input (tensor): [B X D X T] a 2d-tensor representing the input features.
            selected_encoded_layers (list or -1): if list: list of indices of the encoder
            layers (to be used later as coded information in downstream or auxiliary task)
            -1: index of last layer of encoder
            {for the indices, we consider mlp and nonlinearity modules}

        Return:
            (tensor): hiddens (latent) features [B X H or B X (H1+H2+ ...)]
            Concatenation of list of features in encoder layers H1, H2, where
            indices are specified by "selected_encoded_layers"
        """
        input = torch.unsqueeze(input, dim=1)  # adding 1-channel > B X 1 X D X T
        cnn_encout = self.EncoderNet.EncoderCNN(input)
        encoded = cnn_encout.view(-1, self.EncoderNet.lsize)
        if isinstance(selected_encoded_layers, list):
            assert all(
                [
                    layer_num < len(self.EncoderNet.EncoderMLP)
                    for layer_num in selected_encoded_layers
                ]
            ), (
                "selected layer for encoder feature extraction is not"
                " valid (more than number of encoder MLP layers)"
            )
            encoder_layers = []
            for i, layer in enumerate(self.EncoderNet.EncoderMLP):
                if i <= max(selected_encoded_layers):
                    encoded = layer(encoded)
                    encoder_layers.append(encoded)
            encoder_layers = torch.cat(
                list(map(encoder_layers.__getitem__, selected_encoded_layers)), dim=1
            )

        if selected_encoded_layers == -1:
            encoder_layers = self.EncoderNet.EncoderMLP(encoded)  # last layer
        return encoder_layers


class AERNNNetEnc(nn.Module):
    def __init__(
        self,
        freqlen,
        seqlen,
        hidden_size,
        hidden_units,
        RNN_dims,
        stacked_layers_nums,
        bidirectional,
        nonlinearity,
        dropout_prob,
    ):
        """initializing the parameters of the RNN encoder model
        Args:
            freqlen (int): freq dim (number of features)
            seqlen (int): temporal dim (number of frames)
            hidden_size (int): the latent feature size (coded dimension)
            hidden_units (list): a list indicating the number of hidden units of linear
            layers after hidden states of RNNs. If list is empty means we
            have input > output (1 layer only)
            RNN_dims: number of RNN layers (hidden size in RNN layers)
            stacked_layers_nums: number of stacked RNN layers in decoder
            bidirectional: If True, RNN is bidirectional
            nonlinearity (str): str indicating nonlinearity in between layers
            dropout_prob (float): the dropout rate.
        """
        super(AERNNNetEnc, self).__init__()
        self.code_dim = hidden_size
        self.hidden_units = hidden_units
        self.nonlinearity = ACT2FN[nonlinearity]
        self.dropout = nn.Dropout(dropout_prob)
        self.seqlen = seqlen
        self.freqlen = freqlen
        self.all_layers_uints = self.hidden_units + [self.code_dim]
        self.RNN_dims = [freqlen] + RNN_dims
        if bidirectional:
            self.directions = 2
        else:
            self.directions = 1
        rnn_layers = []
        tsize = seqlen
        fsize = freqlen
        for layers_num in range(len(self.RNN_dims) - 1):
            # with batch_first: True,  input should be of size B X T X F
            rnn = nn.LSTM(
                self.RNN_dims[layers_num],
                self.RNN_dims[layers_num + 1],
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout_prob,
            )
            rnn_layers += [rnn]

        self.EncoderRNN = nn.Sequential(*rnn_layers)
        # concatenating outputs and hidden states
        self.lsize = 2 * self.directions * self.RNN_dims[-1]
        layers = []
        for layers_num in range(len(self.all_layers_uints)):
            in_dim = (
                self.lsize if layers_num == 0 else self.all_layers_uints[layers_num - 1]
            )
            out_dim = self.all_layers_uints[layers_num]
            if layers_num != len(self.all_layers_uints) - 1:
                layers += [nn.Linear(in_dim, out_dim), self.nonlinearity]
            else:
                layers += [nn.Linear(in_dim, out_dim)]

        self.EncoderMLP = nn.Sequential(*layers)

    def forward(self, input):
        """
        Input:
            input (tensor): [B X D X T] a 3d-tensor representing the input features.
        Return:
            (tensor): hidden representation [B X H] or the latent representation
        """
        input = input.permute(0, 2, 1)  # B X T X D
        for lstm in self.EncoderRNN:
            input, (hn, cn) = lstm(input[:, :, : lstm.input_size])
        out = input
        hn = hn.permute(1, 0, 2)
        # output of last time step: B X num_directions * hidden_size
        out = out[:, -1, :]
        hn = hn.contiguous().view(hn.size(0), -1)
        out = out.view(out.size(0), -1)
        hidden_in = torch.cat((out, hn), 1)
        hidden_out = self.EncoderMLP(hidden_in.view(-1, self.lsize))
        return hidden_out


class AERNNNetDec(nn.Module):
    def __init__(
        self,
        freqlen,
        seqlen,
        hidden_size,
        hidden_units,
        RNN_dims,
        stacked_layers_nums,
        bidirectional,
        nonlinearity,
        dropout_prob,
    ):
        """initializing the parameters of the RNN decoder model
        Args:
            freqlen (int): freq dim (number of features)
            seqlen (int): temporal dim (number of frames)
            hidden_size (int): the latent feature size (coded dimension)
            hidden_units (list): a list indicating the number of hidden units
            of linear layers after hidden states of RNNs. If list is empty means we have
            input > output (1 layer only)
            RNN_dims: number of RNN layers (hidden size in RNN layers)
            stacked_layers_nums: number of stacked RNN layers in decoder
            bidirectional: If True, RNN is bidirectional (only for encoder)
            nonlinearity (str): str indicating nonlinearity in between layers
            dropout_prob (float): the dropout rate.
        """
        super(AERNNNetDec, self).__init__()
        self.code_dim = hidden_size
        self.hidden_units = hidden_units
        self.nonlinearity = ACT2FN[nonlinearity]
        self.dropout = nn.Dropout(dropout_prob)
        self.seqlen = seqlen
        self.freqlen = freqlen
        self.RNN_dims = [self.code_dim] + RNN_dims[::-1] + [freqlen]

        rnn_layers = []
        tsize = seqlen
        fsize = freqlen
        for layers_num in range(len(self.RNN_dims) - 1):
            # If stacked are used, dropout would be 0 for last stacked models
            if layers_num != len(self.RNN_dims) - 1:
                rnn = nn.LSTM(
                    self.RNN_dims[layers_num],
                    self.RNN_dims[layers_num + 1],
                    batch_first=True,
                    dropout=dropout_prob,
                    num_layers=stacked_layers_nums,
                )
                # with batch_first: True,  input should be of size B X T X F
            else:
                rnn = nn.LSTM(
                    self.RNN_dims[layers_num],
                    self.RNN_dims[layers_num + 1],
                    batch_first=True,
                    dropout=0,
                    num_layers=stacked_layers_nums,
                )
            rnn_layers += [rnn]  # , self.nonlinearity]

        self.DecoderRNN = nn.Sequential(*rnn_layers)

    def forward(self, input):
        """
        Input:
            input (tensor): hidden representation [B X H] (the latent representation)
        Return:
            predicted_output (tensor): [B X D X T] the predicted output
        """
        input = input.unsqueeze(1).expand(-1, self.seqlen, -1)  # B X T X D
        for lstm in self.DecoderRNN:
            input, (hn, cn) = lstm(input[:, :, : lstm.input_size])
        # out: B X T X num_directions * hidden_size, hn: num_layers * num_directions X B X hidden_size
        out = input
        predicted_out = out.permute(0, 2, 1)  # B X D X T
        return predicted_out


class AERNNNet(nn.Module):
    def __init__(self, freqlen, seqlen, **kwargs):
        """initializing the parameters of the RNN AE model"""
        super(AERNNNet, self).__init__()
        self.EncoderNet = AERNNNetEnc(freqlen, seqlen, **kwargs)
        self.DecoderNet = AERNNNetDec(freqlen, seqlen, **kwargs)

    def forward(self, input, selected_encoded_layers=-1):
        """
        Input:
            input B X D X T (tenosr): a 3d-tensor representing the input features.
            selected_encoded_layers (list or -1 ): indices of the encoder layers (to be
            used later for auxilary or downstream task)
        Return:
           (tuple): a tuple containing:
                - predicted_output (tenosr): [B X D X T] the predicted output
                (reconstructed input).
                - (tensor): hiddens (latent) features [B X H or B X (H1+H2+ ...)]:
                Concatenation of list of features in encoder layers H1, H2, where
                indices are specified by "selected_encoded_layers"
        """
        hidden_out = self.EncoderNet(input)
        predicted_out = self.DecoderNet(hidden_out)

        if isinstance(selected_encoded_layers, list):
            input = input.permute(0, 2, 1)  # B X T X D
            for lstm in self.EncoderNet.EncoderRNN:
                input, (hn, cn) = lstm(input[:, :, : lstm.input_size])
            out = input
            # B X num_layers * num_directions X hidden_size
            hn = hn.permute(1, 0, 2)
            # output of last time step: B X num_directions * hidden_size
            out = out[:, -1, :]
            hn = hn.contiguous().view(hn.size(0), -1)
            out = out.view(out.size(0), -1)
            encoded = torch.cat((out, hn), 1).view(-1, self.EncoderNet.lsize)
            assert all(
                [
                    layer_num < len(self.EncoderNet.EncoderMLP)
                    for layer_num in selected_encoded_layers
                ]
            ), (
                "selected layer for encoder feature extraction is not valid"
                " (more than number of encoder MLP layers)"
            )
            hidden_out = []
            for i, layer in enumerate(self.EncoderNet.EncoderMLP):
                if i <= max(selected_encoded_layers):
                    encoded = layer(encoded)
                    hidden_out.append(encoded)
            hidden_out = torch.cat(
                list(map(hidden_out.__getitem__, selected_encoded_layers)), dim=1
            )

        if selected_encoded_layers == -1:
            pass  # last layer

        return predicted_out, hidden_out


class AERNNNetEncoder(AERNNNet):
    """
    Encoder part of RNN AE model for downsrteam usage
    """

    def __init__(self, freqlen, seqlen, **kwargs):
        super(AERNNNetEncoder, self).__init__(freqlen, seqlen, **kwargs)

    def forward(self, input, selected_encoded_layers=-1):
        """
        Input:
            input (tensor): [B X D X T] a 3d-tensor representing the input features.
            selected_encoded_layers (list or -1 ): indices of the encoder layers
            (to be used later for downstream)
            for this netowrk, we only consider mlp layers also we consider
            the index of nonlinearity modules (indexing all modules)
        Return:
            (tensor): hiddens (latent) features [B X H or B X (H1+H2+ ...)] Concatenation
            of list of features in encoder layers H1, H2, where indices are specified by
            "selected_encoded_layers"
        """
        input = input.permute(0, 2, 1)  # B X T X D
        for lstm in self.EncoderNet.EncoderRNN:
            input, (hn, cn) = lstm(input[:, :, : lstm.input_size])
        out = input
        # B X num_layers * num_directions X hidden_size
        hn = hn.permute(1, 0, 2)
        # output of last time step: B X num_directions * hidden_size
        out = out[:, -1, :]
        hn = hn.contiguous().view(hn.size(0), -1)
        out = out.view(out.size(0), -1)
        encoded = torch.cat((out, hn), 1).view(-1, self.EncoderNet.lsize)
        if isinstance(selected_encoded_layers, list):
            assert all(
                [
                    layer_num < len(self.EncoderNet.EncoderMLP)
                    for layer_num in selected_encoded_layers
                ]
            ), (
                "selected layer for encoder feature extraction is"
                " not valid (more than number of encoder MLP layers)"
            )
            encoder_layers = []
            for i, layer in enumerate(self.EncoderNet.EncoderMLP):
                if i <= max(selected_encoded_layers):
                    encoded = layer(encoded)
                    encoder_layers.append(encoded)
            encoder_layers = torch.cat(
                list(map(encoder_layers.__getitem__, selected_encoded_layers)), dim=1
            )

        if selected_encoded_layers == -1:
            encoder_layers = self.EncoderNet.EncoderMLP(encoded)  # last layer
        return encoder_layers


if __name__ == "__main__":
    # # ------------------------------ changing the paths ----------------------------- #
    import sys
    from pathlib import Path

    file = Path(__file__).resolve()
    parent, root, subroot = file.parent, file.parents[1], file.parents[2]
    sys.path.append(str(subroot))
    sys.path.append(str(root))
    os.chdir(root)

    # # ---------------------------------------------------------------------------- #
    from torch.utils.data import DataLoader
    from audio.audio_utils import get_config_args, create_transform
    from audio.audio_dataset import OnlineAcousticDataset, OfflineAcousticDataset
    import matplotlib.pyplot as plt
    import numpy as np
    import random

    cpath = "config/upstream_config.yaml"
    dataloading_config = get_config_args(cpath)
    feat_path = "config/audio_config.yaml"
    feat_config = get_config_args(feat_path)
    file_path = "preprocess/dummy_database/folds/test_fold1_fbank_offline.csv"
    transforms = create_transform(feat_config, 16000)

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

    # offline testing
    dataset = OfflineAcousticDataset(None, dataloading_config, file_path, feat_config)
    freqdim, seqlen = dataset.getDimension()
    Network_config = dataloading_config
    selected_model = Network_config["SelectedAENetwork"]
    model_config = Network_config[selected_model]
    rnnmodel_config = Network_config["AERNNNet"]
    mlpmodel_config = Network_config["AEMLPNet"]
    seed = 0
    seed_torch(seed)
    model = eval(f"{selected_model}")(freqdim, seqlen, **model_config)
    seed_torch(seed)  # check encoder parts
    encoder = eval(f"{selected_model}Encoder")(freqdim, seqlen, **model_config)
    seed_torch(seed)
    model2 = AERNNNet(freqdim, seqlen, **rnnmodel_config)
    model3 = AEMLPNet(freqdim, seqlen, **mlpmodel_config)
    seed_torch(seed)
    encoder_rnn = AERNNNetEncoder(freqdim, seqlen, **rnnmodel_config)
    encoder_mlp = AEMLPNetEncoder(freqdim, seqlen, **mlpmodel_config)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=10, shuffle=True, num_workers=0
    )
    for batch_idx, data_batch in enumerate(data_loader):
        print("batch index: ", batch_idx)
        data, ID, targets = data_batch
        print("input size: ", data.shape)
        output = model(data)
        output2 = model2(data)
        output3 = model3(data)
        rnn_encoded = encoder_rnn(data, selected_encoded_layers=-1)
        mlp_encoded = encoder_mlp(data, selected_encoded_layers=-1)
        encoded = encoder(data, selected_encoded_layers=-1)
        print("output/latent size: ", output[0].shape, output[1].shape, "\n")
        print("cnn latent size:", encoded.shape)
        print("rnn latent size:", rnn_encoded.shape)
        print("mlp latent size:", mlp_encoded.shape)
        print("cnn encoded error", torch.norm(output[1] - encoded))
        print("rnn encoded error", torch.norm(output2[1] - rnn_encoded))
        print("mlp encoded error", torch.norm(output3[1] - mlp_encoded))
        break

    plt.close("all")
    plt.subplot(411)
    plt.imshow(data[0, :, :].detach().numpy(), aspect="auto", cmap="jet")
    plt.title("input data")
    plt.subplot(412)
    plt.imshow(output[0][0, :, :].detach().numpy(), aspect="auto", cmap="jet")
    plt.title("CNN AE output")
    plt.subplot(413)
    plt.imshow(output2[0][0, :, :].detach().numpy(), aspect="auto", cmap="jet")
    plt.title("RNN AE output")
    plt.subplot(414)
    plt.imshow(output3[0][0, :, :].detach().numpy(), aspect="auto", cmap="jet")
    plt.title("MLP AE output")
