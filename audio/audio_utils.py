# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
# Defining audio feature extraction utils
# For this some implementation ideas from https://github.com/s3prl/s3prl/blob/master/upstream/apc/audio.py
# are used.

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
import os.path as op
import torch
import torch.nn as nn
from typing import BinaryIO, Optional, Tuple, Union
import numpy as np
import yaml
import torchaudio.compliance.kaldi as ta_kaldi
from torchaudio import transforms


def get_waveform(
    path_or_fp: Union[str, BinaryIO], normalization=True
) -> Tuple[np.ndarray, int]:
    """Get the waveform and sample rate of a 16-bit mono-channel WAV or FLAC.
    adapted from https://github.com/pytorch/fairseq/blob/master/fairseq/data/audio/audio_utils.py
    Args:
        path_or_fp (str or BinaryIO): the path or file-like object
        normalization (bool): Normalize values to [-1, 1] (Default: True)
    Returns:
        (numpy.ndarray): [n,] waveform array, (int): sample rate
    """
    if isinstance(path_or_fp, str):
        ext = op.splitext(op.basename(path_or_fp))[1]
        if ext not in {".flac", ".wav"}:
            raise ValueError(f"Unsupported audio format: {ext}")
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("Please install soundfile to load WAV/FLAC file")

    waveform, sample_rate = sf.read(path_or_fp, dtype="float32")
    waveform -= np.mean(waveform)
    waveform /= np.max(np.abs(waveform))
    if not normalization:
        waveform *= 2 ** 15  # denormalized to 16-bit signed integers
    return waveform, sample_rate


def _get_torch_feat(waveform, sample_rate, **config) -> Optional[np.ndarray]:
    """Extract features based on config dict, In case of Mel-bank,
    MFCC or spectrogram features TorchAudio is used
    while in case of articulatory features it uses saved CNN models
    (not released in repo!).
    Args:
        waveform (numpy.ndarray): input waveform array
        sample_rate (int): sample rate
    Returns:
        features (numpy.ndarray): extracted features from waveform
    """
    try:
        import torch
        import torchaudio.compliance.kaldi as ta_kaldi

        feat_type = config["feat_type"]
        apply_cmvn = config["postprocess"].get("cmvn")
        apply_delta = config["postprocess"].get("delta")
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        if feat_type == "AP":
            if (
                "AP_AllExtract" not in dir()
            ):  # for articulatory feature extractors (not released yet!)
                from Phonological_posteriors_extraction.posterior_extractor import (
                    AP_AllExtract,
                )
            features = AP_AllExtract(
                waveform, sample_frequency=sample_rate
            )  # tensor T X D
        else:
            extractor = eval(f"ta_kaldi.{feat_type}")
            features = extractor(
                waveform, **config["torchaudio"], sample_frequency=sample_rate
            )
        if apply_cmvn:
            eps = 1e-10
            features = (features - features.mean(dim=0, keepdim=True)) / (
                eps + features.std(dim=0, keepdim=True)
            )
        if apply_delta > 0:
            order = 1
            feats = [features]
            for o in range(order):
                feat = feats[-1].transpose(0, 1).unsqueeze(0)
                Delta = transforms.ComputeDeltas(win_length=apply_delta)
                delta = Delta(feat)
                feats.append(delta.squeeze(0).transpose(0, 1))
            features = torch.cat(feats, dim=-1)
        return features.numpy()
    except ImportError:
        return None


def get_feat(path_or_fp: Union[str, BinaryIO], config_path: str) -> np.ndarray:
    """Compute features based on feature config file. Note that
    TorchAudio requires 16-bit signed integers as inputs and hence the
    waveform should not be normalized.

    Args:
        path_or_fp (Union[str, BinaryIO]): path of wav file
        config_path (str): path of feature extraction config
    Returns:
        (np.ndarray): extracted features
    """
    sound, sample_rate = get_waveform(path_or_fp, normalization=False)
    config = get_config_args(config_path)
    features = _get_torch_feat(sound, sample_rate, **config)
    if features is None:
        raise ImportError(
            "Please install pyKaldi or torchaudio to enable "
            "online filterbank feature extraction"
        )
    return features


def get_config_args(cpath):
    """get contents of yaml file
    Args:
        cpath (str): yaml file
    Returns:
        (dict): Contents of yaml file
    """
    with open(cpath, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


# ------------ classes for online feature extraction with wav inputs ----------- #


class CMVN(nn.Module):
    """Feature normalization module"""

    def __init__(self, eps=1e-10):
        super(CMVN, self).__init__()
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x (tensor): input feature
        Returns:
            (tensor): normalized feature
        """
        x = (x - x.mean(dim=0, keepdim=True)) / (self.eps + x.std(dim=0, keepdim=True))
        return x


class Delta(nn.Module):
    """Computing delta representation of features"""

    def __init__(self, order=1, **kwargs):
        super(Delta, self).__init__()
        self.order = order
        self.compute_delta = transforms.ComputeDeltas(**kwargs)

    def forward(self, x):
        """
        Args:
            x (tensor):  input tensor (feat_seqlen, feat_dim)  [T X D]

        Returns:
            x (tensor): [T X D1] concatenated features input features and
            its deltas according to order number
        """
        feats = [x]
        for o in range(self.order):
            feat = feats[-1].transpose(0, 1).unsqueeze(0)
            delta = self.compute_delta(feat)
            feats.append(delta.squeeze(0).transpose(0, 1))
        x = torch.cat(feats, dim=-1)
        return x


class ExtractAudioFeature(nn.Module):
    """first level audio feature extraction module"""

    def __init__(self, mode="spectrogram", sample_rate=16000, **kwargs):
        super(ExtractAudioFeature, self).__init__()
        assert (
            (mode == "fbank") | (mode == "spectrogram") | (mode == "mfcc")
        ), "only spectrogram, fbank, and mfcc are implemented"
        self.mode = mode
        if mode != "AP":  # for articulatory features (not released yet)
            self.extract_fn = eval(f"ta_kaldi.{self.mode}")
        else:
            # (not released)
            from Phonological_posteriors_extraction.posterior_extractor import (
                AP_AllExtract,
            )

            self.extract_fn = AP_AllExtract

        self.sample_rate = sample_rate
        self.kwargs = kwargs

    def forward(self, waveform):
        """feature computation module
        Args:
            waveform (tensor): input waveform [1 X T]
        Returns:
            (tensor):  features (feat_seqlen, feat_dim) [T X D]
        """
        x = self.extract_fn(
            waveform.view(1, -1), sample_frequency=self.sample_rate, **self.kwargs
        )
        return x


class FeatureExtractor(nn.Module):
    """Full audio feature extraction considering normalization and delta computation"""

    def __init__(
        self,
        mode="spectrogram",
        sample_rate=16000,
        apply_cmvn=True,
        apply_delta=0,
        **kwargs,
    ):
        """feature extractor initialization
        Args:
            mode (str, optional): type of features (spectrogram, fbank, and mfcc).
            Defaults to "spectrogram".
            sample_rate (int, optional): audio sampling rate. Defaults to 16000.
            apply_cmvn (bool, optional): if True applies feature normalization. Defaults to True.
            apply_delta (int, optional): length of delta window length. Defaults to 0
            (0, e.g., not delta computation).
        """
        super(FeatureExtractor, self).__init__()
        # ToDo: Other representation
        self.sample_rate = sample_rate
        self.kwargs = kwargs
        self.apply_cmvn = apply_cmvn
        self.apply_delta = apply_delta
        transforms = [
            ExtractAudioFeature(mode=mode, sample_rate=self.sample_rate, **self.kwargs)
        ]
        if self.apply_cmvn:
            transforms.append(CMVN())
        if self.apply_delta > 0:
            transforms.append(Delta(win_length=apply_delta))
        self.extract_postprocess = nn.Sequential(*transforms)

    def forward(self, waveform):
        y = self.extract_postprocess(waveform)  # TxD
        return y


def create_transform(audio_config, fs):
    """create transform for wav file to be converted to feature domain
    Args:
        audio_config (dict): config file for acoustic feature extraction
        fs (int): wav sample rate
    Returns:
        (FeatureExtractor): feature extractor object to be operated on wav tensor
        of size [1 X time]
    """

    feat_type = audio_config["feat_type"]
    torchaudio_parmas = audio_config["torchaudio"]
    apply_cmvn = audio_config["postprocess"]["cmvn"]
    apply_delta = audio_config["postprocess"]["delta"]
    transforms = FeatureExtractor(
        mode=feat_type,
        sample_rate=fs,
        apply_cmvn=apply_cmvn,
        apply_delta=apply_delta,
        **torchaudio_parmas,
    )
    return transforms


#%%
if __name__ == "__main__":
    #  checking parity between offline and online feature extraction
    import matplotlib.pyplot as plt

    wav_test = "../preprocess/dummy_database/audio_data/sample4.wav"
    wav, fs = get_waveform(wav_test, normalization=False)
    import torchaudio

    wav_torch, fs1 = torchaudio.load(wav_test, normalization=False)
    conf_feat_path = "../config/audio_config.yaml"
    features = get_feat(wav_test, conf_feat_path)
    # this when saved should be, array of size D X T
    plt.subplot(311)
    plt.plot(wav)
    plt.subplot(312)
    plt.imshow(features.transpose(), aspect="auto", cmap="jet")

    cpath = "../config/audio_config.yaml"
    cf = get_config_args(cpath)
    transforms = create_transform(cf, fs)
    wav_tensor = torch.from_numpy(wav).unsqueeze(0)
    feat = transforms(wav_tensor)
    plt.subplot(313)
    plt.imshow(feat.t().detach().numpy(), aspect="auto", cmap="jet")
    print(
        "error feat computations:",
        np.linalg.norm(feat.t().detach().numpy() - features.transpose()),
    )
