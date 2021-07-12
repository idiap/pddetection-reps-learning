# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
# Defining acoustic datasets

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
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchaudio
torchaudio.set_audio_backend("sox_io")
import bisect
import sys
import pandas as pd
# from audio_utils import get_waveform
from .audio_utils import get_waveform


class AcousticDataset(Dataset):
    def __init__(self, extractor, dataloading_config, file_path, feat_config, **kwargs):
        """AcousticDataset initialization
        Args:
            extractor (FeatureExtractor): online feature extraction modules,
            should be None for offline model
            dataloading_config (dict): dataloading config dict
            file_path (str): path of csv file of utterance paths
            feat_config (dict): feature extraction config
        """
        super(AcousticDataset, self).__init__()

        self.extractor = extractor
        self.dataloading_config = dataloading_config["dataloader"]
        self.sample_length = self.dataloading_config["sequence_length"]
        assert self.sample_length > 0, "only segmented inputs are implemented"
        self.overlap = 0.5
        # by default sampling is done with 50% overlap between extracted segments
        self.root = file_path
        if feat_config["feat_type"] != "AP":
            mode = "torchaudio"
        else:
            mode = "APCNN"
        self.frame_length = feat_config[mode]["frame_length"]
        self.frame_shift = feat_config[mode]["frame_shift"]
        if self.dataloading_config["online"]:
            self.sample_rate = self.extractor.sample_rate
            print(
                "[Dataset] - Sampling random segments with sample length:",
                self.sample_length / 1e3,
                "seconds",
            )
        else:
            self.sample_length = 1 + int(
                (self.sample_length - self.frame_length) / (self.frame_shift)
            )  # computing number of frames for segment_length
            print(
                "[Dataset] - Sampling random segments with sample length:",
                self.sample_length,
                "frames",
            )

        table = pd.read_csv(os.path.join(file_path))
        # Remove utterances that are shorter than sample_length
        self.table = table[table.length >= self.sample_length]
        self.X = self.table["file_path"].tolist()  # All paths
        # All utterances length
        X_lens = self.table["length"].tolist()
        self.spkID = self.table["ID"].tolist()
        try:
            self.label = self.table["label"].tolist()
        except:  # If there is no label then it is from healthy speech database without provided labels
            self.label = [1] * len(self.table)
        self.X_lens = (
            np.array(X_lens) - self.sample_length
        )  # Effective length for sampling segments
        print("[Dataset] - Number of individual utterance instances:", len(self.X))

    def standardize(self, tensor):
        """Standardize input data
        Args:
            tensor (tensor): ipnut [D X T] dimension
        Returns:
            (tensor): output [D X T] dimension
        """
        MIN = tensor.min()
        MAX = tensor.max()
        return (tensor - MIN) / (MAX - MIN)

    def _sample(self, x, interval):
        """sampling from data for creating batches
        Args:
            x (tensor): either waveform data [1 X T] or acoustic features [D X T]
            interval (int): sampling start
        Returns:
            (tensor): sampled interval from x
        """
        if not self.dataloading_config["online"]:  # offline dataloader
            return x[:, interval : interval + self.sample_length]
        else:
            interval = int(interval * self.sample_rate // 1e3)
            return x[
                :,
                interval : interval + int(self.sample_length * self.sample_rate // 1e3),
            ]

    def _getindex(self, index):
        """compute the interval and utterance number based on the random index.
        By default sampling is done with 50% overlap between extracted segments
        Args:
            index (int): index

        Returns:
            (tuple): a tuple containing:
                - (int): utterance number (or path number)
                - (int): time interval for the utterance
        """

        index_unit = int(self.sample_length * self.overlap)  # shift size (ms or frames)
        Cumuints = np.cumsum((1 + self.X_lens // index_unit))
        Cumuints1 = np.append(Cumuints, 0)
        uttr_num = bisect.bisect_left(Cumuints, index + 1)
        interval = index - Cumuints1[uttr_num - 1]
        return uttr_num, int((interval) * index_unit)

    def __len__(self):
        """Computing total number of audio segments
        Returns:
            (int): total number of audio segments
        """
        return int(np.sum(1 + (self.X_lens // (self.overlap * self.sample_length))))


class OfflineAcousticDataset(AcousticDataset):
    """Dataset loader for offline feature extraction
    (features are already computed and saved)
    Args:
        AcousticDataset (AcousticDataset)
    """

    def __init__(self, extractor, dataloading_config, file_path, feat_config, **kwargs):
        super(OfflineAcousticDataset, self).__init__(
            extractor, dataloading_config, file_path, feat_config, **kwargs
        )

    def _load_feat(self, npy_path):
        """
        Args:
            npy_path (str): path of numpy file of feature

        Returns:
            (tensor): [D X T] features
        """
        feat = np.load(npy_path, allow_pickle=True)
        return torch.from_numpy(
            feat
        )  # D X T feature (previously saved feature arrays should be of dimension D X T)

    def getDimension(self):
        """get dimension of input feature tensors
        Returns:
            (int): freq dim, (int): seq len
        """
        INTERVAL = 0
        freqlen, seqlen = self._sample(self._load_feat(self.X[0]), INTERVAL).shape
        return freqlen, seqlen

    def __getitem__(self, index):
        """get acoustic feature segments
        Args:
            index (int): index for sampling segments
        Returns:
            (tuple): a tuple containing:
                - (tensor): tensor data of the segment
                - (int): speaker index (from which data is extracted)
                - (int): utterance label (from which data is extracted)
        """
        uttr_num, interval = self._getindex(index)
        feat_file = self.X[uttr_num]
        return (
            self.standardize(self._sample(self._load_feat(feat_file), interval)),
            self.spkID[uttr_num],
            self.label[uttr_num],
        )


class OnlineAcousticDataset(AcousticDataset):
    """Dataset loader for online feature extraction
    Args:
        AcousticDataset (AcousticDataset)
    """

    def __init__(self, extractor, dataloading_config, file_path, feat_config, **kwargs):
        super(OnlineAcousticDataset, self).__init__(
            extractor, dataloading_config, file_path, feat_config, **kwargs
        )

        _, test_fs = get_waveform(self.X[0], normalization=False)
        assert test_fs == extractor.sample_rate, (
            f"the setting for sampling frequency"
            " in config file {extractor.sample_rate} should be {test_fs}"
        )

    def _load_feat(self, wavfile):
        """
        Args:
            wavfile (str): path of wav file

        Returns:
            (tensor): [1 X T] wavform
        """
        wav, _ = get_waveform(wavfile, normalization=False)
        return torch.from_numpy(wav).unsqueeze(0)  # 1 X T

    def _feature_extract(self, wavfile):
        """Extracting features from wavfile
        Args:
            wavfile (tensor): Tensor wavform [1 X T]
        Returns:
            features (tensor): [D X T] extracted features from waveform
        """
        return self.extractor(wavfile).t()  # D X T

    def getDimension(self):
        """getting dimension of input tensors
        Returns:
            (tuple): a tuple containing:
                - (int): freq dim
                - (int): seq len
        """
        INTERVAL = 0
        EXAMPLE_FEAT_SEQLEN = int(self.sample_length * 1e-3 * self.sample_rate * 2)
        pseudo_wav = torch.randn(1, EXAMPLE_FEAT_SEQLEN)
        freqlen, seqlen = self._feature_extract(
            self._sample(pseudo_wav, INTERVAL)
        ).shape
        return freqlen, seqlen

    def __getitem__(self, index):
        """get acoustic feature segments
        Args:
            index (int): index for sampling segments
        Returns:
            (tuple): a tuple containing:
                - (tensor): tensor data of the segment
                - (int): speaker index (from which data is extracted)
                - (int): utterance label (from which data is extracted)
        """
        uttr_num, interval = self._getindex(index)
        wav_file = self.X[uttr_num]
        return (
            self.standardize(
                self._feature_extract(self._sample(self._load_feat(wav_file), interval))
            ),
            self.spkID[uttr_num],
            self.label[uttr_num],
        )


if __name__ == "__main__":
    # checking parity between online and offline feature extraction    
    from audio_utils import get_config_args, create_transform
    import matplotlib.pyplot as plt
    from pathlib import Path
    file = Path(__file__).resolve()
    parent, root, subroot = file.parent, file.parents[1], file.parents[2]
    sys.path.append(str(subroot))
    sys.path.append(str(root))
    os.chdir(root)

    cpath = "config/upstream_config.yaml"
    dataloading_config = get_config_args(cpath)
    dataloading_config["dataloader"]["online"] = True
    import copy
    dataloading_config_off = copy.deepcopy(dataloading_config)
    dataloading_config_off["dataloader"]["online"] = False
    feat_path = "config/audio_config.yaml"
    feat_config = get_config_args(feat_path)
    file_path = "preprocess/dummy_database/folds/test_fold1_online.csv"
    transforms = create_transform(feat_config, 16000)

    # online testing
    assert dataloading_config["dataloader"][
        "online"
    ], 'set "online field" in upstream config to "True"'
    dataset = OnlineAcousticDataset(
        transforms, dataloading_config, file_path, feat_config
    )

    offline_file_path = "preprocess/dummy_database/folds/test_fold1_fbank_offline.csv"
    file_path = "preprocess/dummy_database/folds/test_fold1_fbank_offline.csv"
    dataset_off = OfflineAcousticDataset(
        None, dataloading_config_off, file_path, feat_config
    )

    freqdim, seqlen = dataset.getDimension()
    print(freqdim, seqlen)
    test_indx = np.random.choice(dataset.__len__(), 3, replace=False)
    plt.figure(1)
    plt.title("online")
    plt.subplot(311)
    plt.imshow(
        dataset.__getitem__(test_indx[0])[0].detach().numpy(), aspect="auto", cmap="jet"
    )
    plt.title(f"SPK ID: {dataset.__getitem__(test_indx[0])[1]}, uttr: {test_indx[0]}")
    plt.subplot(312)
    plt.imshow(
        dataset.__getitem__(test_indx[1])[0].detach().numpy(), aspect="auto", cmap="jet"
    )
    plt.title(f"SPK ID: {dataset.__getitem__(test_indx[1])[1]}, uttr: {test_indx[1]}")
    plt.subplot(313)
    plt.imshow(
        dataset.__getitem__(test_indx[2])[0].detach().numpy(), aspect="auto", cmap="jet"
    )
    plt.title(f"SPK ID: {dataset.__getitem__(test_indx[2])[1]}, uttr: {test_indx[2]}")
    plt.figure(2)
    plt.title("offline")
    plt.subplot(311)
    plt.imshow(
        dataset_off.__getitem__(test_indx[0])[0].detach().numpy(),
        aspect="auto",
        cmap="jet",
    )
    plt.title(
        f"SPK ID: {dataset_off.__getitem__(test_indx[0])[1]}, uttr: {test_indx[0]}"
    )
    plt.subplot(312)
    plt.imshow(
        dataset_off.__getitem__(test_indx[1])[0].detach().numpy(),
        aspect="auto",
        cmap="jet",
    )
    plt.title(
        f"SPK ID: {dataset_off.__getitem__(test_indx[1])[1]}, uttr: {test_indx[1]}"
    )
    plt.subplot(313)
    plt.imshow(
        dataset_off.__getitem__(test_indx[2])[0].detach().numpy(),
        aspect="auto",
        cmap="jet",
    )
    plt.title(
        f"SPK ID: {dataset_off.__getitem__(test_indx[2])[1]}, uttr: {test_indx[2]}"
    )
    plt.colorbar(orientation="horizontal")
