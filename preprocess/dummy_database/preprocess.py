# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   preprocess a sample database
#   1) Segmenting: if for the database, separate short utterances (less than 10 seconds)
#   are not available,
#   it is better to segment all speech metarials for each speaker into short (e.g.,
#   8 seconds) utterances
#   2) computing/saving features from utterances (paralell processing)
#   3) making tables (train/test/validation) for wav data (for online feature
#   extraction) and offline feature data (for offline dataloading)
#   Note: we consider cross fold validation here, therefore we create different
#   set of tables for each fold

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

from pathlib import Path
import sys
from sklearn.model_selection import StratifiedKFold
import soundfile as sf
import os

# ------------------------------ Path change ----------------------------- #
file = Path(__file__).resolve()
parent, root, subroot = file.parent, file.parents[1], file.parents[2]
sys.path.append(str(subroot))
sys.path.append(str(root))
os.chdir(root)
# ------------------------------------------------------------------------- #

from downstream.ds_runner import seed_torch
from audio.audio_utils import get_waveform, get_feat, get_config_args
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import basename
from joblib import Parallel, delayed
import shutil


def segmenting(wav_path, saved_path_dir, MaxSegLen=8):
    """Segmenting long utterances
    Args:
        wav_path (str): path of audio file
        saved_path_dir (str): directory path for saving segmented audio files
        MaxSegLen (int, optional): segmenting length. Defaults to 8.

    Yields:
        (tuple): a tuple containing:
            - (str): path of segmented files
            - (float): length of audio (ms)
    """
    wav, fs = get_waveform(wav_path, normalization=True)
    n_samples = len(wav)
    segmentsize = MaxSegLen * fs
    if not os.path.exists(saved_path_dir):
        os.makedirs(saved_path_dir, exist_ok=True)
    for ind, start in enumerate(range(0, n_samples, segmentsize)):
        end = min(start + segmentsize, n_samples)
        new_path = os.path.join(
            saved_path_dir,
            basename(wav_path).split(Path(wav_path).suffix)[0]
            + "_uttr"
            + str(ind)
            + Path(wav_path).suffix,
        )
        sf.write(new_path, wav[start:end], fs)
        yield new_path, (end - start) * 1e3 / fs


def feature_extraction(wav_path, saved_path_dir, feat_config_path):
    """extracting features from audio file
    Args:
        wav_path (str): path of audio file
        saved_path_dir (str): directory path for saving features
        feat_config_path (str): path of feature extraction config file
    Returns:
        (tuple): a tuple containing:
            - (str): path of saved feature file
            - (int): length of feature data (e.g., number of time frames)
    """
    features = get_feat(wav_path, feat_config_path).T
    if not os.path.exists(saved_path_dir):
        os.makedirs(saved_path_dir, exist_ok=True)
    new_filename = basename(wav_path).split(Path(wav_path).suffix)[0]
    new_path = os.path.join(saved_path_dir, new_filename)
    np.save(new_path, features)
    return new_path + ".npy", features.shape[1]


def preprocess_segmenting(
    wav_path,
    ID,
    SPK_ID,
    label,
    save_path_seg_wav,
    save_path_feat,
    feat_config_path,
    MaxSegLen=8,
):
    """audio segmenting and feature extraction
    Args:
        wav_path (str): path of wav file
        ID (int): speaker index
        SPK_ID (str): speaker ID
        label (int): speaker label (healthy or pathological)
        save_path_seg_wav (str): path for saving segmented data
        save_path_feat (str): path for saving segmented features
        feat_config_path (str): path of feature extraction config file
        MaxSegLen (int, optional): Audio segmenting length. Defaults to 8.

    Returns:
        (tuple): a tuple containing:
            - (DataFrame): dataframe including information of raw wav data
            - (DataFrame): dataframe including information of feature data
    """
    minlength = float(
        get_config_args(feat_config_path).get("torchaudio").get("frame_length")
    )
    segmenting_gen = segmenting(wav_path, save_path_seg_wav, MaxSegLen=MaxSegLen)
    df_wav = pd.DataFrame(
        data={
            "ID": [],
            "file_path": [],
            "length": [],
            "label": [],
            "spk_ID": [],
        }
    ).rename_axis("uttr_num")
    df_feat = pd.DataFrame(
        data={
            "ID": [],
            "file_path": [],
            "length": [],
            "label": [],
            "spk_ID": [],
        }
    ).rename_axis("file_num")
    for new_wav_path, wav_length in segmenting_gen:
        if wav_length > minlength:
            feat_path, feat_length = feature_extraction(
                new_wav_path, save_path_feat, feat_config_path
            )
            root_dir = os.path.basename(root)
            new_wav_path = os.path.join(root_dir, new_wav_path)
            feat_path = os.path.join(root_dir, feat_path)
            df_wav_seg = pd.DataFrame(
                data={
                    "ID": ["{:d}".format(ID)],
                    "file_path": [new_wav_path],
                    "length": [wav_length],
                    "label": ["{:d}".format(label)],
                    "spk_ID": [SPK_ID],
                }
            ).rename_axis("uttr_num")
            df_feat_seg = pd.DataFrame(
                data={
                    "ID": ["{:d}".format(ID)],
                    "file_path": [feat_path],
                    "length": [feat_length],
                    "label": ["{:d}".format(label)],
                    "spk_ID": [SPK_ID],
                }
            ).rename_axis("uttr_num")
            df_wav = df_wav.append(df_wav_seg, ignore_index=True)
            df_feat = df_feat.append(df_feat_seg, ignore_index=True)
    return df_wav.rename_axis("uttr_num"), df_feat.rename_axis("uttr_num")


def preprocess(wav_path, ID, SPK_ID, label, save_path_feat, feat_config_path):
    """audio feature extraction (without segmentation)
    Args:
        wav_path (str): path of wav file
        ID (int): speaker index
        SPK_ID (str): speaker ID
        label (int): speaker label (healthy or pathological)
        save_path_feat (str): path for saving segmented features
        feat_config_path (str): path of feature extraction config file

    Returns:
        (DataFrame): dataframe including information of feature data
    """
    minlength = float(
        get_config_args(feat_config_path).get("torchaudio").get("frame_length")
    )
    df_feat = pd.DataFrame(
        data={
            "ID": [],
            "file_path": [],
            "length": [],
            "label": [],
            "spk_ID": [],
        }
    ).rename_axis("file_num")
    wav, fs = get_waveform(wav_path, normalization=True)
    new_wav_path, wav_length = wav_path, len(wav) * 1e3 / fs
    if wav_length > minlength:
        feat_path, feat_length = feature_extraction(
            new_wav_path, save_path_feat, feat_config_path
        )
        root_dir = os.path.basename(root)
        feat_path = os.path.join(root_dir, feat_path)
        df_feat_seg = pd.DataFrame(
            data={
                "ID": ["{:d}".format(ID)],
                "file_path": [feat_path],
                "length": [feat_length],
                "label": ["{:d}".format(label)],
                "spk_ID": [SPK_ID],
            }
        ).rename_axis("uttr_num")
        df_feat = df_feat.append(df_feat_seg, ignore_index=True)
    return df_feat.rename_axis("uttr_num")


def remove_dirs(path):
    """removing files in path"""
    if os.path.exists(path):
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            try:
                shutil.rmtree(filepath)
            except (OSError):
                try:
                    os.remove(filepath)
                except:
                    pass


def folds_making(feat_path_csv, wav_path_csv, folds_path, feat_type, spk_index, labels):
    """making fold-wise tables from table of audio and feature data
    Args:
        feat_path_csv (str): path of feature data table
        wav_path_csv (str): path of audio data table
        folds_path (str): path of directory of folds tables for saving
        feat_type (str): feature type
        spk_index (numpy.ndarray): speaker indices
        labels (numpy.ndarray): speaker labels
    """
    table_feat = pd.read_csv(os.path.join(feat_path_csv))
    table_wav = pd.read_csv(os.path.join(wav_path_csv))
    folds_num = 3
    """no shuffling since preshuffling is done on speakers indices before gender and age 
    balancing, otherwise shuffling is needed"""
    main_Kfold_obj = StratifiedKFold(n_splits=folds_num, shuffle=False)
    val_folds_num = folds_num - 1
    val_Kfold_obj = StratifiedKFold(n_splits=val_folds_num, shuffle=False)

    for test_fold in range(1, folds_num + 1):
        D = main_Kfold_obj.split(spk_index, labels)
        for i in range(test_fold):
            train_index, test_index = next(D)
        train_tot_index = spk_index[train_index]
        train_tot_label = labels[train_index]
        test_fold_label = labels[test_index]
        test_fold_index = spk_index[test_index]

        gen = val_Kfold_obj.split(train_tot_index, train_tot_label)
        for train_nested_index, val_nested_index in gen:
            train_fold_index = train_tot_index[train_nested_index]
            val_fold_index = train_tot_index[val_nested_index]
            train_fold_label = train_tot_label[train_nested_index]
            val_fold_label = train_tot_label[val_nested_index]
            gen.close()

        print("Saving data table for fold: ", test_fold)
        table_wav[table_wav.ID.isin(list(map(str, val_fold_index)))].to_csv(
            os.path.join(folds_path, f"val_fold{test_fold}_online.csv"),
            index=False,
        )
        table_feat[table_feat.ID.isin(list(map(str, val_fold_index)))].to_csv(
            os.path.join(folds_path, f"val_fold{test_fold}_{feat_type}_offline.csv"),
            index=False,
        )

        table_wav[table_wav.ID.isin(list(map(str, test_fold_index)))].to_csv(
            os.path.join(folds_path, f"test_fold{test_fold}_online.csv"),
            index=False,
        )
        table_feat[table_feat.ID.isin(list(map(str, test_fold_index)))].to_csv(
            os.path.join(folds_path, f"test_fold{test_fold}_{feat_type}_offline.csv"),
            index=False,
        )

        table_wav[table_wav.ID.isin(list(map(str, train_fold_index)))].to_csv(
            os.path.join(folds_path, f"train_fold{test_fold}_online.csv"),
            index=False,
        )
        table_feat[table_feat.ID.isin(list(map(str, train_fold_index)))].to_csv(
            os.path.join(folds_path, f"train_fold{test_fold}_{feat_type}_offline.csv"),
            index=False,
        )


def test(uttr_num):
    """testing the data loading

    Args:
        uttr_num (int): test utterance index
    """
    import matplotlib.pyplot as plt

    final_feat_df = pd.read_csv(
        os.path.join(main_dir, f"{feat_type}_features_data.csv")
    )
    rel_path = final_feat_df["file_path"].tolist()[uttr_num]
    rel_path = os.path.relpath(rel_path, os.path.basename(root))
    spectr = np.load(rel_path, allow_pickle=True)
    plt.subplot(211)
    plt.imshow(spectr, cmap="jet", aspect="auto")
    final_wav_df = pd.read_csv(os.path.join(main_dir, "audio_data.csv"))
    rel_path = final_wav_df["file_path"].tolist()[uttr_num]
    rel_path = os.path.relpath(rel_path, os.path.basename(root))
    wav, fs = sf.read(rel_path)
    plt.subplot(212)
    plt.plot(wav)
    plt.xlim([0, len(wav)])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="preprocess arguments for the dataset."
    )
    parser.add_argument(
        "--Database",
        type=str,
        default="dummy_database/audio_data",
        help="database audio directory",
    )
    parser.add_argument(
        "--segmentdata",
        action="store_false",
        default=True,
        help="If True, segments the audio data",
    )
    parser.add_argument(
        "--init",
        action="store_false",
        default=True,
        help="If True, removes previous saved preprocessed data",
    )
    parser.add_argument(
        "--segmentlen",
        default=8,
        type=float,
        help="maximum length of segments [seconds]",
    )
    parser.add_argument(
        "--njobs",
        default=4,
        type=int,
        help="number of parallel jobs for offline feature extraction",
    )
    parser.add_argument(
        "--segmented_data_path",
        default="dummy_database/segmented_audio_data",
        type=str,
        help="Path of segmented data to be saved",
    )
    parser.add_argument(
        "--features_data_path",
        default="dummy_database/features_data",
        type=str,
        help="Path of extracted features to be saved",
    )
    parser.add_argument(
        "--config_path",
        default="../config/audio_config.yaml",
        type=str,
        help="Path of feature extraction config file",
    )
    args = parser.parse_args(args=[])

    seed_torch(0)

    test_path = []
    spk_index = []
    spk_ID = []
    for dirpath, dirnames, filenames in os.walk(args.Database):
        for f_ind, filename in enumerate(
            [f for f in sorted(filenames) if f.endswith(".wav")]
        ):
            ccomplete_path = os.path.join(dirpath, filename)
            test_path.append(ccomplete_path)
            spk_index.append(f_ind)
            spk_ID.append(os.path.splitext(filename)[0])

    spk_index = np.array(spk_index)
    total_num = len(spk_index)  # creating dummy labels for the dummy database
    labels = np.zeros_like(spk_index)
    labels[len(spk_index) // 2 :] = 1  # healthy: 0, pathological: 1

    feat_config = get_config_args(args.config_path)

    save_wav_path = os.path.join(args.segmented_data_path)
    save_feat_path = os.path.join(args.features_data_path, feat_config.get("feat_type"))

    main_dir = os.path.dirname(args.Database)
    if args.init:
        remove_dirs(save_wav_path)  # remove dirs and files inside
        remove_dirs(save_feat_path)

    if not os.path.exists(save_wav_path):
        os.mkdir(save_wav_path)
    if not os.path.exists(save_feat_path):
        os.mkdir(save_feat_path)

    print(len(test_path), f"audio files found in {args.Database}")

    if args.segmentdata:
        DF = Parallel(n_jobs=args.njobs)(
            delayed(preprocess_segmenting)(
                test_path[i],
                spk_index[i],
                spk_ID[i],
                labels[i],
                save_wav_path,
                save_feat_path,
                args.config_path,
                MaxSegLen=8,
            )
            for i in tqdm(range(total_num))
        )
        df_wav_list, df_feat_list = zip(*DF)
        final_wav_df = pd.concat(df_wav_list)
        final_wav_df.to_csv(os.path.join(main_dir, "audio_data.csv"))
    else:
        assert os.path.exists(save_wav_path), "Segmentation is needed"
        assert os.path.exists(
            os.path.join(args.Database, "audio_data.csv")
        ), "Segmented file path (audio_data.csv) is needed"
        table = pd.read_csv(os.path.join(args.Database, "audio_data.csv"))
        test_path_seg = table["file_path"].tolist()  # All paths
        spk_index_seg = table["ID"].tolist()
        spk_IDs_seg = table["spk_ID"].tolist()
        labels_seg = table["label"].tolist()
        total_num = len(test_path_seg)
        df_feat_list = Parallel(n_jobs=args.njobs)(
            delayed(preprocess)(
                test_path_seg[i],
                spk_index_seg[i],
                spk_IDs_seg[i],
                labels_seg[i],
                save_feat_path,
                args.config_path,
            )
            for i in tqdm(range(total_num))
        )

    final_feat_df = pd.concat(df_feat_list)
    feat_type = feat_config.get("feat_type")

    final_feat_df.to_csv(os.path.join(main_dir, f"{feat_type}_features_data.csv"))

    folds_path = os.path.join(main_dir, "folds")
    if not os.path.exists(folds_path):
        os.mkdir(folds_path)

    folds_making(
        os.path.join(main_dir, f"{feat_type}_features_data.csv"),
        os.path.join(main_dir, "audio_data.csv"),
        folds_path,
        feat_type,
        spk_index,
        labels,
    )

    print("loading saved data...")
    test(1)

    # NOTE: the speaker ID database should be created separately, here
    # just copied dummy folds into folds_spkIDtask just for the demo

    if os.path.exists(folds_path.replace("folds", "folds_spkID_task")):
        shutil.rmtree(folds_path.replace("folds", "folds_spkID_task"))

    shutil.copytree(folds_path, folds_path.replace("folds", "folds_spkID_task"))
