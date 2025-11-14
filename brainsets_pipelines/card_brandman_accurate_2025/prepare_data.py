import argparse
import datetime
import logging
import h5py
import os
import re
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.io as sio

from temporaldata import Data, IrregularTimeSeries, Interval, ArrayDict

from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    SubjectDescription,
    DeviceDescription,
)
from brainsets.taxonomy import RecordingTech, Sex, Species, Task
from brainsets import serialize_fn_map

logging.basicConfig(level=logging.INFO)

# 20 ms bins
FREQ = 50

vocab = "abcdefghijklmnopqrstuvwxyz'- "


def clean(sentence):
    sentence = "".join(
        [
            x
            for x in sentence.lower().strip().replace("--", "")
            if x == " " or x in vocab
        ]
    )
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence


def spike_array_to_timestamps(
    arr: np.ndarray, freq: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform a matrix corresponding to a list of threshold crossings into a list
    of spike times and spike ids.
    """
    spike_timestamps = []
    spike_ids = []

    while (arr > 0).any():
        idx_time, idx_spike = np.where(arr > 0)
        # spike_timestamps.append(idx_time / freq)
        spike_timestamps.append((idx_time + 0.5) / freq)
        spike_ids.append(idx_spike)
        arr = (arr - 1) * (arr > 0)

    spike_timestamps = np.concatenate(spike_timestamps).squeeze()
    spike_ids = np.concatenate(spike_ids).squeeze().astype(np.int32)

    idx = np.argsort(spike_timestamps)
    spike_timestamps, spike_ids = spike_timestamps[idx], spike_ids[idx]

    return spike_timestamps, spike_ids

def spike_array_to_timestamps_and_counts(
    arr: np.ndarray, freq: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform a matrix corresponding to a list of threshold crossings into a list
    of spike times and spike ids.
    """
    spike_timestamps = []
    spike_ids = []
    spike_counts = []
    
    for bin_idx in range(arr.shape[0]):
        spike_timestamps.append(np.full(arr.shape[1], (bin_idx + 0.5) / freq))
        spike_ids.append(np.arange(arr.shape[1]))
        spike_counts.append(arr[bin_idx])

    spike_timestamps = np.concatenate(spike_timestamps).squeeze()
    spike_ids = np.concatenate(spike_ids).squeeze().astype(np.int32)
    spike_counts = np.concatenate(spike_counts).squeeze()

    return spike_timestamps, spike_ids, spike_counts

def spike_array_to_timestamps_and_counts_and_power(
    arr: np.ndarray, freq: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform a matrix corresponding to a list of threshold crossings into a list
    of spike times and spike ids.
    """
    spike_timestamps = []
    spike_ids = []
    spike_counts = []
    spike_power = []
    
    for bin_idx in range(arr.shape[0]):
        spike_timestamps.append(np.full(arr.shape[1] // 2, (bin_idx + 0.5) / freq))
        spike_ids.append(np.arange(arr.shape[1] // 2))
        spike_counts.append(arr[bin_idx, :128])
        spike_power.append(arr[bin_idx, 128:])

    spike_timestamps = np.concatenate(spike_timestamps).squeeze()
    spike_ids = np.concatenate(spike_ids).squeeze().astype(np.int32)
    spike_counts = np.concatenate(spike_counts).squeeze()
    spike_power = np.concatenate(spike_power).squeeze()

    return spike_timestamps, spike_ids, spike_counts, spike_power

def normalize_by_block(spikes, blockIdx, mode=3):
    unique_block_idx = np.unique(blockIdx)
    for i in unique_block_idx:
        spikes_concat = np.concatenate(
            [x for j, x in enumerate(spikes) if blockIdx[j] == i], axis=0
        )
        mean = spikes_concat.mean(axis=0, keepdims=True)
        std = spikes_concat.std(axis=0, keepdims=True) + 1e-4
        if mode == 1:
            spikes = (spikes - 0.) / 1.
        elif mode == 2:
            for idx in np.where(unique_block_idx == i)[0]:
                spikes[idx] = (spikes[idx] - mean) / std
        elif mode == 3:
            for idx in np.where(blockIdx == i)[0]:
                spikes[idx] = (spikes[idx] - mean) / std
    return spikes


"""
Layout according to paper

											   ^
											   |
											   |
											Superior
											
				Area 44 Superior 					Area 6v Superior
				192 193 208 216 160 165 178 185     062 051 043 035 094 087 079 078 
				194 195 209 217 162 167 180 184     060 053 041 033 095 086 077 076 
				196 197 211 218 164 170 177 189     063 054 047 044 093 084 075 074 
				198 199 210 219 166 174 173 187     058 055 048 040 092 085 073 072 
				200 201 213 220 168 176 183 186     059 045 046 038 091 082 071 070 
				202 203 212 221 172 175 182 191     061 049 042 036 090 083 069 068 
				204 205 214 223 161 169 181 188     056 052 039 034 089 081 067 066 
				206 207 215 222 163 171 179 190     057 050 037 032 088 080 065 064 
<-- Anterior 															Posterior -->
				Area 44 Inferior 					Area 6v Inferior 
				129 144 150 158 224 232 239 255     125 126 112 103 031 028 011 008 
				128 142 152 145 226 233 242 241     123 124 110 102 029 026 009 005 
				130 135 148 149 225 234 244 243     121 122 109 101 027 019 018 004 
				131 138 141 151 227 235 246 245     119 120 108 100 025 015 012 006 
				134 140 143 153 228 236 248 247     117 118 107 099 023 013 010 003 
				132 146 147 155 229 237 250 249     115 116 106 097 021 020 007 002 
				133 137 154 157 230 238 252 251     113 114 105 098 017 024 014 000 
				136 139 156 159 231 240 254 253     127 111 104 096 030 022 016 001 
				
										    Inferior
											   |
											   |
											   v
"""


def get_unit_metadata():
    recording_tech = RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS
    unit_meta = []
    for i in range(128):
        unit_id = f"group_0/elec{i:03d}/multiunit_0"
        unit_meta.append(
            {"id": unit_id, "unit_number": i, "count": -1, "type": int(recording_tech)}
        )
    unit_meta_df = pd.DataFrame(unit_meta)
    units = ArrayDict.from_dataframe(unit_meta_df, unsigned_to_long=True)

    return units


def stack_trials(mat_data):
    """Stack all the trial data into a single array."""
    # threshold_xings = mat_data["tx1"].squeeze().tolist()
    threshold_xings = mat_data["input_features"]

    threshold_xings = [x[:, :128] for x in threshold_xings]
    trial_times = [x.shape[0] / FREQ for x in threshold_xings]
    trial_bounds = np.cumsum(np.concatenate([[0], trial_times]))
    
    spike_power = mat_data["spikePow"].squeeze().tolist()
    spike_power = [x[:, :128] for x in spike_power]
    sentences_feats = []
    for i in range(len(threshold_xings)):
        sentences_feats.append(
            np.concatenate(
                [
                    threshold_xings[i],
                    spike_power[i],
                ],
                axis=-1
            )
        )
    
    sentences_feats = normalize_by_block(sentences_feats, mat_data["blockIdx"].squeeze())
    sentences_feats = [spike_array_to_timestamps_and_counts_and_power(x, FREQ) for x in sentences_feats]
    # check that all spike time stamps are within the trial bound
    for i, (x, _, _, _) in enumerate(sentences_feats):
        assert (x >= 0.).all() and ((x + trial_bounds[i]) < trial_bounds[i + 1]).all()

    # Create the corresponding intervals for that dataset
    trials = Interval(start=trial_bounds[:-1], end=trial_bounds[1:])

    units = get_unit_metadata()

    timestamps = np.concatenate(
        [x + trial_bounds[i] for i, (x, _, _, _) in enumerate(sentences_feats)]
    )
    unit_index = np.concatenate([y for (_, y, _, _) in sentences_feats])
    unit_count = np.concatenate([z for (_, _, z, _) in sentences_feats])
    unit_power = np.concatenate([k for (_, _, _, k) in sentences_feats])
    
    spikes = IrregularTimeSeries(
        timestamps=np.array(timestamps),
        unit_index=np.array(unit_index),
        unit_count=np.array(unit_count),
        unit_power=np.array(unit_power),
        domain=trials,
    )

    sentences = [clean(x) for x in mat_data["sentenceText"].squeeze().tolist()]

    # Write things out phonetically, which is the actual target.
    phonemes = [
        speech.to_phonemes(x) for x in mat_data["sentenceText"].squeeze().tolist()
    ]
    cypher = [x for x, _ in phonemes]

    # We have to pad phonemes in order to store the data in a regular array.
    phonemes = [x for _, x in phonemes]
    phoneme_len = [len(x) for x in phonemes]
    max_len = max(phoneme_len)
    phonemes = [
        np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=0)
        for x in phonemes
    ]

    # We assign the labels to the start of each trial.
    sentences = IrregularTimeSeries(
        timestamps=trial_bounds[:-1],
        block_num=mat_data["blockIdx"].squeeze(),
        transcript=np.array(sentences),
        phonemes=np.array(phonemes),
        phonemes_len=np.array(phoneme_len),
        phonemes_cypher=np.array(cypher),
        domain="auto",
    )

    return spikes, units, trials, sentences


def get_subject():
    return SubjectDescription(
        id="T12",
        species=Species.from_string("HUMAN"),
        sex=Sex.from_string("M"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_dir", type=str, default="./processed")

    args = parser.parse_args()

    brainset_description = BrainsetDescription(
        id="card_brandman_accurate_2025",
        origin_version="1.0.0",
        derived_version="1.0.0",
        source="https://www.kaggle.com/competitions/brain-to-text-25/data",
        description="The dataset used in this competition consists of 10,948 sentences spoken "
                    "by a single research participant as described in Card et al. “An Accurate "
                    "and Rapidly Calibrating Speech Neuroprosthesis” (2024) New England Journal "
                    "of Medicine. For each sentence, we provide the transcript of what the participant "
                    "was attempting to say, along with the corresponding time series of neural spiking "
                    "activity recorded from 256 microelectrodes in speech motor cortex. The dataset "
                    "contains predefined train, val, and test partitions. The train and val partitions "
                    "include the sentence labels",
    )

    logging.info(f"Processing file: {args.input_file}")

    subject = get_subject()  # Participant pseudonym

    # load_keys = ["sentenceText", "tx1", "blockIdx", "spikePow"]
    # mat_data = sio.loadmat(args.input_file, variable_names=load_keys)
    # num_train_trials = mat_data["tx1"].shape[1]

    # do the above but for h5py
    mat_data = {}
    with h5py.File(args.input_file, "r") as f:
        for trial in f.keys():
            mat_data[trial] = {}
            for key in f[trial].keys():
                mat_data[trial][key] = f[trial][key][:]
            transcript_text = "".join(chr(c) for c in mat_data[trial]["transcription"])
            logging.info(f"Loaded trial {trial} with transcription: {transcript_text}")
    exit()

    # Also concatenate the data from the validation split. Note the validation
    # split is actually called test here.
    valid_path = str.replace(args.input_file, "train", "test")
    logging.info(f"Adding validation data: {valid_path}")
    valid_mat_data = sio.loadmat(valid_path, variable_names=load_keys)
    num_valid_trials = valid_mat_data["tx1"].shape[1]

    for key in load_keys:
        mat_data[key] = np.concatenate(
            [mat_data[key].squeeze(), valid_mat_data[key].squeeze()], axis=0
        )

    session_date = os.path.splitext(os.path.basename(args.input_file))[0]
    _, year, month, day = session_date.split(".")
    recording_date = datetime.datetime(int(year), int(month), int(day))
    device_id = str.replace(session_date, ".", "_")
    session_id = device_id

    session_description = SessionDescription(
        id=session_id,
        recording_date=recording_date,
        task=Task.CONTINUOUS_SPEAKING_SENTENCE,
    )

    device_description = DeviceDescription(
        id=device_id,
        recording_tech=RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS,
    )

    spikes, units, trials, sentences = stack_trials(mat_data)

    data = Data(
        brainset=brainset_description,
        subject=subject,
        session=session_description,
        device=device_description,
        spikes=spikes,
        units=units,
        trials=trials,
        speech=sentences,
        domain=spikes.domain,
    )

    train_mask = np.zeros(num_train_trials + num_valid_trials, dtype=bool)
    train_mask[:num_train_trials] = True

    train_split = trials.select_by_mask(train_mask)
    valid_split = trials.select_by_mask(~train_mask)

    data.set_train_domain(train_split)
    data.set_valid_domain(valid_split)

    path = os.path.join(args.output_dir, f"{session_id}.h5")
    with h5py.File(path, "w") as f:
        data.to_hdf5(f, serialize_fn_map=serialize_fn_map)


if __name__ == "__main__":
    main()
