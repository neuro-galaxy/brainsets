import os
import re
import h5py
import argparse
import logging
import neuroprobe
import neuroprobe.config as neuroprobe_config
import neuroprobe.train_test_splits as neuroprobe_train_test_splits
import numpy as np
import pandas as pd
import torch
from neuroprobe import BrainTreebankSubject
from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries
from brainsets.descriptions import (
    BrainsetDescription,
    SubjectDescription,
)
from brainsets.taxonomy import RecordingTech, Species, Sex
from brainsets import serialize_fn_map
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)


def get_subject_metadata(subject_id: int) -> SubjectDescription:
    return SubjectDescription(
        id=str(subject_id),
        species=Species.HOMO_SAPIENS,
        sex=Sex.UNKNOWN,
    )


def extract_channel_data(subject: BrainTreebankSubject) -> ArrayDict:
    channel_name_basis = np.array(
        list(subject.h5_neural_data_keys.keys()), dtype=np.str_
    )
    channels = ArrayDict(
        id=np.arange(len(channel_name_basis)),
        name=channel_name_basis,  # e.g. T1bIc1
        h5_label=np.array(  # e.g. electrode_76
            [subject.h5_neural_data_keys[name] for name in channel_name_basis]
        ),
        included=np.isin(channel_name_basis, subject.electrode_labels).astype(
            np.bool_
        ),  # excludes corrupted and trigger electrodes
        type=np.ones(len(channel_name_basis)) * int(RecordingTech.STEREO_EEG),
    )
    # register localization data for each channel
    for col in subject.localization_data.columns:
        if col == "Electrode":
            continue
        loc_series = subject.localization_data.set_index("Electrode")[col]
        full_column = loc_series.reindex(channel_name_basis)
        # not all channels have localization data
        if pd.api.types.is_string_dtype(loc_series):
            full_column = full_column.fillna("").to_numpy().astype(np.str_)
        elif pd.api.types.is_numeric_dtype(loc_series):
            full_column = full_column.fillna(np.nan).to_numpy().astype(np.float32)
        else:
            raise ValueError(f"Unsupported dtype: {loc_series.dtype}")
        setattr(channels, f"localization_{col}", full_column)
    return channels


def extract_splits(
    subject_id: int,
    trial_id: int,
    *,
    lite: bool,
    nano: bool,
) -> Tuple[Dict, ArrayDict]:
    split_indices = {}
    subject = BrainTreebankSubject(
        subject_id=subject_id,
        allow_corrupted=False,
        cache=False,
        dtype=torch.float32,
        coordinates_type="lpi",
    )
    channels = extract_channel_data(subject)

    assert (
        len(neuroprobe_config.NEUROPROBE_TASKS_MAPPING) > 0
    ), "No tasks to extract splits for"
    for eval_name in neuroprobe_config.NEUROPROBE_TASKS_MAPPING:
        split_indices[eval_name] = {}

        # Allow selecting lite/full/nano from CLI.
        dtype = torch.float32
        binary_tasks = True
        max_samples = None
        start_neural_data_before_word_onset = 0
        end_neural_data_after_word_onset = neuroprobe_config.SAMPLING_RATE * 1

        folds = neuroprobe_train_test_splits.generate_splits_within_session(
            test_subject=subject,
            test_trial_id=trial_id,
            eval_name=eval_name,
            dtype=dtype,
            lite=lite,
            nano=nano,
            binary_tasks=binary_tasks,
            output_indices=True,
            start_neural_data_before_word_onset=start_neural_data_before_word_onset,
            end_neural_data_after_word_onset=end_neural_data_after_word_onset,
            max_samples=max_samples,
        )

        assert len(folds) > 0, "No folds to extract splits for"
        for fold_idx, fold in enumerate(folds):
            # register included channels for the current fold
            setattr(
                channels,
                f"included_{eval_name}_fold{fold_idx}_train",
                np.isin(
                    channels.name, fold["train_dataset"].dataset.electrode_labels
                ).astype(bool),
            )
            setattr(
                channels,
                f"included_{eval_name}_fold{fold_idx}_test",
                np.isin(
                    channels.name, fold["test_dataset"].dataset.electrode_labels
                ).astype(bool),
            )

            X_train = np.array([item[0] for item in fold["train_dataset"]])
            y_train = np.array([item[1] for item in fold["train_dataset"]])
            X_test = np.array([item[0] for item in fold["test_dataset"]])
            y_test = np.array([item[1] for item in fold["test_dataset"]])

            # derive train and test intervals from the extracted index windows
            train_intervals = Interval(
                start=X_train[:, 0].astype(np.float64)
                / neuroprobe_config.SAMPLING_RATE,
                end=X_train[:, 1].astype(np.float64) / neuroprobe_config.SAMPLING_RATE,
                label=y_train,
            )
            test_intervals = Interval(
                start=X_test[:, 0].astype(np.float64) / neuroprobe_config.SAMPLING_RATE,
                end=X_test[:, 1].astype(np.float64) / neuroprobe_config.SAMPLING_RATE,
                label=y_test,
            )

            split_indices[eval_name][fold_idx] = {
                "train_intervals": train_intervals,
                "test_intervals": test_intervals,
            }

    return split_indices, channels


def extract_neural_data(input_file: str, channels: ArrayDict) -> IrregularTimeSeries:
    with h5py.File(
        os.path.join(os.environ["ROOT_DIR_BRAINTREEBANK"], input_file), "r"
    ) as f:
        data = None
        # read channels in same order as in channels object
        for c, key in enumerate(channels.h5_label):
            if data is None:
                data = np.zeros(
                    (f["data"][key].shape[0], len(channels)), dtype=np.float32
                )
            data[:, c] = f["data"][key][:]

    seeg_data = IrregularTimeSeries(
        timestamps=np.linspace(
            0,
            data.shape[0] / neuroprobe_config.SAMPLING_RATE,
            data.shape[0],
            dtype=np.float64,
        ),
        data=data,
        domain="auto",
    )

    return seeg_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_dir", type=str, default="./processed")
    parser.add_argument(
        "--lite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Neuroprobe-Lite trial/electrode subsets (default: true).",
    )
    parser.add_argument(
        "--nano",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Neuroprobe-Nano trial/electrode subsets (default: false).",
    )
    parser.add_argument(
        "--no_splits",
        action="store_true",
        help="Skip trialization/split extraction; write only processed data.",
    )

    args = parser.parse_args()

    if args.input_file is None:
        logging.error("Input file is required (--input_file)")
        return
    if args.output_dir is None:
        logging.error("Output directory is required (--output_dir)")
        return

    brainset_description = BrainsetDescription(
        id="neuroprobe_2025",
        origin_version="1.0.0",  # btb
        derived_version=neuroprobe.__version__,
        source="https://neuroprobe.dev/",
        description="High-resolution neural datasets enable foundation models for the next generation of "
        "brain-computer interfaces and neurological treatments. The community requires rigorous benchmarks "
        "to discriminate between competing modeling approaches, yet no standardized evaluation frameworks "
        "exist for intracranial EEG (iEEG) recordings. To address this gap, we present Neuroprobe: a suite "
        "of decoding tasks for studying multi-modal language processing in the brain. Unlike scalp EEG, "
        "intracranial EEG requires invasive surgery to implant electrodes that record neural activity directly "
        "from the brain with minimal signal distortion. Neuroprobe is built on the BrainTreebank dataset, which "
        "consists of 40 hours of iEEG recordings from 10 human subjects performing a naturalistic movie viewing "
        "task. Neuroprobe serves two critical functions. First, it is a mine from which neuroscience insights "
        "can be drawn. The high temporal and spatial resolution of the labeled iEEG allows researchers to "
        "systematically determine when and where computations for each aspect of language processing occur "
        "in the brain by measuring the decodability of each feature across time and all electrode locations. "
        "Using Neuroprobe, we visualize how information flows from key language and audio processing sites in "
        "the superior temporal gyrus to sites in the prefrontal cortex. We also demonstrate the progression "
        "from processing simple auditory features (e.g., pitch and volume) to more complex language features "
        "(part of speech and word position in the sentence tree) in a purely data-driven manner. Second, as "
        "the field moves toward neural foundation models trained on large-scale datasets, Neuroprobe provides "
        "a rigorous framework for comparing competing architectures and training protocols. We found that the "
        "linear baseline on spectrogram inputs is surprisingly strong, beating frontier foundation models on "
        "many tasks. Neuroprobe is designed with computational efficiency and ease of use in mind. We make "
        "the code for Neuroprobe openly available and will maintain a public leaderboard of evaluation submissions, "
        "aiming to enable measurable progress in the field of iEEG foundation models.",
    )

    # extract subject_id and trial_id from input_file path
    input_file_basename = os.path.basename(args.input_file)
    match = re.match(r"sub_(\d+)_trial(\d{3})", input_file_basename)
    if not match:
        raise ValueError(
            f"Input file name '{input_file_basename}' does not match expected pattern 'sub_x_trialyyy.h5'"
        )
    subject_id = int(match.group(1))
    trial_id = int(match.group(2))

    logging.info(
        f"Processing file: {args.input_file} (subject_id: {subject_id}, trial_id: {trial_id})"
    )

    # extract all data
    subject = get_subject_metadata(subject_id)
    if args.no_splits:
        subject_obj = BrainTreebankSubject(
            subject_id=subject_id,
            allow_corrupted=False,
            cache=False,
            dtype=torch.float32,
            coordinates_type="lpi",
        )
        channels = extract_channel_data(subject_obj)
        split_indices = {}
        logging.info("Skipping split extraction (--no_splits)")
    else:
        split_indices, channels = extract_splits(
            subject_id,
            trial_id,
            lite=args.lite,
            nano=args.nano,
        )
        logging.info(f"Extracted {len(split_indices)} splits")
    seeg_data = extract_neural_data(args.input_file, channels)

    logging.info(
        f"Loaded and registered {len(seeg_data)} samples of neural data with {len(channels)} channels"
    )

    # register session
    data = Data(
        brainset=brainset_description,
        subject=subject,
        # neural activity
        seeg_data=seeg_data,
        channels=channels,
        # domain
        domain=seeg_data.domain,
    )

    # register all splits as root-level Intervals
    for split_name, split_indices in split_indices.items():
        for fold_idx, fold_indices in split_indices.items():
            setattr(
                data,
                f"{split_name}_fold{fold_idx}_train",
                fold_indices["train_intervals"],
            )
            setattr(
                data,
                f"{split_name}_fold{fold_idx}_test",
                fold_indices["test_intervals"],
            )

    # save data to disk
    path = os.path.join(args.output_dir, input_file_basename)
    with h5py.File(path, "w") as file:
        data.to_hdf5(file, serialize_fn_map=serialize_fn_map)
    logging.info(f"Saved data to {path}")


if __name__ == "__main__":
    main()
