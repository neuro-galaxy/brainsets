# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "neuroprobe==0.1.7",
# ]
# ///

from functools import cache
from itertools import product
import os
import re
import h5py
import logging
import shutil
import urllib.request
import zipfile
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from typing import Any, Dict, List, Literal, Tuple, Optional, get_args

from brainsets.pipeline import BrainsetPipeline
from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries
from brainsets.descriptions import (
    BrainsetDescription,
    SubjectDescription,
)
from brainsets.taxonomy import RecordingTech, Species, Sex
from brainsets import serialize_fn_map


import logging
logging.basicConfig(level=logging.INFO)


BASE_URL = "https://braintreebank.dev"
COMMON_ASSETS = [
    "data/localization.zip",
    "data/subject_timings.zip",
    "data/subject_metadata.zip",
    "data/electrode_labels.zip",
    "data/speaker_annotations.zip",
    "data/scene_annotations.zip",
    "data/transcripts.zip",
    "data/trees.zip",
    "data/movie_frames.zip",
    "data/corrupted_elec.json",
]

FILENAME_MAP = lambda sub_id, trial_id: f"sub_{sub_id}_trial{trial_id:03d}"
ASSET_PATH_MAP = lambda sub_id, trial_id: f"data/subject_data/sub_{sub_id}/trial{trial_id:03d}/{FILENAME_MAP(sub_id, trial_id)}.h5.zip"
FILENAME_PATTERN = r"sub_(\d+)_trial(\d{3})"

INCL_CHANNEL_KEY_PREFIX_MAP = lambda lite, nano, binary_tasks, eval_setting, eval_name, fold_idx: \
    f"included$lite{int(lite)}$nano{int(nano)}$binary_tasks{int(binary_tasks)}${eval_setting}${eval_name}$fold{fold_idx}"
EvalSettingOption = Literal["within_session", "cross_session", "cross_subject"]
ALL_EVAL_SETTINGS = {
    "lite": [True, False],
    "nano": [True, False],
    "binary_tasks": [True, False],
    "eval_setting": get_args(EvalSettingOption),
}
SETTING_SPLIT_KEY_MAP = lambda lite, nano, binary_tasks, eval_setting, key: \
    f"{'lite' if lite else ('nano' if nano else 'full')}.{'binary' if binary_tasks else 'multiclass'}.{eval_setting}.{key}"
SPLIT_KEY_MAP = lambda eval_name, fold_idx, split_type: f"{eval_name}.fold{fold_idx}.{split_type}_intervals"

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")
parser.add_argument(
    "--no_splits",
    action="store_true",
    help="Skip split extraction; write only processed data.",
)


class Pipeline(BrainsetPipeline):
    brainset_id = "neuroprobe_2025"
    parser = parser

    @classmethod
    def get_manifest(
        cls,
        raw_dir: Path,
        args: Optional[Namespace],
    ) -> pd.DataFrame:
        _prepare_neuroprobe_lib(raw_dir)

        raw_dir.mkdir(exist_ok=True, parents=True)

        # construct manifest for selected mode
        manifest_list = [
            {
                "subject_id": subject_id,
                "trial_id": trial_id,
                "filename": FILENAME_MAP(subject_id, trial_id),
            }
            for subject_id, trial_id in neuroprobe_config.NEUROPROBE_FULL_SUBJECT_TRIALS
        ]

        # trials (manifest items) are indexed by filename
        manifest = pd.DataFrame(manifest_list).set_index("filename")
        return manifest

    def download(self, manifest_item):
        # download common assets
        self.update_status("Downloading common assets")
        _ensure_common_assets(
            self.raw_dir,
            COMMON_ASSETS,
            overwrite=bool(self.args and self.args.redownload)
        )
        
        # download subject data
        self.update_status("DOWNLOADING")
        extracted_path = _download_and_extract(
            self.raw_dir,
            ASSET_PATH_MAP(manifest_item.subject_id, manifest_item.trial_id),
            overwrite=bool(self.args and self.args.redownload),
        )
        return extracted_path

    def process(self, fpath):
        self.update_status("Processing")
        self.processed_dir.mkdir(exist_ok=True, parents=True)
        output_path = self.processed_dir / Path(fpath).name
        if output_path.exists() and not (self.args and self.args.reprocess):
            logging.info(f"Skipping processing for {output_path} because it exists")
            self.update_status("Skipped Processing")
            return

        brainset_description = get_brainset_description()

        # extract subject_id and trial_id from input_file path
        input_file_basename = os.path.basename(fpath)
        match = re.match(FILENAME_PATTERN, input_file_basename)
        if not match:
            raise ValueError(
                f"Input file name '{input_file_basename}' does not match expected pattern 'sub_x_trialyyy.h5'"
            )
        subject_id = int(match.group(1))
        trial_id = int(match.group(2))

        logging.info(
            f"Processing {fpath} to {self.processed_dir}\n"
            f"  subject_id: {subject_id}\n"
            f"  trial_id: {trial_id}\n"
            f"  no_splits: {self.args.no_splits}"
        )

        # subject metadata
        self.update_status("Extracting subject metadata")
        subject = _get_subject_metadata(subject_id)
        
        # extract channel data & splits (if not disabled)
        if self.args.no_splits:
            self.update_status("Extracting channel data")
            subject_obj = neuroprobe.BrainTreebankSubject(
                subject_id=subject_id,
                allow_corrupted=False,
                cache=False,
                dtype=torch.float32,
                coordinates_type="lpi",
            )
            channels = _extract_channel_data(subject_obj)
            split_indices = {}
            logging.info("Skipping split extraction (--no_splits)")
        else:
            split_indices = self.iterate_extract_splits(subject_id, trial_id)
            channels = self.all_channels[subject_id]
            logging.info(f"Extracted {len(split_indices)} splits")

        # extract neural data
        self.update_status("Extracting neural data")
        seeg_data = _extract_neural_data(fpath, channels)
        logging.info(
            f"Loaded and registered {len(seeg_data)} samples of neural data with {len(channels)} channels"
        )

        # register session
        self.update_status("Registering session")
        data = Data(
            brainset=brainset_description,
            subject=subject,
            # neural activity
            seeg_data=seeg_data,
            channels=channels,
            # domain
            splits=Data(domain=seeg_data.domain),
            domain=seeg_data.domain,
        )

        # register all splits as root-level Intervals
        if not self.args.no_splits:
            self.update_status("Registering splits")
            for split_key, intervals in split_indices.items():
                _data_set_nested_attribute(data.splits, split_key, intervals)

        # save data to disk
        self.update_status("Storing")
        path = self.processed_dir / Path(fpath).name
        with h5py.File(path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)
        logging.info(f"Saved data to {path}")
    
    def iterate_extract_splits(self, subject_id: int, trial_id: int) -> Tuple[Dict, ArrayDict]:
        if not hasattr(self, "all_subjects"):
            # load all subjects and channels once
            # channels will be populated with subsets for each fold
            self.all_subjects = {
                subject_id: neuroprobe.BrainTreebankSubject(
                    subject_id=subject_id,
                    allow_corrupted=False,
                    cache=False,
                    dtype=torch.float32,
                    coordinates_type="lpi",
                )
                for subject_id, _ in neuroprobe_config.NEUROPROBE_FULL_SUBJECT_TRIALS
            }
            self.all_channels = {
                subject_id: _extract_channel_data(self.all_subjects[subject_id])
                for subject_id in self.all_subjects
            }

        all_combinations = product(*ALL_EVAL_SETTINGS.values())
        split_indices = {}
        for setting_combination in all_combinations:
            lite, nano, binary_tasks, eval_setting = setting_combination
            if lite and nano:  # lite and nano cannot be True at the same time
                continue
            if eval_setting == "cross_session" and nano:  # cross-session splits are not supported for nano
                continue
            if lite and (subject_id, trial_id) not in neuroprobe_config.NEUROPROBE_LITE_SUBJECT_TRIALS:
                continue
            if nano and (subject_id, trial_id) not in neuroprobe_config.NEUROPROBE_NANO_SUBJECT_TRIALS:
                continue
            if eval_setting == "cross_subject" and subject_id == neuroprobe_config.DS_DM_TRAIN_SUBJECT_ID:
                # don't extract test split for the training subject and trial
                continue

            self.update_status(f"Extracting splits (lite={lite}, nano={nano}, binary_tasks={binary_tasks}, eval_setting={eval_setting})")
            _split_indices = _extract_and_structure_splits(
                all_subjects=self.all_subjects,
                all_channels=self.all_channels,
                subject_id=subject_id,
                trial_id=trial_id,
                lite=lite,
                nano=nano,
                binary_tasks=binary_tasks,
                eval_setting=eval_setting,
            )
            for key, value in _split_indices.items():
                split_indices[SETTING_SPLIT_KEY_MAP(lite, nano, binary_tasks, eval_setting, key)] = value
        
        return split_indices


def get_brainset_description() -> BrainsetDescription:
    return BrainsetDescription(
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


def _get_subject_metadata(subject_id: int) -> SubjectDescription:
    return SubjectDescription(
        id=str(subject_id),
        species=Species.HOMO_SAPIENS,
        sex=Sex.UNKNOWN,
    )


def _prepare_neuroprobe_lib(raw_dir: Path) -> None:
    # neuroprobe requires the raw data to be set as an environment variable
    os.environ["ROOT_DIR_BRAINTREEBANK"] = str(raw_dir)
    global neuroprobe, neuroprobe_config, neuroprobe_train_test_splits
    import neuroprobe
    import neuroprobe.config as neuroprobe_config
    import neuroprobe.train_test_splits as neuroprobe_train_test_splits


def _data_set_nested_attribute(data: Data, path: str, value: Any) -> None:
    # Split key by dots, resolve using getattr
    components = path.split(".")
    obj = data
    for c in components[:-1]:
        try:
            obj = getattr(obj, c)
        except AttributeError:
            setattr(obj, c, Data(domain=data.domain))
            obj = getattr(obj, c)

    setattr(obj, components[-1], value)


# subject is neuroprobe.BrainTreebankSubject object
def _extract_channel_data(subject) -> ArrayDict:
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


def _extract_and_structure_splits(
    all_subjects: Dict[int, object],
    all_channels: Dict[int, ArrayDict],
    subject_id: int,
    trial_id: int,
    lite: bool,
    nano: bool,
    binary_tasks: bool,
    eval_setting: EvalSettingOption,
) -> Tuple[Dict, ArrayDict]:
    split_indices = {}

    assert (
        len(neuroprobe_config.NEUROPROBE_TASKS_MAPPING) > 0
    ), "No tasks to extract splits for"
    for eval_name in neuroprobe_config.NEUROPROBE_TASKS_MAPPING:
        # load splits via neuroprobe API
        folds = _extract_splits(
            all_subjects=all_subjects,
            subject_id=subject_id,
            trial_id=trial_id,
            lite=lite,
            nano=nano,
            binary_tasks=binary_tasks,
            eval_name=eval_name,
            eval_setting=eval_setting,
        )

        # load channels for each fold
        assert len(folds) > 0, "No folds to extract splits for"
        channels = all_channels[subject_id]
        for fold_idx, fold in enumerate(folds):
            # register included channels for the current fold
            key_prefix = INCL_CHANNEL_KEY_PREFIX_MAP(
                lite=lite,
                nano=nano,
                binary_tasks=binary_tasks,
                eval_setting=eval_setting,
                eval_name=eval_name,
                fold_idx=fold_idx,
            )
            setattr(
                channels,
                f"{key_prefix}$train",
                np.isin(
                    channels.name, _get_electrode_labels(fold["train_dataset"])
                ).astype(bool),
            )
            setattr(
                channels,
                f"{key_prefix}$val",
                np.isin(
                    channels.name, _get_electrode_labels(fold["val_dataset"])
                ).astype(bool),
            )
            setattr(
                channels,
                f"{key_prefix}$test",
                np.isin(
                    channels.name, _get_electrode_labels(fold["test_dataset"])
                ).astype(bool),
            )

            train_items = [
                _unpack_dataset_item(item) for item in fold["train_dataset"]
            ]
            train_windows, train_labels = zip(*train_items)
            X_train = np.array(train_windows)
            y_train = np.array(train_labels)

            val_items = [_unpack_dataset_item(item) for item in fold["val_dataset"]]
            val_windows, val_labels = zip(*val_items)
            X_val = np.array(val_windows)
            y_val = np.array(val_labels)

            test_items = [_unpack_dataset_item(item) for item in fold["test_dataset"]]
            test_windows, test_labels = zip(*test_items)
            X_test = np.array(test_windows)
            y_test = np.array(test_labels)

            # derive train and test intervals from the extracted index windows
            train_intervals = Interval(
                start=X_train[:, 0].astype(np.float64)
                / neuroprobe_config.SAMPLING_RATE,
                end=X_train[:, 1].astype(np.float64) / neuroprobe_config.SAMPLING_RATE,
                label=y_train,
            )
            val_intervals = Interval(
                start=X_val[:, 0].astype(np.float64)
                / neuroprobe_config.SAMPLING_RATE,
                end=X_val[:, 1].astype(np.float64) / neuroprobe_config.SAMPLING_RATE,
                label=y_val,
            )
            test_intervals = Interval(
                start=X_test[:, 0].astype(np.float64) / neuroprobe_config.SAMPLING_RATE,
                end=X_test[:, 1].astype(np.float64) / neuroprobe_config.SAMPLING_RATE,
                label=y_test,
            )

            split_indices[SPLIT_KEY_MAP(eval_name, fold_idx, "train")] = train_intervals
            split_indices[SPLIT_KEY_MAP(eval_name, fold_idx, "val")] = val_intervals
            split_indices[SPLIT_KEY_MAP(eval_name, fold_idx, "test")] = test_intervals

    return split_indices


def _extract_splits(
    all_subjects: Dict[int, object],
    subject_id: int,
    trial_id: int,
    lite: bool,
    nano: bool,
    binary_tasks: bool,
    eval_name: str,
    eval_setting: EvalSettingOption,
):
    dtype = torch.float32
    max_samples = None
    start_neural_data_before_word_onset = 0
    end_neural_data_after_word_onset = neuroprobe_config.SAMPLING_RATE * 1

    if eval_setting == "within_session":   
        folds = neuroprobe_train_test_splits.generate_splits_within_session(
            test_subject=all_subjects[subject_id],
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
        return folds
    elif eval_setting == "cross_session":
        folds = neuroprobe_train_test_splits.generate_splits_cross_session(
            test_subject=all_subjects[subject_id],
            test_trial_id=trial_id,
            eval_name=eval_name,
            dtype=dtype,
            lite=lite,
            binary_tasks=binary_tasks,
            output_indices=True,
            start_neural_data_before_word_onset=start_neural_data_before_word_onset,
            end_neural_data_after_word_onset=end_neural_data_after_word_onset,
            max_samples=max_samples,
            include_all_other_trials=False,  # TODO double-check this
        )
        return folds
    elif eval_setting == "cross_subject":
        folds = neuroprobe_train_test_splits.generate_splits_cross_subject(
            all_subjects=all_subjects,
            test_subject_id=subject_id,
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
        return folds


def _extract_neural_data(input_file: Path, channels: ArrayDict) -> IrregularTimeSeries:
    with h5py.File(input_file, "r") as f:
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


def _download_file(url: str, dest: Path, *, overwrite: bool) -> None:
    if dest.exists() and not overwrite:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _download_and_extract(raw_dir: Path, href: str, *, overwrite: bool) -> Path:
    url = f"{BASE_URL}/{href}"
    basename = Path(href).name
    zip_path = raw_dir / basename

    if href.endswith(".zip"):
        extracted_path = raw_dir / Path(href).stem
        if extracted_path.exists() and not overwrite:
            return extracted_path

        _download_file(url, zip_path, overwrite=overwrite)
        with zipfile.ZipFile(zip_path, "r") as zip_handle:
            zip_handle.extractall(raw_dir)
        zip_path.unlink()
        return extracted_path

    extracted_path = raw_dir / basename
    _download_file(url, extracted_path, overwrite=overwrite)
    return extracted_path


def _ensure_common_assets(raw_dir: Path, assets: list[str], *, overwrite: bool) -> None:
    for asset in assets:
        _download_and_extract(raw_dir, asset, overwrite=overwrite)


def _get_electrode_labels(dataset) -> np.ndarray:
    if hasattr(dataset, "electrode_labels"):
        return dataset.electrode_labels
    if hasattr(dataset, "dataset"):
        return _get_electrode_labels(dataset.dataset)
    raise AttributeError(
        f"{type(dataset).__name__} has no electrode_labels and no nested dataset"
    )


def _unpack_dataset_item(item) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(item, dict):
        return item["data"], item["label"]
    return item[0], item[1]