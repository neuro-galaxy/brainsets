# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne==1.11.0",
#   "mne-bids==0.18",
#   "scikit-learn==1.7.2",
# ]
# ///

from argparse import ArgumentParser, Namespace

from typing import NamedTuple, Optional

import warnings
import h5py

import numpy as np
import pandas as pd
from datetime import datetime, timezone
import mne
from mne_bids import read_raw_bids, get_entity_vals, get_entities_from_fname

from pathlib import Path

from brainsets.utils.split import (
    generate_string_kfold_assignment,
    generate_stratified_folds,
)

from brainsets.utils.bids_utils import (
    fetch_ieeg_recordings,
    check_ieeg_recording_files_exist,
    group_recordings_by_entity,
    build_bids_path,
    get_subject_info,
    load_json_sidecar,
)
from brainsets.utils.mne_utils import (
    extract_measurement_date,
    extract_channels,
    extract_signal,
    concatenate_recordings,
)

from temporaldata import Data, Interval
from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    DeviceDescription,
    SubjectDescription,
    Species,
)
from brainsets import serialize_fn_map

from brainsets.pipeline import BrainsetPipeline

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")

STIM_VS_REST_TO_ID = {
    "rest": 0,
    "stim": 1,
}

STIM_FREQUENCY_TO_ID = {
    'rest': 0,
    'stim_100Hz': 1,
    'stim_200Hz': 2,
    'stim_300Hz': 3,
    'stim_400Hz': 4,
    'stim_500Hz': 5,
    'stim_800Hz': 6,
    'stim_1000Hz': 7,
    'stim_2000Hz': 8,
    'stim_5000Hz': 9,
    'stim_8000Hz': 10,
    'stim_10000Hz': 11,
    'stim_15000Hz': 12,
    'stim_16000Hz': 13,
    'stim_20000Hz': 14,
    'stim_30000Hz': 15,
    'stim_40000Hz': 16,
}


class Pipeline(BrainsetPipeline):
    brainset_id = "neurosoft_minipigs_2026"
    modality = "ieeg"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir: Path, args: Optional[Namespace]) -> pd.DataFrame:
        """Generates a manifest DataFrame of recordings discovered within a BIDS root directory.

        Args:
            raw_dir: Raw data directory assigned to this brainset
            args: Pipeline-specific arguments parsed from the command line. Expects 'bids_root'
            (path to the BIDS dataset root) as a required argument.

        Returns:
            DataFrame indexed by 'recording_id' with columns:
                - subject_id  : (str) BIDS subject identifier, e.g., 'sub-01'
                - session_id  : (str or None) BIDS session identifier, e.g., 'ses-01', or None if not present
                - task_id     : (str) BIDS task identifier (e.g., 'Sleep')
                - data_file   : (str) Relative path or filename of the iEEG data
                - fpath       : (pathlib.Path) Local directory or filepath for the subject's raw data
        Raises:
            ValueError: If no iEEG recordings are found within the provided BIDS root directory.
        """        
        if not raw_dir.exists():
            raise FileNotFoundError(
                f"BIDS root directory '{raw_dir}' does not exist."
            )
    
        recordings = fetch_ieeg_recordings(raw_dir)
        grouped_recordings = group_recordings_by_entity(
            recordings,
            fixed_entities=["subject", "session", "task", "description"],
        )
                
        # TODO: This is a hack to group by hemisphere. It should be done in a more elegant way.
        # The right way to do this is to use a different run number for each stimulation frequency
        # and use acq to denote hemisphere
        new_grouped_recordings = {}
        for group_id, recordings in grouped_recordings.items():
            hemispheres = {}
            for recording in recordings:
                recording_id = recording['recording_id']
                entities = get_entities_from_fname(recording_id, on_error="raise")
                acquisition = entities.get('acquisition', '')
                hemi = None
                if acquisition and 'L' in acquisition:
                    hemi = 'L'
                elif acquisition and 'R' in acquisition:
                    hemi = 'R'
                else:
                    hemi = 'UNKNOWN'
                hemispheres.setdefault(hemi, []).append(recording)
                        
            if 'UNKNOWN' in hemispheres:
                raise ValueError(f"Unknown hemisphere found for group {group_id}")
            
            # If there's more than one hemisphere in this group, split up
            if len(hemispheres) > 0:
                for hemi, hemi_recordings in hemispheres.items():
                    new_group_id = f"{group_id}_{hemi}H"
                    new_grouped_recordings[new_group_id] = hemi_recordings
        grouped_recordings = new_grouped_recordings
                
        manifest_list = []
        for session_id, recordings in grouped_recordings.items():
            manifest_list.append(
                {
                    "session_id": session_id,
                    "recording_ids": [recording['recording_id'] for recording in recordings],
                }
            )

        if not manifest_list:
            raise ValueError(f"No iEEG recordings found in BODS root {raw_dir}")
        
        manifest = pd.DataFrame(manifest_list).set_index("session_id")
        manifest.to_csv("manifest.csv")
        return manifest
    
    def download(self, manifest_item: pd.Series):
        self.update_status("DOWNLOADING")

        recording_ids = manifest_item.recording_ids
        
        if not getattr(self.args, "redownload", False):
            for recording_id in recording_ids:
                if check_ieeg_recording_files_exist(self.raw_dir, recording_id):
                    self.update_status(f"Already Downloaded")
                else:
                    raise FileNotFoundError(f"Recording {recording_id} not found.")
        
        return {
            "session_id": manifest_item.Index,
            "recording_ids": recording_ids,
        }

    def process(self, download_output: dict) -> Optional[tuple[Data, Path]]:
        """Process a group of recordings and create a Data object.

        This method handles common OpenNeuro processing tasks:
        1. Loads data files using MNE
        2. Extracts metadata (subject, session, device, brainset descriptions)
        3. Extracts signal and channel information
        4. Applies modality-specific channel handling via _build_channels()
        5. Creates a Data object

        Args:
            download_output: Dictionary returned by download()

        Returns:
            Tuple of (Data object, store_path), or None if already processed and skipped
        """
        self.processed_dir.mkdir(exist_ok=True, parents=True)

        session_id = download_output.get("session_id")
        entities = get_entities_from_fname(session_id, on_error="raise")
        subject_id = f"sub-{entities['subject']}"
        recording_ids = download_output.get("recording_ids")

        store_path = self.processed_dir / f"{session_id}.h5"
        if not getattr(self.args, "reprocess", False):
            if store_path.exists():
                self.update_status("Already Processed")
                return None

        # Load all recordings from the same session into a single raw object
        self.update_status(f"Loading {self.modality.upper()} recordings")
        session = {}
        for recording_id in (recording_ids):
            bids_path = build_bids_path(self.raw_dir, recording_id, self.modality)
            raw = read_raw_bids(
                bids_path,
                on_ch_mismatch='reorder',
                verbose='CRITICAL',
            )
                        
            # check if the recording has annotations
            if not raw.annotations or len(raw.annotations) == 0:
                if 'Baseline' in recording_id:
                    # warnings.warn(f"No annotations found in recording {recording_id}, adding baseline annotations")
                    _add_baseline_annotations(raw)
                else:
                    raise ValueError(f"No annotations found in recording {recording_id}")
            else:
                # add rest annotations if not present
                if 'rest' not in np.unique(raw.annotations.description):
                    _add_rest_annotations(raw)
            
            # update meas_date to the original recording timestamp
            meas_date = datetime.fromisoformat(load_json_sidecar(bids_path)['OriginalRecordingTimestamp'])
            if meas_date.tzinfo is None:
                meas_date = meas_date.replace(tzinfo=timezone.utc)
            raw.set_meas_date(meas_date)

            session[recording_id] = raw
        raw = concatenate_recordings(list(session.values()))
        _delete_boundary_annotations(raw)

        self.update_status("Extracting Metadata")
        source = "NeurosoftBioelectronics"
        dataset_description = (
            "This dataset contains electrophysiology data from minipigs undergoing acoustic stimulation at various frequencies. "
            "Each trial consists of 1 second: 0.5 seconds of stimulation followed by 0.5 seconds of rest."
        )

        brainset_description = BrainsetDescription(
            id=self.brainset_id,
            origin_version="0.0.1",
            derived_version="1.0.0",
            source=source,
            description=dataset_description,
        )

        subject_info = get_subject_info(bids_root=self.raw_dir, subject_id=subject_id)
        subject_description = SubjectDescription(
            id=subject_id,
            species=Species.UNKNOWN,
            age=subject_info["age"],
            sex=subject_info["sex"],
        )

        meas_date = extract_measurement_date(raw)
        session_description = SessionDescription(
            id=session_id, recording_date=meas_date
        )

        device_description = DeviceDescription(id=session_id)

        self.update_status(f"Extracting {self.modality.upper()} Signal")
        signal = extract_signal(raw)
        
        self.update_status("Building Channels")
        channels = extract_channels(raw)
                
        self.update_status("Extracting behavior intervals")   
        stim_vs_rest_trials = extract_stim_vs_rest_trials(raw)  
        acoustic_stim_trials = extract_acoustic_stim_trials(raw)
        
        self.update_status("Generating splits")
        splits = generate_splits(subject_id, session_id, stim_vs_rest_trials, acoustic_stim_trials)
        
        self.update_status("Creating Data Object")
        data = Data(
            brainset=brainset_description,
            subject=subject_description,
            session=session_description,
            device=device_description,
            ecog=signal,
            channels=channels,
            stim_vs_rest_trials=stim_vs_rest_trials,
            acoustic_stim_trials=acoustic_stim_trials,
            splits=splits,
            domain=signal.domain,
        )
        
        self.update_status("Storing")
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


def generate_splits(
    subject_id: str,
    session_id: str,
    stim_vs_rest_trials: Interval,
    acoustic_stim_trials: Interval,
) -> Data:
    
    subject_assignments = generate_string_kfold_assignment(
        string_id=subject_id, n_folds=3, val_ratio=0.2, seed=42
    )
    session_assignments = generate_string_kfold_assignment(
        string_id=f"{subject_id}_{session_id}",
        n_folds=3,
        val_ratio=0.2,
        seed=42,
    )
    namespaced_assignments = {
        f"intersubject_fold_{fold_idx}_assignment": assignment
        for fold_idx, assignment in enumerate(subject_assignments)
    }
    namespaced_assignments.update(
        {
            f"intersession_fold_{fold_idx}_assignment": assignment
            for fold_idx, assignment in enumerate(session_assignments)
        }
    )    
    
    stim_vs_rest_splits = {}
    if len(stim_vs_rest_trials) > 0:
        stim_vs_rest_folds = generate_stratified_folds(
            intervals=stim_vs_rest_trials,
            stratify_by="behavior_labels",
            n_folds=3,
            val_ratio=0.2,
            seed=42,
        )
        
        for k, fold_data in enumerate(stim_vs_rest_folds):
            stim_vs_rest_splits[f"stim_vs_rest_fold_{k}_train"] = fold_data.train
            stim_vs_rest_splits[f"stim_vs_rest_fold_{k}_valid"] = fold_data.valid
            stim_vs_rest_splits[f"stim_vs_rest_fold_{k}_test"] = fold_data.test

    acoustic_stim_splits = {}
    if len(acoustic_stim_trials) > 0:
        acoustic_stim_folds = generate_stratified_folds(
            intervals=acoustic_stim_trials,
            stratify_by="behavior_labels",
            n_folds=3,
            val_ratio=0.2,
            seed=42,
        )
    
        for k, fold_data in enumerate(acoustic_stim_folds):
            acoustic_stim_splits[f"acoustic_stim_fold_{k}_train"] = fold_data.train
            acoustic_stim_splits[f"acoustic_stim_fold_{k}_valid"] = fold_data.valid
            acoustic_stim_splits[f"acoustic_stim_fold_{k}_test"] = fold_data.test
            
    return Data(
        **namespaced_assignments,
        **stim_vs_rest_splits,
        **acoustic_stim_splits,
        domain=stim_vs_rest_trials | acoustic_stim_trials,
    )

def extract_stim_vs_rest_trials(raw: mne.io.Raw) -> Interval:
    """Extracts rest and stimulation trials from a raw object.

    Args:
        raw (mne.io.Raw): The raw object to extract the rest and stimulation trials from.

    Returns:
        Interval: The rest and stim trials.
    """
    annotations = raw.annotations
    onset = annotations.onset
    duration = annotations.duration
    description = annotations.description
    
    start_times = []
    end_times = []
    labels = []
    for onset, duration, description in zip(onset, duration, description):
        start_time = onset
        end_time = onset + duration
        if len(end_times) > 0 and start_time < end_times[-1]:
            # the previous trial goes into the next one
            # this happens because of numerical precision issues
            assert (
                end_times[-1] - start_time < 0.1
            ), f"found overlap between trials: start time of trial i: {start_time}, end time of trial i-1: {end_times[-1]}"

            # we can clip the end time of the last trial
            end_times[-1] = start_time

        if description != "baseline":
            start_times.append(start_time)
            end_times.append(end_time)

            if description == "rest":
                labels.append(description)
            else:
                labels.append("stim")

    # Map each unique label to an integer (in increasing order)
    label_ids = [STIM_VS_REST_TO_ID[label] for label in labels]

    return Interval(
        start=np.array(start_times),
        end=np.array(end_times),
        timestamps=(np.array(start_times) + np.array(end_times)) / 2,
        behavior_labels=np.array(labels),
        behavior_ids=np.array(label_ids),
        timekeys=["start", "end", "timestamps"],
    )


def extract_acoustic_stim_trials(raw: mne.io.Raw) -> Interval:
    """Extracts the acoustic stimulation trials from a raw object.

    Args:
        raw (mne.io.Raw): The raw object to extract the acoustic stimulation trials from.

    Returns:
        Interval: The acoustic stimulation trials.
    """
    annotations = raw.annotations
    onset = annotations.onset
    duration = annotations.duration
    description = annotations.description

    start_times = []
    end_times = []
    labels = []
    for onset, duration, description in zip(onset, duration, description):
        start_time = onset
        end_time = onset + duration
        if len(end_times) > 0 and start_time < end_times[-1]:
            # the previous trial goes into the next one
            # this happens because of numerical precision issues
            assert (
                end_times[-1] - start_time < 0.1
            ), f"found overlap between trials: start time of trial i: {start_time}, end time of trial i-1: {end_times[-1]}"

            # we can clip the end time of the last trial
            end_times[-1] = start_time

        if description != "baseline":
            start_times.append(start_time)
            end_times.append(end_time)
            
            if 'Hz' in description:
                frequency = _extract_stim_frequency(description)
                labels.append(f"stim_{frequency}Hz")
            else:
                labels.append(description)

    # Map each unique label to an integer (in increasing order)
    label_ids = [STIM_FREQUENCY_TO_ID[label] for label in labels]
        
    return Interval(
        start=np.array(start_times),
        end=np.array(end_times),
        timestamps=(np.array(start_times) + np.array(end_times)) / 2,
        behavior_labels=np.array(labels),
        behavior_ids=np.array(label_ids),
        timekeys=["start", "end", "timestamps"],
    )


def _add_baseline_annotations(raw: mne.io.Raw):
    """Adds baseline annotations to a raw object.

    Args:
        raw (mne.io.Raw): The raw object to add the baseline annotations to.
    """
    onsets = np.array(raw.times[0])
    durations = np.array(raw.times[-1] - raw.times[0])
    descriptions = np.array(["baseline"])
    
    baseline_annot = mne.Annotations(
        onset=onsets,
        duration=durations,
        description=descriptions,
        orig_time=raw.annotations.orig_time
    )
    
    raw.set_annotations(raw.annotations + baseline_annot)


def _add_rest_annotations(raw: mne.io.Raw):
    """Adds rest annotations to a raw object.

    Args:
        raw (mne.io.Raw): The raw object to add the rest annotations to.
    """
    annot = raw.annotations
    order = np.argsort(annot.onset)

    on = annot.onset[order]
    off = on + annot.duration[order]

    rest_onsets = off[:-1]
    rest_durations = on[1:] - off[:-1]

    mask = rest_durations > 0
    rest_onsets, rest_durations = rest_onsets[mask], rest_durations[mask]
    
    rest_annot = mne.Annotations(
        onset=rest_onsets,
        duration=rest_durations,
        description=["rest"] * len(rest_onsets),
        orig_time=raw.annotations.orig_time
    )
    
    raw.set_annotations(raw.annotations + rest_annot)
    

def _delete_boundary_annotations(raw: mne.io.Raw):
    """Deletes the boundary annotations from a raw object.

    Args:
        raw (mne.io.Raw): The raw object to delete the boundary annotations from.
    """
    boundary_annotations = ["BAD boundary", "EDGE boundary"]
    annot = raw.annotations
    
    # get the indices of boundary annotations to be deleted
    idx_to_be_deleted = [i for i, description in enumerate(annot.description) if description in boundary_annotations]
    
    # delete the boundary annotations    
    raw.annotations.delete(idx_to_be_deleted)


def _extract_stim_frequency(description: str) -> int:
    """Extracts the stimulation frequency as an integer from a string containing 'Hz' or 'kHz'.

    Args:
        description (str): Text containing the frequency (e.g. "stim_120Hz" or "stim_120kHz")

    Returns:
        int: The frequency value as an integer (e.g., 120 or 120000)

    Raises:
        ValueError: If no frequency is found before 'Hz' or 'kHz'.
    """
    import re
    match = re.search(r'(\d+(?:\.\d+)?)\s*Hz', description, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    match = re.search(r'(\d+(?:\.\d+)?)\s*kHz', description, re.IGNORECASE)
    if match:
        return int(match.group(1)) * 1000

    raise ValueError(f"No frequency found in description: {description}")
