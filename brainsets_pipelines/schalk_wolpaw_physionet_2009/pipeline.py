"""
Pipeline for the Schalk & Wolpaw PhysioNet 2009 Motor Imagery dataset.

This pipeline downloads and processes EEG motor imagery data from the PhysioNet dataset
using the MOABB dataset loader. The dataset consists of over 1500 one- and two-minute 
EEG recordings obtained from 109 volunteers performing motor imagery tasks.
"""

from argparse import ArgumentParser
from typing import NamedTuple
import logging
import datetime

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from collections import Counter

import h5py
import mne
from moabb import set_log_level
from moabb.datasets import PhysionetMI, BNCI2014_001
from moabb.paradigms import MotorImagery

from temporaldata import Data, RegularTimeSeries, Interval, ArrayDict
from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    SubjectDescription,
    DeviceDescription,
)
from brainsets.taxonomy import Species
from brainsets.utils.split import generate_train_valid_test_splits
from brainsets.utils.moabb_utils import (
    download_file,
    encode_session_values,
)
from brainsets import serialize_fn_map
from brainsets.pipeline import BrainsetPipeline


logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")


MOVEMENT_ID_MAP = {
    "left_hand": 0,
    "right_hand": 1,
    "hands": 2,
    "feet": 3,
    "rest": 4,
}


class Pipeline(BrainsetPipeline):
    brainset_id = "schalk_wolpaw_physionet_2009"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir, args) -> pd.DataFrame:
        """Generate manifest of all subjects to process.
        
        Parameters
        ----------
        raw_dir : Path
            Directory where raw data will be stored
        args : Namespace
            Command line arguments
            
        Returns
        -------
        pd.DataFrame
            Manifest with session_id as index and subject number
        """
        # Initialize the dataset to get subject list
        dataset = PhysionetMI(imagined=True, executed=False)
        
        manifest_list = []
        for subject in dataset.subject_list:
            for session in range(dataset.n_sessions):
                session_id = f"subj-{subject}_sess-{session}_imagined_movement"

                manifest_list.append({
                    "subject": subject,
                    "session": session,
                    "session_id": session_id,
                })
        
        manifest = pd.DataFrame(manifest_list).set_index("session_id")
        return manifest

    def download(self, manifest_item):
        """Download data for a single subject using MOABB.
        
        Parameters
        ----------
        manifest_item : NamedTuple
            Row from manifest containing subject information
            
        Returns
        -------
        dict
            Dictionary containing raw_data, array_data, labels, and metadata
        """
        if manifest_item.subject <= 1:
            return
        
        self.update_status("DOWNLOADING")
        
        # Initialize dataset and paradigm
        dataset = PhysionetMI(imagined=True, executed=False)
        # dataset = BNCI2014_001()
        paradigm = MotorImagery()
        
        # Download data for this subject using the utility function
        subject = manifest_item.subject
        session = manifest_item.session
        raw_data = download_file(
            dataset=dataset,
            subject=subject,
            raw_dir=self.raw_dir,
            session=session
        )
        print(type(raw_data))
        
        # Get processed data (epochs and labels) using paradigm
        array_data, labels, metadata = paradigm.get_data(
            dataset=dataset,
            subjects=[subject],
            return_epochs=False,
        )
        
        # Encode session values to integers
        session_map = encode_session_values(list(metadata["session"].values), return_dict=True)
        metadata["session"] = metadata["session"].map(session_map)
        
        # Filter data for the specific session
        session_mask = metadata["session"] == session
        array_data = array_data[session_mask]
        labels = labels[session_mask]
        
        array_data_concat = np.concatenate([trial for trial in array_data], axis=1)
        
        print("Array Data Concat:", array_data_concat.shape)
        print("Raw Data:", raw_data.get_data().shape)
        print("All Close:", np.allclose(array_data_concat, raw_data.get_data()))
        exit()
        return {
            "raw_data": raw_data, 
            "array_data": array_data, 
            "labels": labels, 
        }

    def process(self, download_output):
        """Process downloaded data into standardized format.
        
        Parameters
        ----------
        download_output : dict
            Dictionary containing raw_data, array_data, labels, and metadata
        """
        # Extract data from download output dictionary
        raw_data = download_output["raw_data"]
        array_data = download_output["array_data"]
        labels = download_output["labels"]
        
        self.update_status("Processing")
        self.processed_dir.mkdir(exist_ok=True, parents=True)
        
        # Extract subject ID from raw data
        subject_id = f"S{raw_data.info['subject_info']['his_id']}" if 'subject_info' in raw_data.info and raw_data.info['subject_info'] is not None else f"S{self._asset_id.split('_')[0][1:]}"
        
        # Create session ID
        session_id = f"{subject_id}_{raw_data.info['meas_date'].strftime('%Y%m%d')}_imagined_movement"
        
        store_path = self.processed_dir / f"{session_id}.h5"
        if store_path.exists() and not self.args.reprocess:
            self.update_status("Skipped Processing")
            return
        
        # Create brainset description
        brainset_description = BrainsetDescription(
            id="schalk_wolpaw_physionet_2009",
            origin_version="unknown",
            derived_version="1.0.0",
            source="https://moabb.neurotechx.com/docs/generated/moabb.datasets.PhysionetMI.html",
            description="This data set consists of over 1500 one- and two-minute EEG "
            "recordings, obtained from 109 volunteers.",
        )
        
        # Create subject description
        subject_description = SubjectDescription(
            id=subject_id,
            species=Species.HOMO_SAPIENS,
        )
        
        # Create session description
        session_description = SessionDescription(
            id=session_id,
            recording_date=raw_data.info["meas_date"],
        )
        
        # Create device description
        device_id = f"{subject_id}_{session_description.recording_date}"
        device_description = DeviceDescription(
            id=device_id,
        )
        
        self.update_status("Extracting EEG")
        # Extract EEG data
        eeg, units, epochs = extract_eeg_data(raw_data, array_data)
        
        self.update_status("Extracting Trials")
        # Extract trial structure
        imagined_movement_trials = extract_motor_imagery_trials(
            raw_data, array_data, labels
        )
        
        self.update_status("Generating Splits")
        # Generate k-fold CV splits
        imagined_movement_trials.allow_split_mask_overlap()
        generate_kfold_train_test_splits("MotorImagery", imagined_movement_trials)
        generate_kfold_train_test_splits("LeftRightImagery", imagined_movement_trials)
        generate_kfold_train_test_splits("RightHandFeetImagery", imagined_movement_trials)
        
        # Create data object
        data = Data(
            brainset=brainset_description,
            subject=subject_description,
            session=session_description,
            device=device_description,
            # neural activity
            eeg=eeg,
            units=units,
            # behavior
            imagined_movement_trials=imagined_movement_trials,
            domain=eeg.domain,
        )
        
        # Generate train/validation/test split
        train_trials, valid_trials, test_trials = generate_train_valid_test_splits(
            epoch_dict={"all_tasks": imagined_movement_trials},
            grid=imagined_movement_trials,
        )
        
        data.set_train_domain(train_trials)
        data.set_valid_domain(valid_trials)
        data.set_test_domain(test_trials)
        
        self.update_status("Updating Channel Info")
        # Update ch_names and ch_types
        mne_ch_names, mne_ch_types = update_channel_info(units.id)
        data.units.standard_id = mne_ch_names
        data.units.standard_types = mne_ch_types
        
        # Add montage
        montage, _ = detect_montage(mne_ch_names[mne_ch_types == "EEG"], verbose=False)
        data.montage = montage
        
        self.update_status("Extracting Statistics")
        # Extract stats
        median, iqr = extract_stats_info(eeg)
        data.units.mean = median
        data.units.std = iqr
        
        self.update_status("Extracting Group Info")
        # Extract group info
        group_info = extract_group_info(
            data.units.standard_id, data.units.standard_types, data.montage
        )
        data.group_info = group_info
        
        self.update_status("Storing")
        # Save data to disk
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


def extract_eeg_data(raw_data, array_data) -> RegularTimeSeries:
    """
    Extract EEG data from raw_data and array_data.

    Parameters
    ----------
    raw_data : mne.io.Raw
        An MNE Raw object containing EEG data for a single session.
    array_data : list or array-like
        A list or array of numpy arrays, where each array contains the EEG signal for a single trial/epoch.

    Returns
    -------
    eeg : RegularTimeSeries
        A RegularTimeSeries object containing the concatenated EEG signals from all trials, with appropriate
        sampling rate and domain.
    units : ArrayDict
        An ArrayDict containing channel ids and types for the EEG data.
    epochs : Interval
        An Interval object containing the start and end times (in seconds) for each epoch/trial.
    """
    sfreq = raw_data.info["sfreq"]
    trials = [trial for trial in array_data]

    # extract eeg signal
    eeg_signals = np.concatenate(trials, axis=1).T

    # extract epochs - for single session, we'll create synthetic epochs from trials
    num_trials = len(array_data)
    epoch_start = [0.0]
    epoch_end = []
    
    for trial in trials:
        if len(epoch_end) > 0:
            epoch_start.append(epoch_end[-1])
        epoch_end.append(epoch_start[-1] + len(trial.T) / sfreq)

    # make regular time series
    eeg = RegularTimeSeries(
        signal=eeg_signals,
        sampling_rate=sfreq,
        domain=Interval(
            start=np.array([0.0]),
            end=np.array([(len(eeg_signals) - 1) / sfreq]),
        ),
    )

    units = ArrayDict(
        id=np.array(raw_data.ch_names, dtype="U"),
        types=np.array(raw_data.get_channel_types(), dtype="U"),
    )

    epochs = Interval(
        start=np.array(epoch_start),
        end=np.array(epoch_end),
    )

    return eeg, units, epochs


def extract_motor_imagery_trials(raw_data, array_data, labels) -> Interval:
    """
    Extracts the start and end times, as well as movement labels, for each motor imagery trial.

    Parameters
    ----------
    raw_data : mne.io.Raw
        An MNE Raw object containing EEG data for a single session.
    array_data : list or array-like
        List or array of EEG data arrays, one per trial.
    labels : list or array-like
        List of movement labels corresponding to each trial.

    Returns
    -------
    trials : Interval
        An Interval object containing the start and end times (in seconds) for each trial,
        as well as movement labels and movement IDs.
    """
    sfreq = raw_data.info["sfreq"]
    trials = [trial for trial in array_data]

    start_times = []
    end_times = []
    movements = []
    for trial, label in zip(trials, labels):
        if len(start_times) == 0:
            start_time = 0.0
        else:
            start_time = end_times[-1]

        end_time = start_time + ((len(trial.T) - 1) / sfreq)

        if len(end_times) > 0 and start_time < end_times[-1]:
            # the previous trial goes into the next one
            # this happens because of numerical precision issues
            assert (
                end_times[-1] - start_time < 0.1
            ), f"found overlap between trials: start time of trial i: {start_time}, end time of trial i-1: {end_times[-1]}"

            # we can clip the end time of the last trial
            end_times[-1] = start_time

        start_times.append(start_time)
        end_times.append(end_time)
        movements.append(label)

    trials = Interval(
        start=np.array(start_times),
        end=np.array(end_times),
        timestamps=(np.array(start_times) + np.array(end_times)) / 2,
        movements=np.array(movements),
        movement_ids=np.array([MOVEMENT_ID_MAP[m] for m in movements]),
        timekeys=["start", "end", "timestamps"],
    )

    if not trials.is_disjoint():
        raise ValueError("Found overlapping trials")

    return trials


def generate_kfold_train_test_splits(task_name, trials):
    """
    Generate k-fold cross-validation train/test splits for a given task and trials.

    This function creates 5-fold stratified train/test splits for the specified task using the
    provided trials Interval object. The splits are added to the trials object as split masks,
    with names formatted as "{task_name}_fold{k}_train", "{task_name}_fold{k}_valid", and 
    "{task_name}_fold{k}_test" for each fold.

    Args
    ----
    task_name : str
        The name of the task for which to generate splits. Supported values are:
        - "MotorImagery"
        - "LeftRightImagery"
        - "RightHandFeetImagery"
    trials : Interval
        An Interval object containing trial information, including movements.

    Returns
    -------
    None
        The function modifies the trials object in-place by adding split masks.
    """
    print(f"\n\nGenerating {task_name} k-fold train/valid/test splits")

    # select labels to include based on task
    if task_name == "MotorImagery":
        include_labels = ["left_hand", "right_hand", "hands", "feet", "rest"]
    elif task_name == "LeftRightImagery":
        include_labels = ["left_hand", "right_hand"]
    elif task_name == "RightHandFeetImagery":
        include_labels = ["right_hand", "feet"]

    # select task trial indexes and labels based on task
    task_mask = np.isin(trials.movements, include_labels)
    task_trial_idx = np.where(task_mask)[0]
    task_labels = trials.movements[task_trial_idx]

    # apply stratified k-fold cv splits
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    for k, (train_index, test_index) in enumerate(
        cv.split(X=np.zeros(len(task_labels)), y=task_labels)
    ):
        train_mask = np.array(
            [
                True if trial_idx in task_trial_idx[train_index] else False
                for trial_idx in np.arange(len(task_mask))
            ]
        )
        test_mask = np.array(
            [
                True if trial_idx in task_trial_idx[test_index] else False
                for trial_idx in np.arange(len(task_mask))
            ]
        )

        # Split train_mask into train and validation using StratifiedShuffleSplit
        train_trial_idx = task_trial_idx[train_index]
        train_labels_subset = task_labels[train_index]

        # Use StratifiedShuffleSplit to split train into train and validation
        sss = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
        train_subset_index, valid_subset_index = next(
            sss.split(X=np.zeros(len(train_labels_subset)), y=train_labels_subset)
        )

        # Create train and validation masks
        train_subset_mask = np.array(
            [
                True if trial_idx in train_trial_idx[train_subset_index] else False
                for trial_idx in np.arange(len(task_mask))
            ]
        )
        valid_subset_mask = np.array(
            [
                True if trial_idx in train_trial_idx[valid_subset_index] else False
                for trial_idx in np.arange(len(task_mask))
            ]
        )

        # Print the number of labels of each class in each set
        train_labels = task_labels[train_index][train_subset_index]
        valid_labels = task_labels[train_index][valid_subset_index]
        test_labels = task_labels[test_index]

        print(f"Fold {k}:")
        print("  Train label counts:", dict(Counter(train_labels)))
        print("  Valid label counts:", dict(Counter(valid_labels)))
        print("  Test label counts:", dict(Counter(test_labels)))

        trials.add_split_mask(
            f"{task_name}_fold{k}_train", trials.select_by_mask(train_subset_mask)
        )
        trials.add_split_mask(
            f"{task_name}_fold{k}_valid", trials.select_by_mask(valid_subset_mask)
        )
        trials.add_split_mask(
            f"{task_name}_fold{k}_test", trials.select_by_mask(test_mask)
        )


def update_channel_info(ch_names):
    """Create a mapping from original channel names/types to MNE channel names/types."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "ch_names_updated.csv")
    
    on_df = pd.read_csv(csv_path)
    on_df = on_df[on_df["ch_name"].isin(ch_names)]

    # Create list of MNE channel names
    mne_ch_names = []
    for ch in ch_names:
        # Find matching row in filtered df
        match = on_df[on_df.ch_name == ch]
        if not match.empty:
            mne_ch_names.append(match.iloc[0]["std_ch_name"])

    if len(mne_ch_names) != len(ch_names):
        raise ValueError(
            "Number of MNE channel names does not match number of original channel names"
        )

    # Create list of channel types
    mne_ch_types = []
    for ch in ch_names:
        # Find matching row in filtered df
        match = on_df[on_df.ch_name == ch]
        if not match.empty:
            mne_ch_types.append(match.iloc[0]["std_ch_type"])

    if len(mne_ch_types) != len(ch_names):
        raise ValueError(
            "Number of MNE channel names does not match number of original channel names"
        )

    return np.array(mne_ch_names), np.array(mne_ch_types)


def get_mne_montage_info():
    """Get information about MNE's built-in montages.

    This function loads all built-in MNE montages and extracts key information about each one,
    including the montage family (derived from the name) and channel names. It also adds an
    'other' montage category for non-standard montages.

    Returns
    -------
    dict
        Dictionary with montage names as keys, where each value is a dict containing:
        - 'family': str, the montage family name (e.g. 'standard', 'biosemi', etc.)
        - 'ch_names': list of str, the channel names in the montage
    """
    MNE_MONTAGE_NAMES = mne.channels.get_builtin_montages()

    mne_montages = dict()
    for montage_name in MNE_MONTAGE_NAMES:
        # Load the montage to get number of channels
        mont = mne.channels.make_standard_montage(montage_name)

        # Extract family name by removing digits
        family = "".join(c for c in montage_name if not c.isdigit()).rstrip("_")

        # Store montage info in family dictionary
        mne_montages[montage_name] = dict()
        mne_montages[montage_name]["family"] = family
        mne_montages[montage_name]["ch_names"] = mont.ch_names

    # Add 'other' family with 'other' montage
    mne_montages["other"] = dict()
    mne_montages["other"]["family"] = "other"
    mne_montages["other"]["ch_names"] = ""

    # Convert defaultdict to regular dict
    return mne_montages


def find_match_percentage_by_montage(ch_names):
    """Calculate the percentage of channel names that match each MNE montage.
    For each MNE montage, calculates what percentage of the input channel names
    are present in that montage's channel list.
    """
    mne_montages = get_mne_montage_info()

    match_percentage = {}
    if len(ch_names) > 0:
        for montage_name, montage_info in mne_montages.items():
            common_ch = np.intersect1d(ch_names, montage_info["ch_names"])
            match_percentage[montage_name] = len(common_ch) / len(ch_names)

    return match_percentage


def detect_montage(ch_names, verbose=False):
    """Detect the best matching MNE montage for given channel names."""
    mne_montages = get_mne_montage_info()

    if len(ch_names) == 0:
        return "other", 0

    # Get the match percentage to each MNE montage for the given ch_names
    match_percentage_by_montage = find_match_percentage_by_montage(ch_names)
    if max(match_percentage_by_montage.values()) == 0:
        return "other", 0

    # Find montages with maximum match percentage
    max_match_percentage = max(match_percentage_by_montage.values())

    # Find all montages that have the maximum match percentage
    best_matching_montage = [
        montage
        for montage, match in match_percentage_by_montage.items()
        if match == max_match_percentage
    ]

    if len(best_matching_montage) > 1:
        valid_montages = {}
        for montage_name in best_matching_montage:
            diff = len(mne_montages[montage_name]["ch_names"]) - len(ch_names)
            if diff >= 0:  # Only consider montages with enough channels
                valid_montages[montage_name] = diff

        # If valid_montages is not empty, find the montage with the minimum difference
        # in the number of channels
        if valid_montages:
            best_matching_montage = min(valid_montages.items(), key=lambda x: x[1])[0]
            max_match_percentage = match_percentage_by_montage[best_matching_montage]
        else:
            best_matching_montage = "other"
            max_match_percentage = 0
    else:
        best_matching_montage = best_matching_montage[0]

    if verbose:
        print("Valid montages: ", valid_montages)
        print("Best matching montage: ", best_matching_montage)
        print("No. ch_names in session: ", len(ch_names))
        print(
            "No. ch_names in selected montage: ",
            len(mne_montages[best_matching_montage]["ch_names"]),
        )

    return best_matching_montage, max_match_percentage


def get_num_groups(num_channels: int):
    """Returns the number of groups based on the number of channels."""
    num_groups = list(range(1, min(num_channels, 16) + 1))

    if num_channels > 16:
        init = 1
        while num_channels // init > 16:
            num_groups.append(num_channels // init)
            init += 1

    num_groups.sort()
    return num_groups


def group_electrodes_hierarchical(num_groups, positions: np.ndarray):
    """
    Group electrodes hierarchically into specified number of groups.

    This method groups electrodes into a hierarchical structure based on their
    spatial positions using Ward's hierarchical clustering method. The clustering
    is performed on the normalized 3D positions of the electrodes.

    Args:
        num_groups (list[int]): List of number of groups to create. Must be between 1 and
            the number of electrodes.
        positions (np.ndarray): Array of electrode positions with shape (n_electrodes, 3).

    Returns:
        list[int]: A list of group indices.
    """
    # Perform hierarchical clustering
    clusters = linkage(positions, method="ward", metric="euclidean")

    labels = []
    for num_group in num_groups:
        labels.append(fcluster(clusters, num_group, criterion="maxclust"))

    # Convert to 0-based consecutive integers and create output dictionary
    return np.array(labels).T - 1


def extract_stats_info(eeg):
    """Extract statistical information from EEG data."""
    recording_data = eeg.signal

    # get median
    median = np.median(recording_data, axis=0)

    # get IQR
    q25 = np.percentile(recording_data, 25, axis=0)
    q75 = np.percentile(recording_data, 75, axis=0)
    iqr = q75 - q25

    return median, iqr


def extract_group_info(ch_names, ch_types, montage):
    """Extract hierarchical grouping information for channels."""
    eeg_num_groups = get_num_groups(len(ch_names[ch_types == "EEG"]))

    # Get the montage's reference from mne
    montage_ref = mne.channels.make_standard_montage(montage).get_positions()["ch_pos"]

    # Get each channel's position from the reference through the channels names
    ordered_channel_positions = np.array(
        [
            montage_ref[ch_name]
            for idx, ch_name in enumerate(ch_names)
            if ch_types[idx] == "EEG"
        ]
    )

    group_indexes = np.zeros((len(ch_types), len(eeg_num_groups)))
    group_idx = 0
    for group_type in np.unique(ch_types):
        if group_type == "EEG":
            continue
        if group_type == "RM":
            this_idx = -1
        else:
            this_idx = group_idx
            group_idx += 1

        group_indexes[np.where(ch_types == group_type)[0], :] = this_idx

    # Group the EEG channels hierarchically
    eeg_group_indices = group_electrodes_hierarchical(
        eeg_num_groups, ordered_channel_positions
    )
    eeg_group_indices += group_idx

    group_indexes[np.where(ch_types == "EEG")[0]] = eeg_group_indices

    group_info = ArrayDict(
        group_id=group_indexes.T.astype(np.int16),
        num_groups=np.array(eeg_num_groups).astype(np.int16),
    )

    return group_info


if __name__ == "__main__":
    raw_dir = "/network/projects/neuro-galaxy/data/raw/schalk_wolpaw_physionet_2009"
    processed_dir = "/home/mila/s/suarezul/data/processed/schalk_wolpaw_physionet_2009"
    
    # raw_dir = "/home/mila/s/suarezul/data/raw/bnci2014_001"
    # processed_dir = "/home/mila/s/suarezul/data/processed/bnci2014_001"


    from argparse import Namespace
    pipeline = Pipeline(
        raw_dir=raw_dir, 
        processed_dir=processed_dir,
        args=Namespace(redownload=False, reprocess=False)
    )
    manifest = pipeline.get_manifest(raw_dir=raw_dir, args=Namespace(redownload=False, reprocess=False))
    print(manifest)
    
    for item in manifest.itertuples():
        download_output = pipeline.download(item)
        # print(download_output["array_data"].shape)
        # print(len(download_output["labels"]))
        
        # data = pipeline.process(download_output)
        # print(data)
        
    # pipeline.run()