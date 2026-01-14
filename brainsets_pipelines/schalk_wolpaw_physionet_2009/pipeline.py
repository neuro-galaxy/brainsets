# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne==1.11.0",
#   "moabb==1.4.3",
#   "scikit-learn==1.8.0",
# ]
# ///

"""Pipeline for PhysionetMI Motor Imagery dataset using MOABB.

This pipeline downloads and processes EEG motor imagery data from the PhysioNet
dataset using the MOABB dataset loader. The dataset consists of over 1500 one-
and two-minute EEG recordings obtained from 109 volunteers performing motor
imagery tasks.
"""

from argparse import ArgumentParser
from typing import NamedTuple
import logging
import datetime

import h5py
import numpy as np

from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery

from temporaldata import Data, RegularTimeSeries, Interval, ArrayDict
from brainsets import serialize_fn_map
from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    SubjectDescription,
    DeviceDescription,
)
from brainsets.taxonomy import Species, Task
from brainsets.moabb_pipeline import MOABBPipeline
from brainsets.utils.split import generate_stratified_folds


logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")


MOVEMENT_ID_MAP = {
    "left_hand": 0,
    "right_hand": 1,
}


class Pipeline(MOABBPipeline):
    brainset_id = "physionet_mi"
    parser = parser

    dataset_class = PhysionetMI
    paradigm_class = MotorImagery
    dataset_kwargs = {"imagined": True, "executed": False}

    def process(self, download_output):
        """Process downloaded MOABB data into standardized brainsets format.

        Parameters
        ----------
        download_output : dict
            Dictionary containing X, labels, meta, info, epochs from download()
        """
        X = download_output["X"]
        labels = download_output["labels"]
        meta = download_output["meta"]
        info = download_output["info"]
        epochs = download_output["epochs"]

        self.update_status("PROCESSING")
        self.processed_dir.mkdir(exist_ok=True, parents=True)

        subject_id = f"S{meta.iloc[0]['subject']:03d}"
        session_id = f"{subject_id}_sess-{meta.iloc[0]['session']}"

        store_path = self.processed_dir / f"{session_id}.h5"
        if store_path.exists() and not self.args.reprocess:
            self.update_status("Skipped Processing")
            return

        self.update_status("Creating Descriptions")
        brainset_description = BrainsetDescription(
            id="physionet_mi",
            origin_version="unknown",
            derived_version="1.0.0",
            source="https://moabb.neurotechx.com/docs/generated/moabb.datasets.PhysionetMI.html",
            description="PhysioNet Motor Imagery dataset: over 1500 EEG recordings "
            "from 109 volunteers performing motor imagery tasks.",
        )

        subject_description = SubjectDescription(
            id=subject_id,
            species=Species.HOMO_SAPIENS,
        )

        if info is None:
            raise ValueError("No MNE Info object available from epochs")

        recording_date = info.get("meas_date")
        if recording_date is None:
            recording_date = datetime.datetime.now()

        session_description = SessionDescription(
            id=session_id,
            recording_date=recording_date,
            task=Task.MOTOR_IMAGERY,
        )

        device_description = DeviceDescription(
            id=f"{subject_id}_{recording_date.strftime('%Y%m%d')}",
        )

        self.update_status("Extracting EEG")
        eeg, units, epoch_intervals = self._extract_eeg_data(X, info, epochs)

        self.update_status("Extracting Trials")
        trials = self._extract_motor_imagery_trials(X, labels, info)

        self.update_status("Generating Splits")
        folds = generate_stratified_folds(
            trials,
            stratify_by="movements",
            n_folds=5,
            val_ratio=0.2,
            seed=42,
        )

        folds_dict = {f"fold_{i}": fold for i, fold in enumerate(folds)}
        splits = Data(**folds_dict, domain=trials)

        self.update_status("Creating Data Object")
        data = Data(
            brainset=brainset_description,
            subject=subject_description,
            session=session_description,
            device=device_description,
            eeg=eeg,
            units=units,
            motor_imagery_trials=trials,
            splits=splits,
            domain=eeg.domain,
        )

        self.update_status("Storing")
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

        logging.info(f"Saved processed data to: {store_path}")

    def _extract_eeg_data(self, X, info, epochs):
        """Extract EEG data from epoched arrays.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (n_epochs, n_channels, n_samples)
        info : mne.Info
            MNE Info object with channel and sampling rate information
        epochs : mne.Epochs
            MNE Epochs object (used to get channel types)

        Returns
        -------
        eeg : RegularTimeSeries
            Concatenated EEG signals
        units : ArrayDict
            Channel IDs and types
        epoch_intervals : Interval
            Time intervals for each epoch
        """
        sfreq = info["sfreq"]
        n_epochs, n_channels, n_samples = X.shape

        eeg_signals = np.concatenate([X[i].T for i in range(n_epochs)], axis=0)

        epoch_starts = []
        epoch_ends = []
        current_time = 0.0

        for i in range(n_epochs):
            epoch_duration = n_samples / sfreq
            epoch_starts.append(current_time)
            epoch_ends.append(current_time + epoch_duration)
            current_time += epoch_duration

        eeg = RegularTimeSeries(
            signal=eeg_signals,
            sampling_rate=sfreq,
            domain=Interval(
                start=np.array([0.0]),
                end=np.array([(len(eeg_signals) - 1) / sfreq]),
            ),
        )

        ch_names = info["ch_names"]
        if len(epochs) > 0:
            ch_types = epochs[0].get_channel_types()
        else:
            ch_types = []
            for ch_name in ch_names:
                ch_idx = info["ch_names"].index(ch_name)
                ch_kind = info["chs"][ch_idx]["kind"]
                ch_type_map = {
                    2: "EEG",
                    3: "EOG",
                    4: "EMG",
                    5: "ECG",
                    301: "MISC",
                }
                ch_types.append(ch_type_map.get(ch_kind, "MISC"))

        units = ArrayDict(
            id=np.array(ch_names, dtype="U"),
            types=np.array(ch_types, dtype="U"),
        )

        epoch_intervals = Interval(
            start=np.array(epoch_starts),
            end=np.array(epoch_ends),
        )

        return eeg, units, epoch_intervals

    def _extract_motor_imagery_trials(self, X, labels, info):
        """Extract motor imagery trial intervals with movement labels.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (n_epochs, n_channels, n_samples)
        labels : np.ndarray
            Array of shape (n_epochs,) with movement labels
        info : mne.Info
            MNE Info object with sampling rate information

        Returns
        -------
        trials : Interval
            Interval object with start/end times and movement labels
        """
        sfreq = info["sfreq"]
        n_epochs, _, n_samples = X.shape

        start_times = []
        end_times = []
        movements = []
        movement_ids = []

        current_time = 0.0

        for i in range(n_epochs):
            epoch_duration = n_samples / sfreq

            start_times.append(current_time)
            end_times.append(current_time + epoch_duration)

            label = labels[i]
            movements.append(label)
            movement_ids.append(MOVEMENT_ID_MAP.get(label, -1))

            current_time += epoch_duration

        trials = Interval(
            start=np.array(start_times),
            end=np.array(end_times),
            timestamps=(np.array(start_times) + np.array(end_times)) / 2,
            movements=np.array(movements),
            movement_ids=np.array(movement_ids),
            timekeys=["start", "end", "timestamps"],
        )

        if not trials.is_disjoint():
            raise ValueError("Found overlapping trials")

        return trials
