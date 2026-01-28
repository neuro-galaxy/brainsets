"""Base pipeline class for MOABB (Mother of All BCI Benchmarks) datasets.

This module provides a reusable base class for integrating MOABB datasets into
the brainsets pipeline framework. Subclasses define the dataset and paradigm
classes, and implement paradigm-specific processing logic.
"""

from abc import abstractmethod
from typing import Dict, Any, Type, Optional
from pathlib import Path
import os
import pandas as pd
import numpy as np
import datetime
import logging
import h5py

from moabb.datasets.base import BaseDataset
from moabb.paradigms.base import BaseParadigm
from temporaldata import Data, RegularTimeSeries, Interval, ArrayDict
from brainsets.pipeline import BrainsetPipeline
from brainsets import serialize_fn_map
from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    SubjectDescription,
    DeviceDescription,
)
from brainsets.taxonomy import Species, Task
from brainsets.utils.split import generate_trial_folds


class MOABBPipeline(BrainsetPipeline):
    """Base class for MOABB dataset pipelines.

    Subclasses must define:
        - brainset_id: str
        - dataset_class: Type[BaseDataset]
        - paradigm_class: Type[BaseParadigm]
        - dataset_kwargs: Dict[str, Any] (optional, defaults to {})
        - paradigm_kwargs: Dict[str, Any] (optional, defaults to {})
        - task: Task (e.g., Task.MOTOR_IMAGERY, Task.P300)
        - trial_key: str (e.g., "motor_imagery_trials", "p300_trials")
        - label_field: str (e.g., "movements", "targets")
        - id_field: str (e.g., "movement_ids", "target_ids")
        - stratify_field: str (e.g., "movements", "targets")
        - label_map: Dict[str, int] (mapping from label strings to integer IDs)

    Subclasses must implement:
        - get_brainset_description(): Return dataset-specific BrainsetDescription

    The base class handles:
        - Manifest generation from dataset metadata
        - Data download via MOABB paradigm.get_data()
        - Session filtering
        - MNE download directory setup
        - Default process() workflow (EEG extraction, trial extraction, splits, storage)
    """

    dataset_class: Type[BaseDataset]
    paradigm_class: Type[BaseParadigm]
    dataset_kwargs: Dict[str, Any] = {}
    paradigm_kwargs: Dict[str, Any] = {}

    task: Task
    trial_key: str
    label_field: str
    id_field: str
    stratify_field: str
    label_map: Dict[str, int]

    @classmethod
    def get_dataset(cls) -> BaseDataset:
        """Instantiate the MOABB dataset with configured kwargs."""
        return cls.dataset_class(**cls.dataset_kwargs)

    @classmethod
    def get_paradigm(cls) -> BaseParadigm:
        """Instantiate the MOABB paradigm with configured kwargs."""
        kwargs = {k: v for k, v in cls.paradigm_kwargs.items() if v is not None}
        return cls.paradigm_class(**kwargs)

    @classmethod
    def get_manifest(cls, raw_dir: Path, args) -> pd.DataFrame:
        """Auto-generate manifest from MOABB dataset metadata.

        Creates a manifest with one row per subject-session combination.
        The manifest index is 'session_id' formatted as 'subj-{subject:03d}_sess-{session}'.

        Parameters
        ----------
        raw_dir : Path
            Raw data directory (unused but required by base class)
        args : Namespace
            Pipeline arguments (unused but required by base class)

        Returns
        -------
        pd.DataFrame
            Manifest with columns: subject, session, session_id (as index)
        """
        dataset = cls.get_dataset()
        manifest_list = []

        for subject in dataset.subject_list:
            for session in range(dataset.n_sessions):
                # Make sure session is an integer
                session_id = f"subj-{subject:03d}_sess-{session}"
                manifest_list.append(
                    {
                        "subject": subject,
                        "session": session,
                        "session_id": session_id,
                    }
                )

        return pd.DataFrame(manifest_list).set_index("session_id")

    def download(self, manifest_item) -> Dict[str, Any]:
        """Download and extract data using MOABB paradigm.

        This method:
        1. Sets up MNE download directory
        2. Calls paradigm.get_data() to get epoched arrays
        3. Filters results to the specific session from manifest_item

        Parameters
        ----------
        manifest_item : NamedTuple
            Row from manifest containing subject and session info

        Returns
        -------
        dict
            Dictionary containing:
            - X: np.ndarray of shape (n_epochs, n_channels, n_samples)
            - labels: np.ndarray of shape (n_epochs,) with event names
            - meta: pd.DataFrame with columns: subject, session, run
            - info: MNE Info object from first epoch (for channel info)
        """
        self.update_status("DOWNLOADING")

        self.raw_dir.mkdir(exist_ok=True, parents=True)
        os.environ["MNE_DATA"] = str(self.raw_dir.resolve())

        dataset = self.get_dataset()
        paradigm = self.get_paradigm()
        subject = int(manifest_item.subject)
        session = int(manifest_item.session)

        epochs, labels, meta = paradigm.get_data(
            dataset=dataset,
            subjects=[subject],
            return_epochs=True,
        )

        if len(epochs) == 0:
            raise ValueError(
                f"No epochs found for subject {subject}, session {session}"
            )

        session_values = sorted(meta["session"].unique())
        if isinstance(session, int):
            if session < len(session_values):
                session_key = session_values[session]
            else:
                raise ValueError(
                    f"Session index {session} out of range for subject {subject}. "
                    f"Available {len(session_values)} sessions: {list(session_values)}"
                )
        else:
            session_key = str(session)
            if session_key not in session_values:
                raise ValueError(
                    f"Session {session_key} not found for subject {subject}. "
                    f"Available sessions: {list(session_values)}"
                )

        session_mask = meta["session"] == session_key
        if not session_mask.any():
            raise ValueError(
                f"No epochs found for subject {subject}, session {session_key}"
            )

        epochs_filtered = epochs[session_mask]
        labels_filtered = labels[session_mask]
        meta_filtered = meta[session_mask].reset_index(drop=True)

        X_filtered = np.concatenate(epochs_filtered, axis=0)
        info = epochs_filtered[0].info if len(epochs_filtered) > 0 else None

        return {
            "X": X_filtered,
            "labels": labels_filtered,
            "meta": meta_filtered,
            "info": info,
            "epochs": epochs_filtered,
        }

    def _get_channel_types(self, info, epochs):
        """Extract channel types from MNE info and epochs objects.

        Parameters
        ----------
        info : mne.Info
            MNE Info object with channel information
        epochs : mne.Epochs
            MNE Epochs object

        Returns
        -------
        list[str]
            List of channel type strings (e.g., "EEG", "EOG", "EMG")
        """
        if len(epochs) > 0:
            return epochs[0].get_channel_types()

        # MNE channel kind constants (mne.io.constants.FIFF)
        ch_type_map = {
            2: "EEG",  # FIFFV_EEG_CH
            202: "EOG",  # FIFFV_EOG_CH
            302: "EMG",  # FIFFV_EMG_CH
            402: "ECG",  # FIFFV_ECG_CH
            502: "MISC",  # FIFFV_MISC_CH
        }
        return [
            ch_type_map.get(info["chs"][info["ch_names"].index(ch)]["kind"], "MISC")
            for ch in info["ch_names"]
        ]

    def _extract_eeg_data(self, X, info, ch_types, labels):
        """Extract EEG data and trial intervals from epoched arrays.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (n_epochs, n_channels, n_samples)
        info : mne.Info
            MNE Info object with channel and sampling rate information
        ch_types : list[str]
            List of channel type strings for each channel
        labels : np.ndarray
            Array of shape (n_epochs,) with event labels

        Returns
        -------
        eeg : RegularTimeSeries
            Concatenated EEG signals
        channels : ArrayDict
            Channel IDs and types
        trials : Interval
            Trial intervals with start/end times and label fields
        """
        sfreq = info["sfreq"]
        n_epochs, n_channels, n_samples = X.shape

        eeg_signals = X.transpose(0, 2, 1).reshape(-1, n_channels)

        eeg = RegularTimeSeries(
            signal=eeg_signals,
            sampling_rate=sfreq,
            domain=Interval(
                start=np.array([0.0]),
                end=np.array([(len(eeg_signals) - 1) / sfreq]),
            ),
        )

        channels = ArrayDict(
            id=np.array(info["ch_names"], dtype="U"),
            types=np.array(ch_types, dtype="U"),
        )

        sample_boundaries = np.arange(n_epochs + 1) * n_samples
        time_boundaries = sample_boundaries / sfreq
        start_times = time_boundaries[:-1]
        end_times = time_boundaries[1:]
        id_values = np.array([self.label_map.get(label, -1) for label in labels])

        trials = Interval(
            start=start_times,
            end=end_times,
            timestamps=(start_times + end_times) / 2,
            timekeys=["start", "end", "timestamps"],
            **{
                self.label_field: np.asarray(labels),
                self.id_field: id_values,
            },
        )

        if not trials.is_disjoint():
            raise ValueError("Found overlapping trials")

        return eeg, channels, trials

    @abstractmethod
    def get_brainset_description(self) -> BrainsetDescription:
        """Return dataset-specific BrainsetDescription.

        Returns
        -------
        BrainsetDescription
            Description object with dataset metadata
        """
        ...

    def _get_subject_description(self, subject_id: str) -> SubjectDescription:
        """Create subject description from subject ID.

        Parameters
        ----------
        subject_id : str
            Subject identifier

        Returns
        -------
        SubjectDescription
            Subject description object
        """
        return SubjectDescription(
            id=subject_id,
            species=Species.HOMO_SAPIENS,
        )

    def _get_session_description(
        self, session_id: str, info: Any
    ) -> SessionDescription:
        """Create session description from session ID and MNE info.

        Parameters
        ----------
        session_id : str
            Session identifier
        info : mne.Info
            MNE Info object with recording date

        Returns
        -------
        SessionDescription
            Session description object
        """
        if info is None:
            raise ValueError("No MNE Info object available from epochs")

        recording_date = info.get("meas_date")
        if recording_date is None:
            recording_date = datetime.datetime(
                2026, 1, 15, tzinfo=datetime.timezone.utc
            )
            logging.warning(
                f"Missing meas_date for session '{session_id}' (task={self.task}); "
                "using sentinel date 2026-01-15"
            )

        return SessionDescription(
            id=session_id,
            recording_date=recording_date,
            task=self.task,
        )

    def _get_device_description(self, subject_id: str, info: Any) -> DeviceDescription:
        """Create device description from subject ID and MNE info.

        Parameters
        ----------
        subject_id : str
            Subject identifier
        info : mne.Info
            MNE Info object with recording date

        Returns
        -------
        DeviceDescription
            Device description object
        """
        if info is None:
            raise ValueError("No MNE Info object available from epochs")

        recording_date = info.get("meas_date")
        if recording_date is None:
            recording_date = datetime.datetime(
                2026, 1, 15, tzinfo=datetime.timezone.utc
            )

        return DeviceDescription(
            id=f"{subject_id}_{recording_date.strftime('%Y%m%d')}",
        )

    def _generate_splits(self, trials, subject_id: Optional[str] = None):
        """Generate stratified folds for trials.

        Parameters
        ----------
        trials : Interval
            Trial intervals with label fields
        subject_id : str, optional
            Subject identifier for subject-level splits. Default is None.

        Returns
        -------
        splits : Data
            Data object containing fold splits
        """
        folds = generate_trial_folds(
            trials,
            stratify_by=self.stratify_field,
            n_folds=3,
            val_ratio=0.2,
            seed=42,
        )

        folds_dict = {f"fold_{i}": fold for i, fold in enumerate(folds)}
        return Data(**folds_dict, domain=trials)

    def process(self, download_output: Dict[str, Any]) -> None:
        """Transform MOABB data to brainsets format.

        This default implementation handles the common workflow:
        1. Extract EEG data and channel information
        2. Extract trial intervals with labels
        3. Generate stratified splits
        4. Create and store Data object

        Subclasses can override for custom processing, but typically only
        need to implement get_brainset_description().

        Parameters
        ----------
        download_output : dict
            Dictionary returned by download() containing X, labels, meta, info, epochs
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
        safe_reprocess = getattr(getattr(self, "args", None), "reprocess", False)
        if store_path.exists() and not safe_reprocess:
            self.update_status("Skipped Processing")
            return

        self.update_status("Creating Descriptions")
        brainset_description = self.get_brainset_description()
        subject_description = self._get_subject_description(subject_id)
        session_description = self._get_session_description(session_id, info)
        device_description = self._get_device_description(subject_id, info)

        self.update_status("Extracting EEG and Trials")
        ch_types = self._get_channel_types(info, epochs)
        eeg, channels, trials = self._extract_eeg_data(X, info, ch_types, labels)

        self.update_status("Generating Splits")
        splits = self._generate_splits(trials, subject_id=subject_id)

        self.update_status("Creating Data Object")
        data = Data(
            brainset=brainset_description,
            subject=subject_description,
            session=session_description,
            device=device_description,
            eeg=eeg,
            channels=channels,
            **{self.trial_key: trials},
            splits=splits,
            domain=eeg.domain,
        )

        self.update_status("Storing")
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

        logging.info(f"Saved processed data to: {store_path}")
