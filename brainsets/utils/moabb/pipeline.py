"""Base pipeline class for MOABB (Mother of All BCI Benchmarks) datasets.

This module provides a reusable base class for integrating MOABB datasets into
the brainsets pipeline framework. Subclasses define the dataset and paradigm
classes, and implement paradigm-specific processing logic.
"""

from abc import abstractmethod
from typing import Dict, Any, Type, Optional
from pathlib import Path
from argparse import ArgumentParser
import os
import pandas as pd
import numpy as np
import datetime
import logging
import h5py
import mne

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


_base_parser = ArgumentParser(add_help=False)
_base_parser.add_argument(
    "--redownload", action="store_true", help="Force redownload of raw data"
)
_base_parser.add_argument(
    "--reprocess", action="store_true", help="Force reprocessing of data"
)
_base_parser.add_argument(
    "--bandpass-low",
    type=float,
    default=None,
    help="Low cutoff frequency for bandpass filter in Hz (default: None, no filtering)",
)
_base_parser.add_argument(
    "--bandpass-high",
    type=float,
    default=None,
    help="High cutoff frequency for bandpass filter in Hz (default: None, no filtering)",
)
_base_parser.add_argument(
    "--resample",
    type=float,
    default=None,
    help="Target sampling rate in Hz (default: no resampling)",
)


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
    def get_paradigm(cls, args=None) -> BaseParadigm:
        """Instantiate the MOABB paradigm with configured kwargs.

        Parameters
        ----------
        args : Namespace, optional
            CLI arguments containing bandpass_low, bandpass_high, resample params

        Returns
        -------
        BaseParadigm
            Paradigm instance with filtering parameters from args
        """
        kwargs = {k: v for k, v in cls.paradigm_kwargs.items() if v is not None}

        if args is not None:
            if args.bandpass_low is not None:
                kwargs["fmin"] = args.bandpass_low
            if args.bandpass_high is not None:
                kwargs["fmax"] = args.bandpass_high
            if args.resample is not None:
                kwargs["resample"] = args.resample

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

    def _validate_bandpass_params(self, dataset: BaseDataset, subject: int) -> None:
        """Validate bandpass parameters against Nyquist frequency.

        Loads raw data briefly to check sampling rate and raises an error if
        bandpass_high exceeds the Nyquist frequency.

        Parameters
        ----------
        dataset : BaseDataset
            MOABB dataset instance
        subject : int
            Subject ID to check

        Raises
        ------
        ValueError
            If bandpass_high >= Nyquist frequency
        """
        fmax = self.args.bandpass_high
        if fmax is None:
            return

        data_path = dataset.data_path(subject)
        if isinstance(data_path, dict):
            first_path = next(iter(data_path.values()))
            if isinstance(first_path, list):
                first_path = first_path[0]
        elif isinstance(data_path, list):
            first_path = data_path[0]
        else:
            first_path = data_path

        raw_check = mne.io.read_raw(first_path, preload=False, verbose=False)
        sfreq = raw_check.info["sfreq"]
        nyquist = sfreq / 2.0

        if fmax >= nyquist:
            raise ValueError(
                f"Bandpass high ({fmax} Hz) exceeds Nyquist frequency ({nyquist} Hz). "
                f"Use a value less than {nyquist} Hz or omit --bandpass-high for no filtering."
            )

    def download(self, manifest_item) -> Dict[str, Any]:
        """Download and extract data using MOABB paradigm.

        This method:
        1. Sets up MNE download directory
        2. Validates bandpass parameters against Nyquist frequency
        3. Calls paradigm.get_data(return_raws=True) to get filtered Raw objects
        4. Filters results to the specific session from manifest_item

        Parameters
        ----------
        manifest_item : NamedTuple
            Row from manifest containing subject and session info

        Returns
        -------
        dict
            Dictionary containing:
            - raws: list of mne.io.Raw objects (filtered, continuous)
            - meta: pd.DataFrame with columns: subject, session, run
            - dataset: BaseDataset instance
        """
        self.update_status("DOWNLOADING")

        self.raw_dir.mkdir(exist_ok=True, parents=True)
        os.environ["MNE_DATA"] = str(self.raw_dir.resolve())

        dataset = self.get_dataset()
        subject = int(manifest_item.subject)
        session = int(manifest_item.session)

        self._validate_bandpass_params(dataset, subject)
        paradigm = self.get_paradigm(self.args)

        raws, labels, meta = paradigm.get_data(
            dataset=dataset,
            subjects=[subject],
            return_raws=True,
        )

        if len(raws) == 0:
            raise ValueError(f"No data found for subject {subject}, session {session}")

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
                f"No data found for subject {subject}, session {session_key}"
            )

        meta_filtered = meta[session_mask].reset_index(drop=True)

        raws_filtered = [raws[i] for i in range(len(raws)) if session_mask.iloc[i]]

        return {
            "raws": raws_filtered,
            "meta": meta_filtered,
            "dataset": dataset,
        }

    def _get_channel_types(self, info):
        """Extract channel types from MNE info object.

        Parameters
        ----------
        info : mne.Info
            MNE Info object with channel information

        Returns
        -------
        list[str]
            List of channel type strings (e.g., "EEG", "EOG", "EMG")
        """
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

    def _extract_trials_from_raw(self, raw, dataset) -> Interval:
        """Extract trial intervals from Raw annotations using actual trial durations.

        Instead of using a fixed duration, this method calculates trial duration
        by finding the time between consecutive events. Each trial starts at an
        event onset and ends when the next event begins.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw MNE object with annotations
        dataset : BaseDataset
            MOABB dataset instance

        Returns
        -------
        Interval
            Trial intervals with start/end times and label fields
        """
        events, _ = mne.events_from_annotations(
            raw, event_id=dataset.event_id, verbose=False
        )

        if len(events) == 0:
            raise ValueError("No events found in Raw annotations")

        sfreq = raw.info["sfreq"]
        event_id_to_name = {v: k for k, v in dataset.event_id.items()}

        starts = events[:, 0] / sfreq

        ends = np.zeros(len(events))
        for i in range(len(events) - 1):
            ends[i] = events[i + 1, 0] / sfreq

        last_sample = raw.n_times - 1
        ends[-1] = last_sample / sfreq

        labels = np.array(
            [
                event_id_to_name.get(event_code, "unknown")
                for event_code in events[:, 2]
            ],
            dtype="U",
        )

        id_values = np.array([self.label_map.get(label, -1) for label in labels])

        trials = Interval(
            start=starts,
            end=ends,
            timestamps=(starts + ends) / 2,
            timekeys=["start", "end", "timestamps"],
            **{
                self.label_field: labels,
                self.id_field: id_values,
            },
        )

        if not trials.is_disjoint():
            raise ValueError("Found overlapping trials")

        return trials

    def _extract_continuous_eeg(self, raw):
        """Extract continuous EEG signal from Raw object.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw MNE object with continuous data

        Returns
        -------
        eeg : RegularTimeSeries
            Continuous EEG signals
        channels : ArrayDict
            Channel IDs and types
        """
        data, times = raw.get_data(return_times=True)
        info = raw.info

        eeg = RegularTimeSeries(
            signal=data.T,  # (n_samples, n_channels)
            sampling_rate=info["sfreq"],
            domain=Interval(
                start=np.array([times[0]]),
                end=np.array([times[-1]]),
            ),
        )

        ch_types = self._get_channel_types(info)
        channels = ArrayDict(
            id=np.array(info["ch_names"], dtype="U"),
            types=np.array(ch_types, dtype="U"),
        )

        return eeg, channels

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
        1. Concatenate runs into continuous recording
        2. Extract continuous EEG data and channel information
        3. Extract trial intervals from annotations
        4. Generate stratified splits
        5. Create and store Data object

        Subclasses can override for custom processing, but typically only
        need to implement get_brainset_description().

        Parameters
        ----------
        download_output : dict
            Dictionary returned by download() containing raws, meta, dataset
        """
        raws = download_output["raws"]
        meta = download_output["meta"]
        dataset = download_output["dataset"]

        self.update_status("PROCESSING")
        self.processed_dir.mkdir(exist_ok=True, parents=True)

        subject_id = f"S{meta.iloc[0]['subject']:03d}"
        session_id = f"{subject_id}_sess-{meta.iloc[0]['session']}"

        store_path = self.processed_dir / f"{session_id}.h5"
        safe_reprocess = getattr(getattr(self, "args", None), "reprocess", False)
        if store_path.exists() and not safe_reprocess:
            self.update_status("Skipped Processing")
            return

        self.update_status("Concatenating Runs")
        if len(raws) > 1:
            raw = mne.concatenate_raws(raws, verbose=False)
        else:
            raw = raws[0]

        info = raw.info

        self.update_status("Creating Descriptions")
        brainset_description = self.get_brainset_description()
        subject_description = self._get_subject_description(subject_id)
        session_description = self._get_session_description(session_id, info)
        device_description = self._get_device_description(subject_id, info)

        self.update_status("Extracting Continuous EEG")
        eeg, channels = self._extract_continuous_eeg(raw)

        self.update_status("Extracting Trial Intervals")
        trials = self._extract_trials_from_raw(raw, dataset)

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
