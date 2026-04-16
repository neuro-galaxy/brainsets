"""Base pipeline classes for OpenNeuro datasets.

This module provides the OpenNeuroPipeline abstract base class and its
concrete subclasses (OpenNeuroEEGPipeline, OpenNeuroIEEGPipeline) that
handle common functionality for processing datasets from OpenNeuro, including:
- Dynamic recording discovery from OpenNeuro S3
- Downloading EEG/iEEG files with caching
- Common processing workflow with electrode mapping
"""

from abc import ABC
from argparse import ArgumentParser, Namespace
from functools import cached_property
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from temporaldata import Data, Interval

try:
    from mne_bids import read_raw_bids

    MNE_BIDS_AVAILABLE = True
except ImportError:
    read_raw_bids = None
    MNE_BIDS_AVAILABLE = False

from brainsets import serialize_fn_map
from brainsets.descriptions import (
    BrainsetDescription,
    DeviceDescription,
    SessionDescription,
    SubjectDescription,
)
from brainsets.pipeline import BrainsetPipeline
from brainsets.taxonomy import Species
from brainsets.utils.bids_utils import (
    build_bids_path,
    fetch_eeg_recordings,
    fetch_ieeg_recordings,
    check_eeg_recording_files_exist,
    check_ieeg_recording_files_exist,
    get_subject_info,
)
from brainsets.utils.mne_utils import (
    extract_signal,
    extract_measurement_date,
    extract_channels,
)
from brainsets.utils.split import generate_string_kfold_assignment
from brainsets.utils.openneuro import (
    construct_s3_url_from_path,
    download_dataset_description,
    download_recording,
    fetch_all_filenames,
    fetch_participants_tsv,
    fetch_species,
    validate_dataset_id,
    validate_dataset_version,
    validate_subject_ids,
)

_openneuro_parser = ArgumentParser()
_openneuro_parser.add_argument("--redownload", action="store_true")
_openneuro_parser.add_argument("--reprocess", action="store_true")


def _require_mne_bids(func_name: str) -> None:
    """Raise ImportError if mne-bids is not available."""
    if not MNE_BIDS_AVAILABLE:
        raise ImportError(
            f"{func_name} requires mne-bids, which is not installed. "
            "Install it with `pip install mne-bids`."
        )


class OpenNeuroPipeline(BrainsetPipeline, ABC):
    """Abstract base class for OpenNeuro dataset pipelines.

    This class extends BrainsetPipeline and provides common functionality for
    processing datasets from OpenNeuro, supporting both EEG and iEEG modalities.

    Required class attributes:
        - dataset_id (str): OpenNeuro dataset identifier (e.g., "ds005555").
        - brainset_id (str): Unique string identifier for the brainset.
        - origin_version (str): Version of the original source data.

    Optional class attributes:
        - subject_ids (list[str] or None): Restricts processing to these subject IDs
          (default: None, which means all subjects).
        - derived_version (str): Version tag for processed files (default: "1.0.0").
        - description (str, optional): Text description of the dataset.

    Subclass requirements:
        - The modality property (must return either "eeg" or "ieeg").
        - The _build_channels() method, which constructs and returns an ArrayDict of channel objects.

    Class attributes and override points are provided to support common customization
    such as channel name and type remapping, ignored channels, and data splits. See documentation
    for CHANNEL_NAME_REMAPPING, TYPE_CHANNELS_REMAPPING, and IGNORE_CHANNELS for more details.
    """

    parser = _openneuro_parser
    """ArgumentParser with common OpenNeuro pipeline arguments (--redownload, --reprocess)."""

    dataset_id: str
    """OpenNeuro dataset identifier (e.g., "ds005555", "ds006914")."""

    brainset_id: str
    """Unique identifier for the brainset."""

    subject_ids: Optional[list[str]] = None
    """Optional list of subject IDs to process. If None, processes all subjects."""

    origin_version: str
    """Version of the original data."""

    derived_version: str = "1.0.0"
    """Version of the processed data."""

    description: Optional[str] = None
    """Optional description of the dataset."""

    modality: str
    """Data modality for this pipeline. Must be overridden by subclasses."""

    CHANNEL_NAME_REMAPPING: Optional[dict[str, str]] = None
    """Optional dict mapping original channel name to new standardized name.

    For more complex configurations (e.g., per-recording mappings), override
    get_channel_name_remapping() instead.
    """

    TYPE_CHANNELS_REMAPPING: Optional[dict[str, list[str]]] = None
    """Optional dict mapping channel types to lists of channel names.

    For more complex configurations (e.g., per-recording mappings), override
    get_type_channels_remapping() instead.
    """

    IGNORE_CHANNELS: Optional[list[str]] = None
    """Optional list of channel names to ignore.

    Channel names should be specified as they appear in the original namespace of the raw object (i.e., prior to any remapping or type changes).
    """

    split_ratios: tuple[float, float] = (0.9, 0.1)
    """Default train/valid time split ratios for recordings."""

    random_seed: int = 42
    """Random seed for generating splits."""

    @cached_property
    def _participants_data(self) -> Optional[pd.DataFrame]:
        """Lazy-load participants.tsv data, cached for the pipeline run.

        Returns:
            DataFrame with participant information indexed by participant_id,
            or None if participants.tsv doesn't exist.
        """
        return fetch_participants_tsv(self.dataset_id)

    @classmethod
    def get_manifest(cls, raw_dir: Path, args: Optional[Namespace]) -> pd.DataFrame:
        """Generate a manifest DataFrame by discovering recordings from OpenNeuro.

        This implementation queries OpenNeuro S3 to find all recordings matching
        the modality by parsing BIDS-compliant filenames.

        Args:
            raw_dir: Raw data directory assigned to this brainset
            args: Pipeline-specific arguments parsed from the command line

        Returns:
            DataFrame with columns:
                - recording_id: Recording identifier (index)
                - subject_id: Subject identifier (e.g., 'sub-01')
                - session_id: Session identifier or None
                - task_id: Task identifier (e.g., 'Sleep')
                - data_file: Relative path to data file (modality-specific key)
                - s3_url: S3 URL for downloading
                - fpath: Local file path for downloaded data
        """
        dataset_id = validate_dataset_id(cls.dataset_id)
        validate_dataset_version(dataset_id, cls.origin_version)

        if cls.subject_ids is not None:
            subject_ids = validate_subject_ids(dataset_id, cls.subject_ids)
        else:
            subject_ids = None

        all_files = fetch_all_filenames(dataset_id)
        modality = cls.modality

        if modality == "eeg":
            recordings = fetch_eeg_recordings(all_files)
        elif modality == "ieeg":
            recordings = fetch_ieeg_recordings(all_files)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        if subject_ids is not None:
            recordings = [r for r in recordings if r["subject_id"] in subject_ids]
            if not recordings:
                raise ValueError(
                    f"None of the requested subjects {subject_ids} "
                    f"were found in dataset {dataset_id}"
                )

        manifest_list = []
        for rec in recordings:
            s3_url = construct_s3_url_from_path(
                dataset_id,
                rec["fpath"],
                rec["recording_id"],
            )

            manifest_list.append(
                {
                    "subject_id": rec["subject_id"],
                    "recording_id": rec["recording_id"],
                    "s3_url": s3_url,
                }
            )

        if not manifest_list:
            raise ValueError(
                f"No {modality.upper()} recordings found in dataset {dataset_id}"
            )

        manifest = pd.DataFrame(manifest_list)
        return manifest.set_index("recording_id")

    def download(self, manifest_item) -> dict:
        """Download data for a single recording from OpenNeuro S3.

        Args:
            manifest_item: A single row of the manifest

        Returns:
            Dictionary with keys:
                - recording_id: Recording identifier
                - subject_id: Subject identifier
                - session_id: Session identifier or None
                - fpath: Local file path where data was downloaded
        """
        self.update_status("DOWNLOADING")
        self.raw_dir.mkdir(exist_ok=True, parents=True)

        s3_url = manifest_item.s3_url
        recording_id = manifest_item.Index
        subject_id = manifest_item.subject_id
        root_dir = self.raw_dir

        # if the dataset_description.json file does not exist or the redownload flag is set, download it
        # dataset_description.json is required for mne-bids to recognize a valid BIDS dataset
        dataset_description_exists = (
            self.raw_dir / "dataset_description.json"
        ).exists()
        if not dataset_description_exists or getattr(self.args, "redownload", False):
            download_dataset_description(self.dataset_id, root_dir)

        if not getattr(self.args, "redownload", False):
            if self.modality == "eeg":
                if check_eeg_recording_files_exist(self.raw_dir, recording_id):
                    self.update_status("Already Downloaded")
                    return {
                        "subject_id": subject_id,
                        "recording_id": recording_id,
                    }
            elif self.modality == "ieeg":
                if check_ieeg_recording_files_exist(self.raw_dir, recording_id):
                    self.update_status("Already Downloaded")
                    return {
                        "subject_id": subject_id,
                        "recording_id": recording_id,
                    }

        try:
            download_recording(s3_url, root_dir)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download data for {subject_id} from {self.dataset_id}: {str(e)}"
            ) from e

        return {
            "subject_id": subject_id,
            "recording_id": recording_id,
        }

    def _process_common(self, download_output: dict) -> Optional[tuple[Data, Path]]:
        """Process data files and create a Data object.

        This method handles common OpenNeuro processing tasks:
        1. Loads data files using MNE
        2. Extracts metadata (subject, session, device, brainset descriptions)
        3. Extracts signal and channel information
        5. Creates a Data object

        Args:
            download_output: Dictionary returned by download()

        Returns:
            Tuple of (Data object, store_path), or None if already processed and skipped
        """
        self.processed_dir.mkdir(exist_ok=True, parents=True)

        recording_id = download_output["recording_id"]
        subject_id = download_output["subject_id"]

        store_path = self.processed_dir / f"{recording_id}.h5"
        if not getattr(self.args, "reprocess", False):
            if store_path.exists():
                self.update_status("Already Processed")
                return None

        _require_mne_bids("_process_common")
        self.update_status(f"Loading {self.modality.upper()} file")
        bids_path = build_bids_path(self.raw_dir, recording_id, self.modality)
        raw = read_raw_bids(
            bids_path,
            on_ch_mismatch="reorder",
            verbose="CRITICAL",
        )

        self.update_status("Extracting Metadata")
        source = f"https://openneuro.org/datasets/{self.dataset_id}"
        dataset_description = (
            self.description
            if self.description
            else f"OpenNeuro dataset {self.dataset_id}"
        )

        brainset_description = BrainsetDescription(
            id=self.brainset_id,
            # validate and update the dataset version
            origin_version=validate_dataset_version(
                self.dataset_id, self.origin_version
            ),
            derived_version=self.derived_version,
            source=source,
            description=dataset_description,
        )

        subject_info = get_subject_info(subject_id, self._participants_data)
        species = fetch_species(self.dataset_id)
        subject_description = SubjectDescription(
            id=subject_id,
            species=(
                Species.HOMO_SAPIENS if species == "homo sapiens" else Species.UNKNOWN
            ),
            age=subject_info.get("age"),
            sex=subject_info.get("sex"),
        )

        meas_date = extract_measurement_date(raw)

        session_description = SessionDescription(
            id=recording_id, recording_date=meas_date
        )

        device_description = DeviceDescription(id=recording_id)

        self.update_status(f"Extracting {self.modality.upper()} Signal")
        signal = extract_signal(
            raw,
            ignore_channels=self.IGNORE_CHANNELS,
        )

        self.update_status("Building Channels")
        channels = extract_channels(
            raw,
            channel_names_mapping=self.get_channel_name_remapping(recording_id),
            type_channels_mapping=self.get_type_channels_remapping(recording_id),
            ignore_channels=self.IGNORE_CHANNELS,
        )

        self.update_status("Creating Data Object")
        data_kwargs = {
            "brainset": brainset_description,
            "subject": subject_description,
            "session": session_description,
            "device": device_description,
            "channels": channels,
            "domain": signal.domain,
        }
        data_kwargs[self.modality] = signal

        data = Data(**data_kwargs)

        self.update_status("Creating Splits")
        data.splits = self.generate_splits(
            domain=data.domain,
            subject_id=subject_id,
            session_id=session_description.id,
        )

        return data, store_path

    def process(self, download_output: dict) -> None:
        """Process and save the dataset.

        Default implementation calls _process_common() and saves the result.
        Subclasses can override to add dataset-specific processing.

        Args:
            download_output: Dictionary returned by download()
        """
        result = self._process_common(download_output)

        if result is None:
            return

        data, store_path = result

        self.update_status("Storing")
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

    def get_channel_name_remapping(
        self,
        recording_id: str | None = None,
    ) -> Optional[dict[str, str]]:
        """Return channel name remapping for a given recording.

        Override this method to provide per-recording channel name remappings.
        The default implementation returns the class-level CHANNEL_NAME_REMAPPING attribute.

        Args:
            recording_id: The recording identifier

        Returns:
            Dict mapping original channel names to standardized names, or None
        """
        return self.CHANNEL_NAME_REMAPPING

    def get_type_channels_remapping(
        self,
        recording_id: str | None = None,
    ) -> Optional[dict[str, list[str]]]:
        """Return channel type remapping for a given recording.

        Override this method to provide per-recording channel type remappings.
        The default implementation returns the class-level TYPE_CHANNELS_REMAPPING attribute.

        Args:
            recording_id: The recording identifier

        Returns:
            Dict mapping channel types to lists of channel names, or None
        """
        return self.TYPE_CHANNELS_REMAPPING

    def generate_splits(
        self, domain: Interval, subject_id: str, session_id: str
    ) -> Data:
        """
        Generate default intrasession train/valid splits using a causal (sequential) strategy.
        This method also assigns subject and session split labels used for intersubject
        and intersession model training, respectively.

        These splits assume that the data represents a continuous recording
        without any underlying trial structure or event segmentation; these splits are
        mostly suitable for pretraining large models.

        Subclasses can override this method to implement alternative or more task-specific
        splitting behaviors.

        Args:
            domain: The interval domain (e.g., Interval or similar) representing the start and end
                times of the continuous recording to be split.
            subject_id (str): The identifier for the subject.
            session_id (str): The identifier for the session.

        Returns:
            Data: A Data object with train/valid splits assigned, and with split assignment
                labels for both intersubject and intersession strategies.

        Raises:
            ValueError: If split_ratios does not contain exactly two values, contains negative values,
                or does not sum to 1.0.
        """
        if len(self.split_ratios) != 2:
            raise ValueError(
                "split_ratios must contain exactly two values (train, valid)"
            )
        if any(ratio < 0 for ratio in self.split_ratios):
            raise ValueError("split_ratios cannot contain negative values")
        if not np.isclose(sum(self.split_ratios), 1.0):
            raise ValueError("split_ratios must sum to 1.0")

        # intrasession (causal) split
        starts = np.asarray(domain.start)
        ends = np.asarray(domain.end)
        durations = ends - starts

        train_ends = starts + durations * self.split_ratios[0]
        valid_ends = train_ends + durations * self.split_ratios[1]

        train = Interval(start=starts.copy(), end=train_ends)
        valid = Interval(start=train_ends, end=valid_ends)

        # n_folds for assignment
        valid_ratio = self.split_ratios[1]
        if valid_ratio <= 0:
            assignment_n_folds = 1
        else:
            assignment_n_folds = max(1, int(round(1.0 / valid_ratio)))
        assignment_fold_idx = 0

        # intersubject split
        subject_assignments = generate_string_kfold_assignment(
            string_id=subject_id,
            n_folds=assignment_n_folds,
            val_ratio=0.0,
            seed=self.random_seed,
        )
        subject_assignment = subject_assignments[assignment_fold_idx]
        if subject_assignment == "test":
            subject_assignment = "valid"

        # intersession split
        session_assignments = generate_string_kfold_assignment(
            string_id=f"{subject_id}_{session_id}",
            n_folds=assignment_n_folds,
            val_ratio=0.0,
            seed=self.random_seed,
        )
        session_assignment = session_assignments[assignment_fold_idx]
        if session_assignment == "test":
            session_assignment = "valid"

        return Data(
            train=train,
            valid=valid,
            intersubject_assignment=subject_assignment,
            intersession_assignment=session_assignment,
            domain=domain,
        )


class OpenNeuroEEGPipeline(OpenNeuroPipeline):
    """
    Pipeline base class for EEG data processing from OpenNeuro.

    This class provides EEG-specific extensions on top of OpenNeuroPipeline,
    supporting custom channel name and type remappings.

    **Usage:**

    For datasets where all recordings share a common electrode naming scheme
    and channel configuration, specify these as class attributes:

        class Pipeline(OpenNeuroEEGPipeline):
            brainset_id = "your_dataset_id"
            dataset_id = "dsXXXXXX"
            origin_version = "1.0.0"

            CHANNEL_NAME_REMAPPING = {
                "PSG_F3": "F3",
                "PSG_F4": "F4",
            }

            TYPE_CHANNELS_REMAPPING = {
                "EEG": ["F3", "F4"],
                "EOG": ["EOG"]
            }

    For more complex datasets having variable channel naming or multiple
    acquisition schemes, override the methods to dynamically supply
    the necessary channel name and type remapping maps based on the recording id:

        class Pipeline(OpenNeuroEEGPipeline):
            brainset_id = "your_dataset_id"
            dataset_id = "dsXXXXXX"

            def get_channel_name_remapping(self, recording_id):
                if "acq-headband" in recording_id:
                    return {"HB_1": "AF7", "HB_2": "AF8"}
                return {"PSG_F3": "F3", "PSG_F4": "F4"}

            def get_type_channels_remapping(self, recording_id):
                if "acq-headband" in recording_id:
                    return {"EEG": ["AF7", "AF8"]}
                return {"EEG": ["F3", "F4"], "EOG": ["EOG"]}

    **Class Attributes:**
        - CHANNEL_NAME_REMAPPING (dict, optional): Map old channel names to new ones.
        - TYPE_CHANNELS_REMAPPING (dict, optional): Map channel types to channel lists.
    """

    modality = "eeg"
    """Data modality for this pipeline."""


class OpenNeuroIEEGPipeline(OpenNeuroPipeline):
    """
    Pipeline base class for iEEG data processing from OpenNeuro.

    This class provides iEEG-specific extensions on top of OpenNeuroPipeline,
    leveraging BIDS-compliant sidecar files to define channel and electrode
    configuration automatically.

    **Usage:**

    For most iEEG datasets where electrode and channel information are properly
    defined in BIDS sidecar files, subclassing this pipeline is sufficient:

        class Pipeline(OpenNeuroIEEGPipeline):
            brainset_id = "your_dataset_id"
            dataset_id = "dsXXXXXX"
            origin_version = "1.0.0"

    **Changing electrode names and types:**
        - If you wish to change the names or types of electrodes (e.g., for harmonization or custom processing),
          use the same approach as in OpenNeuroEEGPipeline:
            - Set the `CHANNEL_NAME_REMAPPING`/`TYPE_CHANNELS_REMAPPING` class attribute, or
            - Override `get_channel_name_remapping(self, recording_id)` / `get_type_channels_remapping(self, recording_id)`
          with your desired logic.

    **Class Attributes:**
        - CHANNEL_NAME_REMAPPING (dict, optional): Map old electrode/channel names to new ones.
        - TYPE_CHANNELS_REMAPPING (dict, optional): Map types to electrode/channel lists.
    """

    modality = "ieeg"
    """Data modality for this pipeline."""
