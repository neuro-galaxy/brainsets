"""Base pipeline classes for OpenNeuro datasets.

This module provides the OpenNeuroPipeline abstract base class and its
concrete subclasses (OpenNeuroEEGPipeline, OpenNeuroIEEGPipeline) that
handle common functionality for processing datasets from OpenNeuro, including:
- Dynamic recording discovery from OpenNeuro S3
- Downloading EEG/iEEG files with caching
- Common processing workflow with electrode mapping
"""

import logging
from abc import ABC
from argparse import ArgumentParser, Namespace
from functools import cached_property
from pathlib import Path
from typing import Optional

import h5py
import mne
import numpy as np
import pandas as pd
from temporaldata import ArrayDict, Data

try:
    from mne_bids import BIDSPath, read_raw_bids

    MNE_BIDS_AVAILABLE = True
except ImportError:
    BIDSPath = None
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
from brainsets.utils.bids_utils import parse_recording_id
from brainsets.utils.mne_utils import (
    extract_eeg_signal,
    extract_measurement_date,
)
from brainsets.utils.openneuro.dataset import (
    check_recording_files_exist,
    construct_s3_url_from_path,
    download_dataset_description,
    download_recording,
    fetch_eeg_recordings,
    fetch_ieeg_recordings,
    fetch_participants_tsv,
    validate_dataset_id,
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

    **Required class attributes:**
        - dataset_id: OpenNeuro dataset identifier (e.g., "ds005555")
        - brainset_id: Unique identifier for the brainset

    **Optional class attributes:**
        - subject_ids: List of subject IDs to process (defaults to all)
        - derived_version: Version of the processed data (defaults to "1.0.0")
        - description: Description of the dataset

    Subclasses must implement:
        - modality property (returns "eeg" or "ieeg")
        - _build_channels() method (constructs channels ArrayDict)
    """

    parser = _openneuro_parser
    """ArgumentParser with common OpenNeuro pipeline arguments (--redownload, --reprocess)."""

    dataset_id: str
    """OpenNeuro dataset identifier (e.g., "ds005555", "ds006914")."""

    brainset_id: str
    """Unique identifier for the brainset."""

    subject_ids: Optional[list[str]] = None
    """Optional list of subject IDs to process. If None, processes all subjects."""

    derived_version: str = "1.0.0"
    """Version of the processed data."""

    description: Optional[str] = None
    """Optional description of the dataset."""

    modality: str
    """Data modality for this pipeline. Must be overridden by subclasses."""

    def _build_bids_path(self, recording_id: str, data_dir: Path) -> BIDSPath:
        """Build a mne_bids.BIDSPath from recording_id and data directory.

        Args:
            recording_id: Recording identifier (e.g., 'sub-01_ses-01_task-Sleep')
            data_dir: Data directory (not used; BIDS root is set to self.raw_dir)

        Returns:
            BIDSPath configured for reading via mne_bids.read_raw_bids

        Raises:
            ValueError: If recording_id cannot be parsed
        """
        _require_mne_bids("_build_bids_path")
        entities = parse_recording_id(recording_id)

        bids_path = BIDSPath(
            root=self.raw_dir,
            subject=entities["subject"],
            session=entities.get("session"),
            task=entities["task"],
            acquisition=entities.get("acquisition"),
            run=entities.get("run"),
            datatype=self.modality,
            suffix=self.modality,
        )

        return bids_path

    def _build_channels(
        self, raw: mne.io.Raw, recording_id: str, data_dir: Path
    ) -> ArrayDict:
        """Build channels ArrayDict for this recording.

        Subclasses override this method to apply modality-specific channel handling.

        Args:
            raw: The loaded MNE Raw object
            recording_id: Recording identifier (for per-recording config)
            data_dir: Data directory (for accessing sidecar files)

        Returns:
            ArrayDict with 'id' and 'types' fields, plus modality-specific fields
        """
        raise NotImplementedError("Subclasses must implement _build_channels()")

    @cached_property
    def _participants_data(self) -> Optional[pd.DataFrame]:
        """Lazy-load participants.tsv data, cached for the pipeline run.

        Returns:
            DataFrame with participant information indexed by participant_id,
            or None if participants.tsv doesn't exist.
        """
        return fetch_participants_tsv(self.dataset_id)

    def get_subject_info(self, subject_id: str) -> dict:
        """Return subject information dict with 'age' and 'sex' keys.

        Override this method to provide subject information from alternative sources
        (e.g., custom metadata files, databases, etc.).

        The default implementation fetches data from the BIDS participants.tsv file.
        If the file doesn't exist or the subject is not found, returns None values.

        Args:
            subject_id: Subject identifier (e.g., 'sub-01')

        Returns:
            Dictionary with keys 'age' and 'sex', values may be None if not available
        """
        if self._participants_data is None:
            return {"age": None, "sex": None}

        if subject_id not in self._participants_data.index:
            return {"age": None, "sex": None}

        row = self._participants_data.loc[subject_id]

        age = row.get("age", None)
        if pd.isna(age):
            age = None

        sex = row.get("sex", None)
        if pd.isna(sex):
            sex = None

        return {"age": age, "sex": sex}

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

        modality = cls.modality
        if modality == "eeg":
            recordings = fetch_eeg_recordings(dataset_id)
            data_file_key = "eeg_file"
        elif modality == "ieeg":
            recordings = fetch_ieeg_recordings(dataset_id)
            data_file_key = "ieeg_file"
        else:
            raise ValueError(f"Unknown modality: {modality}")

        if cls.subject_ids is not None:
            recordings = [r for r in recordings if r["subject_id"] in cls.subject_ids]
            if not recordings:
                raise ValueError(
                    f"None of the requested subjects {cls.subject_ids} "
                    f"were found in dataset {dataset_id}"
                )

        manifest_list = []
        for rec in recordings:
            s3_url = construct_s3_url_from_path(dataset_id, rec[data_file_key])

            manifest_list.append(
                {
                    "recording_id": rec["recording_id"],
                    "subject_id": rec["subject_id"],
                    "session_id": rec["session_id"],
                    "task_id": rec["task_id"],
                    "data_file": rec[data_file_key],
                    "s3_url": s3_url,
                    "fpath": raw_dir / rec["subject_id"],
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
        session_id = getattr(manifest_item, "session_id", None)

        target_dir = self.raw_dir

        subject_dir = target_dir / subject_id

        if not getattr(self.args, "redownload", False):
            if check_recording_files_exist(recording_id, subject_dir):
                self.update_status("Already Downloaded")
                return {
                    "recording_id": recording_id,
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "fpath": subject_dir,
                }

        try:
            download_recording(s3_url, target_dir)
            download_dataset_description(self.dataset_id, target_dir)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download data for {subject_id} from {self.dataset_id}: {str(e)}"
            ) from e

        return {
            "recording_id": recording_id,
            "subject_id": subject_id,
            "session_id": session_id,
            "fpath": subject_dir,
        }

    def _process_common(self, download_output: dict) -> Optional[tuple[Data, Path]]:
        """Process data files and create a Data object.

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

        recording_id = download_output["recording_id"]
        subject_id = download_output["subject_id"]
        data_dir = Path(download_output["fpath"])

        store_path = self.processed_dir / f"{recording_id}.h5"

        if not getattr(self.args, "reprocess", False):
            if store_path.exists():
                self.update_status("Already Processed")
                return None

        _require_mne_bids("_process_common")
        self.update_status(f"Loading {self.modality.upper()} file")
        bids_path = self._build_bids_path(recording_id, data_dir)
        raw = read_raw_bids(bids_path, verbose=False)

        self.update_status("Extracting Metadata")
        source = f"https://openneuro.org/datasets/{self.dataset_id}"
        dataset_description = (
            self.description
            if self.description
            else f"OpenNeuro dataset {self.dataset_id}"
        )

        brainset_description = BrainsetDescription(
            id=self.brainset_id,
            origin_version=f"openneuro/{self.dataset_id}",
            derived_version=self.derived_version,
            source=source,
            description=dataset_description,
        )

        subject_info = self.get_subject_info(subject_id)
        subject_description = SubjectDescription(
            id=subject_id,
            species=Species.HOMO_SAPIENS,
            age=subject_info.get("age"),
            sex=subject_info.get("sex"),
        )

        meas_date = extract_measurement_date(raw)

        session_description = SessionDescription(
            id=recording_id, recording_date=meas_date
        )

        device_description = DeviceDescription(id=recording_id)

        self.update_status(f"Extracting {self.modality.upper()} Signal")
        signal = extract_eeg_signal(raw)

        self.update_status("Building Channels")
        channels = self._build_channels(raw, recording_id, data_dir)

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


class OpenNeuroEEGPipeline(OpenNeuroPipeline):
    """Concrete pipeline class for OpenNeuro EEG dataset processing.

    This class extends OpenNeuroPipeline with EEG-specific functionality,
    including electrode renaming and modality channel mapping.

    **Optional class attributes:**
        - ELECTRODE_RENAME: Dict mapping old electrode names to new names
        - MODALITY_CHANNELS: Dict mapping modality types to channel name lists

    **Channel configuration:**

    For simple datasets with uniform channels, use class attributes:

        class Pipeline(OpenNeuroEEGPipeline):
            brainset_id = "my_dataset_2024"
            dataset_id = "ds005555"
            ELECTRODE_RENAME = {"PSG_F3": "F3", "PSG_F4": "F4"}
            MODALITY_CHANNELS = {"EEG": ["F3", "F4"], "EOG": ["EOG"]}

    For datasets with multiple channel configurations (e.g., different
    acquisition types), override the methods instead:

        class Pipeline(OpenNeuroEEGPipeline):
            brainset_id = "my_dataset_2024"
            dataset_id = "ds005555"

            def get_electrode_rename(self, recording_id):
                if "acq-headband" in recording_id:
                    return {"HB_1": "AF7", "HB_2": "AF8"}
                return {"PSG_F3": "F3", "PSG_F4": "F4"}

            def get_modality_channels(self, recording_id):
                if "acq-headband" in recording_id:
                    return {"EEG": ["AF7", "AF8"]}
                return {"EEG": ["F3", "F4"], "EOG": ["EOG"]}
    """

    modality = "eeg"
    """Data modality for this pipeline."""

    ELECTRODE_RENAME: Optional[dict[str, str]] = None
    """Optional dict mapping old electrode names to new standardized names.

    For more complex configurations (e.g., per-recording mappings), override
    get_electrode_rename() instead.
    """

    MODALITY_CHANNELS: Optional[dict[str, list[str]]] = None
    """Optional dict mapping modality types to lists of channel names.

    For more complex configurations (e.g., per-recording mappings), override
    get_modality_channels() instead.
    """

    def get_electrode_rename(self, recording_id: str) -> Optional[dict[str, str]]:
        """Return electrode rename mapping for a given recording.

        Override this method to provide per-recording electrode rename mappings.
        The default implementation returns the class-level ELECTRODE_RENAME attribute.

        Args:
            recording_id: The recording identifier

        Returns:
            Dict mapping original electrode names to standardized names, or None
        """
        return self.ELECTRODE_RENAME

    def get_modality_channels(
        self, recording_id: str
    ) -> Optional[dict[str, list[str]]]:
        """Return modality-to-channels mapping for a given recording.

        Override this method to provide per-recording modality mappings.
        The default implementation returns the class-level MODALITY_CHANNELS attribute.

        Args:
            recording_id: The recording identifier

        Returns:
            Dict mapping modality names to lists of channel names, or None
        """
        return self.MODALITY_CHANNELS

    def _build_channels(
        self, raw: mne.io.Raw, recording_id: str, data_dir: Path
    ) -> ArrayDict:
        """Build channels by applying electrode renaming and modality mapping.

        Args:
            raw: The loaded MNE Raw object
            recording_id: Recording identifier (for per-recording config)
            data_dir: Data directory (not used for EEG)

        Returns:
            ArrayDict with renamed channels and modality types
        """
        channel_ids = np.array(raw.ch_names, dtype="U")
        channel_types = np.array(raw.get_channel_types(), dtype="U")

        electrode_rename = self.get_electrode_rename(recording_id)
        if electrode_rename:
            self.update_status("Renaming electrodes")
            for i, ch_name in enumerate(channel_ids):
                if ch_name in electrode_rename:
                    channel_ids[i] = electrode_rename[ch_name]

        modality_channels = self.get_modality_channels(recording_id)
        if modality_channels:
            self.update_status("Setting channel modalities")
            channel_id_set = set(channel_ids)
            for modality, channel_names in modality_channels.items():
                for ch_name in channel_names:
                    if ch_name in channel_id_set:
                        idx = np.where(channel_ids == ch_name)[0]
                        channel_types[idx] = modality

        return ArrayDict(id=channel_ids, types=channel_types)


class OpenNeuroIEEGPipeline(OpenNeuroPipeline):
    """Concrete pipeline class for OpenNeuro iEEG dataset processing.

    This class extends OpenNeuroPipeline with iEEG-specific functionality,
    automatically reading channel and electrode information from BIDS sidecars.

    iEEG datasets provide rich metadata via:
    - _channels.tsv: Channel names, types, sampling rates, and status
    - _electrodes.tsv: Electrode coordinates (x, y, z in MNI space)
    - _coordsystem.json: Coordinate system information

    Channel configuration is automatic; no manual mapping needed. Override
    the sidecar readers if your dataset has non-standard formats.
    """

    modality = "ieeg"
    """Data modality for this pipeline."""

    def _build_channels(
        self, raw: mne.io.Raw, recording_id: str, data_dir: Path
    ) -> ArrayDict:
        """Build channels from iEEG recording.

        Extracts channel information, types, and coordinates from the loaded Raw object.
        mne-bids already populated raw.info with channel types from _channels.tsv
        and electrode coordinates from _electrodes.tsv via the montage.

        Args:
            raw: The loaded MNE Raw object (already processed by mne-bids)
            recording_id: Recording identifier (not used; metadata comes from raw)
            data_dir: Data directory (not used; metadata comes from raw)

        Returns:
            ArrayDict with channel id, type, status, and coordinate fields
        """
        channel_ids = np.array(raw.ch_names, dtype="U")
        channel_count = len(channel_ids)

        channel_types = np.array(raw.get_channel_types(), dtype="U")

        status = np.full(channel_count, "good", dtype="U")
        bad_channels = raw.info.get("bads", [])
        for idx, ch_name in enumerate(channel_ids):
            if ch_name in bad_channels:
                status[idx] = "bad"

        x_coords = np.full(channel_count, np.nan)
        y_coords = np.full(channel_count, np.nan)
        z_coords = np.full(channel_count, np.nan)

        try:
            montage = raw.get_montage()
            if montage is not None:
                positions = montage.get_positions()
                if positions is not None and "ch_pos" in positions:
                    ch_pos = positions["ch_pos"]
                    for idx, ch_name in enumerate(channel_ids):
                        if ch_name in ch_pos:
                            coords = ch_pos[ch_name]
                            x_coords[idx] = float(coords[0])
                            y_coords[idx] = float(coords[1])
                            z_coords[idx] = float(coords[2])
        except Exception as e:
            logging.warning(
                f"Could not extract electrode positions for {recording_id}: {e}"
            )

        return ArrayDict(
            id=channel_ids,
            types=channel_types,
            status=status,
            x=x_coords,
            y=y_coords,
            z=z_coords,
        )
