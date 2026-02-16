"""Base pipeline class for OpenNeuro EEG datasets.

This module provides the OpenNeuroEEGPipeline base class that handles common
functionality for processing EEG datasets from OpenNeuro, including:
- Dynamic recording discovery from OpenNeuro S3
- Downloading EEG files with caching
- Common processing workflow with electrode mapping
"""

import datetime
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

from brainsets import serialize_fn_map
from brainsets.pipeline import BrainsetPipeline
from brainsets.utils.openneuro.data_extraction import (
    extract_brainset_description,
    extract_device_description,
    extract_meas_date,
    extract_session_description,
    extract_signal,
    extract_subject_description,
    read_bids_channels_tsv,
    read_bids_electrodes_tsv,
)
from brainsets.utils.openneuro.dataset import (
    check_recording_files_exist,
    construct_s3_url_from_path,
    download_recording,
    fetch_eeg_recordings,
    fetch_ieeg_recordings,
    fetch_participants_tsv,
    validate_dataset_id,
)

_openneuro_parser = ArgumentParser()
_openneuro_parser.add_argument("--redownload", action="store_true")
_openneuro_parser.add_argument("--reprocess", action="store_true")


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

    FILE_EXTENSIONS: set[str] = {".fif", ".set", ".edf", ".bdf", ".vhdr", ".eeg"}
    """Supported file extensions for loading. Overridable per modality."""

    derived_version: str = "1.0.0"
    """Version of the processed data."""

    description: Optional[str] = None
    """Optional description of the dataset."""

    @property
    def modality(self) -> str:
        """Return the data modality: 'eeg' or 'ieeg'.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement modality property")

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
        except Exception as e:
            raise RuntimeError(
                f"Failed to download data for {subject_id} from {self.dataset_id}: {str(e)}"
            )

        return {
            "recording_id": recording_id,
            "subject_id": subject_id,
            "session_id": session_id,
            "fpath": subject_dir,
        }

    def _find_data_file(self, data_dir: Path, recording_id: str) -> Path:
        """Find the data file (EEG or iEEG) for a recording in the data directory.

        Returns:
            Path to the data file

        Raises:
            RuntimeError: If no file or multiple ambiguous files found
        """
        candidates = [
            p
            for p in data_dir.rglob(f"{recording_id}*")
            if p.suffix.lower() in self.FILE_EXTENSIONS
        ]

        if not candidates:
            raise RuntimeError(
                f"No {self.modality.upper()} files found in {data_dir} for recording {recording_id}."
            )

        if len(candidates) > 1:
            # BIDS files have "_eeg." or "_ieeg." before extension
            # Filter for this pattern to disambiguate from sidecar files
            # (e.g., _channels.tsv, _events.tsv) or auxiliary files.
            if self.modality == "eeg":
                pattern = "_eeg."
            else:
                pattern = "_ieeg."
            bids_files = [f for f in candidates if pattern in f.name]
            if len(bids_files) == 1:
                return bids_files[0]
            raise RuntimeError(
                f"Multiple {self.modality.upper()} files found in {data_dir} for recording {recording_id}: "
                f"{candidates}"
            )

        return candidates[0]

    def _load_data_file(self, data_file: Path) -> mne.io.Raw:
        """Load a data file (EEG or iEEG) using MNE's auto-detecting reader.

        Uses mne.io.read_raw() which automatically selects the appropriate
        reader based on file extension (.fif, .set, .edf, .bdf, .vhdr, etc.).

        Returns:
            MNE Raw object

        Raises:
            RuntimeError: If the file cannot be loaded
            ValueError: If BrainVision .eeg file is missing its .vhdr header
        """
        # BrainVision format consists of three files:
        # - .vhdr: Header file (contains metadata, must be loaded)
        # - .vmrk: Marker file (contains event markers)
        # - .eeg: Binary data file
        # MNE requires loading via the .vhdr header file.
        if data_file.suffix.lower() == ".eeg":
            vhdr_file = data_file.with_suffix(".vhdr")
            if not vhdr_file.exists():
                raise ValueError(f"Could not find .vhdr header file for {data_file}")
            data_file = vhdr_file

        try:
            return mne.io.read_raw(data_file, preload=True, verbose=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {self.modality.upper()} file {data_file}: {e}"
            ) from e

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

        recording_id = download_output.get("recording_id")
        subject_id = download_output["subject_id"]
        data_dir = Path(download_output["fpath"])

        data_file = self._find_data_file(data_dir, recording_id)
        self.update_status(f"Processing {data_file.name}")

        store_path = self.processed_dir / f"{recording_id}.h5"

        if not getattr(self.args, "reprocess", False):
            if store_path.exists():
                self.update_status("Already Processed")
                return None

        self.update_status(f"Loading {self.modality.upper()} file")
        raw = self._load_data_file(data_file)

        self.update_status("Extracting Metadata")
        source = f"https://openneuro.org/datasets/{self.dataset_id}"
        dataset_description = (
            self.description
            if self.description
            else f"OpenNeuro dataset {self.dataset_id}"
        )

        brainset_description = extract_brainset_description(
            dataset_id=self.dataset_id,
            origin_version=f"openneuro/{self.dataset_id}",
            derived_version=self.derived_version,
            source=source,
            description=dataset_description,
        )

        subject_info = self.get_subject_info(subject_id)
        subject_description = extract_subject_description(
            subject_id=subject_id,
            age=subject_info.get("age"),
            sex=subject_info.get("sex"),
        )

        meas_date = extract_meas_date(raw)
        if meas_date is None:
            logging.warning(
                f"No measurement date found for {recording_id}, using Unix epoch as placeholder"
            )
            meas_date = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)

        session_description = extract_session_description(
            session_id=recording_id, recording_date=meas_date
        )

        device_description = extract_device_description(device_id=recording_id)

        self.update_status(f"Extracting {self.modality.upper()} Signal")
        signal = extract_signal(raw)

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

    FILE_EXTENSIONS = {".fif", ".set", ".edf", ".bdf", ".vhdr", ".eeg"}
    """Supported EEG file extensions for loading.

    BIDS-compliant extensions: .edf, .vhdr, .set, .bdf, .eeg
    Non-BIDS extension: .fif (MNE FIFF format, included for processed data)

    Note: .vhdr is the BrainVision header file. When a .eeg file is found,
    the code automatically looks for its .vhdr header (see _load_data_file).
    """

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

    FILE_EXTENSIONS = {".fif", ".set", ".edf", ".bdf", ".vhdr", ".eeg", ".nwb"}
    """Supported iEEG file extensions for loading.

    BIDS iEEG formats: .edf, .vhdr, .set, .bdf, .eeg, .nwb
    Non-BIDS: .fif (MNE FIFF format for processed data)
    """

    def _read_channels_tsv(self, data_dir: Path, recording_id: str) -> pd.DataFrame:
        """Read channel metadata from BIDS _channels.tsv sidecar.

        Override this method to customize channel parsing for non-standard formats.

        Args:
            data_dir: Directory containing the recording files
            recording_id: Recording identifier to find sidecar files

        Returns:
            DataFrame with channel metadata (name, type, status, etc.)

        Raises:
            FileNotFoundError: If _channels.tsv cannot be found
        """
        channels_files = list(data_dir.glob(f"**/{recording_id}*_channels.tsv"))
        if not channels_files:
            raise FileNotFoundError(
                f"No _channels.tsv found for recording {recording_id} in {data_dir}"
            )
        if len(channels_files) > 1:
            logging.warning(
                f"Multiple _channels.tsv files found for {recording_id}, using first: {channels_files[0]}"
            )
        return read_bids_channels_tsv(channels_files[0])

    def _read_electrodes_tsv(self, data_dir: Path, recording_id: str) -> pd.DataFrame:
        """Read electrode coordinates from BIDS _electrodes.tsv sidecar.

        Override this method to customize electrode parsing for non-standard formats.

        Args:
            data_dir: Directory containing the recording files
            recording_id: Recording identifier to find sidecar files

        Returns:
            DataFrame with electrode metadata (name, x, y, z coordinates, etc.)

        Raises:
            FileNotFoundError: If _electrodes.tsv cannot be found
        """
        electrodes_files = list(data_dir.glob(f"**/{recording_id}*_electrodes.tsv"))
        if not electrodes_files:
            raise FileNotFoundError(
                f"No _electrodes.tsv found for recording {recording_id} in {data_dir}"
            )
        if len(electrodes_files) > 1:
            logging.warning(
                f"Multiple _electrodes.tsv files found for {recording_id}, using first: {electrodes_files[0]}"
            )
        return read_bids_electrodes_tsv(electrodes_files[0])

    def _build_channels(
        self, raw: mne.io.Raw, recording_id: str, data_dir: Path
    ) -> ArrayDict:
        """Build channels from BIDS iEEG sidecars.

        Reads _channels.tsv for channel types and status, and _electrodes.tsv for
        coordinates. Merges metadata with MNE channel information.

        Args:
            raw: The loaded MNE Raw object
            recording_id: Recording identifier
            data_dir: Data directory containing sidecar files

        Returns:
            ArrayDict with channel id, type, status, and coordinate fields
        """
        self.update_status("Reading channels metadata")
        channels_df = self._read_channels_tsv(data_dir, recording_id)

        self.update_status("Reading electrodes metadata")
        electrodes_df = self._read_electrodes_tsv(data_dir, recording_id)

        channel_ids = np.array(raw.ch_names, dtype="U")
        channel_count = len(channel_ids)

        # Extract types from channels.tsv, default to MNE types if not found
        channel_types = np.array(raw.get_channel_types(), dtype="U")
        if "name" in channels_df.columns and "type" in channels_df.columns:
            for idx, ch_name in enumerate(channel_ids):
                mask = channels_df["name"] == ch_name
                if mask.any():
                    ch_type = channels_df.loc[mask, "type"].iloc[0]
                    channel_types[idx] = str(ch_type)

        # Extract status (good/bad), default to good if not found
        status = np.full(channel_count, "good", dtype="U")
        if "name" in channels_df.columns and "status" in channels_df.columns:
            for idx, ch_name in enumerate(channel_ids):
                mask = channels_df["name"] == ch_name
                if mask.any():
                    ch_status = channels_df.loc[mask, "status"].iloc[0]
                    status[idx] = str(ch_status)

        # Extract coordinates (x, y, z)
        x_coords = np.full(channel_count, np.nan)
        y_coords = np.full(channel_count, np.nan)
        z_coords = np.full(channel_count, np.nan)

        if "name" in electrodes_df.columns:
            for idx, ch_name in enumerate(channel_ids):
                mask = electrodes_df["name"] == ch_name
                if mask.any():
                    row = electrodes_df.loc[mask].iloc[0]
                    if "x" in row:
                        x_coords[idx] = float(row["x"])
                    if "y" in row:
                        y_coords[idx] = float(row["y"])
                    if "z" in row:
                        z_coords[idx] = float(row["z"])

        return ArrayDict(
            id=channel_ids,
            types=channel_types,
            status=status,
            x=x_coords,
            y=y_coords,
            z=z_coords,
        )
