"""Base pipeline class for OpenNeuro EEG datasets.

This module provides the OpenNeuroEEGPipeline base class that handles common
functionality for processing EEG datasets from OpenNeuro, including:
- Dynamic recording discovery from OpenNeuro S3
- Downloading EEG files with caching
- Common processing workflow with electrode mapping
"""

import datetime
import glob
from abc import ABC
from argparse import Namespace
from pathlib import Path
from typing import Optional

import h5py
import mne
import pandas as pd
from temporaldata import Data

from brainsets import serialize_fn_map
from brainsets.pipeline import BrainsetPipeline
from brainsets.utils.open_neuro import (
    check_recording_files_exist,
    construct_s3_url,
    download_prefix_from_s3,
    fetch_eeg_recordings,
    rename_electrodes,
    set_channel_modalities,
    validate_dataset_id,
)
from brainsets.utils.open_neuro_utils.data_extraction import (
    extract_brainset_description,
    extract_channels,
    extract_device_description,
    extract_meas_date,
    extract_session_description,
    extract_signal,
    extract_subject_description,
)


class OpenNeuroEEGPipeline(BrainsetPipeline, ABC):
    """Abstract base class for OpenNeuro EEG dataset pipelines.

    This class extends BrainsetPipeline and provides common functionality for
    processing EEG datasets from OpenNeuro. It includes default implementations
    for get_manifest(), download(), and process() methods.

    **Required class attributes:**
        - dataset_id: OpenNeuro dataset identifier (e.g., "ds005555")
        - brainset_id: Unique identifier for the brainset

    **Optional class attributes:**
        - ELECTRODE_RENAME: Dict mapping old electrode names to new names
        - MODALITY_CHANNELS: Dict mapping modality types to channel name lists
        - subject_ids: List of subject IDs to process (defaults to all)
        - derived_version: Version of the processed data (defaults to "1.0.0")
        - description: Description of the dataset

    **Example usage:**

        class Pipeline(OpenNeuroEEGPipeline):
            brainset_id = "my_dataset_2024"
            dataset_id = "ds005555"

            ELECTRODE_RENAME = {
                "PSG_F3": "F3",
                "PSG_F4": "F4",
            }

            MODALITY_CHANNELS = {
                "EEG": ["F3", "F4", "C3", "C4"],
                "EOG": ["EOG_L", "EOG_R"],
            }

            def process(self, download_output):
                super().process(download_output)
    """

    dataset_id: str
    """OpenNeuro dataset identifier (e.g., "ds005555")."""

    brainset_id: str
    """Unique identifier for the brainset."""

    ELECTRODE_RENAME: Optional[dict[str, str]] = None
    """Optional dict mapping old electrode names to new standardized names."""

    MODALITY_CHANNELS: Optional[dict[str, list[str]]] = None
    """Optional dict mapping modality types to lists of channel names."""

    subject_ids: Optional[list[str]] = None
    """Optional list of subject IDs to process. If None, processes all subjects."""

    derived_version: str = "1.0.0"
    """Version of the processed data."""

    description: Optional[str] = None
    """Optional description of the dataset."""

    @classmethod
    def get_manifest(cls, raw_dir: Path, args: Optional[Namespace]) -> pd.DataFrame:
        """Generate a manifest DataFrame by discovering recordings from OpenNeuro.

        This implementation queries OpenNeuro S3 to find all EEG recordings
        by parsing BIDS-compliant filenames.

        Args:
            raw_dir: Raw data directory assigned to this brainset
            args: Pipeline-specific arguments parsed from the command line

        Returns:
            DataFrame with columns:
                - recording_id: Recording identifier (index)
                - subject_id: Subject identifier (e.g., 'sub-01')
                - session_id: Session identifier or None
                - task_id: Task identifier (e.g., 'Sleep')
                - eeg_file: Relative path to EEG file
                - s3_url: S3 URL for downloading
                - fpath: Local file path for downloaded data
        """
        dataset_id = validate_dataset_id(cls.dataset_id)

        recordings = fetch_eeg_recordings(dataset_id)

        if cls.subject_ids is not None:
            recordings = [r for r in recordings if r["subject_id"] in cls.subject_ids]
            if not recordings:
                raise ValueError(
                    f"None of the requested subjects {cls.subject_ids} "
                    f"were found in dataset {dataset_id}"
                )

        manifest_list = []
        for rec in recordings:
            recording_id = rec["recording_id"]
            subject_id = rec["subject_id"]

            try:
                s3_url = construct_s3_url(dataset_id, recording_id)
            except RuntimeError:
                continue

            manifest_list.append(
                {
                    "recording_id": recording_id,
                    "subject_id": subject_id,
                    "session_id": rec["session_id"],
                    "task_id": rec["task_id"],
                    "eeg_file": rec["eeg_file"],
                    "s3_url": s3_url,
                    "fpath": raw_dir / subject_id,
                }
            )

        if not manifest_list:
            raise ValueError(f"No EEG recordings found in dataset {dataset_id}")

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
        target_dir.mkdir(exist_ok=True, parents=True)

        subject_dir = target_dir / subject_id

        if check_recording_files_exist(recording_id, subject_dir):
            if not (self.args and getattr(self.args, "redownload", False)):
                self.update_status("Already Downloaded")
                return {
                    "recording_id": recording_id,
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "fpath": subject_dir,
                }

        try:
            self.update_status(f"Downloading from {s3_url}")
            download_prefix_from_s3(s3_url, target_dir)
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

    def _apply_common_mapping(self, raw: mne.io.Raw) -> None:
        """Apply electrode renaming and modality mapping to an MNE Raw object.

        This method applies:
        1. Electrode renaming using ELECTRODE_RENAME (if defined)
        2. Channel type setting using MODALITY_CHANNELS (if defined)

        Args:
            raw: The MNE Raw object to modify (modified in place)
        """
        if self.ELECTRODE_RENAME:
            self.update_status("Renaming electrodes")
            rename_electrodes(raw, self.ELECTRODE_RENAME)

        if self.MODALITY_CHANNELS:
            self.update_status("Setting channel modalities")
            set_channel_modalities(raw, self.MODALITY_CHANNELS)

    def _process_common(self, download_output: dict) -> tuple[Data, Path]:
        """Process EEG files and create a Data object.

        This method handles common OpenNeuro EEG processing tasks:
        1. Loads EEG files using MNE
        2. Applies electrode mapping via _apply_common_mapping (if defined)
        3. Extracts metadata (subject, session, device, brainset descriptions)
        4. Extracts EEG signal and channel information
        5. Creates a Data object

        Args:
            download_output: Dictionary returned by download()

        Returns:
            Tuple of (Data object, store_path)
        """
        self.processed_dir.mkdir(exist_ok=True, parents=True)

        recording_id = download_output.get("recording_id")
        subject_id = download_output["subject_id"]
        data_dir = Path(download_output["fpath"])

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

        eeg_patterns = [
            f"**/{recording_id}*.fif",
            f"**/{recording_id}*.set",
            f"**/{recording_id}*.edf",
            f"**/{recording_id}*.bdf",
            f"**/{recording_id}*.vhdr",
            f"**/{recording_id}*.eeg",
        ]

        eeg_files = []
        for pattern in eeg_patterns:
            eeg_files.extend(glob.glob(str(data_dir / pattern), recursive=True))
            if eeg_files:
                break

        if not eeg_files:
            raise RuntimeError(
                f"No EEG files found in {data_dir} for recording {recording_id}."
            )

        if len(eeg_files) > 1:
            eeg_files = [f for f in eeg_files if "_eeg." in f]
            if len(eeg_files) != 1:
                raise RuntimeError(
                    f"Multiple EEG files found in {data_dir} for recording {recording_id}."
                )

        eeg_file_path = eeg_files[0]
        eeg_file = Path(eeg_file_path)
        self.update_status(f"Processing {eeg_file.name}")

        parts = eeg_file.parts
        session_id = None
        for part in parts:
            if part.startswith("ses-"):
                session_id = part.replace("ses-", "")
                break

        if session_id is None:
            session_id = "1"

        full_session_id = f"{subject_id}_ses-{session_id}"

        store_name = f"{recording_id}.h5" if recording_id else f"{full_session_id}.h5"
        store_path = self.processed_dir / store_name

        if store_path.exists() and not (
            self.args and getattr(self.args, "reprocess", False)
        ):
            raise RuntimeError(
                f"Processed file already exists at {store_path}. "
                f"Use --reprocess flag to reprocess."
            )

        self.update_status("Loading EEG file")
        try:
            if eeg_file.suffix == ".fif":
                raw = mne.io.read_raw_fif(eeg_file, preload=True, verbose=False)
            elif eeg_file.suffix == ".set":
                raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
            elif eeg_file.suffix in [".edf", ".bdf"]:
                raw = mne.io.read_raw_edf(eeg_file, preload=True, verbose=False)
            elif eeg_file.suffix == ".vhdr":
                raw = mne.io.read_raw_brainvision(eeg_file, preload=True, verbose=False)
            elif eeg_file.suffix == ".eeg":
                vhdr_file = eeg_file.with_suffix(".vhdr")
                if vhdr_file.exists():
                    raw = mne.io.read_raw_brainvision(
                        vhdr_file, preload=True, verbose=False
                    )
                else:
                    raise ValueError(f"Could not find .vhdr file for {eeg_file}")
            else:
                raise ValueError(f"Unsupported EEG file format: {eeg_file.suffix}")
        except Exception as e:
            raise RuntimeError(f"Failed to load EEG file {eeg_file}: {str(e)}")

        self._apply_common_mapping(raw)

        subject_description = extract_subject_description(subject_id=subject_id)

        meas_date = extract_meas_date(raw)
        if meas_date is None:
            meas_date = datetime.datetime.now()

        session_description = extract_session_description(
            session_id=full_session_id, recording_date=meas_date
        )

        device_id = f"{subject_id}_{session_id}"
        device_description = extract_device_description(device_id=device_id)

        self.update_status("Extracting EEG Signal")
        eeg_signal = extract_signal(raw)
        channels = extract_channels(raw)

        self.update_status("Creating Data Object")
        data = Data(
            brainset=brainset_description,
            subject=subject_description,
            session=session_description,
            device=device_description,
            eeg=eeg_signal,
            channels=channels,
            domain=eeg_signal.domain,
        )

        return data, store_path

    def process(self, download_output: dict) -> None:
        """Process and save the dataset.

        Default implementation calls _process_common() and saves the result.
        Subclasses can override to add dataset-specific processing.

        Args:
            download_output: Dictionary returned by download()
        """
        data, store_path = self._process_common(download_output)

        self.update_status("Storing")
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)
