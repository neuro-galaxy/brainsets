"""Base class for OpenNeuro EEG pipelines.

This module provides the OpenNeuroEEGPipeline abstract class, which serves as a base class for creating pipelines that process EEG data from OpenNeuro datasets. It includes default implementations for get_manifest(), download(), and process() methods that handle common OpenNeuro EEG dataset processing tasks. Subclasses can extend these methods by calling super() and adding dataset-specific logic.
"""

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional
from argparse import Namespace
import datetime
import glob
import json
import inspect
import subprocess

import h5py
import mne
import pandas as pd

from temporaldata import Data

from brainsets import serialize_fn_map
from brainsets.pipeline import BrainsetPipeline
from brainsets.utils.open_neuro import (
    validate_dataset_id,
    fetch_participants,
    fetch_latest_version_tag,
    fetch_all_filenames,
    fetch_metadata,
)
from brainsets.utils.open_neuro_utils.data_extraction import (
    extract_brainset_description,
    extract_subject_description,
    extract_session_description,
    extract_device_description,
    extract_meas_date,
    extract_signal,
    extract_channels,
)


class OpenNeuroEEGPipeline(BrainsetPipeline, ABC):
    """Abstract base class for OpenNeuro EEG dataset pipelines.

    This class extends BrainsetPipeline and provides common functionality for
    processing EEG datasets from OpenNeuro. It includes default implementations
    for get_manifest(), download(), and process() methods that handle common
    OpenNeuro EEG dataset processing tasks.

    **Required class attributes:**
        - :attr:`brainset_id`: Unique identifier for the brainset
        - :attr:`dataset_id`: OpenNeuro dataset identifier (e.g., "ds005555")

    **Optional class attributes:**
        - :attr:`parser`: ArgumentParser for pipeline-specific CLI arguments
        - :attr:`version_tag`: Specific version tag to use (defaults to latest)
        - :attr:`subject_ids`: List of subject IDs to process (defaults to all)
        - :attr:`derived_version`: Version of the processed data (defaults to "1.0.0")
        - :attr:`description`: Description of the dataset (defaults to fetched from OpenNeuro)

    **The pipeline workflow consists of:**
        1. Generating a manifest (list of assets to process) via :meth:`get_manifest()`.
           This loads recording information from a config file ({dataset_id}_config.json)
           located in the pipeline directory and creates manifest entries organized by recording_id.
        2. Downloading each asset via :meth:`download()`. This downloads files directly from
           OpenNeuro's S3 bucket using AWS CLI (s3://openneuro.org/{dataset_id}/).
        3. Processing each downloaded asset via :meth:`process()`. This default
           implementation handles common OpenNeuro EEG dataset processing tasks, but can be extended
           by subclasses by calling super().process() and adding additional logic.

    **Extending the process() method:**
        Subclasses can override process() and call super().process() to use the
        default implementation, then add dataset-specific processing:

    Examples
    --------
    >>> from argparse import ArgumentParser
    >>> parser = ArgumentParser()
    >>> parser.add_argument("--redownload", action="store_true")
    >>> parser.add_argument("--reprocess", action="store_true")
    >>>
    >>> class Pipeline(OpenNeuroEEGPipeline):
    ...     brainset_name = "klinzing_sleep_ds005555_2024"
    ...     dataset_id = "ds005555"
    ...     parser = parser
    ...
    ...     def process(self, download_output):
    ...         # Call default implementation
    ...         super().process(download_output)
    ...         # Add dataset-specific processing here
    ...         ...
    """

    dataset_id: str
    """OpenNeuro dataset identifier (e.g., "ds005555"). Must be set by the Pipeline subclass."""

    version_tag: Optional[str] = None
    """Optional version tag to use. If None, uses the latest version."""

    subject_ids: Optional[list[str]] = None
    """Optional list of subject IDs to process. If None, processes all subjects."""

    derived_version: str = "1.0.0"
    """Version of the processed data. Defaults to "1.0.0"."""

    description: Optional[str] = None
    """Optional description of the dataset. If None, will be fetched from OpenNeuro metadata."""

    # ------------------------------------------------------------------
    # Abstract hooks to enforce dataset-specific configuration.
    # Subclasses must at least provide a brainset identifier and
    # an OpenNeuro dataset identifier via these methods.
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def get_brainset_id(cls) -> str:
        """Return the unique brainset identifier used for this pipeline."""
        ...

    @classmethod
    @abstractmethod
    def get_dataset_id(cls) -> str:
        """Return the OpenNeuro dataset identifier (e.g., 'ds005555')."""
        ...

    @classmethod
    def _get_config_file_path(cls) -> Path:
        """Get the path to the config file for this pipeline.

        Subclasses can override this method to customize config file location.
        Default implementation expects config file to be named {dataset_id}_config.json
        and located in the same directory as the pipeline module.

        Returns
        -------
        Path
            Path to the config file.
        """
        # Get the file path of the pipeline class module
        pipeline_module = inspect.getmodule(cls)
        if pipeline_module is None or not hasattr(pipeline_module, "__file__"):
            raise RuntimeError(
                f"Could not determine pipeline module file path for {cls.__name__}"
            )

        pipeline_file = Path(pipeline_module.__file__)
        pipeline_dir = pipeline_file.parent

        # Construct config file path: {dataset_id}_config.json
        dataset_id = cls.get_dataset_id()
        config_file = pipeline_dir / f"{dataset_id}_config.json"

        if not config_file.exists():
            raise FileNotFoundError(
                f"Config file not found at {config_file}. "
                f"Expected config file named {dataset_id}_config.json in the pipeline directory."
            )

        return config_file

    @classmethod
    def _load_config(cls) -> dict:
        """Load the config file for this pipeline.

        Returns
        -------
        dict
            Configuration dictionary loaded from the config file.
        """
        config_file = cls._get_config_file_path()
        with open(config_file, "r") as f:
            return json.load(f)

    @classmethod
    def _construct_s3_url(
        cls, dataset_id: str, recording_id: str, version_tag: Optional[str] = None
    ) -> str:
        """Construct the S3 URL for a recording.

        OpenNeuro datasets are hosted at s3://openneuro.org/{dataset_id}/
        The BIDS structure typically follows: {subject_id}/.../{task_id}/...

        The S3 structure for OpenNeuro is:
        s3://openneuro.org/{dataset_id}/{subject_id}/...

        Note: The S3 bucket only contains the latest version of each dataset,
        so the tag parameter is currently ignored. All files from the dataset
        root are stored directly under the dataset root.

        Parameters
        ----------
        dataset_id : str
            OpenNeuro dataset identifier.
        recording_id : str
            Recording identifier (e.g., 'sub-1_task-Sleep_acq-headband').
        version_tag : Optional[str]
            Version tag. Ignored for S3 path construction (OpenNeuro S3 only has latest version).

        Returns
        -------
        str
            S3 URL prefix for the recording (e.g., 's3://openneuro.org/ds005555/sub-1/')
        """
        # Parse recording_id to extract subject_id
        # Format: sub-{N}_task-{TASK}_acq-{ACQ}
        parts = recording_id.split("_")
        subject_id = None

        for part in parts:
            if part.startswith("sub-"):
                subject_id = part
                break

        if not subject_id:
            raise ValueError(
                f"Could not parse subject_id from recording_id: {recording_id}"
            )

        # Construct S3 path: s3://openneuro.org/{dataset_id}/{subject_id}/
        # OpenNeuro S3 structure: files are stored directly under dataset root (latest version only)
        base_path = f"s3://openneuro.org/{dataset_id}"

        # Construct subject-level path
        # We download the entire subject directory, which will include all tasks/acquisitions
        s3_path = f"{base_path}/{subject_id}/"

        return s3_path

    @classmethod
    def get_manifest(cls, raw_dir: Path, args: Optional[Namespace]) -> pd.DataFrame:
        """Generate a manifest DataFrame listing all assets to process.

        This implementation loads the config file to get recording information
        and creates a manifest organized by recording_id. Each row represents
        a recording to be downloaded and processed.

        Parameters
        ----------
        raw_dir : Path
            Raw data directory assigned to this brainset by the pipeline runner.
        args : Optional[Namespace]
            Pipeline-specific arguments parsed from the command line.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
                - 'recording_id': Recording identifier (e.g., 'sub-1_task-Sleep_acq-headband')
                - 'subject_id': Subject identifier (e.g., 'sub-1')
                - 'task_id': Task identifier (e.g., 'Sleep')
                - 'channel_map_id': Channel map identifier
                - 'dataset_id': OpenNeuro dataset identifier
                - 'version_tag': Version tag to download
                - 's3_url': S3 URL prefix for downloading this recording
            The index is set to 'recording_id'.
        """
        # Validate dataset ID
        dataset_id = validate_dataset_id(cls.get_dataset_id())

        # Get version tag (use latest if not specified)
        if cls.version_tag is None:
            version_tag = fetch_latest_version_tag(dataset_id)
        else:
            version_tag = cls.version_tag

        # Load config file
        config = cls._load_config()
        recording_info = (
            config.get("dataset", {})
            .get("recording_info", {})
            .get("recording_info", [])
        )

        if not recording_info:
            raise ValueError(
                f"No recording_info found in config file. "
                f"Expected path: dataset.recording_info.recording_info"
            )

        # Filter recordings if subject_ids is specified
        if cls.subject_ids is not None:
            recording_info = [
                rec
                for rec in recording_info
                if rec.get("subject_id") in cls.subject_ids
            ]
            if not recording_info:
                raise ValueError(
                    f"None of the requested subjects {cls.subject_ids} "
                    f"were found in the config file"
                )

        # Create manifest entries
        manifest_list = []
        for rec in recording_info:
            recording_id = rec.get("recording_id")
            if not recording_id:
                continue

            # Construct S3 URL
            s3_url = cls._construct_s3_url(dataset_id, recording_id, version_tag)

            manifest_list.append(
                {
                    "recording_id": recording_id,
                    "subject_id": rec.get("subject_id"),
                    "task_id": rec.get("task_id"),
                    "channel_map_id": rec.get("channel_map_id"),
                    "dataset_id": dataset_id,
                    "version_tag": version_tag,
                    "s3_url": s3_url,
                }
            )

        if not manifest_list:
            raise ValueError("No recordings found in config file to process")

        # Create DataFrame with recording_id as index
        manifest = pd.DataFrame(manifest_list)
        manifest = manifest.set_index("recording_id")
        return manifest

    def _check_recording_files_exist(
        self, recording_id: str, subject_dir: Path
    ) -> bool:
        """Check if files for a specific recording exist in the subject directory.

        This checks for BIDS-compliant files matching the recording_id pattern.
        Looks for EEG data files.

        Parameters
        ----------
        recording_id : str
            Recording identifier (e.g., 'sub-1_task-Sleep_acq-headband').
        subject_dir : Path
            Subject directory to search in.

        Returns
        -------
        bool
            True if at least one recording file is found, False otherwise.
        """
        if not subject_dir.exists():
            return False

        # Common BIDS file patterns for a recording
        # Check for EEG data files (most important)
        eeg_patterns = [
            f"**/{recording_id}_eeg.edf",
            f"**/{recording_id}_eeg.fif",
            f"**/{recording_id}_eeg.set",
            f"**/{recording_id}_eeg.bdf",
            f"**/{recording_id}_eeg.vhdr",
            f"**/{recording_id}_eeg.eeg",
        ]

        # Check if any EEG data file exists (using any() with generator for efficiency)
        if any(subject_dir.glob(pattern) for pattern in eeg_patterns):
            return True

        return False

    def _download_from_s3(self, s3_url: str, target_dir: Path) -> None:
        """Download files from OpenNeuro S3 bucket using AWS CLI.

        Parameters
        ----------
        s3_url : str
            S3 URL prefix to download from (e.g., 's3://openneuro.org/ds005555/sub-1')
        target_dir : Path
            Local directory to download files to.

        Raises
        ------
        RuntimeError
            If AWS CLI is not available or download fails.
        """
        # Check if AWS CLI is available
        try:
            subprocess.run(
                ["aws", "--version"], check=True, capture_output=True, timeout=5
            )
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            raise RuntimeError(
                "AWS CLI is not available. Please install AWS CLI to download from OpenNeuro S3. "
                "See: https://aws.amazon.com/cli/"
            )

        # Use AWS S3 sync to download files
        # --no-sign-request: OpenNeuro S3 bucket allows public access
        # --exclude: Exclude unnecessary files to speed up download
        # We'll download the entire subject directory structure
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.update_status(f"Downloading from {s3_url}")
            result = subprocess.run(
                [
                    "aws",
                    "s3",
                    "sync",
                    s3_url,
                    str(target_dir),
                    "--no-sign-request",
                    "--quiet",  # Reduce output
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else "Unknown error"
            raise RuntimeError(f"Failed to download from {s3_url}: {error_msg}")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Download from {s3_url} timed out after 1 hour")

    def download(self, manifest_item) -> dict:
        """Download data for a single recording from OpenNeuro S3.

        This implementation downloads EEG data for the recording specified in
        manifest_item using AWS CLI to download directly from OpenNeuro's S3 bucket.

        **Important**: The manifest is per recording, but downloads are per subject.
        When syncing the subject directory (e.g., `s3://.../sub-1/`), `aws s3 sync`
        downloads ALL files recursively for that subject, including all recordings,
        tasks, and acquisitions. Subsequent recordings from the same subject will
        skip the download since the subject directory already exists.

        Parameters
        ----------
        manifest_item : pandas.Series
            A single row of the manifest returned by :meth:`get_manifest()`.
            Should contain 'recording_id', 'subject_id', 'dataset_id', 'version_tag', and 's3_url' fields.

        Returns
        -------
        dict
            Dictionary with keys:
                - 'recording_id': Recording identifier
                - 'subject_id': Subject identifier
                - 'data_dir': Path to downloaded data directory
                - 'dataset_id': OpenNeuro dataset identifier
                - 'version_tag': Version tag that was downloaded
        """
        self.update_status("DOWNLOADING")
        self.raw_dir.mkdir(exist_ok=True, parents=True)

        # Access fields from manifest_item (pandas Series)
        # The index is 'recording_id', accessible via .Index (set by runner)
        recording_id = manifest_item.Index
        subject_id = manifest_item.subject_id
        dataset_id = manifest_item.dataset_id
        version_tag = manifest_item.version_tag
        s3_url = manifest_item.s3_url

        target_dir = self.raw_dir
        # Create subject-specific directory
        # Use subject_id to ensure unique directories for each subject
        # All recordings for the same subject will be stored in the same directory
        subject_dir = target_dir / subject_id
        subject_dir.mkdir(exist_ok=True, parents=True)

        # Check if files for this specific recording already exist
        # Since we download the entire subject directory (all recordings/tasks/acquisitions)
        # in one sync operation, we check if the specific recording files exist rather than
        # just checking if the subject directory exists. This ensures we only skip downloads
        # when the actual recording data is present.
        if self._check_recording_files_exist(recording_id, subject_dir):
            if not (self.args and getattr(self.args, "redownload", False)):
                self.update_status("Already Downloaded")
                return {
                    "recording_id": recording_id,
                    "subject_id": subject_id,
                    "data_dir": subject_dir,
                    "dataset_id": dataset_id,
                    "version_tag": version_tag,
                }
        # TODO: add download of the missing recording if subject_dir exists.

        # Download entire subject directory from S3 using AWS CLI
        # This downloads ALL files for the subject (all recordings, tasks, acquisitions)
        # The s3_url points to the subject directory (e.g., s3://.../sub-1/)
        # and aws s3 sync recursively downloads all files under that prefix
        try:
            self._download_from_s3(s3_url, subject_dir)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download data for {subject_id} from {dataset_id}: {str(e)}"
            )

        return {
            "recording_id": recording_id,
            "subject_id": subject_id,
            "data_dir": subject_dir,
            "dataset_id": dataset_id,
            "version_tag": version_tag,
        }

    def process(self, download_output: dict):
        """Process and save the dataset.

        This default implementation handles common OpenNeuro EEG processing tasks:
        1. Finds EEG files in BIDS structure
        2. Loads them using MNE
        3. Extracts metadata (subject, session, device, brainset descriptions)
        4. Extracts EEG signal and channel information
        5. Creates a Data object and saves it to disk

        Subclasses can override this method and call super().process() to use
        the default implementation, then add dataset-specific processing logic.

        Parameters
        ----------
        download_output : dict
            Dictionary returned by :meth:`download()` containing:
                - 'recording_id': Recording identifier (if available)
                - 'subject_id': Subject identifier
                - 'data_dir': Path to downloaded data directory
                - 'dataset_id': OpenNeuro dataset identifier
                - 'version_tag': Version tag that was downloaded
        """
        self.processed_dir.mkdir(exist_ok=True, parents=True)

        recording_id = download_output.get("recording_id")
        subject_id = download_output["subject_id"]
        data_dir = download_output["data_dir"]
        dataset_id = download_output["dataset_id"]
        version_tag = download_output["version_tag"]

        # Use recording_id if available, otherwise fall back to subject_id
        identifier = recording_id if recording_id else subject_id

        # Find EEG files in BIDS structure
        # Look for common EEG file formats: .fif, .set, .edf, .bdf, .vhdr, .eeg
        eeg_patterns = [
            "**/*.fif",
            "**/*.set",
            "**/*.edf",
            "**/*.bdf",
            "**/*.vhdr",
            "**/*.eeg",
        ]

        eeg_files = []
        for pattern in eeg_patterns:
            eeg_files.extend(glob.glob(str(data_dir / pattern), recursive=True))
            if eeg_files:
                break  # Use first format found

        if not eeg_files:
            raise RuntimeError(
                f"No EEG files found in {data_dir}. "
                f"Expected BIDS structure with EEG files in sub-*/ses-*/eeg/ directories."
            )

        # Process each EEG file (typically one per session)
        for eeg_file_path in eeg_files:
            eeg_file = Path(eeg_file_path)
            self.update_status(f"Processing {eeg_file.name}")

            # Determine session ID from BIDS structure
            # Path structure: sub-XX/ses-YY/eeg/file.fif
            parts = eeg_file.parts
            session_id = None
            for i, part in enumerate(parts):
                if part.startswith("ses-"):
                    session_id = part.replace("ses-", "")
                    break

            # If no session found, use subject_id as session_id
            if session_id is None:
                session_id = subject_id.replace("sub-", "")

            # Create full session identifier
            full_session_id = f"{subject_id}_ses-{session_id}"

            # Check if already processed
            # Use recording_id if available for more specific naming
            if recording_id:
                store_name = f"{recording_id}.h5"
            else:
                store_name = f"{full_session_id}.h5"
            store_path = self.processed_dir / store_name
            if store_path.exists() and not (
                self.args and getattr(self.args, "reprocess", False)
            ):
                self.update_status("Skipped Processing")
                continue

            # Load EEG file with MNE
            self.update_status("Loading EEG file")
            try:
                if eeg_file.suffix == ".fif":
                    raw = mne.io.read_raw_fif(eeg_file, preload=True, verbose=False)
                elif eeg_file.suffix == ".set":
                    raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
                elif eeg_file.suffix in [".edf", ".bdf"]:
                    raw = mne.io.read_raw_edf(eeg_file, preload=True, verbose=False)
                elif eeg_file.suffix == ".vhdr":
                    raw = mne.io.read_raw_brainvision(
                        eeg_file, preload=True, verbose=False
                    )
                elif eeg_file.suffix == ".eeg":
                    # .eeg files are often paired with .vhdr, try to find .vhdr
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

            # Extract metadata
            self.update_status("Extracting Metadata")

            # Get dataset metadata from OpenNeuro
            try:
                metadata = fetch_metadata(dataset_id)
                dataset_description = (
                    self.description
                    if self.description is not None
                    else metadata.get("datasetName", f"OpenNeuro dataset {dataset_id}")
                )
                source = f"https://openneuro.org/datasets/{dataset_id}"
            except Exception:
                # Fallback if metadata fetch fails
                dataset_description = (
                    self.description
                    if self.description is not None
                    else f"OpenNeuro dataset {dataset_id}"
                )
                source = f"https://openneuro.org/datasets/{dataset_id}"

            # Create brainset description
            brainset_description = extract_brainset_description(
                dataset_id=type(self).get_brainset_id(),
                origin_version=f"openneuro/{dataset_id}/{version_tag}",
                derived_version=self.derived_version,
                source=source,
                description=dataset_description,
            )

            # Extract subject description
            subject_description = extract_subject_description(subject_id)

            # Extract measurement date
            meas_date = extract_meas_date(raw)
            if meas_date is None:
                # Fallback to current date if not available
                meas_date = datetime.datetime.now()

            # Create session description
            session_description = extract_session_description(
                session_id=full_session_id, recording_date=meas_date
            )

            # Create device description
            device_id = f"{subject_id}_{session_id}"
            device_description = extract_device_description(device_id=device_id)

            # Extract EEG signal
            self.update_status("Extracting EEG Signal")
            eeg_signal = extract_signal(raw)

            # Extract channel information
            channels = extract_channels(raw)

            # Create Data object
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

            # Save to disk
            self.update_status("Storing")
            with h5py.File(store_path, "w") as file:
                data.to_hdf5(file, serialize_fn_map=serialize_fn_map)
