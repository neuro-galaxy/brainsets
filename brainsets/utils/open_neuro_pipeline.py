"""Base class for OpenNeuro EEG pipelines.

This module provides the OpenNeuroEEGPipeline abstract class, which serves as a base class for creating pipelines that process EEG data from OpenNeuro datasets. It includes default implementations for get_manifest(), download(), and process() methods that handle common OpenNeuro EEG dataset processing tasks. Subclasses can extend these methods by calling super() and adding dataset-specific logic.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Optional
from argparse import Namespace
import datetime
import glob

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
from brainsets.utils.open_neuro_utils import download
from brainsets.utils.open_neuro_utils.data_extraction import (
    extract_brainset_description,
    extract_subject_description,
    extract_session_description,
    extract_device_description,
    extract_meas_date,
    extract_signal,
    extract_channels,
)


class OpenNeuroEEGPipeline(BrainsetPipeline):
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
           This uses OpenNeuro utilities to fetch participant and file information.
        2. Downloading each asset via :meth:`download()`. This downloads files from
           OpenNeuro S3 using the openneuro-py library.
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

    @classmethod
    def get_manifest(cls, raw_dir: Path, args: Optional[Namespace]) -> pd.DataFrame:
        """Generate a manifest DataFrame listing all assets to process.

        This implementation uses OpenNeuro utilities to fetch participant
        information and create a manifest. Each row represents a subject
        to be downloaded and processed.

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
                - 'subject_id': Subject identifier (e.g., 'sub-01')
                - 'dataset_id': OpenNeuro dataset identifier
                - 'version_tag': Version tag to download
            The index is set to 'subject_id'.
        """
        # Validate dataset ID
        dataset_id = validate_dataset_id(cls.dataset_id)

        # Get version tag (use latest if not specified)
        if cls.version_tag is None:
            version_tag = fetch_latest_version_tag(dataset_id)
        else:
            version_tag = cls.version_tag

        # Get list of participants
        if cls.subject_ids is None:
            participants = fetch_participants(dataset_id, version_tag)
        else:
            # Validate that requested subjects exist
            all_participants = fetch_participants(dataset_id, version_tag)
            participants = [
                subj for subj in cls.subject_ids if subj in all_participants
            ]
            if not participants:
                raise ValueError(
                    f"None of the requested subjects {cls.subject_ids} "
                    f"were found in dataset {dataset_id}"
                )

        # Create manifest
        # Note: subject_id is used as the index, so we don't include it as a column
        manifest_list = [
            {
                "dataset_id": dataset_id,
                "version_tag": version_tag,
            }
            for subj_id in participants
        ]

        manifest = pd.DataFrame(manifest_list, index=participants)
        manifest.index.name = "subject_id"
        return manifest

    def download(self, manifest_item) -> dict:
        """Download data for a single subject from OpenNeuro.

        This implementation downloads EEG data for the subject specified in
        manifest_item using the openneuro-py download utility.

        Parameters
        ----------
        manifest_item : NamedTuple
            A single row of the manifest returned by :meth:`get_manifest()`.
            Should contain 'subject_id', 'dataset_id', and 'version_tag' fields.

        Returns
        -------
        dict
            Dictionary with keys:
                - 'subject_id': Subject identifier
                - 'data_dir': Path to downloaded data directory
                - 'dataset_id': OpenNeuro dataset identifier
                - 'version_tag': Version tag that was downloaded
        """
        self.update_status("DOWNLOADING")
        self.raw_dir.mkdir(exist_ok=True, parents=True)

        # Access fields from manifest_item (pandas Series)
        # The index is 'subject_id', accessible via .Index (set by runner)
        subject_id = manifest_item.Index
        dataset_id = manifest_item.dataset_id
        version_tag = manifest_item.version_tag

        # Create subject-specific directory
        subject_dir = self.raw_dir / subject_id
        subject_dir.mkdir(exist_ok=True, parents=True)

        # Check if already downloaded (unless redownload is requested)
        if subject_dir.exists() and any(subject_dir.iterdir()):
            if not (self.args and getattr(self.args, "redownload", False)):
                self.update_status("Already Downloaded")
                return {
                    "subject_id": subject_id,
                    "data_dir": subject_dir,
                    "dataset_id": dataset_id,
                    "version_tag": version_tag,
                }

        # Download subject data using openneuro-py
        # Include pattern for EEG files (BIDS structure)
        include_pattern = f"{subject_id}/**/*"

        try:
            download(
                dataset=dataset_id,
                tag=version_tag,
                target_dir=str(subject_dir),
                include=[include_pattern],
                verify_hash=True,
                verify_size=True,
                max_retries=5,
                max_concurrent_downloads=5,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download data for {subject_id} from {dataset_id}: {str(e)}"
            )

        return {
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
                - 'subject_id': Subject identifier
                - 'data_dir': Path to downloaded data directory
                - 'dataset_id': OpenNeuro dataset identifier
                - 'version_tag': Version tag that was downloaded
        """
        self.processed_dir.mkdir(exist_ok=True, parents=True)

        subject_id = download_output["subject_id"]
        data_dir = download_output["data_dir"]
        dataset_id = download_output["dataset_id"]
        version_tag = download_output["version_tag"]

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
            store_path = self.processed_dir / f"{full_session_id}.h5"
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
                dataset_id=self.brainset_id,
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
