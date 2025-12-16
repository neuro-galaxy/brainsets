from abc import ABC, ABCMeta, abstractmethod
from argparse import Namespace
from pathlib import Path
from typing import Optional
import datetime
import glob
import json
import inspect

import h5py
import mne
import pandas as pd

from temporaldata import Data

from brainsets import serialize_fn_map
from brainsets.pipeline import BrainsetPipeline
from brainsets.utils.open_neuro import (
    validate_dataset_id,
    construct_s3_url,
    download_prefix_from_s3,
    check_recording_files_exist,
    fetch_latest_version_tag,
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


class AutoLoadIdentifiersMeta(ABCMeta):
    """
    Metaclass to automatically load brainset_id, dataset_id, and config_file_path during class creation.
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        config_file_path = cls._get_config_file_path()
        cls.config_file_path = config_file_path

        brainset_id, dataset_id = cls._get_identifiers()
        cls.brainset_id = brainset_id
        cls.dataset_id = dataset_id

        return cls


class OpenNeuroEEGPipeline(BrainsetPipeline, ABC):
    """Abstract base class for OpenNeuro EEG dataset pipelines.

    This class extends BrainsetPipeline and provides common functionality for
    processing EEG datasets from OpenNeuro. It includes default implementations
    for get_manifest(), download(), and process() methods that handle common
    OpenNeuro EEG dataset processing tasks.

    **Required class attributes:**
        - :attr:`brainset_id`: Unique identifier for the brainset
        - :attr:`dataset_id`: OpenNeuro dataset identifier

    **Optional class attributes:**
        - :attr:`parser`: ArgumentParser for pipeline-specific CLI arguments
        - :attr:`version_tag`: Specific version tag
        - :attr:`subject_ids`: List of subject IDs to process (defaults to all)
        - :attr:`derived_version`: Version of the processed data (defaults to "1.0.0")
        - :attr:`description`: Description of the dataset

    **The pipeline workflow consists of:**
        1. Generating a manifest (list of assets to process) via :meth:`get_manifest()`.
           This loads recording information from a config file ({dataset_id}_config.json)
           located in the pipeline directory and creates manifest entries organized by recording_id.
        2. Downloading each asset via :meth:`download()`. This downloads files directly from
           OpenNeuro's S3 bucket using boto3 (s3://openneuro.org/{dataset_id}/).
        3. Processing each downloaded asset via :meth:`process()`. This default
           implementation handles common OpenNeuro EEG dataset processing tasks, but can be extended
           by subclasses by calling super().process() and adding additional logic.

    **Extending the process() method:**
        Subclasses can override process() and call super().process() to use the
        default implementation, then add dataset-specific processing:
    """

    dataset_id: str
    """OpenNeuro dataset identifier. Set according to config file metadata."""

    brainset_id: str
    """Brainset identifier. Set according to config file metadata."""

    config_file_path: Path
    """Path to the config file for this pipeline. Set automatically."""

    version_tag: Optional[str] = None
    """Optional version tag to use. Set according to config file metadata."""

    subject_ids: Optional[list[str]] = None
    """Optional list of subject IDs to process. If None, processes all subjects."""

    derived_version: str = "1.0.0"
    """Version of the processed data. Defaults to "1.0.0"."""

    description: Optional[str] = None
    """Optional description of the dataset. Set according to dataset `name` in config file."""

    @classmethod
    @abstractmethod
    def _get_identifiers(cls) -> tuple[str, str]:
        """Get brainset_id and dataset_id for this pipeline.

        This method must be implemented by subclasses. It can either:
        - Load identifiers from a config file
        - Return hardcoded values

        """
        ...

    @classmethod
    def _get_config_file_path(cls) -> Path:
        """
        Searches for a file matching the pattern *_config.json
        in the same directory as the pipeline module. It expects exactly one such file.

        Returns
        -------
        Path
            Path to the brainset config file.
        """
        pipeline_file = None
        stack = inspect.stack()

        current_file = Path(__file__).resolve()
        for frame_info in stack[1:]:
            filename = frame_info.filename
            # Skip if it's this file, a built-in, or internal Python files
            if (
                filename
                and not filename.startswith("<")
                and not filename.startswith("<frozen")
            ):
                try:
                    frame_file = Path(filename).resolve()
                    if frame_file.exists() and frame_file != current_file:
                        pipeline_file = frame_file
                        break
                except (OSError, ValueError):
                    continue

        if pipeline_file is None or not pipeline_file.exists():
            raise RuntimeError(
                f"Could not determine pipeline module file path for {cls.__name__}. "
                f"Tried inspect.getfile(), module.__file__, and stack inspection."
            )

        pipeline_dir = pipeline_file.parent

        config_files = list(pipeline_dir.glob("*_config.json"))

        if not config_files:
            raise FileNotFoundError(
                f"Config file not found in {pipeline_dir}. "
                f"Expected a file named *_config.json in the pipeline directory."
            )

        if len(config_files) > 1:
            raise RuntimeError(
                f"Multiple config files found in {pipeline_dir}: {config_files}. "
                f"Expected exactly one *_config.json file."
            )

        return config_files[0]

    @classmethod
    def _load_config(cls, config_file: Path) -> dict:
        """Load the config file for this pipeline.

        Returns
        -------
        dict
            Configuration dictionary loaded from the config file.
        """
        with open(config_file, "r") as f:
            return json.load(f)

    @classmethod
    def get_manifest(cls, raw_dir: Path, args: Optional[Namespace]) -> pd.DataFrame:
        """Generate a manifest DataFrame listing all assets to process.

        This implementation loads the config file to get recording information
        and creates a manifest organized by recording_id.

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
                - 'recording_id': Recording/Session identifier (e.g., 'sub-1_task-Sleep_acq-headband')
                - 'subject_id': Subject identifier (e.g., 'sub-1')
                - 'task_id': Task identifier (e.g., 'Sleep')
                - 'channel_map_id': Channel map identifier
                - 'dataset_id': OpenNeuro dataset identifier
                - 'version_tag': Version tag to download
                - 's3_url': S3 URL for downloading
                - 'fpath': Local file path where the manifest item will be downloaded
            The index is set to 'recording_id'.
        """
        dataset_id = validate_dataset_id(cls.dataset_id)

        config = cls._load_config(cls.config_file_path)

        if cls.version_tag is None:
            try:
                version_tag = (
                    config.get("dataset", {}).get("metadata", {}).get("version", None)
                )
                cls.version_tag = version_tag
            except Exception as e:
                raise ValueError(
                    f"Version number not found in config file metadata. {e}"
                )
        else:
            version_tag = cls.version_tag

        cls.description = (
            config.get("dataset", {}).get("metadata", {}).get("name", None)
        )

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
        # FIXME
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

        manifest_list = []
        for rec in recording_info:
            recording_id = rec.get("recording_id")
            if not recording_id:
                continue

            s3_url = construct_s3_url(dataset_id, recording_id)

            subject_id = rec.get("subject_id")
            fpath = raw_dir / subject_id / recording_id if subject_id else None
            # FIXME fpath = raw_dir / subject_id / eeg /recording_id

            manifest_list.append(
                {
                    "recording_id": recording_id,
                    "subject_id": subject_id,
                    "task_id": rec.get("task_id"),
                    "channel_map_id": rec.get("channel_map_id"),
                    "dataset_id": dataset_id,
                    "version_tag": version_tag,
                    "s3_url": s3_url,
                    "fpath": fpath,
                }
            )

        if not manifest_list:
            raise ValueError("No recordings found in config file to process")

        manifest = pd.DataFrame(manifest_list)
        manifest = manifest.set_index("recording_id")
        return manifest

    def download(self, manifest_item) -> dict:
        """Download data for a single recording from OpenNeuro S3.

        Parameters
        ----------
        manifest_item : pandas.Series
            A single row of the manifest returned by :meth:`get_manifest()`.
            Should contain 'recording_id', 'subject_id', 'dataset_id', 'version_tag', 's3_url' and 'fpath' fields.

        Returns
        -------
        dict
            Dictionary with keys:
                - 'recording_id': Recording identifier
                - 'subject_id': Subject identifier
                - 'fpath': Local file path where the manifest item was downloaded
        """
        self.update_status("DOWNLOADING")
        self.raw_dir.mkdir(exist_ok=True, parents=True)

        dataset_id = manifest_item.dataset_id
        version_tag = manifest_item.version_tag
        s3_url = manifest_item.s3_url

        recording_id = manifest_item.Index
        subject_id = manifest_item.subject_id

        target_dir = self.raw_dir
        target_dir.mkdir(exist_ok=True, parents=True)

        subject_dir = target_dir / subject_id
        if check_recording_files_exist(recording_id, subject_dir):
            if not (self.args and getattr(self.args, "redownload", False)):
                self.update_status("Already Downloaded")
                fpath = subject_dir
                return {
                    "recording_id": recording_id,
                    "subject_id": subject_id,
                    "fpath": fpath,
                }

        try:
            self.update_status(f"Downloading from {s3_url}")

            download_prefix_from_s3(s3_url, target_dir)
            fpath = subject_dir
        except Exception as e:
            fpath = None
            raise RuntimeError(
                f"Failed to download data for {subject_id} from {dataset_id}: {str(e)}"
            )

        return {
            "recording_id": recording_id,
            "subject_id": subject_id,
            "fpath": fpath,
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
                - 'fpath': Local file path where the manifest item was downloaded
        """

        self.processed_dir.mkdir(exist_ok=True, parents=True)

        recording_id = download_output.get("recording_id")
        subject_id = download_output["subject_id"]
        data_dir = Path(download_output["fpath"])

        self.update_status("Extracting Metadata")
        source = f"https://openneuro.org/datasets/{self.dataset_id}"
        if self.description is not None:
            dataset_description = self.description
        else:
            dataset_description = f"OpenNeuro dataset {self.dataset_id}"

        brainset_description = extract_brainset_description(
            dataset_id=self.dataset_id,
            origin_version=f"openneuro/{self.dataset_id}/{self.version_tag}",
            derived_version=self.derived_version,
            source=source,
            description=dataset_description,
        )

        # FIXME: verif pattern /eeg/*.ext in BIDS ?
        # there are more possible extension like .mat or .set
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
                f"No EEG files found in {data_dir}. "
                f"Expected BIDS structure with EEG files in sub-*/eeg/ directories."
            )

        for eeg_file_path in eeg_files:
            eeg_file = Path(eeg_file_path)
            self.update_status(f"Processing {eeg_file.name}")

            # TODO: reverif ses- patterns in BIDS ?
            parts = eeg_file.parts
            session_id = None
            for i, part in enumerate(parts):
                if part.startswith("ses-"):
                    session_id = part.replace("ses-", "")
                    break

            if session_id is None:
                session_id = subject_id.replace("sub-", "")

            full_session_id = f"{subject_id}_ses-{session_id}"

            if recording_id:
                store_name = f"{recording_id}.h5"
            else:
                store_name = f"{full_session_id}.h5"
            store_path = self.processed_dir / store_name
            # TODO should we add subject directory ?
            if store_path.exists() and not (
                self.args and getattr(self.args, "reprocess", False)
            ):
                self.update_status("Skipped Processing")
                continue

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

            subject_description = extract_subject_description(subject_id)

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
            # TODO need to map channels to standard channel names in config file

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
            # TODO IMPORTANT : get the channel maps from the config file

            self.update_status("Storing")
            with h5py.File(store_path, "w") as file:
                data.to_hdf5(file, serialize_fn_map=serialize_fn_map)
