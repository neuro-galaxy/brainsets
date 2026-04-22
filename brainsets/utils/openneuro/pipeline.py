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
from pathlib import Path
from typing import Literal, Optional, TypedDict
import logging
import sys

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
    fetch_latest_snapshot_tag,
)

_openneuro_parser = ArgumentParser()
_openneuro_parser.add_argument("--redownload", action="store_true")
_openneuro_parser.add_argument("--reprocess", action="store_true")
_openneuro_parser.add_argument(
    "--on-version-mismatch",
    choices=["abort", "continue", "prompt"],
    default="prompt",
    help=(
        "Behavior when origin_version differs from latest OpenNeuro version: "
        "'abort' raises an error, 'continue' proceeds with warning, "
        "'prompt' asks for confirmation in interactive sessions."
    ),
)


def _require_mne_bids(func_name: str) -> None:
    """Raise ImportError if mne-bids is not available."""
    if not MNE_BIDS_AVAILABLE:
        raise ImportError(
            f"{func_name} requires mne-bids, which is not installed. "
            "Install it with `pip install mne-bids`."
        )


class OpenNeuroContext(TypedDict):
    """Typed structure for shared OpenNeuro metadata cached per dataset."""

    dataset_id: str
    latest_snapshot_tag: str
    species: str
    participants_data: Optional[pd.DataFrame]


class OpenNeuroPipeline(BrainsetPipeline, ABC):
    """Abstract base class for OpenNeuro dataset pipelines.

    This class extends BrainsetPipeline and provides common functionality for
    processing datasets from OpenNeuro, supporting both EEG and iEEG modalities.

    Required class attributes:
        - dataset_id (str): OpenNeuro dataset identifier (e.g., "ds005555").
        - brainset_id (str): Unique string identifier for the brainset.
        - origin_version (str): Version of the original source data.

    Optional class attributes:
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
    """Argument parser for common OpenNeuro pipeline flags."""

    dataset_id: str
    """OpenNeuro dataset identifier (e.g., "ds005555", "ds006914")."""

    brainset_id: str
    """Unique identifier for the brainset."""

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

    _cached_openneuro_context: dict[str, OpenNeuroContext] = {}
    """Class-level cache of shared OpenNeuro context, keyed by dataset_id."""

    @staticmethod
    def validate_dataset_id(dataset_id: str) -> None:
        """Validate OpenNeuro dataset identifier format.

        OpenNeuro dataset IDs follow the format 'ds' followed by exactly 6 digits,
        where the numeric portion ranges from 000001 to 009999.

        Args:
            dataset_id: The dataset identifier in strict format:
                - Must be lowercase 'ds' followed by exactly 6 digits.
                - Numeric portion must be between 000001 and 009999.

        Raises:
            ValueError: If the dataset ID format is invalid, does not match strict format,
                or the numeric part is outside the valid range.
        """
        if not isinstance(dataset_id, str) or len(dataset_id) != 8:
            raise ValueError(
                f"Invalid dataset ID format: '{dataset_id}'. Expected 'ds' followed by exactly 6 digits."
            )

        if not dataset_id.startswith("ds"):
            raise ValueError(
                f"Invalid dataset ID format: '{dataset_id}'. Expected 'ds' followed by exactly 6 digits."
            )

        numeric_part = dataset_id[2:]
        if not numeric_part.isdigit():
            raise ValueError(
                f"Invalid dataset ID format: '{dataset_id}'. Expected 'ds' followed by exactly 6 digits."
            )

        num = int(numeric_part)
        if num < 1 or num > 9999:
            raise ValueError(
                f"Dataset ID '{dataset_id}' has invalid numeric portion. Must be between 000001 and 009999."
            )

    @classmethod
    def _validate_dataset_version(
        cls,
        latest_snapshot_tag: str,
        on_mismatch: Literal["abort", "continue", "prompt"] = "prompt",
    ) -> None:
        """Validate origin version against the latest OpenNeuro snapshot tag.

        Args:
            latest_snapshot_tag: The latest snapshot tag available on OpenNeuro for this dataset.
            on_mismatch: Policy when ``origin_version`` differs from latest
                (``"abort"``, ``"continue"``, or ``"prompt"``). If a mismatch is detected, the
                ``on_mismatch`` parameter determines the behavior (default: ``"prompt"``):
                    - ``"abort"``: Raises an error and exits the pipeline.
                    - ``"continue"``: Logs a warning and proceeds with the latest version.
                    - ``"prompt"``: Prompts the user for confirmation and proceeds if confirmed.

        Raises:
            SystemExit: If mismatch policy aborts execution or user declines prompt.
        """

        def user_confirms(
            prompt: str,
        ) -> bool:
            """Return True if the user confirms continuation, False otherwise."""
            answer = input(prompt).strip().lower()
            return answer in {"y", "yes"}

        if latest_snapshot_tag != cls.origin_version:
            if on_mismatch == "continue":
                logging.warning(
                    f"⚠️ Dataset version '{cls.origin_version}' was used to create the brainset pipeline for dataset '{cls.dataset_id}', "
                    f"but the latest available version on OpenNeuro is '{latest_snapshot_tag}'. "
                    "Downloading data or running the pipeline now will use the latest version, "
                    "which may differ from the original version used, potentially causing errors or inconsistencies. "
                    "Check the CHANGES file of the dataset for details about the differences between versions."
                )
            elif on_mismatch == "abort":
                raise SystemExit(
                    "🛑 Aborting pipeline due to dataset version mismatch."
                )
            elif on_mismatch == "prompt":
                prompt_message = (
                    f"⚠️ Dataset '{cls.dataset_id}' pipeline version is '{cls.origin_version}', "
                    f"but latest on OpenNeuro is '{latest_snapshot_tag}'. "
                    "👉 Continue with latest version? [y/N]: "
                )
                if not user_confirms(prompt_message):
                    raise SystemExit(
                        "🛑 Aborted by user due to dataset version mismatch."
                    )

    @staticmethod
    def _validate_on_mismatch_policy(on_version_mismatch: str) -> None:
        """Validate that on_version_mismatch policy is compatible with execution mode.

        In non-interactive sessions, the 'prompt' policy is invalid because it requires
        user input. This validation runs early to provide a clear error message.

        Args:
            on_version_mismatch: Policy value ('abort', 'continue', or 'prompt').

        Raises:
            ValueError: If on_version_mismatch='prompt' in non-interactive mode.
        """
        if on_version_mismatch == "prompt" and not sys.stdin.isatty():
            raise ValueError(
                "Cannot use --on-version-mismatch='prompt' in non-interactive mode. "
                "The program is running without a TTY and cannot prompt for user input. "
                "Set --on-version-mismatch to either 'continue' (warn and proceed) or 'abort' (fail on mismatch)."
            )

    @staticmethod
    def _normalize_species(species: str | None) -> str:
        """Normalize species names to ``"homo sapiens"`` or ``"unknown"``.

        Args:
            species: The input species name (string or None).

        Returns:
            ``"homo sapiens"`` for recognized human aliases, otherwise
            ``"unknown"``.
        """
        if not isinstance(species, str):
            return "unknown"

        normalized_species = species.strip().lower()
        homo_sapiens_aliases = {
            "homo",
            "homo sapiens",
            "human",
            "humans",
            "h. sapiens",
        }
        if normalized_species in homo_sapiens_aliases:
            return "homo sapiens"
        return "unknown"

    @classmethod
    def _openneuro_context(
        cls,
        on_version_mismatch: str = "prompt",
    ) -> OpenNeuroContext:
        """Return cached OpenNeuro metadata for this pipeline class.

        Metadata is cached per dataset_id (within current process):
        - latest_snapshot_tag
        - species
        - participants_data

        Returns:
            OpenNeuro context with latest snapshot tag, normalized species, and
            participants table.
        """
        cls.validate_dataset_id(cls.dataset_id)
        if cls.dataset_id in cls._cached_openneuro_context:
            return cls._cached_openneuro_context[cls.dataset_id]

        latest_snapshot_tag = fetch_latest_snapshot_tag(cls.dataset_id)
        cls._validate_dataset_version(
            latest_snapshot_tag, on_mismatch=on_version_mismatch
        )

        species = fetch_species(cls.dataset_id)

        ctx: OpenNeuroContext = {
            "latest_snapshot_tag": latest_snapshot_tag,
            "species": cls._normalize_species(species),
            "participants_data": fetch_participants_tsv(cls.dataset_id),
        }
        cls._cached_openneuro_context[cls.dataset_id] = ctx
        return ctx

    @classmethod
    def get_manifest(cls, raw_dir: Path, args: Optional[Namespace]) -> pd.DataFrame:
        """Generate a manifest DataFrame by discovering recordings from OpenNeuro.

        This implementation queries OpenNeuro S3 and parses BIDS-compliant
        filenames to discover recordings for the pipeline modality.

        Args:
            raw_dir: Raw data directory assigned to this brainset
            args: Pipeline-specific arguments parsed from the command line

        Returns:
            DataFrame with columns:
                - subject_id: Subject identifier (e.g., 'sub-01')
                - recording_id: Recording identifier (index)
                - s3_url: S3 URL for downloading
        """
        on_version_mismatch = (
            getattr(args, "on_version_mismatch", "prompt")
            if args is not None
            else "prompt"
        )
        cls._validate_on_mismatch_policy(on_version_mismatch)
        cls._openneuro_context(on_version_mismatch=on_version_mismatch)

        all_files = fetch_all_filenames(cls.dataset_id)

        if cls.modality == "eeg":
            recordings = fetch_eeg_recordings(all_files)
        elif cls.modality == "ieeg":
            recordings = fetch_ieeg_recordings(all_files)
        else:
            raise ValueError(f"Unknown modality: {cls.modality}")

        manifest_list = []
        for rec in recordings:
            s3_url = construct_s3_url_from_path(
                cls.dataset_id,
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
                f"No {cls.modality.upper()} recordings found in dataset {cls.dataset_id}"
            )

        manifest = pd.DataFrame(manifest_list)
        return manifest.set_index("recording_id")

    def download(self, manifest_item) -> dict:
        """Download data for a single recording from OpenNeuro S3.

        Args:
            manifest_item: A single row of the manifest

        Returns:
            Dictionary containing ``subject_id`` and ``recording_id``.
        """
        self.update_status("DOWNLOADING")
        self.raw_dir.mkdir(exist_ok=True, parents=True)

        subject_id = manifest_item.subject_id
        recording_id = manifest_item.Index
        s3_url = manifest_item.s3_url
        root_dir = self.raw_dir

        # if the dataset_description.json file does not exist or the redownload flag is set, download it
        # dataset_description.json is required for mne-bids to recognize a valid BIDS dataset
        dataset_description_exists = (root_dir / "dataset_description.json").exists()
        if not dataset_description_exists or getattr(self.args, "redownload", False):
            download_dataset_description(self.dataset_id, root_dir)

        if not getattr(self.args, "redownload", False):
            if self.modality == "eeg":
                if check_eeg_recording_files_exist(root_dir, recording_id):
                    self.update_status("Already Downloaded")
                    return {
                        "subject_id": subject_id,
                        "recording_id": recording_id,
                    }
            elif self.modality == "ieeg":
                if check_ieeg_recording_files_exist(root_dir, recording_id):
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
        1. Loads BIDS-structured data files using MNE-BIDS
        2. Extracts metadata (subject, session, device, brainset descriptions)
        3. Extracts signal and channel information
        5. Creates a Data object

        Args:
            download_output: Dictionary returned by download()

        Returns:
            Tuple of ``(data, store_path)``, or ``None`` if processing is skipped.
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
        ctx = self._cached_openneuro_context[self.dataset_id]
        source = f"https://openneuro.org/datasets/{self.dataset_id}"
        dataset_description = (
            self.description
            if self.description
            else f"OpenNeuro dataset {self.dataset_id}"
        )

        brainset_description = BrainsetDescription(
            id=self.brainset_id,
            origin_version=ctx["latest_snapshot_tag"],
            derived_version=self.derived_version,
            source=source,
            description=dataset_description,
        )

        subject_info = get_subject_info(subject_id, ctx["participants_data"])
        species = ctx["species"]
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

        Default implementation calls :meth:`_process_common` and persists the
        result. Subclasses can override to add dataset-specific processing.

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
            Mapping from original channel names to standardized names, or
            ``None``.
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
            Mapping from channel type to channel name list, or ``None``.
        """
        return self.TYPE_CHANNELS_REMAPPING

    def generate_splits(
        self, domain: Interval, subject_id: str, session_id: str
    ) -> Data:
        """Generate default intrasession train/valid splits.

        This method uses a causal sequential split over the recording domain and
        also assigns intersubject and intersession split labels.

        Args:
            domain: Interval domain representing recording start/end times.
            subject_id: Subject identifier.
            session_id: Session identifier.

        Returns:
            Data object with ``train``/``valid`` intervals and assignment labels
            for intersubject and intersession strategies.

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
    """Base pipeline for EEG data processing from OpenNeuro.

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

    Class Attributes:
        - CHANNEL_NAME_REMAPPING (dict, optional): Map old channel names to new ones.
        - TYPE_CHANNELS_REMAPPING (dict, optional): Map channel types to channel lists.
    """

    modality = "eeg"
    """Data modality for this pipeline."""


class OpenNeuroIEEGPipeline(OpenNeuroPipeline):
    """Base pipeline for iEEG data processing from OpenNeuro.

    This class provides iEEG-specific extensions on top of OpenNeuroPipeline,
    leveraging BIDS-compliant sidecar files to define channel and electrode
    configuration automatically.

    Usage:

    For most iEEG datasets where electrode and channel information are properly
    defined in BIDS sidecar files, subclassing this pipeline is sufficient:

        class Pipeline(OpenNeuroIEEGPipeline):
            brainset_id = "your_dataset_id"
            dataset_id = "dsXXXXXX"
            origin_version = "1.0.0"

    Changing electrode names and types:
        - If you wish to change the names or types of electrodes (e.g., for harmonization or custom processing),
          use the same approach as in OpenNeuroEEGPipeline:
            - Set the `CHANNEL_NAME_REMAPPING`/`TYPE_CHANNELS_REMAPPING` class attribute, or
            - Override `get_channel_name_remapping(self, recording_id)` / `get_type_channels_remapping(self, recording_id)`
          with your desired logic.

    Class Attributes:
        - CHANNEL_NAME_REMAPPING (dict, optional): Map old electrode/channel names to new ones.
        - TYPE_CHANNELS_REMAPPING (dict, optional): Map types to electrode/channel lists.
    """

    modality = "ieeg"
    """Data modality for this pipeline."""
