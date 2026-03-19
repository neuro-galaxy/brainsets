"""Brain Imaging Data Structure (BIDS) utilities.

This module provides utility functions to parse BIDS-compliant filenames, discover BIDS recordings in a dataset, and check for the existence of BIDS-conformant data files.

For more information about BIDS, see the BIDS specification: https://bids-specification.readthedocs.io/en/stable/
"""

from collections import defaultdict
from typing import Optional
from pathlib import Path
import warnings
import re
import json
import pandas as pd

try:
    from mne_bids import get_bids_path_from_fname, get_entities_from_fname, BIDSPath

    MNE_BIDS_AVAILABLE = True
except ImportError:
    get_bids_path_from_fname = None
    get_entities_from_fname = None
    BIDSPath = None
    MNE_BIDS_AVAILABLE = False

# BIDS EEG supported formats (BIDS v1.10.1):
# - European Data Format (.edf): Single file per recording. edf+ files permitted.
# - BrainVision (.vhdr): Header file; requires .vmrk (markers) and .eeg (data) files.
# - EEGLAB (.set): MATLAB format; optional .fdt file contains float data.
# - Biosemi (.bdf): Single file per recording. bdf+ files permitted.
# Reference: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html
EEG_EXTENSIONS = {".edf", ".vhdr", ".set", ".bdf"}

# BIDS iEEG supported formats (BIDS v1.10.1):
# - European Data Format (.edf): Single file per recording.
# - BrainVision (.vhdr): Header file; requires .vmrk (markers) and .eeg (data) files.
# - EEGLAB (.set): MATLAB format; optional .fdt file contains float data.
# - Biosemi (.bdf): Single file per recording.
# - NWB (.nwb): Neurodata Without Borders format for standardized neurophysiology storage.
# Reference: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/intracranial-electroencephalography.html
IEEG_EXTENSIONS = {".edf", ".vhdr", ".set", ".bdf", ".nwb"}

# BIDS entity short names (BIDS v1.10.1):
# - subject: 'sub'
# - session: 'ses'
# - task: 'task'
# - acquisition: 'acq'
# - run: 'run'
# - description: 'desc'
# Reference: https://bids-specification.readthedocs.io/en/stable/glossary.html#entity
# Note: The short names are used to group recordings by entity.
BIDS_ENTITY_SHORT_NAMES = {
    "subject": "sub",
    "sub": "sub",
    "session": "ses",
    "ses": "ses",
    "task": "task",
    "acquisition": "acq",
    "acq": "acq",
    "run": "run",
    "description": "desc",
    "desc": "desc",
}


def _check_mne_bids_available(func_name: str) -> None:
    """Raise ImportError if mne-bids is not available."""
    if not MNE_BIDS_AVAILABLE:
        raise ImportError(
            f"{func_name} requires mne-bids, which is not installed. "
            "Install it with `pip install mne-bids`."
        )


def fetch_eeg_recordings(
    source: BIDSPath | Path | str | list[BIDSPath | Path | str],
) -> list[dict]:
    """Discover all EEG recordings in a dataset by parsing BIDS filenames.

    Args:
        source: BIDS root directory as a string, BIDSPath, or Path, or a list of those types.

    Returns:
        List of dicts with key/value pairs for BIDS entities:
            - recording_id: Full recording identifier (e.g., 'sub-01_ses-01_task-Sleep')
            - subject_id: Subject identifier (e.g., 'sub-01')
            - session_id: Session identifier or None (e.g., 'ses-01')
            - task_id: Task identifier (e.g., 'Sleep')
            - acquisition_id: Acquisition identifier or None (e.g., 'headband')
            - run_id: Run identifier or None (e.g., '01')
            - description_id: Description identifier or None (e.g., 'preproc')
            - fpath: Relative path to EEG file
    """
    _check_mne_bids_available("fetch_eeg_recordings")
    return _fetch_recordings(source, EEG_EXTENSIONS, "eeg")


def fetch_ieeg_recordings(
    source: BIDSPath | Path | str | list[BIDSPath | Path | str],
) -> list[dict]:
    """Discover all iEEG recordings in a dataset by parsing BIDS filenames.

    Args:
        source: BIDS root directory as a string, BIDSPath, or Path, or a list of those types.

    Returns:
        List of dicts with key/value pairs for BIDS entities:
            - recording_id: Full recording identifier (e.g., 'sub-01_ses-01_task-VisualNaming')
            - subject_id: Subject identifier (e.g., 'sub-01')
            - session_id: Session identifier or None (e.g., 'ses-01')
            - task_id: Task identifier (e.g., 'VisualNaming')
            - acquisition_id: Acquisition identifier or None (e.g., 'ecog')
            - run_id: Run identifier or None (e.g., '01')
            - description_id: Description identifier or None (e.g., 'preproc')
            - fpath: Relative path to iEEG file
    """
    _check_mne_bids_available("fetch_ieeg_recordings")
    return _fetch_recordings(source, IEEG_EXTENSIONS, "ieeg")


def group_recordings_by_entity(
    recordings: list[dict],
    fixed_entities: Optional[list[str]] = None,
) -> dict[str, list[dict]]:
    """Group BIDS-compliant recordings by specified fixed entities.

    Group keys are constructed using only the entities listed in
    `fixed_entities`; all other entities are implicitly allowed to vary within
    a group.

    By default (`fixed_entities=None`), groups are created by all entities except 'run'.

    Entities can be provided in long form (e.g., `subject`, `session`) or short
    BIDS form (e.g., `sub`, `ses`).

    Args:
        recordings: List of recording dictionaries that include a `recording_id`
            key.
        fixed_entities: Entities that must remain fixed within each group.
            If None, all entities except `run` are kept in the grouping key.

    Returns:
        Dictionary mapping a grouping key to the list of recordings in that group.

    Raises:
        ValueError: If an entity name is unsupported.
    """
    _check_mne_bids_available("group_recordings_by_entity")

    def _normalize_entity_list(entities: list[str], arg_name: str) -> list[str]:
        normalized = []
        for entity in entities:
            short_name = BIDS_ENTITY_SHORT_NAMES.get(entity.lower())
            if short_name is None:
                raise ValueError(
                    f"Unsupported BIDS entity '{entity}' in '{arg_name}'. "
                    f"Expected one of: {sorted(BIDS_ENTITY_SHORT_NAMES)}"
                )
            normalized.append(short_name)
        return normalized

    normalized_fixed = (
        set(_normalize_entity_list(fixed_entities, "fixed_entities"))
        if fixed_entities is not None
        else None
    )

    entity_groups: dict[str, list[dict]] = defaultdict(list)
    token_pattern = re.compile(r"^(?P<entity>[a-z]+)-[^_]+$")

    for recording in recordings:
        recording_id = recording["recording_id"]
        components = recording_id.split("_")
        key_components = []

        for component in components:
            match = token_pattern.match(component)
            if match is None:
                continue

            entity_short_name = match.group("entity")

            if normalized_fixed is None:
                if entity_short_name == "run":
                    continue
                key_components.append(component)
                continue

            if entity_short_name in normalized_fixed:
                key_components.append(component)

        entity_key = "_".join(key_components)
        entity_groups[entity_key].append(recording)

    return dict(entity_groups)


def check_eeg_recording_files_exist(
    bids_root: str | Path,
    recording_id: str,
) -> bool:
    """Check if EEG data files corresponding to a BIDS recording_id exist in the BIDS root directory.

    Args:
        bids_root: BIDS root directory (e.g., '/path/to/bids/root')
        recording_id: Recording identifier (e.g., 'sub-1_task-Sleep_acq-headband')

    Returns:
        True if at least one EEG data file is found, False otherwise.
    """
    _check_mne_bids_available("check_eeg_recording_files_exist")
    return _check_recording_files_exist(bids_root, recording_id, "eeg", EEG_EXTENSIONS)


def check_ieeg_recording_files_exist(
    bids_root: str | Path,
    recording_id: str,
) -> bool:
    """Check if iEEG data files corresponding to a BIDS recording_id exist in the BIDS root directory.

    Args:
        bids_root: BIDS root directory (e.g., '/path/to/bids/root')
        recording_id: Recording identifier (e.g., 'sub-1_task-Sleep_acq-headband')

    Returns:
        True if at least one iEEG data file is found, False otherwise.
    """
    _check_mne_bids_available("check_ieeg_recording_files_exist")
    return _check_recording_files_exist(
        bids_root, recording_id, "ieeg", IEEG_EXTENSIONS
    )


def build_bids_path(
    bids_root: str | Path, recording_id: str, modality: str
) -> BIDSPath:
    """Build a mne_bids.BIDSPath for a given recording_id, modality, and BIDS root directory.

    Args:
        bids_root: BIDS root directory (e.g., '/path/to/bids/root')
        recording_id: Recording identifier (e.g., 'sub-01_ses-01_task-Sleep')
        modality: Modality (e.g., 'eeg', 'ieeg', 'meg', 'anat', 'func', 'beh', 'dwi', 'fmap', 'pet')

    Returns:
        BIDSPath configured for reading via mne_bids.read_raw_bids.

    Raises:
        ValueError: If recording_id cannot be parsed.
    """
    _check_mne_bids_available("build_bids_path")
    entities = get_entities_from_fname(recording_id, on_error="raise")

    return BIDSPath(
        root=bids_root,
        subject=entities.get("subject"),
        session=entities.get("session"),
        task=entities.get("task"),
        acquisition=entities.get("acquisition"),
        run=entities.get("run"),
        description=entities.get("description"),
        datatype=modality,
        suffix=modality,
    )


def load_json_sidecar(bids_path: BIDSPath) -> dict:
    """Load the JSON sidecar file for a given BIDS file.

    Args:
        bids_path: BIDS path as a string, Path, or BIDSPath.

    Returns:
        Dictionary containing the JSON sidecar data.

    Raises:
        FileNotFoundError: If no JSON sidecar file is found for the BIDS path.
    """
    _check_mne_bids_available("load_json_sidecar")
    try:
        sidecar_path = bids_path.find_matching_sidecar(
            extension=".json", on_error="raise"
        )
        with open(sidecar_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except RuntimeError:
        raise FileNotFoundError(f"No JSON sidecar file found for {bids_path}.")


def load_participants_tsv(bids_root: Path | str) -> Optional[pd.DataFrame]:
    """Load participants.tsv data from a BIDS root directory.

    Args:
        bids_root: The path to the BIDS root directory.

    Returns:
        pd.DataFrame: Participant information indexed by participant_id,
            or None if participants.tsv is missing the 'participant_id' column.

    Raises:
        FileNotFoundError: If participants.tsv file is not found in the BIDS root directory.
    """
    if not (Path(bids_root) / "participants.tsv").exists():
        raise FileNotFoundError(f"participants.tsv file not found in {bids_root}.")

    df = pd.read_csv(
        Path(bids_root) / "participants.tsv",
        sep="\t",
        na_values=["n/a", "N/A"],
        keep_default_na=True,
    )

    if "participant_id" not in df.columns:
        warnings.warn(
            f"No participant_id column found in participants.tsv file in BIDS root directory {bids_root}. "
            "Returning None."
        )
        return None

    df = df.set_index("participant_id")
    return df


def get_subject_info(
    subject_id: str,
    participants_data: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Retrieve demographic information (age, sex) for a given subject from a participants DataFrame.

    Looks up the subject's 'age' and 'sex' fields in the provided participants DataFrame.
    Returns a dictionary with those keys. If the data is missing or not found, values will be None.
    If no DataFrame is provided, returns None for both age and sex.

    Args:
        subject_id: BIDS subject identifier (e.g., 'sub-01').
        participants_data: Optional DataFrame of participants.tsv data.

    Returns:
        dict: {'age': value or None, 'sex': value or None}
    """
    if participants_data is None:
        warnings.warn(
            "The participants.tsv file was not provided. No subject information can be retrieved. "
            "Returning None for age and sex. Please provide a valid participants.tsv file."
        )
        return {"age": None, "sex": None}

    if subject_id not in participants_data.index:
        warnings.warn(
            f"Subject {subject_id} not found in participants.tsv file. "
            "Setting age and sex to None."
        )
        return {"age": None, "sex": None}

    row = participants_data.loc[subject_id]

    age = row.get("age", None)
    if pd.isna(age):
        warnings.warn(
            f"Age for subject {subject_id} is NaN in participants.tsv file. "
            "Setting age to None."
        )
        age = None

    sex = row.get("sex", None)
    if pd.isna(sex):
        warnings.warn(
            f"Sex for subject {subject_id} is NaN in participants.tsv file. "
            "Setting sex to None."
        )
        sex = None

    return {"age": age, "sex": sex}


def _fetch_recordings(
    source: BIDSPath | Path | str | list[BIDSPath | Path | str],
    extensions: set[str],
    modality: str,
) -> list[dict]:
    """
    Internal helper for discovering BIDS recordings that match provided file extensions and modality.

    Args:
        source: BIDS root directory as a string, BIDSPath, or Path, or a list of those types.
        extensions: Set of allowed file extensions (e.g., EEG_EXTENSIONS).
        modality: Modality to filter by (e.g., 'eeg', 'ieeg').

    Returns:
        List of dicts with key/value pairs for BIDS entities:
            - recording_id: Full recording identifier (e.g., 'sub-01_ses-01_task-Sleep')
            - subject_id: Subject identifier (e.g., 'sub-01')
            - session_id: Session identifier or None (e.g., 'ses-01')
            - task_id: Task identifier (e.g., 'Sleep')
            - acquisition_id: Acquisition identifier or None (e.g., 'headband')
            - run_id: Run identifier or None (e.g., '01')
            - description_id: Description identifier or None (e.g., 'preproc')
            - fpath: Relative path to the recording file
    """
    # Determine the files to analyze
    if source is None:
        raise TypeError(
            "source must be a BIDSPath, Path, or string, or a list of those types. None was provided."
        )

    if isinstance(source, BIDSPath):
        source = source.root

    if not isinstance(source, list):
        source = BIDSPath(root=source, datatype=modality).match()

    if len(source) == 0:
        return []

    recordings = []
    seen_recording_ids = set()

    for filepath in source:
        ext = Path(filepath).suffix.lower()
        if ext not in extensions:
            continue

        if not isinstance(filepath, BIDSPath):
            filepath = get_bids_path_from_fname(filepath, check=False)

        if filepath.datatype != modality:
            continue

        components = []
        entities = filepath.entities
        if entities["subject"]:
            components.append(f"sub-{entities['subject']}")
        if entities["session"]:
            components.append(f"ses-{entities['session']}")
        if entities["task"]:
            components.append(f"task-{entities['task']}")
        if entities["acquisition"]:
            components.append(f"acq-{entities['acquisition']}")
        if entities["run"]:
            components.append(f"run-{entities['run']}")
        if entities["description"]:
            components.append(f"desc-{entities['description']}")
        recording_id = "_".join(components)

        if recording_id in seen_recording_ids:
            continue
        seen_recording_ids.add(recording_id)

        recordings.append(
            {
                "recording_id": recording_id,
                "subject_id": (
                    f"sub-{entities['subject']}" if entities["subject"] else None
                ),
                "session_id": (
                    f"ses-{entities['session']}" if entities["session"] else None
                ),
                "task_id": entities["task"] if entities["task"] else None,
                "acquisition_id": (
                    entities["acquisition"] if entities["acquisition"] else None
                ),
                "run_id": entities["run"] if entities["run"] else None,
                "description_id": (
                    entities["description"] if entities["description"] else None
                ),
                "fpath": filepath,
            }
        )

    return recordings


def _check_recording_files_exist(
    bids_root: str | Path,
    recording_id: str,
    modality: str,
    extensions: set[str],
) -> bool:
    """Check if any data file for a BIDS recording exists in the BIDS root directory.

    This searches for any file belonging to the recording in the proper BIDS directory structure,
    matching any of the supported file extensions. It supports all BIDS-compliant formats (e.g., .edf, .vhdr, .set, .bdf, .eeg, .nwb)
    plus .fif for MNE-processed files.

    Args:
        bids_root: BIDS root directory (e.g., '/path/to/bids/root')
        recording_id: Recording identifier (e.g., 'sub-1_task-Sleep_acq-headband')
        modality: Modality (e.g., 'eeg', 'ieeg')
        extensions: Set of allowed file extensions (e.g., EEG_EXTENSIONS or IEEG_EXTENSIONS)

    Returns:
        True if at least one data file is found, False otherwise.
    """
    bids_path = build_bids_path(bids_root, recording_id, modality)
    subject_id = f"sub-{bids_path.entities['subject']}"
    subject_dir = bids_path.root / subject_id

    for file in subject_dir.rglob(f"{recording_id}_*"):
        if file.suffix.lower() in extensions:
            return True
    return False
