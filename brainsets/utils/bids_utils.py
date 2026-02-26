"""Brain Imaging Data Structure (BIDS) utilities.

This module provides functions to parse BIDS-compliant filenames, discover BIDS recordings, and check if data files exist.

To know more about BIDS, please refer to the BIDS specification: https://bids-specification.readthedocs.io/en/stable/
"""

from typing import Optional
from pathlib import Path

from mne_bids import get_bids_path_from_fname, get_entities_from_fname, BIDSPath

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


def parse_recording_id(recording_id: str) -> dict:
    """Parse a recording_id string to extract BIDS entities.

    Recording IDs are built from BIDS entities in the format:
    sub-XX[_ses-YY]_task-ZZ[_acq-AA][_run-RR]

    Args:
        recording_id: The recording identifier to parse

    Returns:
        Dictionary with keys: subject, session, task, acquisition, run
        All values are prefix-free (e.g., subject='01', not 'sub-01')
        session, acquisition, and run may be None if not present

    Raises:
        ValueError: If the recording_id format is invalid or missing 
        required subject or task entities
    """
    entities = get_entities_from_fname(recording_id, on_error="ignore")

    if not entities.get("subject"):
        raise ValueError(f"Invalid recording_id format: {recording_id}")
    if not entities.get("task"):
        raise ValueError(f"Recording ID missing task entity: {recording_id}")

    return {
        "subject": entities["subject"],
        "session": entities.get("session"),
        "task": entities["task"],
        "acquisition": entities.get("acquisition"),
        "run": entities.get("run"),
        "description": entities.get("description"),
    }


def build_bids_path(bids_root: str | Path, recording_id: str, modality: str) -> BIDSPath:
    """Build a mne_bids.BIDSPath from recording_id and data directory.

    Args:
        bids_root: BIDS root directory (e.g., '/path/to/bids/root')
        recording_id: Recording identifier (e.g., 'sub-01_ses-01_task-Sleep')
        modality: Modality (e.g., 'eeg', 'ieeg', 'meg', 'anat', 'func', 'beh', 'dwi', 'fmap', 'pet')

    Returns:
        BIDSPath configured for reading via mne_bids.read_raw_bids

    Raises:
        ValueError: If recording_id cannot be parsed
    """
    entities = parse_recording_id(recording_id)

    bids_path = BIDSPath(
        root=bids_root,
        subject=entities["subject"],
        session=entities.get("session"),
        task=entities["task"],
        acquisition=entities.get("acquisition"),
        run=entities.get("run"),
        description=entities.get("description"),
        datatype=modality,
        suffix=modality,
    )

    return bids_path


def parse_bids_fname(
    fname: str | Path,
    modality: Optional[str] = None,
) -> Optional[dict]:
    """Parse a BIDS filename to extract entities if suffix matches modality.

    Uses mne_bids to parse BIDS-compliant filenames without regex patterns.

    Args:
        fname: The BIDS filename to parse (e.g., 'sub-01_task-Sleep_eeg.edf') or a Path object
        modality: The anticipated modality suffix (e.g., 'eeg', 'ieeg'). If set to None, filenames will be parsed regardless of their suffix.

    Returns:
        Dictionary with keys: subject, session, task, acquisition, run, description
        All values are prefix-free (e.g., subject='01', not 'sub-01')
        Returns None if the filename doesn't match BIDS pattern or modality
    """
    try:
        bids_path = get_bids_path_from_fname(fname, check=False)
    except (ValueError, IndexError, RuntimeError, KeyError):
        return None
    
    # If modality is provided, check if the suffix matches
    if modality is not None and bids_path.suffix != modality:
        return None

    return {
        "subject": bids_path.subject,
        "session": bids_path.session,
        "task": bids_path.task,
        "acquisition": bids_path.acquisition,
        "run": bids_path.run,
        "description": bids_path.description,
    }


def _fetch_recordings(
    extensions: set[str],
    modality: str,
    bids_root: Optional[Path | str] = None,
    candidate_files: Optional[list[Path | str]] = None,
) -> list[dict]:
    """
    Internal helper to discover BIDS recordings matching file extensions and modality.

    Args:
        extensions: Set of allowed file extensions (e.g., EEG_EXTENSIONS).
        modality: Modality to filter by ('eeg', 'ieeg', etc).
        bids_root: BIDS root directory is required if candidate_files is not provided.
        candidate_files: Optional list of Path objects explicitly specifying files to analyze.

    Returns:
        List of dicts where each dict contains:
            - recording_id: Identifier constructed from BIDS entities
            - subject_id: BIDS subject string (e.g., 'sub-XX')
            - session_id: BIDS session string or None (e.g., 'ses-YY')
            - task_id: Task identifier
            - acq_id: Acquisition identifier or None
            - run_id: Run identifier or None
            - desc_id: Description identifier or None
            - <modality>_file: Path to the data file

    Raises:
        ValueError: If both bids_root and candidate_files are provided or neither is provided.
        ValueError: If candidate_files contains non-Path objects.
    """
    # Determine the files to analyze
    if bids_root is not None and candidate_files is not None:
        raise ValueError(
            "Both 'bids_root' and 'candidate_files' were provided, but these arguments are mutually exclusive. "
            "Please specify only one: set either 'bids_root' (for searching from a root BIDS directory) or 'candidate_files' "
            "(for supplying an explicit list of files), not both."
        )
    if candidate_files is None:
        if bids_root is None:
            raise ValueError("'bids_root' is required if 'candidate_files' is not provided")  
        # Construct a BIDSPath for the desired modality
        bids_path = BIDSPath(root=bids_root).update(datatype=modality)
        candidate_files = bids_path.match()
    
    if len(candidate_files) == 0:
        return []
            
    recordings = []
    seen_recording_ids = set()

    for filepath in candidate_files:
        ext = Path(filepath).suffix.lower()
        if ext not in extensions:
            continue
        
        parsed = parse_bids_fname(filepath, modality)
        if not parsed:
            continue

        components = []
        if parsed["subject"]:
            components.append(f"sub-{parsed['subject']}")
        if parsed["session"]:
            components.append(f"ses-{parsed['session']}")
        if parsed["task"]:
            components.append(f"task-{parsed['task']}")
        if parsed["acquisition"]:
            components.append(f"acq-{parsed['acquisition']}")
        if parsed["run"]:
            components.append(f"run-{parsed['run']}")
        if parsed["description"]:
            components.append(f"desc-{parsed['description']}")
        recording_id = "_".join(components)
        if recording_id in seen_recording_ids:
            continue
        seen_recording_ids.add(recording_id)

        recordings.append(
            {
                "recording_id": recording_id,
                "subject_id": f"sub-{parsed['subject']}",
                "session_id": f"ses-{parsed['session']}" if parsed["session"] else None,
                "task_id": parsed["task"],
                "acq_id": parsed["acquisition"],
                "run_id": parsed["run"],
                "desc_id": parsed["description"],
                f"{modality}_file": filepath,
            }
        )

    return recordings


def fetch_eeg_recordings(bids_root: Optional[Path] = None, candidate_files: Optional[list[Path]] = None) -> list[dict]:
    """Discover all EEG recordings in a dataset by parsing BIDS filenames.

    Args:
        bids_root: BIDS root directory is required if candidate_files is not provided.
        candidate_files: Optional list of Path objects explicitly specifying files to analyze.
    Returns:
        List of dicts with keys:
            - recording_id: Full recording identifier (e.g., 'sub-01_ses-01_task-Sleep')
            - subject_id: Subject identifier (e.g., 'sub-01')
            - session_id: Session identifier or None (e.g., 'ses-01')
            - task_id: Task identifier (e.g., 'Sleep')
            - acq_id: Acquisition identifier or None (e.g., 'headband')
            - run_id: Run identifier or None (e.g., '01')
            - eeg_file: Relative path to EEG file
    """
    return _fetch_recordings(EEG_EXTENSIONS, "eeg", bids_root, candidate_files)


def fetch_ieeg_recordings(bids_root: Optional[Path] = None, candidate_files: Optional[list[Path]] = None) -> list[dict]:
    """Discover all iEEG recordings in a dataset by parsing BIDS filenames.

    Args:
        bids_root: BIDS root directory is required if candidate_files is not provided.
        candidate_files: Optional list of Path objects explicitly specifying files to analyze.
    Returns:
        List of dicts with keys:
            - recording_id: Full recording identifier (e.g., 'sub-01_ses-01_task-VisualNaming')
            - subject_id: Subject identifier (e.g., 'sub-01')
            - session_id: Session identifier or None (e.g., 'ses-01')
            - task_id: Task identifier (e.g., 'VisualNaming')
            - acq_id: Acquisition identifier or None (e.g., 'ecog')
            - run_id: Run identifier or None (e.g., '01')
            - ieeg_file: Relative path to iEEG file
    """
    return _fetch_recordings(IEEG_EXTENSIONS, "ieeg", bids_root, candidate_files)


def check_recording_files_exist(recording_id: str, subject_dir: str | Path) -> bool:
    """Check if data files matching the recording_id pattern exist locally.

    Searches for any data file (EEG or iEEG) in the BIDS-structured directory.
    Supports all BIDS-compliant formats (.edf, .vhdr, .set, .bdf, .eeg, .nwb)
    plus .fif for MNE-processed files.

    Args:
        recording_id: Recording identifier (e.g., 'sub-1_task-Sleep_acq-headband')
        subject_dir: Subject directory to search in, or a BIDSPath object

    Returns:
        True if at least one data file is found, False otherwise
    """
    if isinstance(subject_dir, str):
        subject_dir = Path(subject_dir)
    if not subject_dir.exists():
        return False

    supported_extensions = {".edf", ".set", ".bdf", ".vhdr", ".eeg", ".nwb", ".fif"}

    for file in subject_dir.rglob(f"{recording_id}_*"):
        if file.suffix.lower() in supported_extensions:
            return True

    return False


