"""BIDS filename parsing utilities.

This module provides functions to parse BIDS-compliant filenames,
reusable by any BIDS-compliant pipeline.
"""

import re
from pathlib import Path
from typing import Optional

# BIDS EEG filename pattern (BIDS v1.10.1):
# Required entities: sub-<label>, task-<label>
# Optional entities: ses-<label>, acq-<label>, run-<index>
# Modality suffix: _eeg
# Extension: format-specific (.edf, .vhdr, .set, .bdf)
# Reference: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html
BIDS_EEG_PATTERN = re.compile(
    r"^(?P<subject>sub-[^_]+)"
    r"(?:_(?P<session>ses-[^_]+))?"
    r"_task-(?P<task>[^_]+)"
    r"(?:_acq-(?P<acq>[^_]+))?"
    r"(?:_run-(?P<run>[^_]+))?"
    r"_eeg"
    r"\.(?P<ext>\w+)$"
)

# BIDS EEG supported formats (BIDS v1.10.1):
# - European Data Format (.edf): Single file per recording. edf+ files permitted.
# - BrainVision (.vhdr): Header file; requires .vmrk (markers) and .eeg (data) files.
# - EEGLAB (.set): MATLAB format; optional .fdt file contains float data.
# - Biosemi (.bdf): Single file per recording. bdf+ files permitted.
# Reference: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html
EEG_EXTENSIONS = {".edf", ".vhdr", ".set", ".bdf"}

# BIDS iEEG filename pattern (BIDS v1.10.1):
# Required entities: sub-<label>, task-<label>
# Optional entities: ses-<label>, acq-<label>, run-<index>
# Modality suffix: _ieeg
# Extension: format-specific (.edf, .vhdr, .set, .bdf, .nwb)
# Reference: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/intracranial-electroencephalography.html
BIDS_IEEG_PATTERN = re.compile(
    r"^(?P<subject>sub-[^_]+)"
    r"(?:_(?P<session>ses-[^_]+))?"
    r"_task-(?P<task>[^_]+)"
    r"(?:_acq-(?P<acq>[^_]+))?"
    r"(?:_run-(?P<run>[^_]+))?"
    r"_ieeg"
    r"\.(?P<ext>\w+)$"
)

# BIDS iEEG supported formats (BIDS v1.10.1):
# - European Data Format (.edf): Single file per recording.
# - BrainVision (.vhdr): Header file; requires .vmrk (markers) and .eeg (data) files.
# - EEGLAB (.set): MATLAB format; optional .fdt file contains float data.
# - Biosemi (.bdf): Single file per recording.
# - NWB (.nwb): Neurodata Without Borders format for standardized neurophysiology storage.
# Reference: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/intracranial-electroencephalography.html
IEEG_EXTENSIONS = {".edf", ".vhdr", ".set", ".bdf", ".nwb"}


def parse_bids_eeg_filename(filename: str) -> Optional[dict]:
    """Parse a BIDS-compliant EEG filename to extract components.

    Args:
        filename: The filename to parse (e.g., 'sub-01_task-Sleep_eeg.edf')

    Returns:
        Dictionary with keys: subject_id, session_id, task_id, acq_id, run_id
        Returns None if the filename doesn't match BIDS EEG pattern
    """
    basename = Path(filename).name
    match = BIDS_EEG_PATTERN.match(basename)
    if not match:
        return None

    return {
        "subject_id": match.group("subject"),
        "session_id": match.group("session"),
        "task_id": match.group("task"),
        "acq_id": match.group("acq"),
        "run_id": match.group("run"),
    }


def parse_bids_ieeg_filename(filename: str) -> Optional[dict]:
    """Parse a BIDS-compliant iEEG filename to extract components.

    Args:
        filename: The filename to parse (e.g., 'sub-01_task-VisualNaming_ieeg.edf')

    Returns:
        Dictionary with keys: subject_id, session_id, task_id, acq_id, run_id
        Returns None if the filename doesn't match BIDS iEEG pattern
    """
    basename = Path(filename).name
    match = BIDS_IEEG_PATTERN.match(basename)
    if not match:
        return None

    return {
        "subject_id": match.group("subject"),
        "session_id": match.group("session"),
        "task_id": match.group("task"),
        "acq_id": match.group("acq"),
        "run_id": match.group("run"),
    }


def parse_recording_id(recording_id: str) -> dict:
    """Parse a recording_id string to extract BIDS entities.

    Recording IDs are built from BIDS entities in the format:
    sub-XX[_ses-YY]_task-ZZ[_acq-AA][_run-RR]

    Args:
        recording_id: The recording identifier to parse

    Returns:
        Dictionary with keys: subject_id, session_id, task_id, acq_id, run_id
        (session_id, acq_id, run_id may be None if not present)

    Raises:
        ValueError: If the recording_id format is invalid
    """
    parts = recording_id.split("_")
    if not parts or not parts[0].startswith("sub-"):
        raise ValueError(f"Invalid recording_id format: {recording_id}")

    subject_id = parts[0]
    session_id = None
    task_id = None
    acq_id = None
    run_id = None

    idx = 1
    if idx < len(parts) and parts[idx].startswith("ses-"):
        session_id = parts[idx]
        idx += 1

    if idx < len(parts) and parts[idx].startswith("task-"):
        task_id = parts[idx]
        idx += 1
    else:
        raise ValueError(f"Recording ID missing task entity: {recording_id}")

    while idx < len(parts):
        if parts[idx].startswith("acq-"):
            acq_id = parts[idx]
        elif parts[idx].startswith("run-"):
            run_id = parts[idx]
        idx += 1

    return {
        "subject_id": subject_id,
        "session_id": session_id,
        "task_id": task_id,
        "acq_id": acq_id,
        "run_id": run_id,
    }
