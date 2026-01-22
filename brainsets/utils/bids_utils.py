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
