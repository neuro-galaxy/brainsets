"""BIDS filename parsing utilities.

This module provides functions to parse BIDS-compliant filenames,
reusable by any BIDS-compliant pipeline.
"""

from typing import Optional

try:
    from mne_bids import get_bids_path_from_fname, get_entities_from_fname

    MNE_BIDS_AVAILABLE = True
except ImportError:
    get_bids_path_from_fname = None
    get_entities_from_fname = None
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


def _require_mne_bids(func_name: str) -> None:
    """Raise ImportError if mne-bids is not available."""
    if not MNE_BIDS_AVAILABLE:
        raise ImportError(
            f"{func_name} requires mne-bids, which is not installed. "
            "Install it with `pip install mne-bids`."
        )


def parse_bids_filename(filename: str, modality: str) -> Optional[dict]:
    """Parse a BIDS filename to extract entities if suffix matches modality.

    Uses mne_bids to parse BIDS-compliant filenames without regex patterns.

    Args:
        filename: The filename to parse (e.g., 'sub-01_task-Sleep_eeg.edf')
        modality: The expected modality suffix (e.g., 'eeg', 'ieeg')

    Returns:
        Dictionary with keys: subject, session, task, acquisition, run
        All values are prefix-free (e.g., subject='01', not 'sub-01')
        Returns None if the filename doesn't match BIDS pattern or modality
    """
    _require_mne_bids("parse_bids_filename")
    try:
        bids_path = get_bids_path_from_fname(filename, check=False)
    except (ValueError, IndexError, RuntimeError):
        return None

    if bids_path.suffix != modality:
        return None

    return {
        "subject": bids_path.subject,
        "session": bids_path.session,
        "task": bids_path.task,
        "acquisition": bids_path.acquisition,
        "run": bids_path.run,
    }


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
        ValueError: If the recording_id format is invalid or missing required entities
    """
    _require_mne_bids("parse_recording_id")
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
    }
