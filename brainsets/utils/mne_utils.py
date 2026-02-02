"""Data extraction utilities.

This module provides functions to extract metadata and signal data from
MNE Raw objects and convert them to brainsets data structures.
"""

import datetime
from typing import Union

import mne
import numpy as np
from temporaldata import ArrayDict, Interval, RegularTimeSeries

from brainsets.descriptions import (
    SubjectDescription,
)
from brainsets.taxonomy import Sex, Species


def extract_subject_description(
    subject_id: str,
    age: Union[float, int, str, None] = None,
    sex: Union[str, int, Sex, None] = None,
) -> SubjectDescription:
    """Create a SubjectDescription object for a human subject.

    Args:
        subject_id: Unique identifier for the subject
        age: Age of the subject
        sex: Sex of the subject (0=U=UNKNOWN, 1=M=MALE, 2=F=FEMALE, 3=O=OTHER)

    Returns:
        SubjectDescription object with species set to Homo sapiens
    """
    if age is None:
        age_normalized = 0.0
    elif isinstance(age, (int, float)):
        age_normalized = float(age)
    elif isinstance(age, str):
        try:
            age_normalized = float(age)
        except (ValueError, TypeError):
            age_normalized = 0.0
    else:
        age_normalized = 0.0

    if sex is None:
        sex_normalized = Sex.UNKNOWN
    elif isinstance(sex, Sex):
        sex_normalized = sex
    elif isinstance(sex, str):
        try:
            sex_normalized = Sex.from_string(sex)
        except ValueError:
            sex_normalized = Sex.UNKNOWN
    elif isinstance(sex, int):
        try:
            sex_normalized = Sex(sex)
        except ValueError:
            sex_normalized = Sex.UNKNOWN
    else:
        sex_normalized = Sex.UNKNOWN

    return SubjectDescription(
        id=subject_id,
        species=Species.HOMO_SAPIENS,
        age=age_normalized,
        sex=sex_normalized,
    )


def extract_meas_date(
    recording_data: mne.io.Raw,
) -> datetime.datetime:
    """Extract the measurement date from MNE Raw recording data.

    Args:
        recording_data: The MNE Raw object containing EEG data and metadata

    Returns:
        The measurement date if present, otherwise Unix epoch (1970-01-01 UTC)
    """
    if recording_data.info["meas_date"] is not None:
        return recording_data.info["meas_date"]
    return datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)


def extract_signal(
    recording_data: mne.io.Raw,
) -> RegularTimeSeries:
    """Extract the EEG signal as a RegularTimeSeries from MNE Raw data.

    Args:
        recording_data: The MNE Raw object containing EEG data

    Returns:
        RegularTimeSeries object with the EEG signal
    """
    sfreq = recording_data.info["sfreq"]
    eeg_signal = recording_data.get_data().T
    if len(eeg_signal) == 0:
        raise ValueError("Recording contains no samples")

    return RegularTimeSeries(
        signal=eeg_signal,
        sampling_rate=sfreq,
        domain=Interval(
            start=np.array([0.0]),
            end=np.array([(len(eeg_signal) - 1) / sfreq]),
        ),
    )


def extract_channels(
    recording_data: mne.io.Raw,
) -> ArrayDict:
    """Extract channel names and types from MNE Raw data.

    Args:
        recording_data: The MNE Raw object containing EEG data

    Returns:
        ArrayDict with fields 'id' (channel names) and 'types' (channel types)
    """
    return ArrayDict(
        id=np.array(recording_data.ch_names, dtype="U"),
        types=np.array(recording_data.get_channel_types(), dtype="U"),
    )
