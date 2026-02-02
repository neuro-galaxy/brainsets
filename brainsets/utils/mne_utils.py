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


def generate_train_valid_splits_one_epoch(
    epoch: Interval, split_ratios: list[float] = None
) -> tuple[Interval, Interval]:
    """Split a single time interval into training and validation intervals.

    Args:
        epoch: The full time interval to split (must contain a single interval)
        split_ratios: List of two ratios [train_ratio, valid_ratio] that sum to 1.0.
            Defaults to [0.9, 0.1].

    Returns:
        Tuple of (train_intervals, valid_intervals)

    Raises:
        ValueError: If the epoch does not contain a single interval or
            if split_ratios do not sum to 1
    """
    if split_ratios is None:
        split_ratios = [0.9, 0.1]

    if len(epoch) != 1:
        raise ValueError("Epoch must contain a single interval")

    if not np.isclose(sum(split_ratios), 1.0):
        raise ValueError("Split ratios must sum to 1")

    epoch_start = epoch.start[0]
    epoch_end = epoch.end[0]

    train_split_time = epoch_start + split_ratios[0] * (epoch_end - epoch_start)
    val_split_time = train_split_time + split_ratios[1] * (epoch_end - epoch_start)

    train_intervals = Interval(start=epoch_start, end=train_split_time)
    valid_intervals = Interval(start=train_intervals.end[0], end=val_split_time)

    return train_intervals, valid_intervals
