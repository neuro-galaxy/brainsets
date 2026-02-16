"""Data extraction utilities for OpenNeuro EEG pipelines.

This module provides functions to extract metadata and signal data from
MNE Raw objects and convert them to brainsets data structures.
"""

import datetime
from pathlib import Path
from typing import Union

import mne
import numpy as np
import pandas as pd
from temporaldata import ArrayDict, Interval, RegularTimeSeries

from brainsets.descriptions import (
    BrainsetDescription,
    DeviceDescription,
    SessionDescription,
    SubjectDescription,
)
from brainsets.taxonomy import Sex, Species


def extract_brainset_description(
    dataset_id: str,
    origin_version: str,
    derived_version: str,
    source: str,
    description: str,
) -> BrainsetDescription:
    """Create a BrainsetDescription object from dataset metadata.

    Args:
        dataset_id: Unique identifier for the dataset
        origin_version: Version of the original dataset
        derived_version: Version of the derived dataset
        source: Source or origin of the dataset
        description: Textual description of the dataset

    Returns:
        BrainsetDescription object
    """
    return BrainsetDescription(
        id=dataset_id,
        origin_version=origin_version,
        derived_version=derived_version,
        source=source,
        description=description,
    )


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


def extract_session_description(
    session_id: str,
    recording_date: datetime.datetime,
) -> SessionDescription:
    """Create a SessionDescription object from session metadata.

    Args:
        session_id: Unique identifier for the session
        recording_date: Date and time of the recording

    Returns:
        SessionDescription object
    """
    return SessionDescription(
        id=session_id,
        recording_date=recording_date,
    )


def extract_device_description(
    device_id: str,
) -> DeviceDescription:
    """Create a DeviceDescription object from device metadata.

    Args:
        device_id: Unique identifier for the device

    Returns:
        DeviceDescription object
    """
    return DeviceDescription(
        id=device_id,
    )


def extract_meas_date(
    recording_data: mne.io.Raw,
) -> Union[datetime.datetime, None]:
    """Extract the measurement date from MNE Raw recording data.

    Args:
        recording_data: The MNE Raw object containing EEG data and metadata

    Returns:
        The measurement date if present, otherwise None
    """
    if recording_data.info["meas_date"] is not None:
        return recording_data.info["meas_date"]
    return None


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


def read_bids_channels_tsv(tsv_path: Path) -> pd.DataFrame:
    """Read BIDS _channels.tsv file.

    Parses the BIDS channels metadata file containing channel information
    including name, type, sampling rate, and status (good/bad).

    Args:
        tsv_path: Path to the _channels.tsv file

    Returns:
        DataFrame with channel information, indexed by channel name.
        Columns typically include: type, sampling_rate, status, low_cutoff, etc.

    Raises:
        FileNotFoundError: If the TSV file doesn't exist
        ValueError: If the file cannot be parsed
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"Channels TSV file not found: {tsv_path}")

    try:
        df = pd.read_csv(tsv_path, sep="\t", na_values=["n/a", "N/A"])
        return df
    except Exception as e:
        raise ValueError(f"Failed to parse channels TSV file {tsv_path}: {e}") from e


def read_bids_electrodes_tsv(tsv_path: Path) -> pd.DataFrame:
    """Read BIDS _electrodes.tsv file.

    Parses the BIDS electrodes metadata file containing electrode coordinates
    and properties (name, x, y, z coordinates, size, etc.).

    Args:
        tsv_path: Path to the _electrodes.tsv file

    Returns:
        DataFrame with electrode information, indexed by electrode name.
        Columns typically include: x, y, z (coordinates), size, and other properties.

    Raises:
        FileNotFoundError: If the TSV file doesn't exist
        ValueError: If the file cannot be parsed
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"Electrodes TSV file not found: {tsv_path}")

    try:
        df = pd.read_csv(tsv_path, sep="\t", na_values=["n/a", "N/A"])
        return df
    except Exception as e:
        raise ValueError(f"Failed to parse electrodes TSV file {tsv_path}: {e}") from e


def read_bids_coordsystem_json(json_path: Path) -> dict:
    """Read BIDS _coordsystem.json file.

    Parses the BIDS coordinate system metadata file containing information
    about the coordinate system used for electrode positions.

    Args:
        json_path: Path to the _coordsystem.json file

    Returns:
        Dictionary with coordinate system metadata.

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        ValueError: If the file cannot be parsed
    """
    import json

    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Coordinate system JSON file not found: {json_path}")

    try:
        with open(json_path) as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(
            f"Failed to parse coordinate system JSON file {json_path}: {e}"
        ) from e
