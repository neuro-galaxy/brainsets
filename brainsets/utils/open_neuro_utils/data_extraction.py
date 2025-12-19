import mne
import numpy as np
import datetime
from typing import Union

from temporaldata import (
    RegularTimeSeries,
    ArrayDict,
    Interval,
)

from brainsets.taxonomy import Species, Sex
from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    SubjectDescription,
    DeviceDescription,
)


def extract_brainset_description(
    dataset_id: str,
    origin_version: str,
    derived_version: str,
    source: str,
    description: str,
) -> BrainsetDescription:
    """
    Create a BrainsetDescription object from dataset metadata.

    Args:
        dataset_id : str
            Unique identifier for the dataset.
        origin_version : str
            Version of the original dataset.
        derived_version : str
            Version of the derived dataset.
        source : str
            Source or origin of the dataset.
        description : str
            Textual description of the dataset.

    Returns:
        BrainsetDescription
            An object containing the provided dataset metadata.
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
    age: float,
    sex: Union[str, int, Sex],
) -> SubjectDescription:
    """
    Create a SubjectDescription object for a human subject.

    Args:
        subject_id : str
            Unique identifier for the subject.
        age : float
            Age of the subject in days.
        sex : str, int, or Sex
            Sex of the subject. Can be a string (e.g., "M", "Male", "F", "Female"),
            an integer (0=UNKNOWN, 1=MALE, 2=FEMALE, 3=OTHER), or a Sex enum.
    Returns:
        SubjectDescription
            An object describing the subject, with species set to Homo sapiens.
    """
    if isinstance(sex, str):
        sex = Sex.from_string(sex)
    elif isinstance(sex, int):
        sex = Sex(sex)

    return SubjectDescription(
        id=subject_id,
        species=Species.HOMO_SAPIENS,
        age=age,
        sex=sex,
    )


def extract_session_description(
    session_id: str,
    recording_date: datetime.datetime,
) -> SessionDescription:
    """
    Create a SessionDescription object from session metadata.

    Args:
        session_id : str
            Unique identifier for the session.
        recording_date : datetime.datetime
            Date and time of the recording.

    Returns:
        SessionDescription
            An object describing the session, with the recording date formatted as YYYYMMDD.
    """
    return SessionDescription(
        id=session_id,
        recording_date=recording_date,
    )


def extract_device_description(
    device_id: str,
) -> DeviceDescription:
    """
    Create a DeviceDescription object from device metadata.

    Args:
        device_id : str
            Unique identifier for the device.

    Returns:
        DeviceDescription
            An object describing the device.
    """
    return DeviceDescription(
        id=device_id,
        # recording_tech=RecordingTech.EEG, # TODO: add EEG to the taxonomy
    )


def extract_meas_date(
    recording_data: mne.io.Raw,
) -> datetime.datetime:
    """
    Extract the measurement date from MNE Raw recording data.

    Args:
        recording_data : mne.io.Raw
            The MNE Raw object containing EEG data and metadata.

    Returns:
        datetime.datetime or None
            The measurement date if present, otherwise None.
    """
    if recording_data.info["meas_date"] is not None:
        return recording_data.info["meas_date"]

    return None


def extract_signal(
    recording_data: mne.io.Raw,
) -> RegularTimeSeries:
    """
    Extract the EEG signal as a RegularTimeSeries from MNE Raw data.

    Args:
        recording_data : mne.io.Raw
            The MNE Raw object containing EEG data.

    Returns:
        RegularTimeSeries
            The EEG signal as a RegularTimeSeries object, with time domain and sampling rate.
    """
    sfreq = recording_data.info["sfreq"]
    eeg_signal = recording_data.get_data().T

    eeg = RegularTimeSeries(
        signal=eeg_signal,
        sampling_rate=sfreq,
        domain=Interval(
            start=np.array([0.0]),
            end=np.array([(len(eeg_signal) - 1) / sfreq]),
        ),
    )

    return eeg


def extract_channels(
    recording_data: mne.io.Raw,
) -> ArrayDict:
    """
    Extract channel names and types from MNE Raw data.

    Args:
        recording_data : mne.io.Raw
            The MNE Raw object containing EEG data.

    Returns:
        ArrayDict
            An ArrayDict with fields:
                - id: array of channel names (dtype "U")
                - types: array of channel types (dtype "U")
    """
    channels = ArrayDict(
        id=np.array(recording_data.ch_names, dtype="U"),
        types=np.array(recording_data.get_channel_types(), dtype="U"),
    )

    return channels


def generate_train_valid_splits_one_epoch(
    epoch: Interval, split_ratios: list[float] = [0.9, 0.1]
) -> tuple[Interval, Interval]:
    """
    Split a single time interval into training and validation intervals.

    Args:
        epoch : Interval
            The full time interval to split (must contain a single interval).
        split_ratios : list of float, optional
            List of two ratios [train_ratio, valid_ratio] that sum to 1.0.
            Defaults to [0.9, 0.1].

    Returns:
        tuple of Interval
            (train_intervals, valid_intervals)
            - train_intervals: Interval for training data.
            - valid_intervals: Interval for validation data.

    Raises:
        ValueError
            If the epoch does not contain a single interval or if split_ratios do not sum to 1.

    Example:
        >>> epoch = Interval(start=0, end=100)
        >>> train_int, valid_int = generate_train_valid_splits_one_epoch(epoch, [0.8, 0.2])
    """
    if len(epoch) != 1:
        raise ValueError("Epoch must contain a single interval")

    if split_ratios[0] + split_ratios[1] != 1:
        raise ValueError("Split ratios must sum to 1")

    epoch_start = epoch.start[0]
    epoch_end = epoch.end[0]

    train_split_time = epoch_start + split_ratios[0] * (epoch_end - epoch_start)
    val_split_time = train_split_time + split_ratios[1] * (epoch_end - epoch_start)

    train_intervals = Interval(start=epoch_start, end=train_split_time)
    valid_intervals = Interval(start=train_intervals.end[0], end=val_split_time)

    return train_intervals, valid_intervals
