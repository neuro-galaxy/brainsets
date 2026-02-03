"""Data extraction utilities.

This module provides functions to extract metadata and signal data from
MNE Raw objects and convert them to brainsets data structures.
"""

import mne
import datetime
import numpy as np
from temporaldata import ArrayDict, Interval, RegularTimeSeries


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
