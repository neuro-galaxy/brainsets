"""Data extraction utilities.

This module provides functions to extract metadata and signal data from
MNE Raw objects and convert them to brainsets data structures.
"""

import mne
import datetime
import warnings
import numpy as np
import pandas as pd
from typing import Tuple
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
    warnings.warn("No measurement date found, using Unix epoch as placeholder")
    return datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)


def extract_eeg_signal(
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
        ids=np.array(recording_data.ch_names, dtype="U"),
        types=np.array(recording_data.get_channel_types(), dtype="U"),
    )


def extract_psg_signal(raw_psg: mne.io.Raw) -> Tuple[RegularTimeSeries, ArrayDict]:
    """Extract physiological signals from PSG EDF file as a RegularTimeSeries."""
    data, times = raw_psg.get_data(return_times=True)
    ch_names = raw_psg.ch_names

    signal_list = []
    unit_meta = []

    for idx, ch_name in enumerate(ch_names):
        ch_name_lower = ch_name.lower()
        signal_data = data[idx, :]

        modality = None
        if (
            "eeg" in ch_name_lower
            or "fpz-cz" in ch_name_lower
            or "pz-oz" in ch_name_lower
        ):
            modality = "EEG"
        elif "eog" in ch_name_lower:
            modality = "EOG"
        elif "emg" in ch_name_lower:
            modality = "EMG"
        elif "resp" in ch_name_lower:
            modality = "RESP"
        elif "temp" in ch_name_lower:
            modality = "TEMP"
        else:
            continue

        signal_list.append(signal_data)

        unit_meta.append(
            {
                "id": str(ch_name),
                "modality": modality,
            }
        )

    if not signal_list:
        raise ValueError("No signals extracted from PSG file")

    stacked_signals = np.stack(signal_list, axis=1)

    signals = RegularTimeSeries(
        signal=stacked_signals,
        sampling_rate=raw_psg.info["sfreq"],
        domain=Interval(start=times[0], end=times[-1]),
    )

    units_df = pd.DataFrame(unit_meta)
    units = ArrayDict.from_dataframe(units_df)

    return signals, units
