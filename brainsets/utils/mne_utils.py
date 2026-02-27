"""Data extraction utilities.

This module provides functions to extract metadata and signal data from
MNE Raw objects and convert them to brainsets data structures.
"""

import datetime
import warnings
import numpy as np
from typing import Tuple
from temporaldata import ArrayDict, Interval, RegularTimeSeries
from pathlib import Path
import logging
try:
    import mne

    MNE_AVAILABLE = True
except ImportError:
    mne = None
    MNE_AVAILABLE = False


def _check_mne_available(func_name: str) -> None:
    """Raise ImportError if MNE is not available."""
    if not MNE_AVAILABLE:
        raise ImportError(
            f"{func_name} requires the MNE library which is not installed. "
            "Install it with `pip install mne`"
        )


def extract_measurement_date(
    recording_data: "mne.io.Raw",
) -> datetime.datetime:
    """Extract the measurement date from MNE Raw recording data.

    Args:
        recording_data: The MNE Raw object containing EEG data and metadata

    Returns:
        The measurement date if present, otherwise Unix epoch (1970-01-01 UTC)

    Raises:
        ImportError: If MNE is not installed.
    """
    _check_mne_available("extract_measurement_date")
    if recording_data.info["meas_date"] is not None:
        return recording_data.info["meas_date"]
    warnings.warn("No measurement date found, using Unix epoch as placeholder")
    return datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)


def extract_eeg_signal(
    recording_data: "mne.io.Raw",
) -> RegularTimeSeries:
    """Extract the EEG signal as a RegularTimeSeries from MNE Raw data.

    Args:
        recording_data: The MNE Raw object containing EEG data

    Returns:
        RegularTimeSeries object with the EEG signal

    Raises:
        ImportError: If MNE is not installed.
    """
    _check_mne_available("extract_eeg_signal")
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
    channels_name_mapping: dict[str, str] | None = None,
    channels_type_mapping: dict[str, list[str]] | None = None,
) -> ArrayDict:
    """Extract channel metadata from an MNE Raw object, optionally applying name and type mappings.

    This function generates channel metadata including channel IDs, types, status, 
    and (if available) spatial coordinates by combining information from the MNE Raw object
    and optional user-provided mappings.

    The process includes:
        - Extracting channel names and original types from `raw.ch_names` and `raw.get_channel_types()`
        - Optionally applying a channel name mapping (`channels_name_mapping`) to rename channels
        - Optionally applying a channel type mapping (`channels_type_mapping`) to change channel types
        - Marking channels in `raw.info["bads"]` as "bad" in the status array (all others are "good")
        - Extracting x, y, z coordinates from the Raw object's montage if available

    Args:
        recording_data: The MNE Raw object containing signal data and channel metadata.
        channels_name_mapping: Optional; dict mapping original channel names to new names.
        channels_type_mapping: Optional; dict mapping desired channel type (str) to a list of channel names to assign to that type.

    Returns:
        ArrayDict containing the channel metadata with fields:
            - 'id': channel names (possibly renamed)
            - 'type': channel types (possibly remapped)
            - 'status': 'good' or 'bad' for each channel
            - 'x', 'y', 'z': spatial coordinates (only if available)

    Raises:
        ImportError: If MNE is not installed.
        ValueError: If no channels are extracted from the recording.
    """
    _check_mne_available("extract_channels")
    channel_ids = np.array(recording_data.ch_names, dtype="U")
    channel_count = len(channel_ids)
    channel_types = np.array(recording_data.get_channel_types(), dtype="U")

    # Optional: apply channels name re-mapping
    if channels_name_mapping:
        channel_ids = np.array([channels_name_mapping.get(ch, ch) for ch in channel_ids], dtype="U")

    # Optional: apply channels type re-mapping
    if channels_type_mapping:
        # Build a mapping from channel name to new type using a dictionary comprehension
        ch_type_lookup = {ch: new_type for new_type, ch_list in channels_type_mapping.items() for ch in ch_list}
        # Use the lookup, defaulting to original type if channel not remapped
        channel_types = np.array([ch_type_lookup.get(ch, orig_type) for ch, orig_type in zip(channel_ids, channel_types)], dtype="U")

    # status extraction (defaults to "good", mark raw.info["bads"] as "bad")
    status = np.full(channel_count, "good", dtype="U")
    bad_channels = recording_data.info.get("bads", [])
    for idx, ch_name in enumerate(channel_ids):
        if ch_name in bad_channels:
            status[idx] = "bad"

    # coordinate extraction from montage (x, y, z in meters)
    x_coords = np.full(channel_count, np.nan)
    y_coords = np.full(channel_count, np.nan)
    z_coords = np.full(channel_count, np.nan)

    try:
        montage = recording_data.get_montage()
        if montage is not None:
            positions = montage.get_positions()
            ch_pos = positions.get("ch_pos") if positions is not None else None
            if ch_pos is not None:
                for idx, ch_name in enumerate(channel_ids):
                    if ch_name in ch_pos:
                        coords = ch_pos[ch_name]
                        x_coords[idx] = float(coords[0])
                        y_coords[idx] = float(coords[1])
                        z_coords[idx] = float(coords[2])
    except Exception as e:
        logging.warning(
            f"Could not extract channel coordinates: {e}"
        )

    channel_fields = {
        "id": channel_ids,
        "type": channel_types,
        "status": status,
    }
    if not np.all(np.isnan(x_coords)):
        channel_fields["x"] = x_coords
    if not np.all(np.isnan(y_coords)):
        channel_fields["y"] = y_coords
    if not np.all(np.isnan(z_coords)):
        channel_fields["z"] = z_coords

    return ArrayDict(**channel_fields)

def extract_psg_signal(raw_psg: "mne.io.Raw") -> Tuple[RegularTimeSeries, ArrayDict]:
    """Extract physiological signals from polysomnography (PSG) recording as a RegularTimeSeries.

    Args:
        raw_psg: The MNE Raw object containing PSG data from an EDF file

    Returns:
        A tuple containing:
        - RegularTimeSeries: The extracted physiological signals with
          sampling rate and time domain information
        - ArrayDict: Channel metadata with fields 'id' (channel names)
          and 'type' (channel types: EEG, EOG, EMG, RESP, or TEMP)

    Raises:
        ImportError: If MNE is not installed.
        ValueError: If no signals are extracted from the PSG file.
    """
    _check_mne_available("extract_psg_signal")
    data, times = raw_psg.get_data(return_times=True)
    ch_names = raw_psg.ch_names

    signal_list = []
    channel_meta = []

    for idx, ch_name in enumerate(ch_names):
        ch_name_lower = ch_name.lower()
        signal_data = data[idx, :]

        ch_type = None
        if (
            "eeg" in ch_name_lower
            or "fpz-cz" in ch_name_lower
            or "pz-oz" in ch_name_lower
        ):
            ch_type = "EEG"
        elif "eog" in ch_name_lower:
            ch_type = "EOG"
        elif "emg" in ch_name_lower:
            ch_type = "EMG"
        elif "resp" in ch_name_lower:
            ch_type = "RESP"
        elif "temp" in ch_name_lower:
            ch_type = "TEMP"
        else:
            continue

        signal_list.append(signal_data)

        channel_meta.append(
            {
                "id": str(ch_name),
                "type": ch_type,
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

    channels = ArrayDict(
        id=np.array([ch["id"] for ch in channel_meta], dtype="U"),
        type=np.array([ch["type"] for ch in channel_meta], dtype="U"),
    )

    return signals, channels
