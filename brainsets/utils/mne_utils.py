"""Data extraction utilities.

This module provides functions to extract metadata and signal data from
MNE Raw objects and convert them to brainsets data structures.
"""

import datetime
import warnings
import numpy as np
from typing import Tuple, Literal
from temporaldata import (
    ArrayDict,
    Interval,
    RegularTimeSeries,
)
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


def concatenate_recordings(
    recordings: list["mne.io.Raw"],
    on_mismatch: Literal["ignore", "warn", "raise"] = "raise",
) -> "mne.io.Raw":
    """Concatenate a list of MNE Raw objects into a single MNE Raw object.

    This function concatenates multiple MNE Raw recordings in temporal order (sorted by
    measurement date). Before concatenation, it validates that all recordings have:
    - identical measurement dates (raw.info["meas_date"]),
    - identical channel names and order.

    The function then normalizes each recording's timeline so that its first sample
    corresponds to its measurement date, then concatenates them chronologically.

    Args:
        recordings: A list of MNE Raw objects to concatenate.
        on_mismatch: How to handle mismatches in measurement date or channels.
            - "raise": raise ValueError on any mismatch (default),
            - "warn": issue a warning and continue,
            - "ignore": silently continue with mismatches.

    Returns:
        A single MNE Raw object containing the concatenated recordings in temporal order.

    Raises:
        ImportError: If MNE is not installed.
        ValueError: If recordings list is empty, contains non-Raw objects, on_mismatch
            is invalid, or (if on_mismatch="raise") mismatches are detected.
    """
    _check_mne_available("concatenate_recordings")

    if not recordings:
        raise ValueError("Recordings list cannot be empty")

    if not isinstance(recordings, list):
        raise ValueError("Recordings must be a list")

    valid_policies = {"ignore", "warn", "raise"}
    if on_mismatch not in valid_policies:
        raise ValueError(
            f"on_mismatch must be one of {valid_policies}, got '{on_mismatch}'"
        )

    for idx, rec in enumerate(recordings):
        if not hasattr(rec, "info") or not hasattr(rec, "ch_names"):
            raise ValueError(
                f"Recordings[{idx}] is not an MNE Raw-like object "
                "(missing 'info' or 'ch_names' attributes)"
            )

    # Validate that all recordings have the same measurement date
    mismatch_messages = []
    meas_dates = [rec.info["meas_date"] for rec in recordings]
    meas_days = [
        d.date() if hasattr(d, "date") and d is not None else None for d in meas_dates
    ]
    if len(set(meas_days)) > 1:
        mismatch_messages.append(
            f"Measurement days are not uniform: {meas_days} (full datetimes: {meas_dates})"
        )

    # Validate that all recordings have the same channel names and order
    ch_names_list = [tuple(rec.ch_names) for rec in recordings]
    if len(set(ch_names_list)) > 1:
        mismatch_details = []
        for idx, ch_names in enumerate(ch_names_list):
            mismatch_details.append(f"Recording {idx}: {ch_names}")
        mismatch_message = (
            "Mismatch in channel names and/or order across recordings.\n"
            "Each tuple below shows the channel names for one recording in the given order:\n"
            + "\n".join(mismatch_details)
            + "\n"
            "All recordings must have identical channel lists and order for concatenation."
        )
        mismatch_messages.append(mismatch_message)

    if mismatch_messages:
        full_message = "; ".join(mismatch_messages)
        if on_mismatch == "raise":
            raise ValueError(full_message)
        elif on_mismatch == "warn":
            warnings.warn(full_message)

    # Sort recordings by measurement date
    indexed_recordings = [
        (idx, rec, meas_dates[idx]) for idx, rec in enumerate(recordings)
    ]
    sorted_recordings = sorted(
        indexed_recordings,
        key=lambda x: x[2] if x[2] is not None else datetime.datetime.min,
    )

    copies = []
    for _, rec, _ in sorted_recordings:
        copies.append(rec.copy())

    concatenated = mne.concatenate_raws(copies)

    return concatenated


def extract_signal(
    recording_data: "mne.io.Raw",
) -> RegularTimeSeries:
    """Extract signal from MNE Raw data as a RegularTimeSeries.

    Args:
        recording_data: The MNE Raw object containing signal data

    Returns:
        RegularTimeSeries object with the signal data

    Raises:
        ImportError: If MNE is not installed.
    """
    _check_mne_available("extract_signal")

    sfreq = recording_data.info["sfreq"]
    signal = recording_data.get_data().T
    if len(signal) == 0:
        raise ValueError("Recording contains no samples")

    return RegularTimeSeries(
        signal=signal,
        sampling_rate=sfreq,
        domain=Interval(
            start=np.array([0.0]),
            end=np.array([(len(signal) - 1) / sfreq]),
        ),
    )


def has_consecutive_time_points(
    raw: "mne.io.Raw",
    tolerance: float = 1e-6,
) -> bool:
    """Verify whether a raw object has consecutive time points based on sampling frequency.

    This function checks if the time points in a recording are evenly spaced according to
    the sampling frequency. It detects gaps, duplicates, or irregularities in the temporal
    sampling by comparing expected time intervals (based on sampling rate) with actual intervals.

    Args:
        raw: The MNE Raw object to verify
        tolerance: Relative tolerance for comparing time intervals. The expected interval
            (1/sfreq) is compared to actual intervals, and they are considered equal if:
            |actual - expected| / expected <= tolerance. Defaults to 1e-6 (0.0001% tolerance).

    Returns:
        True if all time points are consecutive with uniform spacing according to sampling
        frequency; False if gaps, duplicates, or irregular spacing are detected.

    Raises:
        ImportError: If MNE is not installed.
        ValueError: If the recording contains fewer than 2 samples.

    Examples:
        >>> raw = mne.io.read_raw_edf('recording.edf')
        >>> is_consecutive = has_consecutive_time_points(raw)
        >>> if not is_consecutive:
        ...     print("Recording has gaps or irregular sampling")
    """
    _check_mne_available("has_consecutive_time_points")

    n_samples = raw.n_times
    if n_samples < 2:
        raise ValueError(
            "Recording must contain at least 2 samples to verify consecutiveness"
        )

    sfreq = raw.info["sfreq"]
    expected_interval = 1.0 / sfreq

    times = raw.times
    actual_intervals = np.diff(times)

    max_deviation = np.max(np.abs(actual_intervals - expected_interval))
    relative_max_deviation = max_deviation / expected_interval

    return relative_max_deviation <= tolerance


def extract_channels(
    recording_data: "mne.io.Raw",
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
        channel_ids = np.array(
            [channels_name_mapping.get(ch, ch) for ch in channel_ids], dtype="U"
        )

    # Optional: apply channels type re-mapping
    if channels_type_mapping:
        # Build a mapping from channel name to new type using a dictionary comprehension
        ch_type_lookup = {
            ch: new_type
            for new_type, ch_list in channels_type_mapping.items()
            for ch in ch_list
        }
        # Use the lookup, defaulting to original type if channel not remapped
        channel_types = np.array(
            [
                ch_type_lookup.get(ch, orig_type)
                for ch, orig_type in zip(channel_ids, channel_types)
            ],
            dtype="U",
        )

    # status extraction (defaults to "good", mark raw.info["bads"] as "bad")
    status = np.full(channel_count, "good", dtype="U")
    bad_channels = recording_data.info.get("bads", [])
    for idx, ch_name in enumerate(channel_ids):
        if ch_name in bad_channels:
            status[idx] = "bad"

    # coordinate extraction from montage (x, y, z in meters)
    coords_arr = np.full((channel_count, 3), np.nan)
    try:
        montage = recording_data.get_montage()
        if montage is not None:
            positions = montage.get_positions()
            ch_pos = positions.get("ch_pos") if positions is not None else None
            if ch_pos is not None:
                for idx, ch_name in enumerate(channel_ids):
                    if ch_name in ch_pos:
                        coords = ch_pos[ch_name]
                        coords_arr[idx, 0] = float(coords[0])
                        coords_arr[idx, 1] = float(coords[1])
                        coords_arr[idx, 2] = float(coords[2])
    except Exception as e:
        logging.warning(f"Could not extract channel coordinates: {e}")

    channel_fields = {
        "id": channel_ids,
        "type": channel_types,
        "status": status,
    }
    if not np.all(np.isnan(coords_arr)):
        channel_fields["coords"] = coords_arr

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

