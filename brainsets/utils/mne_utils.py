"""Data extraction utilities.

This module provides functions to extract metadata and signal data from
MNE Raw objects and convert them to brainsets data structures.
"""

import datetime
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
        recording_data: The MNE Raw object containing recording data and metadata.

    Returns:
        The measurement date as a datetime object if present, otherwise
        the Unix epoch (1970-01-01 UTC) as a placeholder.

    Raises:
        ImportError: If MNE is not installed.
    """
    _check_mne_available("extract_measurement_date")
    if recording_data.info["meas_date"] is not None:
        return recording_data.info["meas_date"]
    logging.warning("No measurement date found, using Unix epoch as placeholder")
    return datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)


def concatenate_recordings(
    recordings: list["mne.io.Raw"],
    on_mismatch: Literal["ignore", "warn", "raise"] = "raise",
    on_offset: Literal["ignore", "warn", "raise"] = "warn",
) -> "mne.io.Raw":
    """Concatenate a list of MNE Raw objects into one, validating metadata.

    This function concatenates multiple MNE Raw recordings in temporal order
    (sorted by measurement date). Before concatenation, it validates that all
    recordings have:
      - identical measurement days (raw.info["meas_date"]),
      - identical channel names and order.

    It then normalizes each recording's timeline so that its first sample
    corresponds to its measurement date, then concatenates them.

    Args:
        recordings: List of MNE Raw objects to concatenate.
        on_mismatch: How to handle mismatches in measurement date or channels.
            - "raise": raise ValueError on any mismatch (default),
            - "warn": issue a warning and continue,
            - "ignore": silently continue with mismatches.

    Returns:
        An MNE Raw object containing the concatenated recordings in temporal order.

    Raises:
        ImportError: If MNE is not installed.
        ValueError: If recordings is empty, contains non-Raw objects,
            on_mismatch is invalid, or (if on_mismatch="raise") mismatches are detected.
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
    
    if on_offset not in valid_policies:
        raise ValueError(
            f"on_offset must be one of {valid_policies}, got '{on_offset}'"
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
            logging.warning(full_message)

    # Sort recordings by measurement date
    indexed_recordings = [
        (idx, rec, meas_dates[idx]) for idx, rec in enumerate(recordings)
    ]
    sorted_recordings = sorted(
        indexed_recordings,
        key=lambda x: x[2] if x[2] is not None else datetime.datetime.min,
    )

    # Validate that offset between consecutive recordings is within 1 hour
    if on_offset == "raise":
        # More efficient: use zip to pair consecutive elements, avoid repeated indexing
        for (idx1, _, date1), (idx2, _, date2) in zip(sorted_recordings, sorted_recordings[1:]):
            offset = (date2 - date1).total_seconds()
            if offset > 3600:
                raise ValueError(
                    f"Offset between recordings {idx1} and {idx2} is greater than 1 hour: {offset} seconds"
                )
    elif on_offset == "warn":
        # A more efficient way: use zip to iterate through consecutive pairs directly.
        for (idx1, _, date1), (idx2, _, date2) in zip(sorted_recordings, sorted_recordings[1:]):
            offset = (date2 - date1).total_seconds()
            if offset > 3600:
                logging.warning(
                    f"Offset between recordings {idx1} and {idx2} is greater than 1 hour: {offset} seconds"
                )

    copies = []
    for _, rec, _ in sorted_recordings:
        copies.append(rec.copy())

    concatenated = mne.concatenate_raws(copies)

    return concatenated


def extract_signal(
    recording_data: "mne.io.Raw",
) -> RegularTimeSeries:
    """Extract entire time-series signal from an MNE Raw object.

    Args:
        recording_data: The MNE Raw object containing signal data.

    Returns:
        RegularTimeSeries object containing the signal matrix
        and time information.

    Raises:
        ImportError: If MNE is not installed.
        ValueError: If the recording contains no samples.
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


def extract_psg_signal(raw_psg: "mne.io.Raw") -> Tuple[RegularTimeSeries, ArrayDict]:
    """Extract physiological (PSG) signals and channel metadata from an MNE Raw (EDF) recording.

    Args:
        raw_psg: The MNE Raw object containing PSG data (e.g., from an EDF file).

    Returns:
        Tuple containing:
        - RegularTimeSeries: The extracted physiological signal data with
          sampling rate and domain info.
        - ArrayDict: Channel metadata with fields 'id' (channel names)
          and 'type' (channel types: EEG, EOG, EMG, RESP, or TEMP).

    Raises:
        ImportError: If MNE is not installed.
        ValueError: If no physiological signals are extracted from the PSG recording.
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


def extract_channels(
    recording_data: "mne.io.Raw",
    channels_name_mapping: dict[str, str] | None = None,
    channels_type_mapping: dict[str, list[str]] | None = None,
) -> ArrayDict:
    """Extract channel metadata from an MNE Raw object with optional renaming and type assignment.

    This function generates channel metadata including ids, types, 'bad' status,
    and (if present) spatial coordinates, by combining information from the MNE Raw object
    and optional user-provided mappings.

    The process includes:
        - Extracting channel names and original types from `raw.ch_names` and `raw.get_channel_types()`
        - Optionally applying a channel name mapping (`channels_name_mapping`) to rename channels
        - Optionally applying a channel type mapping (`channels_type_mapping`) to reassign channel types
        - Marking channels in `raw.info["bads"]` as "bad" in the boolean array (others are "good")
        - Extracting coordinates (x, y, z) from the Raw object's montage if available

    Args:
        recording_data: The MNE Raw object containing signal data and channel metadata.
        channels_name_mapping: Optional; dict mapping original channel names to new names.
        channels_type_mapping: Optional; dict mapping desired channel type (str) to a list of channel names to assign that type.

    Returns:
        ArrayDict containing channel metadata with fields:
            - 'id': channel names (renamed if applicable)
            - 'type': channel types (remapped if applicable)
            - 'bad': boolean array, True if bad channel
            - 'coord': spatial coordinates, shape (n_channels, 3), if available

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
        # We need to decide whether to apply channel type mappings based on the original or (optionally) renamed channel names.
        # - If channels_name_mapping is applied, channel_ids may be different from the raw input ch_names.
        # - channels_type_mapping dictionary assigns a desired type to a list of channel names.
        # We want to detect whether these names in the type mapping refer to the original raw names or the mapped names.

        # Original channel names before any mapping
        original_channel_names = np.array(recording_data.ch_names, dtype="U")

        # Gather all channel names referenced in the type mapping
        type_mapping_channel_names = set(
            ch for ch_list in channels_type_mapping.values() for ch in ch_list
        )

        # Decide whether to use the mapped names or the original names, for type assignment
        # If all current channel_ids (already renamed if applicable) are in the type mapping reference list,
        # we assume the type mapping is meant for the renamed channels.
        use_renamed_for_type = all(
            ch in type_mapping_channel_names for ch in channel_ids
        )

        # Select which set of names (renamed or original) should be used for the type lookup
        channel_names_for_typing = (
            channel_ids if use_renamed_for_type else original_channel_names
        )

        # Walk through channels and assign types from mapping, defaulting as needed
        channel_type_lookup = {
            ch_name: ch_type
            for ch_type, ch_list in channels_type_mapping.items()
            for ch_name in ch_list
        }
        # Use the lookup, defaulting to original type if channel not remapped
        channel_types = np.array(
            [
                channel_type_lookup.get(name_for_type, orig_type)
                for name_for_type, orig_type in zip(
                    channel_names_for_typing, channel_types
                )
            ],
            dtype="U",
        )

    # bad channel extraction
    bad_channels = recording_data.info.get("bads", [])
    if bad_channels:
        is_bad_channel = np.array([ch in bad_channels for ch in channel_ids], dtype=bool)
    else:
        is_bad_channel = None

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
    }

    if is_bad_channel is not None:
        channel_fields["bad"] = is_bad_channel

    if not np.all(np.isnan(coords_arr)):
        channel_fields["coord"] = coords_arr

    return ArrayDict(**channel_fields)
