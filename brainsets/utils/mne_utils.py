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

    Offset validation:
        The function checks for temporal offsets in the measurement dates of the recordings.
        If the measurement dates are separated by notable amounts of time, this can indicate
        temporal discontinuity. The `on_offset` parameter controls how such offsets are handled:
            - "raise": raise ValueError if an offset is detected,
            - "warn": issue a warning and continue (default),
            - "ignore": silently continue.
        This is useful to ensure recordings are truly continuous or to be notified about gaps between sessions.

    Args:
        recordings: List of MNE Raw objects to concatenate.
        on_mismatch: How to handle mismatches in measurement date or channels.
            - "raise": raise ValueError on any mismatch (default),
            - "warn": issue a warning and continue,
            - "ignore": silently continue with mismatches.
        on_offset: How to handle temporal offsets between recordings' measurement dates.
            - "raise": raise ValueError if offsets are detected,
            - "warn": issue a warning and continue (default),
            - "ignore": silently continue with offsets.

    Returns:
        An MNE Raw object containing the concatenated recordings in temporal order.

    Raises:
        ImportError: If MNE is not installed.
        ValueError: If recordings is empty, contains non-Raw objects,
            on_mismatch or on_offset is invalid, or (if set to "raise") mismatches or time offsets are detected.
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
        for (idx1, _, date1), (idx2, _, date2) in zip(
            sorted_recordings, sorted_recordings[1:]
        ):
            offset = (date2 - date1).total_seconds()
            if offset > 3600:
                raise ValueError(
                    f"Offset between recordings {idx1} and {idx2} is greater than 1 hour: {offset} seconds"
                )
    elif on_offset == "warn":
        # A more efficient way: use zip to iterate through consecutive pairs directly.
        for (idx1, _, date1), (idx2, _, date2) in zip(
            sorted_recordings, sorted_recordings[1:]
        ):
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
    channels_pos_mapping: dict[str, np.ndarray] | None = None,
) -> ArrayDict:
    """Extract channel metadata from an MNE Raw object with optional renaming and type assignment.

    This function generates channel metadata including ids, types, 'bad' status,
    and (if present) spatial positions, by combining information from the MNE Raw object
    and optional user-provided mappings.

    The process includes:
        - Extracting channel names and original types from `raw.ch_names` and `raw.get_channel_types()`
        - Optionally applying a channel name mapping (`channels_name_mapping`) to rename channels
        - Optionally applying a channel type mapping (`channels_type_mapping`) to reassign channel types
        - Marking channels in `raw.info["bads"]` as "bad" in the boolean array (others are "good")
        - Extracting spatial positions (x, y, z) from `pos_mapping` or the Raw object's montage if available

    Args:
        recording_data: The MNE Raw object containing signal data and channel metadata.
        channels_name_mapping: Optional; dict mapping original channel names to new names.
        channels_type_mapping: Optional; dict mapping desired channel type (str) to a list of channel names to assign that type.
        channels_pos_mapping: Optional; dict mapping channel names to 1D numpy arrays of shape (3,) containing (x, y, z) positions.

    Returns:
        ArrayDict containing channel metadata with fields:
            - 'id': channel names (renamed if applicable)
            - 'type': channel types (remapped if applicable)
            - 'bad': boolean array, True if bad channel
            - 'pos': spatial positions, shape (n_channels, 3), if available (from pos_mapping or montage)

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
        # We need to detect whether channel names in channels_type_mapping
        # refer to the original raw names or the mapped channel_ids (after name mapping).

        # Original channel names before any name mapping
        original_ch_names = np.array(recording_data.ch_names, dtype="U")

        # Gather all channel names referenced in the type mapping
        ch_names_in_type_mapping = set(
            ch for ch_list in channels_type_mapping.values() for ch in ch_list
        )
        ch_names_for_type_mapping = _resolve_channel_names_for_mapping(
            original_ch_names=original_ch_names,
            renamed_ch_names=channel_ids,
            ch_names_in_mapping=ch_names_in_type_mapping,
        )
        ch_type_lookup = {
            ch_name: ch_type
            for ch_type, ch_list in channels_type_mapping.items()
            for ch_name in ch_list
        }
        channel_types = np.array(
            [
                ch_type_lookup.get(ch_name_for_type, orig_type)
                for ch_name_for_type, orig_type in zip(
                    ch_names_for_type_mapping, channel_types
                )
            ],
            dtype="U",
        )

    # bad channel extraction
    bad_channels = recording_data.info.get("bads", [])
    if bad_channels:
        is_bad_channel = np.array(
            [ch in bad_channels for ch in channel_ids], dtype=bool
        )
    else:
        is_bad_channel = None

    # position extraction: prioritize pos_mapping, fall back to montage (x, y, z in mm)
    pos_arr = np.full((channel_count, 3), np.nan)
    if channels_pos_mapping is not None:
        original_ch_names = np.array(recording_data.ch_names, dtype="U")
        ch_names_in_pos_mapping = set(channels_pos_mapping.keys())
        ch_names_for_pos_mapping = _resolve_channel_names_for_mapping(
            original_ch_names=original_ch_names,
            renamed_ch_names=channel_ids,
            ch_names_in_mapping=ch_names_in_pos_mapping,
        )
        # Fill the existing pos_arr for each channel
        for i, ch_name in enumerate(ch_names_for_pos_mapping):
            if ch_name in channels_pos_mapping:
                pos_arr[i] = channels_pos_mapping[ch_name]
    else:
        # Fallback to montage-based extraction if no pos_mapping provided
        try:
            montage = recording_data.get_montage()
            if montage is not None:
                ch_pos_mapping = montage.get_positions()["ch_pos"]
                if ch_pos_mapping is not None:
                    # Fill pos_arr for each channel using montage-based positions if available
                    for i, ch_name in enumerate(recording_data.ch_names):
                        if ch_name in ch_pos_mapping:
                            pos_arr[i] = ch_pos_mapping[ch_name]
        except Exception as e:
            logging.warning(f"Could not extract channel positions from montage: {e}")

    channel_fields = {
        "id": channel_ids,
        "type": channel_types,
    }

    if is_bad_channel is not None:
        channel_fields["bad"] = is_bad_channel

    if np.any(~np.isnan(pos_arr)):
        channel_fields["pos"] = pos_arr

    return ArrayDict(**channel_fields)


def _resolve_channel_names_for_mapping(
    original_ch_names: np.ndarray,
    renamed_ch_names: np.ndarray,
    ch_names_in_mapping: set,
) -> np.ndarray:
    """Determine which channel names to use for mapping lookups (original or remapped).

    When a mapping is provided (e.g., type or position mapping), this helper decides
    whether the mapping keys refer to original channel names or the renamed channel
    names (after optional name mapping). The validation requires that all mapping keys
    exist in either the original or renamed channel names (but not split between them).

    Args:
        original_ch_names: Array of channel names from the raw recording.
        renamed_ch_names: Array of channel names after optional name mapping.
        ch_names_in_mapping: Set of all channel names referenced in the mapping.

    Returns:
        Array of channel names to use for mapping lookups (either original or renamed).

    Raises:
        ValueError: If mapping names are not consistent with either original or remapped channel names.
    """
    renamed_ch_names_set = set(renamed_ch_names)
    original_ch_names_set = set(original_ch_names)

    # Check if all mapping references are in renamed channel names
    all_in_renamed = ch_names_in_mapping.issubset(renamed_ch_names_set)

    # Check if all mapping references are in original names
    all_in_original = ch_names_in_mapping.issubset(original_ch_names_set)

    if all_in_renamed:
        return renamed_ch_names
    elif all_in_original:
        return original_ch_names
    else:
        # Neither original nor remapped contains all mapping names - inconsistent
        raise ValueError(
            f"Channel name mismatch in the mapping keys must refer to either "
            f"all original channel names or all renamed channel names, but not a mix. "
            f"Mapping keys: {sorted(ch_names_in_mapping)}. "
            f"Renamed channel names: {sorted(renamed_ch_names_set)}. "
            f"Original channel names: {sorted(original_ch_names_set)}."
        )
