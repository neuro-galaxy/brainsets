"""Data extraction utilities.

This module provides functions to extract metadata and signal data from
MNE Raw objects and convert them to brainsets data structures.
"""

import datetime
import numpy as np
from typing import Tuple, Literal
from collections import Counter

from temporaldata import (
    ArrayDict,
    Interval,
    RegularTimeSeries,
)
import warnings

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
    recording_data: "mne.io.BaseRaw",
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
    warnings.warn("No measurement date found, using Unix epoch as placeholder")
    return datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)


def concatenate_recordings(
    recordings: list["mne.io.BaseRaw"],
    max_gap: float = 1.0,
    on_mismatch: Literal["ignore", "warn", "raise"] = "raise",
    on_gap: Literal["ignore", "warn", "raise"] = "warn",
    on_missing_meas_date: Literal["ignore", "warn", "raise"] = "warn",
) -> "mne.io.BaseRaw":
    """Concatenate a list of MNE Raw objects into one, validating metadata.

    This function concatenates multiple MNE Raw recordings, prioritizing temporal order
    by default: recordings are sorted by measurement date before concatenation.

    Channel validation (always enforced):
        All recordings must have identical channel names and order.

    Measurement date validation:
        The function validates that all recordings have identical measurement days.
        The `on_mismatch` parameter controls how such mismatches are handled; default is "raise".

        If one or more recordings are missing a measurement date (`meas_date` is None), temporal order cannot be established.
        By default, the function will concatenate the recordings in the given input order rather than sorting by measurement date.
        The `on_missing_meas_date` parameter controls how this is handled; default is "warn".

    Offset validation:
        The function checks for temporal offsets in the measurement dates of the recordings.
        If the measurement dates are separated by notable amounts of time (as defined by the `max_gap`
        parameter, in hours), this can indicate temporal discontinuity.
        The `on_gap` parameter controls how such offsets are handled when the offset exceeds `max_gap`; default is "warn".
        This is useful to ensure recordings are truly continuous or to be notified about gaps between sessions.

    Args:
        recordings: List of MNE Raw objects to concatenate.
        max_gap: Maximum allowed gap in hours between consecutive measurement dates for the recordings to be considered continuous.
        on_mismatch: How to handle measurement date mismatches (channel mismatches always raise).
            - "raise": raise ValueError if measurement days are not uniform (default),
            - "warn": issue a warning and continue,
            - "ignore": silently continue with measurement day mismatches.
        on_gap: How to handle temporal offsets between recordings' measurement dates.
            - "raise": raise ValueError if offsets are detected,
            - "warn": issue a warning and continue (default),
            - "ignore": silently continue with offsets.
        on_missing_meas_date: How to handle missing (None) measurement dates.
            - "raise": raise ValueError if any measurement date is None,
            - "warn": issue a warning and continue in input order (default),
            - "ignore": silently continue in input order.

    Returns:
        An MNE Raw object containing the concatenated recordings in temporal order
        (or input order if measurement dates are missing or mixed).

    Raises:
        ImportError: If MNE is not installed.
        ValueError: If recordings is empty, contains non-Raw objects, has channel mismatches,
            on_mismatch, on_gap, or on_missing_meas_date is invalid, or (if set to "raise")
            measurement date mismatches, time offsets, or missing measurement dates are detected.
    """
    _check_mne_available("concatenate_recordings")

    def _normalize_meas_date(
        meas_date: datetime.datetime | None,
    ) -> datetime.datetime | None:
        """Normalize measurement date to naive UTC datetime for consistent comparison.

        Converts timezone-aware datetimes to naive UTC. Naive datetimes are returned as-is.
        None values are preserved as None.

        Args:
            meas_date: A datetime object that may be timezone-aware or naive, or None.

        Returns:
            A naive UTC datetime, a naive datetime (unchanged), or None.
        """
        if meas_date is None:
            return None
        # Convert timezone-aware datetime to naive UTC datetime
        return meas_date.astimezone(datetime.timezone.utc).replace(tzinfo=None)

    if not isinstance(recordings, list):
        raise TypeError(f"Recordings must be a list, got {type(recordings).__name__}.")

    if not recordings:
        raise ValueError("Recordings list cannot be empty")

    valid_policies = {"ignore", "warn", "raise"}
    if on_mismatch not in valid_policies:
        raise ValueError(
            f"on_mismatch must be one of {valid_policies}, got '{on_mismatch}'"
        )

    if on_gap not in valid_policies:
        raise ValueError(f"on_gap must be one of {valid_policies}, got '{on_gap}'")

    if on_missing_meas_date not in valid_policies:
        raise ValueError(
            f"on_missing_meas_date must be one of {valid_policies}, got '{on_missing_meas_date}'"
        )
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")

    for idx, rec in enumerate(recordings):
        if not hasattr(rec, "info") or not hasattr(rec, "ch_names"):
            raise ValueError(
                f"Recordings[{idx}] is not an MNE Raw-like object "
                "(missing 'info' or 'ch_names' attributes)"
            )

    # Validate that all recordings have the same channel names and order (always enforced)
    ch_names_list = [tuple(rec.ch_names) for rec in recordings]
    if len(set(ch_names_list)) > 1:
        mismatch_details = []
        for idx, ch_names in enumerate(ch_names_list):
            mismatch_details.append(f"Recording {idx}: {ch_names}")
        raise ValueError(
            "Mismatch in channel names and/or order across recordings.\n"
            "Each tuple below shows the channel names for one recording in the given order:\n"
            + "\n".join(mismatch_details)
            + "\n"
            "All recordings must have identical channel lists and order for concatenation."
        )

    # Normalize measurement dates before meas_date validation
    raw_meas_dates = [rec.info["meas_date"] for rec in recordings]
    meas_dates = [_normalize_meas_date(d) for d in raw_meas_dates]

    # Check for missing measurement dates
    has_missing = any(d is None for d in meas_dates)
    if has_missing:
        if on_missing_meas_date == "raise":
            raise ValueError(
                "One or more recordings have missing measurement dates (meas_date=None). "
                "Cannot establish temporal order. Use on_missing_meas_date='warn' or 'ignore' to concatenate in input order."
            )
        elif on_missing_meas_date == "warn":
            warnings.warn(
                "One or more recordings have missing measurement dates (meas_date=None). "
                "Concatenating in input order; measurement date validation and temporal sorting will be skipped."
            )
        # For both 'warn' and 'ignore', skip the date-based validation and sort by input order
        copies = []
        for rec in recordings:
            copies.append(rec.copy())
        concatenated = mne.concatenate_raws(copies)
        return concatenated

    # All dates are present; extract measurement days for validation
    meas_days = [
        d.date() if hasattr(d, "date") and d is not None else None for d in meas_dates
    ]
    if len(set(meas_days)) > 1:
        msg = f"Measurement days are not uniform: {meas_days} (full datetimes: {meas_dates})"
        if on_mismatch == "raise":
            raise ValueError(msg)
        elif on_mismatch == "warn":
            warnings.warn(msg)

    # Sort recordings by measurement date
    indexed_recordings = [
        (idx, rec, meas_dates[idx]) for idx, rec in enumerate(recordings)
    ]
    sorted_recordings = sorted(
        indexed_recordings,
        key=lambda x: x[2] if x[2] is not None else datetime.datetime.min,
    )

    # Validate that gap between consecutive recordings is within max_gap
    for (idx1, rec1, date1), (idx2, rec2, date2) in zip(
        sorted_recordings, sorted_recordings[1:]
    ):
        # Gap is the difference between the meas_date (date2) of the next recording (rec2)
        # and the last time point of the previous recording (rec1), offset by its meas_date (date1).
        rec1_duration_s = rec1.n_times / rec1.info["sfreq"]
        rec1_end_time = date1 + datetime.timedelta(seconds=rec1_duration_s)
        gap = (date2 - rec1_end_time).total_seconds()  # convert to seconds

        if gap > max_gap * 3600:  # convert hours to seconds
            msg = f"Gap between recordings {idx1} and {idx2} is greater than {max_gap} hours: {(gap / 3600):.2f} hours"
            if on_gap == "raise":
                raise ValueError(msg)
            elif on_gap == "warn":
                warnings.warn(msg)

    copies = []
    for _, rec, _ in sorted_recordings:
        copies.append(rec.copy())

    concatenated = mne.concatenate_raws(copies)

    return concatenated


def extract_signal(
    recording_data: "mne.io.BaseRaw",
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


def extract_channels(
    recording_data: "mne.io.BaseRaw",
    channel_names_mapping: dict[str, str] | None = None,
    channel_types_mapping: dict[str, list[str]] | None = None,
    channel_pos_mapping: dict[str, np.ndarray] | None = None,
) -> ArrayDict:
    """
    Extract channel metadata from an MNE Raw object, with support for custom channel name, type, and position mappings.

    This function returns a channel-level ArrayDict containing (at minimum) the unique channel IDs (`id`)
    and types (`type`) for each channel in the provided MNE Raw object. Optionally, it can also include
    the 3D channel locations (`pos`) and "bad" channel labels, depending on mappings and recording metadata.

    Channel name, type, and/or position dictionaries, if provided, can be used to override or map the values
    from the raw file using either original or renamed channel names (but not a mix).

    Args:
        recording_data: MNE Raw object containing the electrophysiological data and metadata.
        channel_names_mapping: Optional dictionary mapping original channel names to new names
            (e.g., {"EEG01": "Fp1"}). Ensures renaming is unique.
        channel_types_mapping: Optional dictionary mapping types (e.g., "eeg") to lists of channel names
            (e.g., {"eeg": ["C3", "C4"]}). See `_validate_channel_types_mapping` for remapping logic.
        channel_pos_mapping: Optional dictionary mapping channel names to 3D position numpy arrays.
            Falls back to using montage positions if not provided.

    Returns:
        ArrayDict containing channel information with fields:
            - id: np.ndarray of (new) channel identifiers.
            - type: np.ndarray of channel types (lowercased).
            - pos: (optional) np.ndarray of shape (n_channels, 3) giving 3D coordinates for each channel.
            - bad: (optional) np.ndarray[bool] mask for bad channels (from MNE info['bads']).

    Raises:
        ImportError: If MNE is not installed.
        TypeError: If input recording or mappings do not match expected types.
        ValueError: If mapping keys mix original and renamed names, or have duplicate new channel names.

    Examples:
        >>> from mne.io import read_raw_edf
        >>> raw = read_raw_edf("example.edf", preload=True)
        >>> metadata = extract_channels(raw)
        >>> print(metadata.keys())
        ['id', 'type', 'pos', 'bad']

        >>> # Remap channel names, types, and positions
        >>> name_map = {"EEG F3-M2": "F3", "EEG F4-M1": "F4"}
        >>> type_map = {"eeg": ["F3", "F4"]}
        >>> pos_map = {"F3": np.array([0.0, 0.7, 0.0]), "F4": np.array([0.6, 0.7, 0.0])}
        >>> metadata = extract_channels(raw, name_map, type_map, pos_map)
    """
    _check_mne_available("extract_channels")

    if not isinstance(recording_data, mne.io.BaseRaw):
        raise TypeError(
            f"recording_data must be a mne.io.BaseRaw object, got {type(recording_data).__name__}."
        )

    if channel_names_mapping is not None and not isinstance(
        channel_names_mapping, dict
    ):
        raise TypeError(
            f"channel_names_mapping must be a dictionary, got {type(channel_names_mapping).__name__}."
        )
    channel_names_mapping = _validate_channel_names_mapping(
        recording_data, channel_names_mapping
    )

    if channel_types_mapping is not None and not isinstance(
        channel_types_mapping, dict
    ):
        raise TypeError(
            f"channel_types_mapping must be a dictionary, got {type(channel_types_mapping).__name__}."
        )
    channel_types_mapping = _validate_channel_types_mapping(
        recording_data, channel_names_mapping, channel_types_mapping
    )

    if channel_pos_mapping is not None and not isinstance(channel_pos_mapping, dict):
        raise TypeError(
            f"channel_pos_mapping must be a dictionary, got {type(channel_pos_mapping).__name__}."
        )
    channel_pos_mapping = _validate_channel_pos_mapping(
        recording_data, channel_names_mapping, channel_pos_mapping
    )

    raw_ch_names = np.array(recording_data.ch_names, dtype="U")
    raw_ch_types = np.array(recording_data.get_channel_types(), dtype="U")

    # Apply channel name mapping
    channel_ids = np.array(
        [channel_names_mapping.get(ch_name, ch_name) for ch_name in raw_ch_names],
        dtype="U",
    )

    # Apply channel type mapping
    channel_types = np.array(
        [
            channel_types_mapping.get(
                ch_name, ch_type
            ).lower()  # to be compatible with MNE's channel type validation
            for ch_name, ch_type in zip(raw_ch_names, raw_ch_types)
        ],
        dtype="U",
    )

    # Apply channel position mapping
    channel_count = len(raw_ch_names)
    channel_pos = np.full((channel_count, 3), np.nan)
    if channel_pos_mapping is None:
        # Fallback to montage-based extraction if no channel_pos_mapping is provided
        try:
            montage = recording_data.get_montage()
            if montage is not None:
                channel_pos_mapping = montage.get_positions()["ch_pos"]
        except Exception as e:
            warnings.warn(f"Could not extract channel positions from montage: {e}")

    if channel_pos_mapping is not None:
        channel_pos = np.array(
            [
                channel_pos_mapping.get(ch_name, ch_pos)
                for ch_name, ch_pos in zip(raw_ch_names, channel_pos)
            ]
        )

    # Bad channel extraction
    bad_channels = recording_data.info.get("bads", [])
    if bad_channels:
        is_bad_channel = np.array(
            [ch_name in bad_channels for ch_name in raw_ch_names], dtype=bool
        )
    else:
        is_bad_channel = None

    # Extract channel fields
    channel_fields = {
        "id": channel_ids,
        "type": channel_types,
    }

    if np.any(~np.isnan(channel_pos)):
        channel_fields["pos"] = channel_pos

    if is_bad_channel is not None:
        channel_fields["bad"] = is_bad_channel

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

    all_in_renamed = ch_names_in_mapping.issubset(renamed_ch_names_set)
    all_in_original = ch_names_in_mapping.issubset(original_ch_names_set)

    if all_in_renamed:
        return renamed_ch_names
    elif all_in_original:
        return original_ch_names
    else:
        raise ValueError(
            f"Channel name mismatch: mapping keys must refer to either "
            f"all original channel names or all renamed channel names, but not a mix. "
            f"Mapping keys: {sorted(ch_names_in_mapping)}. "
            f"Renamed channel names: {sorted(renamed_ch_names_set)}. "
            f"Original channel names: {sorted(original_ch_names_set)}."
        )


def _validate_channel_names_mapping(
    raw_data: "mne.io.BaseRaw",
    channel_names_mapping: dict[str, str] | None = None,
) -> dict[str, str]:
    """Validate and return a channel name mapping.

    Returns identity map (each name maps to itself) if mapping is None.
    Otherwise validates that all mapping keys exist in raw channel names
    and detects ambiguous mappings (e.g., {"A": "B", "B": "A"}).

    Args:
        raw_data: MNE Raw object containing original channel names.
        channel_names_mapping: Optional dict mapping original names to new names.

    Returns:
        Validated mapping dict or identity map if input is None.

    Raises:
        ValueError: If any mapping keys are not present in the raw data channel names,
            if the mapping introduces ambiguous swaps (e.g., {"A": "B", "B": "A"}),
            or if the resulting mapped channel names are not unique.
    """
    raw_ch_names = np.array(raw_data.ch_names, dtype="U")

    if channel_names_mapping is None:
        return {ch_name: ch_name for ch_name in raw_ch_names}

    raw_ch_names_set = set(raw_ch_names)
    mapping_keys_set = set(channel_names_mapping.keys())
    mapping_values_set = set(channel_names_mapping.values())

    if not mapping_keys_set.issubset(raw_ch_names_set):
        missing_keys = mapping_keys_set - raw_ch_names_set
        raise ValueError(
            f"Channel names in the mapping are not present in the raw data: {missing_keys}"
        )

    # Detect ambiguous mappings where a key also appears as a value with different order
    if mapping_keys_set & mapping_values_set:
        key_idx = {
            ch_name: idx for idx, ch_name in enumerate(channel_names_mapping.keys())
        }
        value_idx = {
            ch_name: idx for idx, ch_name in enumerate(channel_names_mapping.values())
        }
        ambiguous = [
            ch_name
            for ch_name in mapping_keys_set
            if key_idx.get(ch_name) != value_idx.get(ch_name)
        ]
        if ambiguous:
            raise ValueError(
                f"Ambiguous channel name mapping detected: {ambiguous}. Keys and values overlap or swap, e.g. {{'A': 'B', 'B': 'A'}}. Use unique, non-overlapping names."
            )

    # Check for duplicate channel names in channel_names_mapping
    if len(mapping_keys_set) != len(mapping_values_set):
        duplicates = [
            ch_name
            for ch_name in set(channel_names_mapping.values())
            if list(channel_names_mapping.values()).count(ch_name) > 1
        ]
        raise ValueError(
            f"Duplicate channel names in channel_names_mapping detected: {duplicates}. "
            f"Ensure that your channel name mapping creates unique identifiers."
        )

    return channel_names_mapping


def _validate_channel_types_mapping(
    raw_data: "mne.io.BaseRaw",
    channel_names_mapping: dict[str, str] | None = None,
    channel_types_mapping: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    """Validate and return a channel type mapping.

    Returns a mapping of original channel names to their types. If no type
    mapping is provided, returns the types from the raw data. If a type
    mapping is provided, it resolves whether the mapping keys refer to
    original or renamed channel names and remaps accordingly.

    Args:
        raw_data: MNE Raw object containing channel names and types.
        channel_names_mapping: Optional channel name remapping (used for resolution).
        channel_types_mapping: Optional dict mapping type (str) to list of channel names.

    Returns:
        Dict mapping original channel names to types.

    Raises:
        ValueError: If mapping keys are not consistent (mixed original/renamed).
    """
    raw_ch_names = np.array(raw_data.ch_names, dtype="U")
    raw_ch_types = np.array(raw_data.get_channel_types(), dtype="U")

    if channel_types_mapping is None:
        return {
            ch_name: ch_type for ch_name, ch_type in zip(raw_ch_names, raw_ch_types)
        }

    # Build a lookup from channel names to types (from type_mapping)
    ch_type_lookup = {
        ch_name: ch_type
        for ch_type, ch_list in channel_types_mapping.items()
        for ch_name in ch_list
    }

    # Determine if mapping keys refer to original or renamed names
    ch_names_in_type_mapping = set(ch_type_lookup.keys())
    renamed_ch_names = (
        np.array([channel_names_mapping.get(ch, ch) for ch in raw_ch_names], dtype="U")
        if channel_names_mapping
        else raw_ch_names
    )

    # Resolve which namespace the mapping uses
    ch_names_for_type_mapping = _resolve_channel_names_for_mapping(
        original_ch_names=raw_ch_names,
        renamed_ch_names=renamed_ch_names,
        ch_names_in_mapping=ch_names_in_type_mapping,
    )

    # If mapping used renamed names, convert back to original names
    if (
        not np.array_equal(ch_names_for_type_mapping, raw_ch_names)
        and channel_names_mapping
    ):
        inv_name_mapping = {v: k for k, v in channel_names_mapping.items()}
        ch_type_lookup = {
            inv_name_mapping.get(ch_name, ch_name): ch_type
            for ch_name, ch_type in ch_type_lookup.items()
        }

    return ch_type_lookup


def _validate_channel_pos_mapping(
    raw_data: "mne.io.BaseRaw",
    channel_names_mapping: dict[str, str] | None = None,
    channel_pos_mapping: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray] | None:
    """Validate and return a channel position mapping.

    Returns a mapping of original channel names to their positions (x, y, z).
    If no position mapping is provided, returns an empty dict. If a position
    mapping is provided, it resolves whether the mapping keys refer to
    original or renamed channel names and remaps accordingly.

    Args:
        raw_data: MNE Raw object containing channel names.
        channel_names_mapping: Optional channel name remapping (used for resolution).
        channel_pos_mapping: Optional dict mapping channel names to position arrays.

    Returns:
        Dict mapping original channel names to position arrays (shape: (3,)).

    Raises:
        ValueError: If mapping keys are not consistent (mixed original/renamed).
    """
    raw_ch_names = np.array(raw_data.ch_names, dtype="U")

    if channel_pos_mapping is None:
        return None

    # Determine if mapping keys refer to original or renamed names
    ch_names_in_pos_mapping = set(channel_pos_mapping.keys())
    renamed_ch_names = (
        np.array([channel_names_mapping.get(ch, ch) for ch in raw_ch_names], dtype="U")
        if channel_names_mapping
        else raw_ch_names
    )

    # Resolve which namespace the mapping uses
    ch_names_for_pos_mapping = _resolve_channel_names_for_mapping(
        original_ch_names=raw_ch_names,
        renamed_ch_names=renamed_ch_names,
        ch_names_in_mapping=ch_names_in_pos_mapping,
    )

    # If mapping used renamed names, convert back to original names
    if (
        not np.array_equal(ch_names_for_pos_mapping, raw_ch_names)
        and channel_names_mapping
    ):
        inv_name_mapping = {v: k for k, v in channel_names_mapping.items()}
        return {
            inv_name_mapping.get(ch_name, ch_name): pos
            for ch_name, pos in channel_pos_mapping.items()
        }

    return channel_pos_mapping
