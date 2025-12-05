from functools import lru_cache
import json
import os
import mne
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Union
import logging

from brainsets.ds_wizard.context import CONTEXT_PATH


@lru_cache
def get_builtin_montages():
    return mne.channels.get_builtin_montages()


@lru_cache
def get_standard_mapping():
    return pd.read_csv(os.path.join(CONTEXT_PATH, "ch_names_updated.csv"))


@lru_cache
def get_biosemi_to_std_mapping():
    with open(os.path.join(CONTEXT_PATH, "biosemi_to_std.json")) as f:
        return json.load(f)


def names_to_standard_names(ch_names: Union[list[str], np.ndarray]) -> np.ndarray:
    """
    Map original channel names to standardized MNE channel names.

    Args:
        ch_names : list of str or np.ndarray
            List or array of original channel names to be mapped.

    Returns:
        np.ndarray
            Array of standardized MNE channel names.
    """
    name_to_std_name_map = dict(
        zip(get_standard_mapping()["ch_name"], get_standard_mapping()["std_ch_name"])
    )
    return np.array([name_to_std_name_map[ch_name] for ch_name in ch_names])


def names_to_standard_types(ch_names: Union[list[str], np.ndarray]) -> np.ndarray:
    """
    Map channel names to their standardized channel types.

    Args:
        ch_names : list of str or np.ndarray
            List or array of channel names to be mapped.

    Returns:
        np.ndarray
            Array of standardized channel types.
    """
    name_to_std_type_map = dict(
        zip(get_standard_mapping()["ch_name"], get_standard_mapping()["std_ch_type"])
    )
    return np.array([name_to_std_type_map[ch_name] for ch_name in ch_names])


def get_mne_montages_info() -> Dict[str, List[str]]:
    """
    Retrieve information about all built-in MNE montages.

    Returns:
        dict
            Dictionary mapping montage names to lists of channel names for each montage.
    """
    mne_montages: Dict[str, List[str]] = dict()
    for montage_name in get_builtin_montages():
        # Load the montage to get number of channels
        mont = mne.channels.make_standard_montage(montage_name)

        # Store montage info in dictionary
        mne_montages[montage_name] = mont.ch_names

    return mne_montages


def find_match_percentage_by_montage(
    eeg_ch_names: Union[list[str], np.ndarray],
) -> tuple[dict, dict]:
    """
    Calculate the percentage of input EEG channel names that match each MNE montage.

    For each MNE montage, computes:
      - The fraction of input channel names present in the montage (match_to_self).
      - The fraction of montage channel names present in the input (match_to_mne).

    Args:
        eeg_ch_names : list of str or np.ndarray
            List or array of EEG channel names to match against MNE montages.

    Returns:
        tuple of dict
            (match_to_self, match_to_mne)
            - match_to_self: dict mapping montage name to fraction of input channels present in the montage.
            - match_to_mne: dict mapping montage name to fraction of montage channels present in the input.

    Raises:
        TypeError
            If eeg_ch_names is not a list or numpy array.
        ValueError
            If eeg_ch_names is empty.
    """
    # check if eeg_ch_names is a list or numpy array
    if not (isinstance(eeg_ch_names, list) or isinstance(eeg_ch_names, np.ndarray)):
        raise TypeError("eeg_ch_names must be a list or numpy array.")

    # check if eeg_ch_names is a non-empty list
    if len(eeg_ch_names) == 0:
        raise ValueError(
            "No EEG channels provided. EEG channels should be a non-empty list."
        )

    # check for duplicates in eeg_ch_names
    if len(eeg_ch_names) != len(set(eeg_ch_names)):
        raise ValueError(
            "EEG channel names should be unique. Duplicate channel names are not allowed."
        )

    mne_montages = get_mne_montages_info()
    match_to_self = {}
    match_to_mne = {}

    # remap biosemi64 channels in case they are not already in the
    # standard MNE nomenclature
    if all(ch in get_biosemi_to_std_mapping().keys() for ch in eeg_ch_names):
        eeg_ch_names = [get_biosemi_to_std_mapping()[ch] for ch in eeg_ch_names]

    if len(eeg_ch_names) > 0:
        for montage_name, montage_ch_names in mne_montages.items():
            common_ch = np.intersect1d(eeg_ch_names, montage_ch_names)
            match_to_self[montage_name] = len(common_ch) / len(eeg_ch_names)
            match_to_mne[montage_name] = len(common_ch) / len(montage_ch_names)

    return match_to_self, match_to_mne


def get_standard_montage(
    eeg_ch_names: Union[list[str], np.ndarray],
    logger: logging.Logger = None,
) -> Tuple[str, float, list[str]]:
    """
    Identify the best-matching MNE montage for a given list of EEG channel names.

    The function compares the provided channel names to all standard MNE montages,
    prioritizing montages where all input channels are present (match_to_self = 1.0),
    and then by the coverage of the montage channels by the input (match_to_mne).

    Args:
        eeg_ch_names : list of str or np.ndarray
            List or array of EEG channel names to match.
        logger : logging.Logger, optional
            Logger for info/warning messages.

    Returns:
        best_montage : str
            Name of the best matching MNE montage, or "other" if no good match is found.
        match_percentage : float
            Fraction of input channels matched to the montage (match_to_self).
        unmatched_ch_names : list of str
            List of input channel names not found in the best montage.

    Raises:
        TypeError
            If eeg_ch_names is not a list or numpy array.
        ValueError
            If eeg_ch_names is empty.
    """
    # check if eeg_ch_names is a list or numpy array
    if not (isinstance(eeg_ch_names, list) or isinstance(eeg_ch_names, np.ndarray)):
        raise TypeError("eeg_ch_names must be a list or numpy array.")

    # check if eeg_ch_names is a non-empty list
    if len(eeg_ch_names) == 0:
        raise ValueError(
            "No EEG channels provided. EEG channels should be a non-empty list."
        )

    # check for duplicates in eeg_ch_names
    if len(eeg_ch_names) != len(set(eeg_ch_names)):
        raise ValueError(
            "EEG channel names should be unique. Duplicate channel names are not allowed."
        )

    match_to_self, match_to_mne = find_match_percentage_by_montage(eeg_ch_names)

    # If no matches at all, return "other"
    if not any(match_to_self.values()) and not any(match_to_mne.values()):
        best_montage = "other"
        match_to_self[best_montage] = 0.0
        match_to_mne[best_montage] = 0.0
        unmatched_ch_names = eeg_ch_names
        montage_ch_names = []
    else:
        # Find montages with perfect match_to_self (all input channels present)
        perfect_self = [m for m, v in match_to_self.items() if v == 1.0]
        if perfect_self:
            # If multiple, pick the one with highest match_to_mne (montage coverage)
            best_montage = max(perfect_self, key=lambda m: match_to_mne[m])
        else:
            # Otherwise, pick montage with highest match_to_self, break ties with match_to_mne
            best_self_val = max(match_to_self.values())
            best_candidates = [
                m for m, v in match_to_self.items() if v == best_self_val
            ]
            best_montage = max(best_candidates, key=lambda m: match_to_mne[m])

        # Get montage channel names for unmatched calculation
        montage_ch_names = get_mne_montages_info()[best_montage]

        # If the best montage is biosemi64, map channel names to standard MNE nomenclature
        if best_montage in ["biosemi16", "biosemi32", "biosemi64"]:
            biosemi_eeg_ch_names = [
                (
                    get_biosemi_to_std_mapping()[ch]
                    if ch in get_biosemi_to_std_mapping().keys()
                    else ch
                )
                for ch in eeg_ch_names
            ]
            unmatched_ch_names = np.setdiff1d(biosemi_eeg_ch_names, montage_ch_names)

            # Map unmatched channels to original nomenclature
            INVERSE_MAPPING = {v: k for k, v in get_biosemi_to_std_mapping().items()}
            unmatched_ch_names = np.array(
                [
                    INVERSE_MAPPING[ch] if ch in INVERSE_MAPPING.keys() else ch
                    for ch in unmatched_ch_names
                ]
            )
        else:
            unmatched_ch_names = np.setdiff1d(eeg_ch_names, montage_ch_names)

    if logger:
        logger.info(f"Best montage: {best_montage}")
        logger.info(f"Match percentage to self: {match_to_self[best_montage]:.3f}")
        logger.info(f"Match percentage to MNE: {match_to_mne[best_montage]:.3f}")
        logger.info(
            f"{len(eeg_ch_names)} channels in recording vs. {len(montage_ch_names)} in montage"
        )
        logger.info(
            f"{len(unmatched_ch_names)} unmatched channels: {unmatched_ch_names}"
        )

    return best_montage, match_to_self[best_montage], unmatched_ch_names


def get_all_montage_matches(
    eeg_ch_names: Union[list[str], np.ndarray],
) -> Dict[str, Dict[str, Union[float, List[str], Dict]]]:
    """
    Get all possible montage matches for a given list of EEG channel names.

    This function returns comprehensive information about how the input channels
    match against all available MNE built-in montages, including match percentages,
    electrode positions, and detailed statistics.

    Args:
        eeg_ch_names : list of str or np.ndarray
            List or array of EEG channel names to match against all montages.

    Returns:
        dict
            Dictionary with montage names as keys and detailed match information as values.
            Each montage entry contains:
            - 'match_to_input': fraction of input channels present in the montage
            - 'match_to_montage': fraction of montage channels present in the input
            - 'matched_channels': list of channels that match
            - 'unmatched_input_channels': input channels not in this montage
            - 'unmatched_montage_channels': montage channels not in input
            - 'montage_channels': all channels in this montage
            - 'positions': 3D coordinates of matched channels (if available)
            - 'total_input_channels': total number of input channels
            - 'total_montage_channels': total number of channels in this montage

    Raises:
        TypeError
            If eeg_ch_names is not a list or numpy array.
        ValueError
            If eeg_ch_names is empty or contains duplicates.
    """
    # Input validation (same as other functions)
    if not (isinstance(eeg_ch_names, list) or isinstance(eeg_ch_names, np.ndarray)):
        raise TypeError("eeg_ch_names must be a list or numpy array.")

    if len(eeg_ch_names) == 0:
        raise ValueError(
            "No EEG channels provided. EEG channels should be a non-empty list."
        )

    if len(eeg_ch_names) != len(set(eeg_ch_names)):
        raise ValueError(
            "EEG channel names should be unique. Duplicate channel names are not allowed."
        )

    # Get MNE montage information
    mne_montages = get_mne_montages_info()

    # Handle biosemi mapping
    original_ch_names = list(eeg_ch_names)
    mapped_ch_names = list(eeg_ch_names)
    if all(ch in get_biosemi_to_std_mapping().keys() for ch in eeg_ch_names):
        mapped_ch_names = [get_biosemi_to_std_mapping()[ch] for ch in eeg_ch_names]

    all_matches = {}

    # Process MNE montages
    for montage_name, montage_ch_names in mne_montages.items():
        # Convert to lowercase for case-insensitive comparison
        mapped_ch_names_lower = [ch.lower() for ch in mapped_ch_names]
        montage_ch_names_lower = [ch.lower() for ch in montage_ch_names]

        # Find matches using lowercase comparison
        matched_indices = []
        for i, ch_lower in enumerate(mapped_ch_names_lower):
            if ch_lower in montage_ch_names_lower:
                matched_indices.append(i)

        # Get original case matched channels
        matched_channels = [mapped_ch_names[i] for i in matched_indices]
        unmatched_input = [
            ch for i, ch in enumerate(mapped_ch_names) if i not in matched_indices
        ]

        # Find unmatched montage channels (case-insensitive)
        matched_montage_indices = []
        for i, ch_lower in enumerate(montage_ch_names_lower):
            if ch_lower in mapped_ch_names_lower:
                matched_montage_indices.append(i)
        unmatched_montage = [
            ch
            for i, ch in enumerate(montage_ch_names)
            if i not in matched_montage_indices
        ]

        match_to_input = len(matched_channels) / len(mapped_ch_names)
        match_to_montage = len(matched_channels) / len(montage_ch_names)

        # Get electrode positions for matched channels
        positions = {}
        try:
            montage = mne.channels.make_standard_montage(montage_name)
            montage_positions = montage.get_positions()["ch_pos"]
            positions = {
                ch: montage_positions[ch].tolist()
                for ch in matched_channels
                if ch in montage_positions
            }
        except Exception:
            positions = {}

        # Map back to original channel names if biosemi
        if all(ch in get_biosemi_to_std_mapping().keys() for ch in original_ch_names):
            inverse_mapping = {v: k for k, v in get_biosemi_to_std_mapping().items()}
            matched_channels = [inverse_mapping.get(ch, ch) for ch in matched_channels]
            unmatched_input = [inverse_mapping.get(ch, ch) for ch in unmatched_input]
            # Update positions keys to original names
            positions = {
                inverse_mapping.get(ch, ch): pos for ch, pos in positions.items()
            }

        all_matches[montage_name] = {
            "match_to_input": match_to_input,
            "match_to_montage": match_to_montage,
            "matched_channels": matched_channels,
            "unmatched_input_channels": unmatched_input,
            "unmatched_montage_channels": unmatched_montage,
            "montage_channels": montage_ch_names,
            "positions": positions,
            "total_input_channels": len(original_ch_names),
            "total_montage_channels": len(montage_ch_names),
        }

    return all_matches


def get_standard_ch_info(
    original_ch_names: list[str],
    logger: logging.Logger = None,
) -> Tuple[list[str], list[str], str]:
    """
    Standardize channel names and types, and determine the best-matching MNE montage.

    This function maps the original channel names to standard MNE channel names and types,
    and identifies the best-matching standard montage. If some EEG channels cannot be matched
    to a known montage, their type is set to "EEG-OTHER" and a warning is logged.

    Args:
        original_ch_names : list of str
            List of original channel names from the dataset/session.
        logger : logging.Logger, optional
            Logger instance for logging information and warnings.

    Returns:
        std_ch_names : np.ndarray
            Array of standardized MNE channel names.
        std_ch_types : np.ndarray
            Array of standardized channel types (e.g., "EEG", "EOG", etc.).
        std_montage : str
            Name of the best-matching standard MNE montage, or "other" if no good match is found.
    """
    # check if original_ch_names is a list or numpy array
    if not (
        isinstance(original_ch_names, list) or isinstance(original_ch_names, np.ndarray)
    ):
        raise TypeError("original_ch_names must be a list or numpy array.")

    # check if original_ch_names is a non-empty list
    if len(original_ch_names) == 0:
        raise ValueError(
            "No original channel names provided. Original channel names should be a non-empty list."
        )

    # check for duplicates in original_ch_names
    if len(original_ch_names) != len(set(original_ch_names)):
        raise ValueError(
            "Original channel names should be unique. Duplicate channel names are not allowed."
        )

    # get standard MNE channel names
    std_ch_names = names_to_standard_names(original_ch_names)

    # get standard channel types
    std_ch_types = names_to_standard_types(original_ch_names)

    # get standard MNE montage
    eeg_ch_names = std_ch_names[std_ch_types == "EEG"]

    if len(eeg_ch_names) > 0:
        std_montage, match_percentage, unmatch_ch_names = get_standard_montage(
            eeg_ch_names, logger=logger
        )

        # if there are unmatched channels, set std_ch_type
        # to EEG-OTHER for those channels
        if len(unmatch_ch_names) > 0:
            std_ch_types = np.array(
                [
                    ch_type if ch_name not in unmatch_ch_names else "EEG-OTHER"
                    for ch_name, ch_type in zip(std_ch_names, std_ch_types)
                ]
            )

            if logger is not None:
                logger.warning(
                    f"Setting standard ch_type to EEG-OTHER for unmatched EEG channels: {unmatch_ch_names}."
                )
    else:
        std_montage = "other"

    return std_ch_names, std_ch_types, std_montage


def get_all_electrode_names_to_montage_mapping() -> Dict[str, str]:
    """
    Get a dictionary which maps all electrode names to all the montages they belong to.

    Returns:
        dict
            Dictionary mapping electrode names to a comma-separated string of montage names
            that contain that electrode. For example: {'Fp1': 'standard_1020,standard_1005', ...}
    """
    mne_montages = get_mne_montages_info()
    electrode_to_montages = {}

    # Iterate through all montages and their electrode names
    for montage_name, electrode_names in mne_montages.items():
        for electrode_name in electrode_names:
            if electrode_name in electrode_to_montages:
                electrode_to_montages[electrode_name] += f",{montage_name}"
            else:
                electrode_to_montages[electrode_name] = montage_name

    return electrode_to_montages
