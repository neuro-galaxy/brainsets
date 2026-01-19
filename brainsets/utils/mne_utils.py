"""MNE channel mapping utilities.

This module provides functions to map modalities and rename channels
in MNE Raw objects, reusable by any EEG/MEG pipeline.
"""

import mne

MODALITY_TO_MNE_TYPE = {
    "EEG": "eeg",
    "SCALP": "eeg",
    "SCALP_EEG": "eeg",
    "EOG": "eog",
    "HEOG": "eog",
    "VEOG": "eog",
    "EMG": "emg",
    "MUSCLE": "emg",
    "ECG": "ecg",
    "EKG": "ecg",
    "CARDIAC": "ecg",
    "ECOG": "ecog",
    "SEEG": "seeg",
    "STEREO_EEG": "seeg",
    "RESP": "resp",
    "RESPIRATORY": "resp",
    "RESPIRATION": "resp",
    "BREATHING": "resp",
    "TMP": "temperature",
    "TEMP": "temperature",
    "THERM": "temperature",
    "TEMPERATURE": "temperature",
    "MEG": "meg",
    "MAG": "meg",
    "REF_MEG": "ref_meg",
    "MEG_REF": "ref_meg",
    "STIM": "stim",
    "STI": "stim",
    "EVENTS": "stim",
    "TRIGGER": "stim",
    "GSR": "gsr",
    "SKIN": "gsr",
    "GALVANIC": "gsr",
    "GALVANIC_SKIN_RESPONSE": "gsr",
    "BIO": "bio",
    "CHPI": "chpi",
    "DBS": "dbs",
    "DIPOLE": "dipole",
    "EXCI": "exci",
    "EYETRACK": "eyetrack",
    "FNIRS": "fnirs",
    "GOF": "gof",
    "GOODNESS_OF_FIT": "gof",
    "IAS": "ias",
    "SYST": "syst",
    "SYSTEM": "syst",
    "MISC": "misc",
}


def modality_to_mne_type(modality: str) -> str:
    """Map modality string to MNE channel type.

    Args:
        modality: Modality string (case-insensitive)

    Returns:
        MNE channel type string, defaults to "misc" if not found
    """
    return MODALITY_TO_MNE_TYPE.get((modality or "").upper(), "misc")


def rename_electrodes(raw: "mne.io.Raw", rename_map: dict[str, str]) -> None:
    """Rename channels in an MNE Raw object.

    Args:
        raw: The MNE Raw object to modify (modified in place)
        rename_map: Dictionary mapping old channel names to new names

    Example:
        >>> rename_map = {"PSG_F3": "F3", "PSG_F4": "F4"}
        >>> rename_electrodes(raw, rename_map)
    """
    if not rename_map:
        return

    current_ch_names = raw.ch_names
    valid_rename = {
        old: new for old, new in rename_map.items() if old in current_ch_names
    }

    if valid_rename:
        raw.rename_channels(valid_rename, allow_duplicates=False)


def set_channel_modalities(
    raw: "mne.io.Raw", modality_map: dict[str, list[str]]
) -> None:
    """Set channel types from a modality mapping.

    Args:
        raw: The MNE Raw object to modify (modified in place)
        modality_map: Dictionary mapping modality types to lists of channel names

    Example:
        >>> modality_map = {
        ...     "EEG": ["F3", "F4", "C3", "C4"],
        ...     "EOG": ["EOG_L", "EOG_R"],
        ... }
        >>> set_channel_modalities(raw, modality_map)
    """
    if not modality_map:
        return

    current_ch_names = set(raw.ch_names)
    type_dict = {}

    for modality, channel_names in modality_map.items():
        mne_type = modality_to_mne_type(modality)
        for ch_name in channel_names:
            if ch_name in current_ch_names:
                type_dict[ch_name] = mne_type

    if type_dict:
        raw.set_channel_types(type_dict)
