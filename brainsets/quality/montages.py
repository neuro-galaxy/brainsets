from typing import Union
from collections import OrderedDict

import mne

from .native_montages import make_tuh_montage

MNE_MONTAGE_NAMES = mne.channels.get_builtin_montages()
NATIVE_MONTAGES = {
    "tuh": make_tuh_montage,
}
VALID_MONTAGE_NAMES = MNE_MONTAGE_NAMES + list(NATIVE_MONTAGES.keys())


def make_montage(
    montage_name: Union[str, None] = None, ch_names: Union[list[str], None] = None
) -> mne.channels.DigMontage:
    """Create a montage from a predefined set or custom channel names.
    This function creates an MNE DigMontage object either from a standard/native montage
    name or from a list of custom channel names.

    Parameters
    ----------
    montage_name : str or None
        Name of the standard montage to use. Must be one of the valid montage names
        defined in VALID_MONTAGE_NAMES. If None, ch_names must be provided.
    ch_names : list[str] or None
        List of channel names to create a custom montage. If provided, montage_name
        is ignored.

    Returns
    -------
    mne.channels.DigMontage
        The digital montage object with the specified configuration.
    Raises
    ------

    ValueError
        If montage_name is not in VALID_MONTAGE_NAMES and ch_names is None.
    Examples
    --------
    >>> montage = make_montage(montage_name="standard_1020")
    >>> custom_montage = make_montage(ch_names=["Fp1", "Fp2", "C3", "C4"])
    """
    if ch_names is not None:
        return make_custom_montage(ch_names)
    elif montage_name in VALID_MONTAGE_NAMES:
        return make_standard_montage(montage_name)
    else:
        raise ValueError(
            f"Invalid montage_name '{montage_name}'. It must be one of {VALID_MONTAGE_NAMES} or ch_names should be provided."
        )


def make_custom_montage(
    ch_names: Union[list[str], None] = None
) -> mne.channels.DigMontage:
    """Create a custom EEG montage from a list of channel names.

    This function creates a custom montage by mapping channel names to their standard
    positions in the 10-05 system. It preserves only the channels that exist in both
    the input list and the standard 10-05 system.

    Parameters
    ----------
    ch_names : list of str
        List of channel names to include in the custom montage.

    Returns
    -------
    mne.channels.DigMontage
        Custom digital montage containing only the specified channels with their
        corresponding positions from the standard 10-05 system, including fiducial
        points (nasion, LPA, RPA) and head shape points.

    Notes
    -----
    The function uses the standard_1005 montage as reference for channel positions
    and fiducial points. Channels in ch_names that don't exist in the 10-05 system
    will be excluded from the resulting montage.
    """
    standard_1005 = mne.channels.make_standard_montage("standard_1005")
    ch_pos_1005 = standard_1005.get_positions()["ch_pos"]
    ch_pos_custom = {ch: ch_pos_1005[ch] for ch in ch_names if ch in ch_pos_1005}

    custome_montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos_custom,
        nasion=standard_1005.get_positions()["nasion"],
        lpa=standard_1005.get_positions()["lpa"],
        rpa=standard_1005.get_positions()["rpa"],
        hsp=standard_1005.get_positions()["hsp"],
        hpi=standard_1005.get_positions()["hpi"],
        coord_frame="head",
    )

    return custome_montage


def make_standard_montage(
    montage_name: Union[str, None] = None,
) -> mne.channels.DigMontage:
    """Return a standard montage either from MNE or from native montages.

    This function creates a standard EEG montage either from MNE's built-in montages
    or from a predefined set of native montages.

    Parameters
    ----------
    montage_name : str
        Name of the montage to create. Can be either an MNE standard montage name
        or a custom native montage name.

    Returns
    -------
    mne.channels.DigMontage
        The digital montage object containing channel positions.

    Notes
    -----
    The function first checks if the montage name exists in MNE's standard montages.
    If not found, it looks up in the NATIVE_MONTAGES dictionary.

    Examples
    --------
    >>> montage = make_standard_montage('standard_1005')
    >>> montage = make_standard_montage('custom_64')
    """
    if montage_name in MNE_MONTAGE_NAMES:
        return mne.channels.make_standard_montage(montage_name)
    else:
        return NATIVE_MONTAGES[montage_name]
