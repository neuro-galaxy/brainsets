from typing import Union
from collections import OrderedDict

import mne

def make_tuh_montage():
    """Create a custom montage for the TUH (Temple University Hospital) EEG dataset.

    This function creates a montage that combines the standard 10-05 electrode positions
    with T1 and T2 positions from the standard postfixed montage. The resulting montage
    includes all standard 10-05 positions plus the temporal electrodes T1 and T2.

    Returns
    -------
    mne.channels.DigMontage
        A custom digital montage containing standard 10-05 positions plus T1 and T2,
        with channel names capitalized. The montage includes fiducial points (nasion, LPA, RPA)
        and head shape points.

    Notes
    -----
    The function performs the following steps:
    1. Loads the standard 10-05 montage
    2. Gets T1 and T2 positions from standard_postfixed montage
    3. Combines these positions
    4. Standardizes channel naming (capitalizes first letter)
    5. Creates a new DigMontage with combined positions
    """
    # Make custom montage for TUH dataset
    # Load the standard_1005 montage
    standard_1005 = mne.channels.make_standard_montage("standard_1005")

    # Load the standard_postfixed montage to find T1 and T2 positions
    standard_postfixed = mne.channels.make_standard_montage("standard_postfixed")

    # Extract T1 and T2 positions from standard_postfixed
    t1_pos = standard_postfixed.get_positions()["ch_pos"]["T1"]
    t2_pos = standard_postfixed.get_positions()["ch_pos"]["T2"]

    # Combine standard_1005 positions with T1 and T2
    pos = standard_1005.get_positions()

    ch_pos = pos["ch_pos"]
    ch_pos["T1"] = t1_pos
    ch_pos["T2"] = t2_pos

    ch_pos = OrderedDict({ch.lower().capitalize(): pos for ch, pos in ch_pos.items()})

    # Now create the new montage
    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        nasion=pos["nasion"],
        lpa=pos["lpa"],
        rpa=pos["rpa"],
        hsp=pos["hsp"],
        hpi=pos["hpi"],
        coord_frame=pos["coord_frame"],
    )

    return montage