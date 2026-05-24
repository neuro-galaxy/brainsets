"""Signal processing functions. Inspired by Stavisky et al. (2015).

https://dx.doi.org/10.1088/1741-2560/12/3/036009
"""

_functions = [
    "downsample_wideband",
    "extract_bands",
]

__all__ = _functions

from typing import List, Tuple

import numpy as np
import tqdm
from scipy import signal

from temporaldata import Data, IrregularTimeSeries, ArrayDict


def downsample_wideband(
    wideband: np.ndarray,
    timestamps: np.ndarray,
    wideband_Fs: float,
    lfp_Fs: float = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsample wideband signal to LFP sampling rate.
    """
    assert wideband.shape[0] == timestamps.shape[0], "Time should be first dimension."
    # Decimate by a factor of 4
    dec_factor = 4
    if wideband.shape[0] % dec_factor != 0:
        wideband = wideband[: -(wideband.shape[0] % dec_factor), :]
        timestamps = timestamps[: -(timestamps.shape[0] % dec_factor)]
    wideband = wideband.reshape(-1, dec_factor, wideband.shape[1])
    wideband = wideband.mean(axis=1)

    timestamps = timestamps[::dec_factor]

    nyq = 0.5 * wideband_Fs / dec_factor  # Nyquist frequency
    cutoff = 0.333 * lfp_Fs  # remove everything above 170 Hz.
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(4, normal_cutoff, btype="low", analog=False, output="ba")

    # Interpolation to achieve the desired sampling rate
    t_new = np.arange(timestamps[0], timestamps[-1], 1 / lfp_Fs)
    lfp = np.zeros((len(t_new), wideband.shape[1]))
    for i in range(wideband.shape[1]):
        # We do this one channel at a time to save memory.
        broadband_low = signal.filtfilt(b, a, wideband[:, i], axis=0)
        lfp[:, i] = np.interp(t_new, timestamps, broadband_low)

    return lfp, t_new


def extract_bands(
    lfps: np.ndarray, ts: np.ndarray, Fs: float = 1000, notch: float = 60
) -> Tuple[np.ndarray, np.ndarray, List]:
    """Extract bands from LFP

    We prefer to extract bands from the LFP upstream rather than downstream, because
    it can be difficult to estimate e.g. the phase of low-frequency LFPs from
    short segments.

    We use the proposed bands from Stravisky et al. (2015), but we use the MNE toolbox
    rather than straight scipy signal.
    """
    try:
        import mne
    except ImportError:
        raise ImportError(
            "This function requires the MNE library which you can install with "
            "`pip install mne`"
        )

    target_Fs = 50
    assert (
        Fs % target_Fs == 0
    ), "Sampling rate must be a multiple of the target frequency"

    assert lfps.shape[0] == ts.shape[0], "Time should be first dimension."
    info = mne.create_info(
        ch_names=lfps.shape[1], sfreq=Fs, ch_types=["eeg"] * lfps.shape[1]
    )
    data = mne.io.RawArray(lfps.T, info)
    data = data.notch_filter(np.arange(notch, notch * 5 + 1, notch), n_jobs=4)

    filtered = []
    band_names = ["delta", "theta", "alpha", "beta", "gamma", "lmp"]
    bands = [(1, 4), (3, 10), (12, 23), (27, 38), (50, 300)]
    for band_low, band_hi in bands:
        band = data.copy().filter(band_low, band_hi, fir_design="firwin", n_jobs=4)
        band = band.apply_function(lambda x: x**2, n_jobs=4)

        band = band.filter(18, None, fir_design="firwin", n_jobs=4)
        # It seems resample overwrites the original data, so we copy it first.
        band = band.resample(target_Fs, npad="auto", n_jobs=4)

        filtered.append(band.get_data().T)

    lmp = data.copy().filter(0.1, 20, fir_design="firwin", n_jobs=4)
    lmp = lmp.resample(target_Fs, npad="auto", n_jobs=4)
    filtered.append(lmp.get_data().T)

    ts = ts[int(Fs / target_Fs / 2) :: int(Fs / target_Fs)]
    stacked = np.stack(filtered, axis=2)

    # There can be off by one errors.
    if stacked.shape[0] != len(ts):
        stacked = stacked[: len(ts), :, :]

    return stacked, ts, band_names
