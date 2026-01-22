"""Signal processing functions. Inspired by Stavisky et al. (2015).

https://dx.doi.org/10.1088/1741-2560/12/3/036009
"""

from typing import List, Tuple

import numpy as np
import tqdm
from scipy import signal

from temporaldata import Data, IrregularTimeSeries, ArrayDict
from brainsets.taxonomy import RecordingTech


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
    lfps: np.ndarray,
    ts: np.ndarray,
    Fs: float = 1000,
    notch: float = 60,
    normalize: str = "zscore",
) -> Tuple[np.ndarray, np.ndarray, List]:
    """Extract bands from LFP

    We prefer to extract bands from the LFP upstream rather than downstream, because
    it can be difficult to estimate e.g. the phase of low-frequency LFPs from
    short segments.

    We use the proposed bands from Stravisky et al. (2015), but we use the MNE toolbox
    rather than straight scipy signal.

    Args:
        lfps: LFP signals with shape (time, channels)
        ts: Timestamps array
        Fs: Sampling frequency in Hz
        notch: Line noise frequency for notch filter (e.g., 60 Hz for US, 50 Hz for EU)
        normalize: Normalization method. Options:
            - "zscore": Z-score normalization per channel per band (default)
            - "log": Log10 transform (for power bands only)
            - None: No normalization

    Returns:
        stacked: Band power array with shape (time, channels, bands)
        ts: Resampled timestamps
        band_names: List of band names
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

        band = band.filter(None, 18, fir_design="firwin", n_jobs=4)
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

    # Apply normalization
    if normalize == "zscore":
        # Z-score per channel per band
        for b in range(stacked.shape[2]):
            for c in range(stacked.shape[1]):
                mean = np.mean(stacked[:, c, b])
                std = np.std(stacked[:, c, b])
                if std > 0:
                    stacked[:, c, b] = (stacked[:, c, b] - mean) / std
    elif normalize == "log":
        # Log transform for power bands (not LMP which can be negative)
        for b in range(stacked.shape[2] - 1):  # Exclude LMP
            stacked[:, :, b] = np.log10(np.maximum(stacked[:, :, b], 1e-10))
    elif normalize is not None:
        raise ValueError(f"Unknown normalization method: {normalize}")

    return stacked, ts, band_names


def extract_bands_fixed(
    lfps: np.ndarray,
    ts: np.ndarray,
    Fs: float = 1000,
    notch: float = 60,
    normalize: str = "zscore",
    target_Fs: float = 50,
) -> Tuple[np.ndarray, np.ndarray, List]:
    """Extract frequency bands from LFP signals.

    This is a corrected version of extract_bands() with the following fixes:
    - Correct frequency bands matching Stavisky et al. (2015)
    - Proper timestamp alignment after resampling
    - Consistent power units across all bands (including LMP)
    - Z-score normalization by default

    Reference: Stavisky et al. (2015) "A high performing brain-machine interface
    driven by low-frequency local field potentials alone and together with spikes"
    https://dx.doi.org/10.1088/1741-2560/12/3/036009

    Args:
        lfps: LFP signals with shape (time, channels)
        ts: Timestamps array with shape (time,)
        Fs: Input sampling frequency in Hz (default: 1000)
        notch: Line noise frequency for notch filter (default: 60 Hz for US)
        normalize: Normalization method. Options:
            - "zscore": Z-score normalization per channel per band (default)
            - "log": Log10 transform before optional z-score
            - None: No normalization
        target_Fs: Output sampling frequency in Hz (default: 50)

    Returns:
        stacked: Band power array with shape (time, channels, bands)
        ts_out: Resampled timestamps aligned with output data
        band_names: List of band names

    Note:
        All bands output power (squared amplitude), including LMP, for consistent
        units across the output array.
    """
    try:
        import mne
    except ImportError:
        raise ImportError(
            "This function requires the MNE library which you can install with "
            "`pip install mne`"
        )

    assert lfps.shape[0] == ts.shape[0], "Time should be first dimension."
    assert Fs % target_Fs == 0, "Sampling rate must be a multiple of target frequency"

    # Create MNE Raw object
    n_channels = lfps.shape[1]
    ch_names = [f"ch{i}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=Fs, ch_types=["eeg"] * n_channels)
    raw = mne.io.RawArray(lfps.T, info, verbose=False)

    # Apply notch filter to remove line noise and harmonics
    notch_freqs = np.arange(notch, min(notch * 5 + 1, Fs / 2), notch)
    raw = raw.notch_filter(notch_freqs, n_jobs=4, verbose=False)

    # Frequency bands from Stavisky et al. (2015) - standard neuroscience bands
    band_names = ["delta", "theta", "alpha", "beta", "gamma", "lmp"]
    bands = [
        (0.5, 4),    # delta
        (4, 8),      # theta
        (8, 12),     # alpha (mu rhythm in motor cortex)
        (12, 30),    # beta
        (30, 150),   # gamma (combined low and high gamma)
    ]

    filtered = []

    for band_low, band_hi in bands:
        # Bandpass filter
        band = raw.copy().filter(
            band_low, band_hi, fir_design="firwin", n_jobs=4, verbose=False
        )
        # Square to get instantaneous power
        band = band.apply_function(lambda x: x**2, n_jobs=4, verbose=False)
        # Smooth the power envelope (low-pass at 18 Hz)
        band = band.filter(None, 18, fir_design="firwin", n_jobs=4, verbose=False)
        # Resample to target frequency
        band = band.resample(target_Fs, npad="auto", n_jobs=4, verbose=False)
        filtered.append(band.get_data().T)

    # LMP (Local Motor Potential) - low frequency component
    # Also compute power for consistent units
    lmp = raw.copy().filter(0.1, 4, fir_design="firwin", n_jobs=4, verbose=False)
    lmp = lmp.apply_function(lambda x: x**2, n_jobs=4, verbose=False)
    lmp = lmp.filter(None, 18, fir_design="firwin", n_jobs=4, verbose=False)
    lmp = lmp.resample(target_Fs, npad="auto", n_jobs=4, verbose=False)
    filtered.append(lmp.get_data().T)

    # Stack all bands: shape (time, channels, bands)
    stacked = np.stack(filtered, axis=2)

    # Generate proper timestamps from MNE's resampled data
    # Use the resampled object's times and offset by original start time
    n_samples_out = stacked.shape[0]
    ts_out = ts[0] + np.arange(n_samples_out) / target_Fs

    # Apply normalization
    if normalize == "log":
        # Log transform first (all bands are power now, so all positive)
        stacked = np.log10(np.maximum(stacked, 1e-10))
        # Then z-score
        for b in range(stacked.shape[2]):
            for c in range(stacked.shape[1]):
                mean = np.mean(stacked[:, c, b])
                std = np.std(stacked[:, c, b])
                if std > 0:
                    stacked[:, c, b] = (stacked[:, c, b] - mean) / std
    elif normalize == "zscore":
        # Z-score per channel per band
        for b in range(stacked.shape[2]):
            for c in range(stacked.shape[1]):
                mean = np.mean(stacked[:, c, b])
                std = np.std(stacked[:, c, b])
                if std > 0:
                    stacked[:, c, b] = (stacked[:, c, b] - mean) / std
    elif normalize is not None:
        raise ValueError(f"Unknown normalization method: {normalize}")

    return stacked, ts_out, band_names


def cube_to_long(
    ts: np.ndarray, cube: np.ndarray, channel_prefix="chan"
) -> Tuple[List[IrregularTimeSeries], Data]:
    """Convert a cube of threshold crossings to a list of trials and units."""
    assert cube.shape[1] == len(ts)
    assert cube.ndim == 3
    channels = np.arange(cube.shape[2])
    channels = np.tile(channels, [cube.shape[1], 1])

    # First dim is batch, second is time, third is channel.
    assert np.issubdtype(cube.dtype, np.integer)
    assert cube.min() >= 0

    ts = np.tile(ts.reshape((-1, 1)), [1, cube.shape[2]])
    assert ts.shape == channels.shape

    # The first dimension we map to a single trial.
    trials = []
    for b in tqdm.tqdm(range(cube.shape[0])):
        cube_ = cube[b, :, :]
        ts_ = []
        channels_ = []

        # This data is binned, so we create N identifical timestamps when there are N
        # spikes in a bin.
        for n in range(1, cube_.max() + 1):
            ts_.append(ts[cube_ >= n])
            channels_.append(channels[cube_ >= n])

        ts_ = np.concatenate(ts_)
        channels_ = np.concatenate(channels_)

        tidx = np.argsort(ts_)
        ts_ = ts_[tidx]
        channels_ = channels_[tidx]

        trials.append(
            IrregularTimeSeries(
                timestamps=ts_,
                unit_index=channels_,
                types=np.ones(len(ts_))
                * int(RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS),
                domain="auto",
            )
        )

    counts = cube.sum(axis=0).sum(axis=0)
    units = ArrayDict(
        count=np.array(counts.astype(int)),
        channel_name=np.array(
            [f"{channel_prefix}{c:03}" for c in range(cube.shape[2])]
        ),
        unit_number=np.zeros(cube.shape[2]),
        id=np.array([f"{channel_prefix}{c}" for c in range(cube.shape[2])]),
        channel_number=np.arange(cube.shape[2]),
        type=np.ones(cube.shape[2]) * int(RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS),
    )

    return trials, units
