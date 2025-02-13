import numpy as np
from scipy import signal

from typing import Union


# utils
def moving_average(
    data: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Computes the moving average of a 2D array along rows.

    This function applies a moving average filter to each row of the input data
    using a specified window size. The convolution is performed with mode="same",
    which means the output array has the same length as the input array.

    Parameters
    ----------
    data : numpy.ndarray
        2D input array where each row will be filtered
    window_size : int
        Size of the moving average window

    Returns
    -------
    numpy.ndarray
        Array of same shape as input, containing the moving averages

    Raises
    ------
    ValueError
        If input data is not a 2D array

    Examples
    --------
    ```
    >>> data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> moving_average(data, 3)
    array([[1.33333333, 2.        , 3.        , 4.        , 4.66666667],
           [6.33333333, 7.        , 8.        , 9.        , 9.66666667]])
    ```
    """
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array")

    return np.apply_along_axis(
        lambda row: np.convolve(row, np.ones(window_size) / window_size, mode="same"),
        axis=1,
        arr=data,
    )


def min_max_norm(
    data: np.ndarray,
    min_val: Union[float, None] = None,
    max_val: Union[float, None] = None,
    clip: bool = False,
) -> np.ndarray:
    """
    Normalize the data to a given range [min_val, max_val].

    Parameters
    ----------
    data : np.ndarray
        Input data to be normalized.
    min_val : float, optional
        Minimum value of the normalized data. If None, the minimum value of the input data is used.
    max_val : float, optional
        Maximum value of the normalized data. If None, the maximum value of the input data is used.
    clip : bool, optional
        If True, clip the normalized data to the range [0, 1].

    Returns
    -------
    np.ndarray
        Normalized data.
    """
    if min_val is None:
        min_val = np.min(data)

    if max_val is None:
        max_val = np.max(data)

    if max_val == min_val:
        raise ValueError("max_val and min_val cannot be equal")

    data = (data - min_val) / (max_val - min_val)

    if clip:
        data = np.clip(data, 0, 1)

    return data


# metrics
def compute_hilbert_features(
    signal_data: np.ndarray,
    sfreq: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute Hilbert transform features from a signal.
    This function calculates three key features using the Hilbert transform:
    instantaneous amplitude (envelope), instantaneous phase, and instantaneous frequency.

    Args:
        signal_data : Input signal array with shape (n_channels, n_timepoints)
        sfreq : Sampling frequency of the signal in Hz

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]
        Contains three numpy arrays:
        - amplitude : np.ndarray
            Instantaneous amplitude (envelope) of the signal
        - phase : np.ndarray
            Instantaneous phase of the signal (unwrapped)
        - frequency : np.ndarray
            Instantaneous frequency of the signal in Hz

    Notes
    -----
    The instantaneous frequency is computed as the derivative of the unwrapped phase
    and is padded at the end to match the original signal length.
    """

    # Compute analytic signal using Hilbert transform
    analytic_signal = signal.hilbert(signal_data)

    # Extract instantaneous amplitude (envelope)
    amplitude = np.abs(analytic_signal)

    # Extract instantaneous phase
    phase = np.unwrap(np.angle(analytic_signal))

    # Compute instantaneous frequency (derivative of phase)
    frequency = np.diff(phase) / (2.0 * np.pi) * sfreq

    # Pad frequency array to match original size
    frequency = np.pad(
        frequency,
        (
            (0, 0),
            (0, 1),
        ),
        "edge",
    )

    return amplitude, phase, frequency


def hilbert_distance(
    raw_signal: np.ndarray,
    reconstructed_signal: np.ndarray,
    sfreq: float,
    smooth: bool = True,
    norm: bool = True,
) -> dict[str, np.ndarray]:
    r"""Compute similarity scores between raw and reconstructed signals.
    This function calculates various similarity metrics between original and reconstructed
    signals based on their Hilbert transform features (amplitude, phase, and frequency).
    It returns a composite quality score along with individual similarity measures.

    Args:
        raw_signal: The original input signal.
        reconstructed_signal: The reconstructed/processed signal to compare against the raw signal.
        sfreq: Sampling frequency of the signals in Hz.
        smooth_scores: Whether to apply smoothing to the composite score (default is True).

    Returns:
        dict
        Dictionary containing the following keys:
            - 'composite_score': Overall quality score (weighted combination of all metrics)
            - 'amplitude_similarity': Similarity score based on signal amplitudes (0-1)
            - 'phase_similarity': Similarity score based on signal phases (0-1)
            - 'frequency_similarity': Similarity score based on instantaneous frequencies (0-1)
            - 'raw_amplitude': Amplitude values of the raw signal
            - 'reconstructed_amplitude': Amplitude values of the reconstructed signal

    Notes
    -----
    The composite score is calculated using default weights:
        - Amplitude: 0.4
        - Phase: 0.4
        - Frequency: 0.2
    All similarity scores are normalized to range from 0 (completely different) to
    1 (identical signals).
    """
    # Compute Hilbert features for both signals
    raw_amp, raw_phase, raw_freq = compute_hilbert_features(raw_signal, sfreq)
    recon_amp, recon_phase, recon_freq = compute_hilbert_features(
        reconstructed_signal, sfreq
    )

    # Compute various similarity metrics

    # 1. Amplitude similarity (normalized between 0 and 1)
    amp_diff = np.abs(raw_amp - recon_amp)
    amp_similarity = 1 / (1 + amp_diff)

    # 2. Phase coherence
    phase_diff = np.abs(raw_phase - recon_phase)
    phase_coherence = np.cos(
        phase_diff
    )  # Will be 1 for perfect match, -1 for opposite phase
    phase_similarity = (phase_coherence + 1) / 2  # Normalize to 0-1 range

    # 3. Frequency similarity
    freq_diff = np.abs(raw_freq - recon_freq)
    freq_similarity = 1 / (1 + freq_diff)

    # 4. Compute composite quality score
    # You can adjust these weights based on what's most important for your analysis
    weights = {"amplitude": 0.45, "phase": 0.1, "frequency": 0.45}

    composite_score = (
        weights["amplitude"] * amp_similarity
        + weights["phase"] * phase_similarity
        + weights["frequency"] * freq_similarity
    )

    # 5. Apply light smoothing to reduce noise
    if smooth:
        composite_score = moving_average(composite_score, window_size=5)

    if norm:
        composite_score = min_max_norm(
            composite_score, min_val=0.4, max_val=1.0, clip=True
        )

    # return {
    #     'composite_score': composite_score,
    #     # 'amplitude_similarity': amp_similarity,
    #     # 'phase_similarity': phase_similarity,
    #     # 'frequency_similarity': freq_similarity,
    #     # 'raw_amplitude': raw_amp,
    #     # 'reconstructed_amplitude': recon_amp
    # }
    return composite_score


def rmse(
    raw_signal: np.ndarray,
    reconstructed_signal: np.ndarray,
    smooth: bool = True,
    norm: bool = True,
) -> np.ndarray:
    """
    Computes the Root Mean Squared Error (RMSE) between a raw and reconstructed EEG signal,
    then smooths the RMSE values using a moving average filter.

    Parameters:
    raw_signal (np.ndarray): The original EEG signal.
    reconstructed_signal (np.ndarray): The reconstructed EEG signal.
    window_size (int, optional): The size of the moving average window for smoothing. Default is 10.

    Returns:
    np.ndarray: The smoothed RMSE values over the signal.
    """
    if raw_signal.shape != reconstructed_signal.shape:
        raise ValueError("Raw and reconstructed signals must have the same shape.")

    rmse = np.sqrt((raw_signal - reconstructed_signal) ** 2)

    if smooth:
        rmse = moving_average(rmse, window_size=10)

    if norm:
        rmse = min_max_norm(rmse, min_val=0.0, max_val=1e-4, clip=True)

    return 1 - rmse
