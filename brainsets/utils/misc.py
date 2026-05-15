_functions = [
    "calc_sampling_rate",
]

__all__ = _functions

import numpy as np


def calc_sampling_rate(timestamps: np.ndarray, rtol: float = 1e-3) -> float:
    """Calculate sampling rate from an array of timestamps.

    Args:
        timestamps: 1D array of timestamps in seconds, expected to be monotonically increasing.
        rtol: Maximum allowed relative variation in sampling interval, defined as
            (max_diff - min_diff) / median_diff. Defaults to 1e-3.

    Returns:
        float: Sampling rate in Hz.

    Raises:
        ValueError: If any of the following is true:
            - fewer than 2 timestamps are provided.
            - the timestamps are not strictly monotonically increasing.
            - the timestamps are not uniformly sampled within the given relative tolerance.
    """

    if timestamps.size < 2:
        raise ValueError(
            f"Need at least 2 timestamps to compute a sampling rate, got {timestamps.size}"
        )

    diffs = np.diff(timestamps)

    if np.any(diffs <= 0):
        raise ValueError(
            "Timestamps must be strictly monotonically increasing "
            "(found duplicate or out-of-order values)"
        )

    dt = np.median(diffs)
    relative_variation = np.abs((np.max(diffs) - np.min(diffs)) / dt)
    if relative_variation >= rtol:
        raise ValueError(
            f"Timestamps are not uniformly sampled (relative variation={relative_variation:.2e} >= rtol={rtol}). "
            "Use an Irregular TimeSeries to store the data."
        )

    return 1.0 / dt
