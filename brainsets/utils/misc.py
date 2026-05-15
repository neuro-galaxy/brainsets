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
        Sampling rate in Hz.

    Raises:
        ValueError: If the median interval between timestamps is <= 0, which indicates
            identical or non-monotonic timestamps.
        AssertionError: If the sampling interval is not consistent within the given relative tolerance.
    """
    diffs = np.diff(timestamps)
    dt = np.median(diffs)

    if dt <= 0:
        raise ValueError(
            f"Invalid timestamps: median interval is {dt}, "
            "timestamps may be identical or non-monotonic"
        )

    relative_variation = np.abs((np.max(diffs) - np.min(diffs)) / dt)
    assert relative_variation < rtol, (
        f"Timestamps are not uniformly sampled (relative variation={relative_variation:.2e} >= rtol={rtol}). "
        "Use an Irregular TimeSeries to store the data."
    )

    return 1.0 / dt
