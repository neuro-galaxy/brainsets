_functions = [
    "calculate_sampling_rate",
    "fill_gappy_timeseries",
]

__all__ = _functions

from typing import Iterable
import numpy as np


def calculate_sampling_rate(timestamps: np.ndarray, rtol: float = 1e-3) -> float:
    """Calculates median sampling rate from an array of timestamps.

    Args:
        timestamps: 1D array of timestamps in seconds, expected to be monotonically increasing.
        rtol: Maximum allowed relative variation in sampling interval, defined as
            (max_diff - min_diff) / median_diff. Defaults to 1e-3.

    Returns:
        float: Sampling rate in Hz.

    Raises:
        ValueError: If fewer than 2 timestamps are provided.
        ValueError: If the timestamps are not strictly monotonically increasing.
        ValueError: If the timestamps are not uniformly sampled within the given relative tolerance.
    """

    if timestamps.ndim != 1:
        raise ValueError(
            f"Timestamps must be a 1D array, got {timestamps.ndim}D array with shape {timestamps.shape}"
        )

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
    if relative_variation > rtol:
        raise ValueError(
            f"Timestamps are not uniformly sampled (relative variation={relative_variation:.2e} >= rtol={rtol}). "
            "Use IrregularTimeSeries to store the data."
        )

    return 1.0 / dt


def fill_gappy_timeseries(
    timestamps: np.ndarray,
    values: np.ndarray | Iterable[np.ndarray],
    sampling_rate: float | None = None,
    gap_value: float = np.nan,
    rtol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray | list[np.ndarray]]:
    """Fill gaps in an regular-but-gappy time series

    Fills the gaps in 'almost regular' time series which has gaps in
    the sampling times.

    Args:
        timestamps: Array of sampling timestamps; shape :math:`(T,)`
        value: Either a single array of values with shape :math:`(T, ...)`,
            or a list of arrays with the first dimension having the same size
            as timestamps.
        sampling_rate: The sampling rate of the time series. If None,
            sampling rate is calculated as the median time difference.
            (default :obj:`None`)
        gap_value: Value to fill in the gaps (default :obj:`np.nan`)

    Returns:
        timestamps, values
    """

    if timestamps.ndim != 1:
        raise ValueError(
            f"Timestamps must be a 1D array, got {timestamps.ndim}D array with shape {timestamps.shape}"
        )

    if len(timestamps) < 2:
        raise ValueError(
            f"This function needs at least 2 timestamps, got {timestamps.size}"
        )

    if not (np.diff(timestamps) > 0).all():
        raise ValueError("Input timestamps are not monotonic")

    if sampling_rate is None:
        # use mode time difference
        # numpy does not have a direct mode() function
        diffs = np.diff(timestamps)
        uniq_diffs, counts = np.unique(diffs, return_counts=True)
        mode_diff = uniq_diffs[np.argmax(counts)]
        sampling_rate = 1.0 / mode_diff

    start_time, end_time = timestamps[0], timestamps[-1]
    rel_ts = timestamps - start_time
    clean_time_idx = np.round(rel_ts * sampling_rate).astype(int)

    # Check for rtol
    relative_variation = clean_time_idx - (rel_ts * sampling_rate)
    max_relative_varation = np.max(np.abs(relative_variation))
    if max_relative_varation > rtol:
        raise ValueError(
            "Timestamps are not uniformly sampled "
            f"(max relative variation={max_relative_varation:.2e} >= rtol={rtol}). "
            f"Perhaps {sampling_rate=} is not a good sampling rate, or "
            f"this timeseries is inherently irregular."
        )

    num_timesteps = round((end_time - start_time) * sampling_rate) + 1

    def fill_gaps(values):
        if values.ndim > 1:
            shape = (num_timesteps, *values.shape[1:])
        else:
            shape = (num_timesteps,)

        ans = np.full(shape, fill_value=gap_value)
        ans[clean_time_idx] = values
        return ans

    clean_timestamps = fill_gaps(timestamps)

    if isinstance(values, np.ndarray):
        if len(values) != len(timestamps):
            raise ValueError(f"Shape mismatch: {len(timestamps)=} != {len(values)=}")
        clean_values = fill_gaps(values)

    elif isinstance(values, (list, tuple)):
        clean_values = []
        for i, v in enumerate(values):
            if len(v) != len(timestamps):
                raise ValueError(
                    f"Shape mismatch on {i}th values: {len(timestamps)=} != {len(values)=}"
                )
            clean_values.append(fill_gaps(v))

    else:
        raise ValueError(
            f"Incorrect type of `values`: {type(values)}."
            " This function accepts a single numpy array,"
            " or a list of numpy arrays."
        )

    return clean_timestamps, clean_values
