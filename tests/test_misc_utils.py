import numpy as np
import pytest
from brainsets.utils.misc import calc_sampling_rate


def test_uniform_timestamps():
    timestamps = np.arange(100) / 1000.0  # 1000 Hz
    assert calc_sampling_rate(timestamps) == pytest.approx(1000.0, rel=1e-3)


def test_non_integer_sampling_rate():
    timestamps = np.arange(100) / 250.0  # 250 Hz
    assert calc_sampling_rate(timestamps) == pytest.approx(250.0, rel=1e-3)


def test_gapped_timestamps_raises():
    timestamps = np.arange(100) / 1000.0
    timestamps[50] += 0.1  # introduce a gap
    with pytest.raises(AssertionError):
        calc_sampling_rate(timestamps)


def test_non_monotonic_timestamps_raises():
    timestamps = np.zeros(10)
    with pytest.raises(ValueError):
        calc_sampling_rate(timestamps)


def test_custom_tolerance():
    rng = np.random.default_rng(0)
    # jitter small enough to pass default rtol=1e-3 but large enough to fail rtol=1e-6
    jitter = rng.uniform(-1e-7, 1e-7, size=99)
    diffs = 1 / 1000.0 + jitter
    timestamps = np.concatenate([[0.0], np.cumsum(diffs)])
    calc_sampling_rate(timestamps)  # passes with default rtol
    with pytest.raises(AssertionError):
        calc_sampling_rate(timestamps, rtol=1e-6)
