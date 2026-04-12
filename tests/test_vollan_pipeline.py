import numpy as np
import pytest
from temporaldata import Interval

from brainsets_pipelines.vollan_moser_alternating_2025.pipeline import (
    build_domain_from_timestamps,
    build_sleep_domain,
    extract_navigation_behavior,
    extract_navigation_units_and_spikes,
)


# ---------------------------------------------------------------------------
# build_domain_from_timestamps
# ---------------------------------------------------------------------------


def test_build_domain_contiguous_timestamps():
    """A contiguous timeseries with no gaps should produce a single interval."""
    t = np.arange(0.0, 1.0, 0.01)  # 100 samples at 10ms
    domain = build_domain_from_timestamps(t)

    assert len(domain.start) == 1
    assert domain.start[0] == pytest.approx(0.0)
    assert domain.end[0] == pytest.approx(t[-1] + 0.01, abs=1e-6)


def test_build_domain_with_gaps():
    """Gaps larger than 2x the sampling interval should split into segments."""
    t1 = np.arange(0.0, 0.5, 0.01)  # 0.0 to 0.49
    t2 = np.arange(1.0, 1.5, 0.01)  # 1.0 to 1.49 (gap of 0.51s >> 0.02s)
    t = np.concatenate([t1, t2])
    domain = build_domain_from_timestamps(t)

    assert len(domain.start) == 2
    assert domain.start[0] == pytest.approx(0.0)
    assert domain.start[1] == pytest.approx(1.0)
    assert domain.is_sorted()
    assert domain.is_disjoint()


def test_build_domain_single_sample():
    """A single timestamp should produce one interval."""
    t = np.array([5.0])
    domain = build_domain_from_timestamps(t)

    assert len(domain.start) == 1
    assert domain.start[0] == pytest.approx(5.0)
    assert domain.end[0] == pytest.approx(5.01)


def test_build_domain_empty():
    """Empty timestamps should return an empty domain."""
    t = np.array([], dtype=np.float64)
    domain = build_domain_from_timestamps(t)

    assert len(domain.start) == 0
    assert len(domain.end) == 0


# ---------------------------------------------------------------------------
# build_sleep_domain
# ---------------------------------------------------------------------------


def _make_sleep_ds(sws_times, rem_times):
    """Helper to create a minimal sleep dataset dict."""
    return {
        "times": {
            "sws": np.array(sws_times, dtype=np.float64),
            "rem": np.array(rem_times, dtype=np.float64),
        }
    }


def test_build_sleep_domain_sws_and_rem():
    """Domain should include both SWS and REM epochs, sorted by start time."""
    ds = _make_sleep_ds(
        sws_times=[[10.0, 20.0], [30.0, 40.0]],
        rem_times=[[25.0, 28.0]],
    )
    domain = build_sleep_domain(ds)

    assert len(domain.start) == 3
    assert domain.is_sorted()
    assert domain.is_disjoint()
    np.testing.assert_array_equal(domain.start, [10.0, 25.0, 30.0])
    np.testing.assert_array_equal(domain.end, [20.0, 28.0, 40.0])


def test_build_sleep_domain_sws_only():
    """Sleep session with only SWS epochs (empty REM)."""
    ds = _make_sleep_ds(
        sws_times=[[5.0, 15.0]],
        rem_times=[],
    )
    domain = build_sleep_domain(ds)

    assert len(domain.start) == 1
    np.testing.assert_array_equal(domain.start, [5.0])


def test_build_sleep_domain_empty_both_raises():
    """Empty SWS and REM should raise (np.concatenate on empty lists)."""
    ds = _make_sleep_ds(sws_times=[], rem_times=[])
    with pytest.raises(ValueError):
        build_sleep_domain(ds)


# ---------------------------------------------------------------------------
# extract_navigation_behavior
# ---------------------------------------------------------------------------


def _make_nav_ds(n=100, include_id=True, include_theta=True):
    """Helper to create a minimal navigation Dsession dict."""
    t = np.arange(0, n * 0.01, 0.01, dtype=np.float64)[:n]
    ds = {
        "t": t,
        "x": np.random.randn(n).astype(np.float32),
        "y": np.random.randn(n).astype(np.float32),
        "z": np.random.randn(n).astype(np.float32),
        "hd": np.random.randn(n).astype(np.float32),
        "id": np.random.randn(n).astype(np.float32) if include_id else np.array([]),
        "theta": (
            np.random.randn(n).astype(np.float32) if include_theta else np.array([])
        ),
    }
    return ds, t


def test_extract_behavior_all_fields_present():
    ds, t = _make_nav_ds(n=50)
    domain = build_domain_from_timestamps(t)
    behavior = extract_navigation_behavior(ds, domain)

    assert behavior.timestamps.shape[0] == 50
    assert behavior.x.shape[0] == 50
    assert not np.any(np.isnan(behavior.id))
    assert not np.any(np.isnan(behavior.theta))


def test_extract_behavior_missing_id_fills_nan():
    """When LMT model was not fitted, id should be filled with NaN."""
    ds, t = _make_nav_ds(n=50, include_id=False)
    domain = build_domain_from_timestamps(t)
    behavior = extract_navigation_behavior(ds, domain)

    assert behavior.id.shape[0] == 50
    assert np.all(np.isnan(behavior.id))


def test_extract_behavior_missing_theta_fills_nan():
    ds, t = _make_nav_ds(n=50, include_theta=False)
    domain = build_domain_from_timestamps(t)
    behavior = extract_navigation_behavior(ds, domain)

    assert behavior.theta.shape[0] == 50
    assert np.all(np.isnan(behavior.theta))


# ---------------------------------------------------------------------------
# extract_navigation_units_and_spikes
# ---------------------------------------------------------------------------


def _make_units_ds(mec_units, hc_units):
    """Helper: build a minimal Dsession dict with units."""
    t = np.arange(0, 1.0, 0.01)
    ds = {
        "t": t,
        "units": {"mec": mec_units, "hc": hc_units},
    }
    return ds, t


def _make_unit(
    spike_times, probe_id=1, shank=1, shank_pos=100.0, mean_rate=1.0, is_grid=0
):
    return {
        "probeId": probe_id,
        "shank": shank,
        "shankPos": shank_pos,
        "meanRate": mean_rate,
        "isGrid": is_grid,
        "spikeTimes": np.array(spike_times, dtype=np.float64),
    }


def test_extract_units_list_of_dicts():
    """Standard case: units are a list of dicts."""
    mec = [_make_unit([0.1, 0.2], is_grid=1), _make_unit([0.3])]
    hc = [_make_unit([0.4, 0.5, 0.6])]
    ds, t = _make_units_ds(mec, hc)
    domain = build_domain_from_timestamps(t)

    units, spikes = extract_navigation_units_and_spikes(ds, domain)

    assert len(units.id) == 3
    assert spikes.timestamps.shape[0] == 6


def test_extract_units_single_dict_not_skipped():
    """A single MATLAB struct (returned as dict, not list) should still be handled."""
    mec = _make_unit([0.1, 0.2], is_grid=1)  # bare dict, not wrapped in list
    hc = [_make_unit([0.3])]
    ds, t = _make_units_ds(mec, hc)
    domain = build_domain_from_timestamps(t)

    units, spikes = extract_navigation_units_and_spikes(ds, domain)

    assert len(units.id) == 2  # 1 mec + 1 hc
    assert spikes.timestamps.shape[0] == 3


def test_extract_units_empty_region():
    """An empty region (np array size 0) should be skipped gracefully."""
    mec = np.array([])
    hc = [_make_unit([0.5])]
    ds, t = _make_units_ds(mec, hc)
    domain = build_domain_from_timestamps(t)

    units, spikes = extract_navigation_units_and_spikes(ds, domain)

    assert len(units.id) == 1
    assert spikes.timestamps.shape[0] == 1


def test_extract_units_spikes_are_sorted():
    """Spikes should be sorted by timestamp after extraction."""
    mec = [_make_unit([0.9, 0.1])]
    hc = [_make_unit([0.5, 0.3])]
    ds, t = _make_units_ds(mec, hc)
    domain = build_domain_from_timestamps(t)

    _, spikes = extract_navigation_units_and_spikes(ds, domain)

    timestamps = spikes.timestamps
    assert np.all(timestamps[:-1] <= timestamps[1:])
