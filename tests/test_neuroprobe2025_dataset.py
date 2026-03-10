from pathlib import Path

import h5py
import numpy as np
import pytest
from temporaldata import Interval

from brainsets.datasets.Neuroprobe2025 import (
    Neuroprobe2025,
    _from_recording_id,
    _to_recording_id,
)


def _mock_dataset_dir(tmp_path: Path) -> Path:
    """Return the expected Neuroprobe dataset directory under tmp_path."""
    return tmp_path / "neuroprobe_2025"


def _split_key(
    *,
    subset_tier: str,
    label_mode: str,
    h5_regime: str,
    task: str,
    fold: int,
    split: str,
) -> str:
    return f"{subset_tier}${label_mode}${h5_regime}${task}$fold{fold}${split}"


def _write_mock_h5(
    path: Path,
    *,
    subset_tier: str,
    label_mode: str,
    h5_regime: str,
    task: str = "speech",
    fold: int = 0,
    splits: tuple[str, ...] = ("train", "val", "test"),
    compatible: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        if not compatible:
            return

        channels = h5.create_group("channels")
        channels.create_dataset("id", data=np.array(["ch0"], dtype="S8"))
        for split in splits:
            key = _split_key(
                subset_tier=subset_tier,
                label_mode=label_mode,
                h5_regime=h5_regime,
                task=task,
                fold=fold,
                split=split,
            )
            channels.create_dataset(key, data=np.array([True], dtype=bool))

        splits_group = h5.create_group("splits")
        for split in splits:
            interval_key = _split_key(
                subset_tier=subset_tier,
                label_mode=label_mode,
                h5_regime=h5_regime,
                task=task,
                fold=fold,
                split=split,
            )
            interval_group = splits_group.create_group(interval_key)
            interval_group.create_dataset("start", data=np.array([0.0], dtype=float))
            interval_group.create_dataset("end", data=np.array([1.0], dtype=float))
            interval_group.create_dataset("label", data=np.array([1], dtype=int))


def _write_mock_recordings(
    tmp_path: Path,
    recording_ids: tuple[str, ...],
    *,
    subset_tier: str,
    h5_regime: str,
    label_mode: str = "binary",
    task: str = "speech",
    fold: int = 0,
) -> None:
    """Create compatible mock H5 recordings for the given ids."""
    dataset_dir = _mock_dataset_dir(tmp_path)
    for recording_id in recording_ids:
        _write_mock_h5(
            dataset_dir / f"{recording_id}.h5",
            subset_tier=subset_tier,
            label_mode=label_mode,
            h5_regime=h5_regime,
            task=task,
            fold=fold,
        )


def _make_dataset(tmp_path: Path, **overrides) -> Neuroprobe2025:
    """Build a Neuroprobe2025 instance with concise defaults for tests."""
    kwargs = {
        "root": str(tmp_path),
        "subset_tier": "full",
        "label_mode": "binary",
        "task": "speech",
        "regime": "SS-SM",
        "test_subject": 1,
        "test_session": 1,
        "split": "train",
        "keep_files_open": False,
    }
    kwargs.update(overrides)
    return Neuroprobe2025(**kwargs)


@pytest.fixture(autouse=True)
def _disable_prewarm(monkeypatch):
    # Keep tests lightweight and independent of full temporaldata recording schema.
    monkeypatch.setattr(
        Neuroprobe2025, "_initialize_seeg_mixin_caches", lambda self: None
    )
    monkeypatch.setattr(
        Neuroprobe2025, "_prime_selected_recording_caches", lambda self: None
    )


def test_recording_id_roundtrip():
    recording_id = _to_recording_id(2, 4)
    assert recording_id == "sub_2_trial004"
    assert _from_recording_id(recording_id) == (2, 4)


def test_recording_id_invalid_format_raises():
    with pytest.raises(ValueError, match="Invalid recording_id"):
        _from_recording_id("subject2_trial4")


def test_recording_id_requires_zero_padded_session():
    with pytest.raises(ValueError, match="zero-padded 3-digit session"):
        _from_recording_id("sub_1_trial4")


def test_nano_cross_x_rejected_fast(tmp_path):
    with pytest.raises(ValueError, match="not compatible with cross_x"):
        Neuroprobe2025(
            root=str(tmp_path),
            recording_ids=["sub_1_trial001"],
            subset_tier="nano",
            label_mode="binary",
            task="speech",
            regime="SS-DM",
            test_subject=1,
            test_session=1,
            split="train",
            keep_files_open=False,
        )


def test_cross_x_fold_must_be_zero(tmp_path):
    _write_mock_recordings(
        tmp_path,
        ("sub_1_trial001",),
        subset_tier="full",
        h5_regime="cross_x",
    )

    with pytest.raises(ValueError, match="must be 0"):
        _make_dataset(
            tmp_path,
            subset_tier="full",
            regime="SS-DM",
            split="test",
            fold=1,
        )


def test_fold_bool_rejected_fast(tmp_path):
    _write_mock_recordings(
        tmp_path,
        ("sub_1_trial001",),
        subset_tier="full",
        h5_regime="within_session",
    )

    with pytest.raises(TypeError, match="fold must be an int"):
        _make_dataset(tmp_path, fold=True)


def test_ss_dm_lite_train_selects_other_trial(tmp_path):
    _write_mock_recordings(
        tmp_path,
        ("sub_1_trial001", "sub_1_trial002"),
        subset_tier="lite",
        h5_regime="cross_x",
    )

    ds = _make_dataset(
        tmp_path,
        subset_tier="lite",
        regime="SS-DM",
        split="train",
    )
    assert ds.describe_selection()["selected_recording_ids"] == ["sub_1_trial002"]


def test_ds_dm_train_selects_fixed_anchor(tmp_path):
    _write_mock_recordings(
        tmp_path,
        ("sub_3_trial001", "sub_2_trial004"),
        subset_tier="full",
        h5_regime="cross_x",
    )

    ds = _make_dataset(
        tmp_path,
        regime="DS-DM",
        test_subject=3,
        test_session=1,
        split="train",
    )
    assert ds.describe_selection()["selected_recording_ids"] == ["sub_2_trial004"]


def test_get_sampling_rate_is_fixed_constant(tmp_path):
    _write_mock_recordings(
        tmp_path,
        ("sub_1_trial001",),
        subset_tier="full",
        h5_regime="within_session",
    )

    ds = _make_dataset(tmp_path)
    assert ds.get_sampling_rate() == Neuroprobe2025.DEFAULT_SAMPLING_RATE_HZ


def test_uniquify_channel_ids_option_sets_seeg_mixin_components(tmp_path):
    _write_mock_recordings(
        tmp_path,
        ("sub_1_trial001",),
        subset_tier="full",
        h5_regime="within_session",
    )

    ds = _make_dataset(tmp_path, uniquify_channel_ids={"subject_id"})
    assert ds.seeg_dataset_mixin_uniquify_channel_ids == frozenset({"subject_id"})


def test_uniquify_channel_ids_option_rejects_non_set(tmp_path):
    _write_mock_recordings(
        tmp_path,
        ("sub_1_trial001",),
        subset_tier="full",
        h5_regime="within_session",
    )

    with pytest.raises(TypeError, match="must be a set/frozenset"):
        _make_dataset(tmp_path, uniquify_channel_ids=True)


def test_get_sampling_intervals_uses_instance_split_path(tmp_path, monkeypatch):
    _write_mock_recordings(
        tmp_path,
        ("sub_1_trial001",),
        subset_tier="full",
        h5_regime="within_session",
    )

    ds = _make_dataset(tmp_path)

    class _FakeRecording:
        def __init__(self):
            self.paths = []

        def get_nested_attribute(self, path: str):
            self.paths.append(path)
            return Interval(
                start=np.array([0.0]), end=np.array([1.0]), label=np.array([1])
            )

    fake_recording = _FakeRecording()
    monkeypatch.setattr(ds, "get_recording", lambda _rid: fake_recording)

    intervals = ds.get_sampling_intervals()
    assert list(intervals.keys()) == ["sub_1_trial001"]
    assert isinstance(intervals["sub_1_trial001"], Interval)
    assert fake_recording.paths == [
        "splits.full$binary$within_session$speech$fold0$train"
    ]


def test_get_channel_view_included_only_filters_channels(tmp_path):
    _write_mock_recordings(
        tmp_path,
        ("sub_1_trial001",),
        subset_tier="full",
        h5_regime="within_session",
    )

    ds = _make_dataset(tmp_path)

    ds.seeg_dataset_mixin_channel_views = {
        "sub_1_trial001": ds.ChannelView(
            ids=np.array(["c0", "c1"]),
            names=np.array(["A", "B"]),
            included_mask=np.array([True, False]),
            lip=np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=float),
        )
    }

    full_view = ds.get_channel_view("sub_1_trial001", included_only=False)
    included_view = ds.get_channel_view("sub_1_trial001", included_only=True)
    assert full_view.ids.tolist() == ["c0", "c1"]
    assert included_view.ids.tolist() == ["c0"]
    assert included_view.names.tolist() == ["A"]
    assert included_view.lip is not None
    assert included_view.lip.shape == (1, 3)


def test_compatibility_cache_invalidation_on_file_change(tmp_path):
    recording_id = "sub_1_trial001"
    h5_path = _mock_dataset_dir(tmp_path) / f"{recording_id}.h5"
    _write_mock_h5(
        h5_path,
        subset_tier="full",
        label_mode="binary",
        h5_regime="within_session",
        compatible=True,
    )

    ds = _make_dataset(tmp_path)
    issue = ds._recording_compatibility_issue(recording_id)
    assert issue is None

    _write_mock_h5(
        h5_path,
        subset_tier="full",
        label_mode="binary",
        h5_regime="within_session",
        compatible=False,
    )

    issue = ds._recording_compatibility_issue(recording_id)
    assert issue == "missing 'splits' group; missing 'channels' group"
