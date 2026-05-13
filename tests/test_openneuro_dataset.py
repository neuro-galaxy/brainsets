"""Unit tests for OpenNeuroDataset class."""

from types import SimpleNamespace

import numpy as np
import pytest
from unittest.mock import patch
from temporaldata import Interval

from brainsets.datasets.OpenNeuroDataset import OpenNeuroDataset
from brainsets.utils.split import generate_string_kfold_assignment


# ============================================================================
# Helpers and Fakes
# ============================================================================


def _assert_intervals_close(
    left: Interval, right: Interval, rtol: float = 1e-7, atol: float = 0.0
) -> None:
    """Assert two Interval objects represent the same bounds (array-safe)."""
    np.testing.assert_allclose(
        np.asarray(left.start, dtype=float),
        np.asarray(right.start, dtype=float),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(left.end, dtype=float),
        np.asarray(right.end, dtype=float),
        rtol=rtol,
        atol=atol,
    )


class FakeRecording:
    """Minimal fake recording object for testing."""

    def __init__(
        self,
        recording_id: str,
        attributes: dict | None = None,
        domain=None,
        *,
        subject_id: str = "sub-default",
        session_id: str = "ses-default",
    ):
        """
        Args:
            recording_id: Identifier for this recording.
            attributes: Dict mapping nested attribute paths to values (unused by
                current ``get_sampling_intervals`` behavior-agnostic path).
            domain: Value for ``rec.domain`` (temporal ``Interval``).
            subject_id: ``recording.subject.id`` for k-fold string ids.
            session_id: ``recording.session.id`` for intersession string ids.
        """
        self.recording_id = recording_id
        self.attributes = attributes or {}
        self.domain = domain or Interval(start=0.0, end=1.0)
        self.subject = SimpleNamespace(id=subject_id)
        self.session = SimpleNamespace(id=session_id)

    def get_nested_attribute(self, path: str):
        """Return attribute or raise KeyError if missing."""
        if path in self.attributes:
            return self.attributes[path]
        raise KeyError(f"Attribute '{path}' not found")


def _expected_intrasession_intervals(
    domain: Interval,
    split: str,
    split_ratios: tuple[float, float, float],
) -> Interval:
    """Mirror ``OpenNeuroDataset._get_behavior_agnostic_intervals`` intrasession math."""
    starts = np.asarray(domain.start, dtype=float)
    ends = np.asarray(domain.end, dtype=float)
    durations = ends - starts
    train_ends = starts + durations * split_ratios[0]
    val_ends = train_ends + durations * split_ratios[1]
    test_ends = val_ends + durations * split_ratios[2]
    if split == "train":
        return Interval(start=starts, end=train_ends)
    if split == "val":
        return Interval(start=train_ends, end=val_ends)
    if split == "test":
        return Interval(start=val_ends, end=test_ends)
    raise AssertionError(split)


def _expected_kfold_assignment(
    string_id: str,
    *,
    test_ratio: float,
    seed: int,
    fold_idx: int = 0,
) -> str:
    """Expected per-recording assignment for intersubject/intersession (fold 0)."""
    if test_ratio <= 0:
        n_folds = 1
    else:
        n_folds = max(1, int(round(1.0 / test_ratio)))
    assignments = generate_string_kfold_assignment(
        string_id=string_id,
        n_folds=n_folds,
        val_ratio=0.0,
        seed=seed,
    )
    assignment = assignments[fold_idx]
    if assignment == "test":
        assignment = "val"
    return assignment


def _make_dataset(
    split_type: str = "intrasession",
    recording_ids: list[str] | None = None,
    **overrides,
) -> OpenNeuroDataset:
    """Build an OpenNeuroDataset instance with concise defaults for tests.

    Args:
        split_type: One of "intrasession", "intersubject", "intersession".
        recording_ids: List of recording IDs, or None for defaults.
        **overrides: Additional kwargs to pass to OpenNeuroDataset constructor.

    Returns:
        OpenNeuroDataset instance with mocked parent initialization.
    """
    kwargs = {
        "root": "/fake/root",
        "dataset_dir": "dataset",
        "split_type": split_type,
    }
    if recording_ids is not None:
        kwargs["recording_ids"] = recording_ids
    elif split_type == "intrasession":
        kwargs["recording_ids"] = ["rec-001", "rec-002"]
    elif split_type == "intersubject":
        kwargs["recording_ids"] = ["rec-001", "rec-002", "rec-003"]
    elif split_type == "intersession":
        kwargs["recording_ids"] = ["rec-001", "rec-002"]

    kwargs.update(overrides)
    ds = OpenNeuroDataset(**kwargs)
    # Since parent Dataset.__init__ is mocked, we need to set _recording_ids manually
    ds._recording_ids = kwargs["recording_ids"]
    return ds


def _make_recording(
    recording_id: str,
    *,
    attributes: dict | None = None,
    domain: Interval | None = None,
    subject_id: str = "sub-default",
    session_id: str = "ses-default",
) -> FakeRecording:
    """Build a FakeRecording with optional attributes and domain."""
    return FakeRecording(
        recording_id,
        attributes=attributes,
        domain=domain or Interval(start=0.0, end=1.0),
        subject_id=subject_id,
        session_id=session_id,
    )


@pytest.fixture
def mock_parent_init():
    """Patch parent Dataset.__init__ to avoid filesystem access and set recording_ids."""
    with patch("torch_brain.dataset.Dataset.__init__", return_value=None):
        yield


# ============================================================================
# Tests for Constructor
# ============================================================================


class TestOpenNeuroDatasetInit:
    """Tests for OpenNeuroDataset.__init__."""

    @pytest.mark.parametrize(
        "split_type", ["intrasession", "intersubject", "intersession"]
    )
    def test_accepts_valid_split_type(self, split_type, mock_parent_init):
        """Constructor accepts each valid split_type."""
        ds = _make_dataset(split_type=split_type)
        assert ds.split_type == split_type

    @pytest.mark.parametrize(
        ("invalid_split_type", "expected_msg_fragment"),
        [
            ("invalid", "Invalid split_type 'invalid'"),
            ("intra_session", "Invalid split_type 'intra_session'"),
            ("", "Invalid split_type ''"),
        ],
    )
    def test_rejects_invalid_split_type(
        self, invalid_split_type, expected_msg_fragment, mock_parent_init
    ):
        """Constructor raises ValueError for invalid split_type."""
        with pytest.raises(ValueError, match=expected_msg_fragment):
            _make_dataset(split_type=invalid_split_type)

    def test_constructor_with_optional_args(self, mock_parent_init):
        """Constructor accepts optional recording_ids and transform."""
        recording_ids = ["rec-001", "rec-002"]
        transform = lambda x: x  # noqa: E731

        ds = _make_dataset(
            recording_ids=recording_ids, transform=transform, split_type="intrasession"
        )
        assert ds.split_type == "intrasession"


# ============================================================================
# Tests for get_sampling_intervals - Basic & Delegation
# ============================================================================


class TestGetSamplingIntervalsBasic:
    """Tests for basic get_sampling_intervals behavior."""

    def test_split_assignment_none_delegates_to_parent(
        self, mock_parent_init, monkeypatch
    ):
        """When split=None, delegates to parent implementation."""
        ds = _make_dataset(split_type="intrasession")
        expected_result = {"rec-001": Interval(0.0, 10.0)}

        with patch.object(
            OpenNeuroDataset.__bases__[1],
            "get_sampling_intervals",
            return_value=expected_result,
        ) as mock_parent:
            result = ds.get_sampling_intervals(split=None)

        assert result == expected_result
        mock_parent.assert_called_once()

    @pytest.mark.parametrize(
        ("invalid_assignment", "expected_msg_fragment"),
        [
            ("invalid", "Invalid split 'invalid'"),
            ("train_split", "Invalid split 'train_split'"),
            ("valid", "Invalid split 'valid'"),
            ("", "Invalid split ''"),
        ],
    )
    def test_rejects_invalid_split_assignment(
        self, invalid_assignment, expected_msg_fragment, mock_parent_init
    ):
        """Invalid split raises ValueError before touching recordings."""
        ds = _make_dataset(split_type="intrasession")
        with pytest.raises(ValueError, match=expected_msg_fragment):
            ds.get_sampling_intervals(split=invalid_assignment)

    def test_behavior_relevant_task_paradigm_not_implemented(
        self, mock_parent_init, monkeypatch
    ):
        """With ``task_paradigm`` set, delegates to ``get_behavior_relevant_intervals``."""
        ds = _make_dataset(
            split_type="intrasession",
            task_paradigm="some_paradigm",
            recording_ids=["rec-001"],
        )
        rec = _make_recording("rec-001")
        monkeypatch.setattr(ds, "get_recording", lambda rid: rec)

        with pytest.raises(
            NotImplementedError, match="get_behavior_relevant_intervals"
        ):
            ds.get_sampling_intervals(split="train")


# ============================================================================
# Tests for get_sampling_intervals - Intrasession
# ============================================================================


class TestGetSamplingIntervalsIntrasession:
    """Tests for intrasession split strategy (domain-based causal chunks)."""

    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_intrasession_splits_domain_by_ratios(
        self, split, mock_parent_init, monkeypatch
    ):
        """Intrasession returns time sub-intervals derived from ``domain`` and ratios."""
        split_ratios = (0.8, 0.1, 0.1)
        ds = _make_dataset(split_type="intrasession", split_ratios=split_ratios)
        domain_1 = Interval(start=0.0, end=10.0)
        domain_2 = Interval(start=0.0, end=100.0)

        recordings = {
            "rec-001": _make_recording("rec-001", domain=domain_1),
            "rec-002": _make_recording("rec-002", domain=domain_2),
        }
        monkeypatch.setattr(ds, "get_recording", lambda rid: recordings[rid])

        result = ds.get_sampling_intervals(split=split)
        exp_1 = _expected_intrasession_intervals(domain_1, split, split_ratios)
        exp_2 = _expected_intrasession_intervals(domain_2, split, split_ratios)

        assert set(result) == {"rec-001", "rec-002"}
        _assert_intervals_close(result["rec-001"], exp_1)
        _assert_intervals_close(result["rec-002"], exp_2)


# ============================================================================
# Tests for get_sampling_intervals - Intersubject
# ============================================================================


class TestGetSamplingIntervalsIntersubject:
    """Tests for intersubject split strategy (k-fold assignment strings, fold 0)."""

    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_intersubject_returns_kfold_assignment_per_recording(
        self, split, mock_parent_init, monkeypatch
    ):
        """Each recording maps to ``'train'`` or ``'val'`` from ``generate_string_kfold_assignment``."""
        ds = _make_dataset(split_type="intersubject", seed=42)
        recordings = {
            "rec-001": _make_recording("rec-001", subject_id="sub-01"),
            "rec-002": _make_recording("rec-002", subject_id="sub-02"),
            "rec-003": _make_recording("rec-003", subject_id="sub-03"),
        }
        monkeypatch.setattr(ds, "get_recording", lambda rid: recordings[rid])

        test_ratio = ds.split_ratios[2]
        expected = {
            rid: _expected_kfold_assignment(
                rec.subject.id, test_ratio=test_ratio, seed=42
            )
            for rid, rec in recordings.items()
        }

        result = ds.get_sampling_intervals(split=split)
        assert result == expected

    def test_intersubject_split_argument_does_not_change_assignments(
        self, mock_parent_init, monkeypatch
    ):
        """``split`` name does not alter k-fold string (same dict for train/val/test)."""
        ds = _make_dataset(split_type="intersubject", recording_ids=["rec-001"], seed=0)
        rec = _make_recording("rec-001", subject_id="sub-xyz")
        monkeypatch.setattr(ds, "get_recording", lambda rid: rec)

        r_train = ds.get_sampling_intervals(split="train")
        r_val = ds.get_sampling_intervals(split="val")
        r_test = ds.get_sampling_intervals(split="test")
        assert r_train == r_val == r_test


# ============================================================================
# Tests for get_sampling_intervals - Intersession
# ============================================================================


class TestGetSamplingIntervalsIntersession:
    """Tests for intersession split strategy (k-fold on ``subject_session`` id)."""

    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_intersession_returns_kfold_assignment_per_recording(
        self, split, mock_parent_init, monkeypatch
    ):
        """String id is ``f\"{subject.id}_{session.id}\"`` for k-fold hashing."""
        ds = _make_dataset(split_type="intersession", seed=42)
        recordings = {
            "rec-001": _make_recording(
                "rec-001", subject_id="sub-01", session_id="ses-a"
            ),
            "rec-002": _make_recording(
                "rec-002", subject_id="sub-01", session_id="ses-b"
            ),
        }
        monkeypatch.setattr(ds, "get_recording", lambda rid: recordings[rid])

        test_ratio = ds.split_ratios[2]
        expected = {}
        for rid, rec in recordings.items():
            sid = f"{rec.subject.id}_{rec.session.id}"
            expected[rid] = _expected_kfold_assignment(
                sid, test_ratio=test_ratio, seed=42
            )

        result = ds.get_sampling_intervals(split=split)
        assert result == expected


# ============================================================================
# Tests for Defensive Behavior
# ============================================================================


class TestDefensiveBehavior:
    """Tests for defensive/edge case behavior."""

    def test_invalid_split_type_at_runtime_raises_value_error(
        self, mock_parent_init, monkeypatch
    ):
        """If ``split_type`` is mutated to an invalid value, raises ValueError."""
        ds = _make_dataset(split_type="intrasession")
        ds.split_type = "bad_split_type"
        rec = _make_recording("rec-001")
        monkeypatch.setattr(ds, "get_recording", lambda rid: rec)

        with pytest.raises(
            ValueError,
            match="Invalid split_type 'bad_split_type'",
        ):
            ds.get_sampling_intervals(split="train")
