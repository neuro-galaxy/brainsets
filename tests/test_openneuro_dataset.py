"""Unit tests for OpenNeuroDataset class."""

import pytest
from unittest.mock import MagicMock, patch
from temporaldata import Interval

from brainsets.datasets.OpenNeuroBase import OpenNeuroDataset


# ============================================================================
# Helpers and Fakes
# ============================================================================


class FakeRecording:
    """Minimal fake recording object for testing."""

    def __init__(self, recording_id: str, attributes: dict | None = None, domain=None):
        """
        Args:
            recording_id: Identifier for this recording.
            attributes: Dict mapping nested attribute paths to values, e.g.,
                        {"splits.train": interval_obj, "splits.intersubject_assignment": "train"}.
            domain: Value to return for rec.domain.
        """
        self.recording_id = recording_id
        self.attributes = attributes or {}
        self.domain = domain or Interval(start=0.0, end=1.0)

    def get_nested_attribute(self, path: str):
        """Return attribute or raise KeyError if missing."""
        if path in self.attributes:
            return self.attributes[path]
        raise KeyError(f"Attribute '{path}' not found")


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
) -> FakeRecording:
    """Build a FakeRecording with optional attributes and domain.

    Args:
        recording_id: Identifier for the recording.
        attributes: Dict of nested attribute paths to values.
        domain: Interval representing the recording's full domain.

    Returns:
        FakeRecording instance.
    """
    return FakeRecording(
        recording_id,
        attributes=attributes,
        domain=domain or Interval(start=0.0, end=1.0),
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
        """When split_assignment=None, delegates to parent implementation."""
        ds = _make_dataset(split_type="intrasession")
        expected_result = {"rec-001": Interval(0.0, 10.0)}

        with patch.object(
            OpenNeuroDataset.__bases__[1],
            "get_sampling_intervals",
            return_value=expected_result,
        ) as mock_parent:
            result = ds.get_sampling_intervals(split_assignment=None)

        assert result == expected_result
        mock_parent.assert_called_once()

    @pytest.mark.parametrize(
        ("invalid_assignment", "expected_msg_fragment"),
        [
            ("invalid", "Invalid split_assignment 'invalid'"),
            ("train_split", "Invalid split_assignment 'train_split'"),
            ("", "Invalid split_assignment ''"),
        ],
    )
    def test_rejects_invalid_split_assignment(
        self, invalid_assignment, expected_msg_fragment, mock_parent_init
    ):
        """Invalid split_assignment raises ValueError."""
        ds = _make_dataset(split_type="intrasession")
        with pytest.raises(ValueError, match=expected_msg_fragment):
            ds.get_sampling_intervals(split_assignment=invalid_assignment)


# ============================================================================
# Tests for get_sampling_intervals - Intrasession
# ============================================================================


class TestGetSamplingIntervalsIntrasession:
    """Tests for intrasession split strategy."""

    @pytest.mark.parametrize("split_assignment", ["train", "valid", "test"])
    def test_intrasession_happy_path(
        self, split_assignment, mock_parent_init, monkeypatch
    ):
        """Intrasession returns per-recording interval from splits.<assignment>."""
        ds = _make_dataset(split_type="intrasession")
        interval_1 = Interval(start=0.0, end=5.0)
        interval_2 = Interval(start=5.0, end=10.0)

        recordings = {
            "rec-001": _make_recording(
                "rec-001",
                attributes={f"splits.{split_assignment}": interval_1},
            ),
            "rec-002": _make_recording(
                "rec-002",
                attributes={f"splits.{split_assignment}": interval_2},
            ),
        }

        monkeypatch.setattr(ds, "get_recording", lambda rid: recordings[rid])
        result = ds.get_sampling_intervals(split_assignment=split_assignment)

        assert result == {
            "rec-001": interval_1,
            "rec-002": interval_2,
        }

    @pytest.mark.parametrize("split_assignment", ["train", "valid", "test"])
    def test_intrasession_missing_split_raises_key_error(
        self, split_assignment, mock_parent_init, monkeypatch
    ):
        """Intrasession raises KeyError with helpful message when split is missing."""
        ds = _make_dataset(split_type="intrasession")
        recording = _make_recording("rec-001", attributes={})
        monkeypatch.setattr(ds, "get_recording", lambda rid: recording)

        with pytest.raises(
            KeyError,
            match="Missing required split attribute.*rec-001.*splits."
            + split_assignment,
        ):
            ds.get_sampling_intervals(split_assignment=split_assignment)

    @pytest.mark.parametrize("split_assignment", ["train", "valid", "test"])
    def test_intrasession_wraps_attribute_error_as_key_error(
        self, split_assignment, mock_parent_init, monkeypatch
    ):
        """Intrasession wraps AttributeError as KeyError."""
        ds = _make_dataset(split_type="intrasession")
        fake_rec = MagicMock()
        fake_rec.get_nested_attribute.side_effect = AttributeError("No such attr")
        monkeypatch.setattr(ds, "get_recording", lambda rid: fake_rec)

        with pytest.raises(
            KeyError,
            match="Missing required split attribute.*rec-001.*splits."
            + split_assignment,
        ):
            ds.get_sampling_intervals(split_assignment=split_assignment)


# ============================================================================
# Tests for get_sampling_intervals - Intersubject
# ============================================================================


class TestGetSamplingIntervalsIntersubject:
    """Tests for intersubject split strategy."""

    @pytest.mark.parametrize("split_assignment", ["train", "valid", "test"])
    def test_intersubject_happy_path(
        self, split_assignment, mock_parent_init, monkeypatch
    ):
        """Intersubject filters by assignment and returns rec.domain."""
        ds = _make_dataset(split_type="intersubject")
        domain_1 = Interval(start=0.0, end=100.0)
        domain_2 = Interval(start=100.0, end=200.0)
        domain_3 = Interval(start=200.0, end=300.0)

        # Set up recordings where only rec-001 and rec-003 match split_assignment
        # rec-002 always has a different assignment for contrast
        other_assignment = "test" if split_assignment != "test" else "train"

        recordings = {
            "rec-001": _make_recording(
                "rec-001",
                attributes={"splits.intersubject_assignment": split_assignment},
                domain=domain_1,
            ),
            "rec-002": _make_recording(
                "rec-002",
                attributes={"splits.intersubject_assignment": other_assignment},
                domain=domain_2,
            ),
            "rec-003": _make_recording(
                "rec-003",
                attributes={"splits.intersubject_assignment": split_assignment},
                domain=domain_3,
            ),
        }

        monkeypatch.setattr(ds, "get_recording", lambda rid: recordings[rid])
        result = ds.get_sampling_intervals(split_assignment=split_assignment)

        # Only rec-001 and rec-003 should match the assignment
        assert result == {
            "rec-001": domain_1,
            "rec-003": domain_3,
        }

    def test_intersubject_empty_result_when_no_matches(
        self, mock_parent_init, monkeypatch
    ):
        """Intersubject returns empty dict when no recordings match assignment."""
        ds = _make_dataset(split_type="intersubject")
        recordings = {
            "rec-001": _make_recording(
                "rec-001",
                attributes={"splits.intersubject_assignment": "train"},
                domain=Interval(0.0, 100.0),
            ),
            "rec-002": _make_recording(
                "rec-002",
                attributes={"splits.intersubject_assignment": "train"},
                domain=Interval(100.0, 200.0),
            ),
            "rec-003": _make_recording(
                "rec-003",
                attributes={"splits.intersubject_assignment": "train"},
                domain=Interval(200.0, 300.0),
            ),
        }

        monkeypatch.setattr(ds, "get_recording", lambda rid: recordings[rid])
        result = ds.get_sampling_intervals(split_assignment="test")

        assert result == {}

    def test_intersubject_assignment_coerced_to_string(
        self, mock_parent_init, monkeypatch
    ):
        """Intersubject coerces assignment to string for comparison."""
        ds = _make_dataset(split_type="intersubject", recording_ids=["rec-001"])
        domain = Interval(0.0, 100.0)

        # Mock recording returns a non-string object that str() converts to "train"
        assignment_obj = MagicMock()
        assignment_obj.__str__ = lambda self: "train"

        recording = _make_recording(
            "rec-001",
            attributes={"splits.intersubject_assignment": assignment_obj},
            domain=domain,
        )

        monkeypatch.setattr(ds, "get_recording", lambda rid: recording)
        result = ds.get_sampling_intervals(split_assignment="train")

        assert result == {"rec-001": domain}

    @pytest.mark.parametrize("split_assignment", ["train", "valid", "test"])
    def test_intersubject_missing_assignment_raises_key_error(
        self, split_assignment, mock_parent_init, monkeypatch
    ):
        """Intersubject raises KeyError with helpful message when assignment is missing."""
        ds = _make_dataset(split_type="intersubject")
        recording = _make_recording("rec-001", attributes={})
        monkeypatch.setattr(ds, "get_recording", lambda rid: recording)

        with pytest.raises(
            KeyError,
            match="Missing required split attribute.*rec-001.*splits.intersubject_assignment",
        ):
            ds.get_sampling_intervals(split_assignment=split_assignment)


# ============================================================================
# Tests for get_sampling_intervals - Intersession
# ============================================================================


class TestGetSamplingIntervalsIntersession:
    """Tests for intersession split strategy."""

    @pytest.mark.parametrize("split_assignment", ["train", "valid", "test"])
    def test_intersession_happy_path(
        self, split_assignment, mock_parent_init, monkeypatch
    ):
        """Intersession filters by assignment and returns rec.domain."""
        ds = _make_dataset(split_type="intersession")
        domain_1 = Interval(start=0.0, end=50.0)
        domain_2 = Interval(start=50.0, end=100.0)

        # Set up recordings where only rec-001 matches split_assignment
        # rec-002 always has a different assignment for contrast
        other_assignment = "test" if split_assignment != "test" else "train"

        recordings = {
            "rec-001": _make_recording(
                "rec-001",
                attributes={"splits.intersession_assignment": split_assignment},
                domain=domain_1,
            ),
            "rec-002": _make_recording(
                "rec-002",
                attributes={"splits.intersession_assignment": other_assignment},
                domain=domain_2,
            ),
        }

        monkeypatch.setattr(ds, "get_recording", lambda rid: recordings[rid])
        result = ds.get_sampling_intervals(split_assignment=split_assignment)

        assert result == {"rec-001": domain_1}

    @pytest.mark.parametrize("split_assignment", ["train", "valid", "test"])
    def test_intersession_missing_assignment_raises_key_error(
        self, split_assignment, mock_parent_init, monkeypatch
    ):
        """Intersession raises KeyError with helpful message when assignment is missing."""
        ds = _make_dataset(split_type="intersession")
        recording = _make_recording("rec-001", attributes={})
        monkeypatch.setattr(ds, "get_recording", lambda rid: recording)

        with pytest.raises(
            KeyError,
            match="Missing required split attribute.*rec-001.*splits.intersession_assignment",
        ):
            ds.get_sampling_intervals(split_assignment=split_assignment)


# ============================================================================
# Tests for Defensive Behavior
# ============================================================================


class TestDefensiveBehavior:
    """Tests for defensive/edge case behavior."""

    def test_invalid_split_type_at_runtime_raises_value_error(
        self, mock_parent_init, monkeypatch
    ):
        """If split_type is mutated to invalid value, get_sampling_intervals raises ValueError."""
        ds = _make_dataset(split_type="intrasession")
        ds.split_type = "bad_split_type"

        with pytest.raises(
            ValueError,
            match="Invalid split_type 'bad_split_type'",
        ):
            ds.get_sampling_intervals(split_assignment="train")
