from pathlib import Path
from typing import Callable, Literal, Optional, get_args

from temporaldata import Interval
from torch_brain.dataset import MultiChannelDatasetMixin, Dataset

SplitType = Literal["intrasession", "intersubject", "intersession"]
SplitAssignmentType = Literal["train", "valid", "test"]

VALID_SPLIT_TYPES = get_args(SplitType)
VALID_SPLIT_ASSIGNMENT_TYPES = get_args(SplitAssignmentType)


class OpenNeuroDataset(MultiChannelDatasetMixin, Dataset):
    """
    Base class for OpenNeuro datasets.

    This class provides an interface for loading, representing, and manipulating
    OpenNeuro datasets using the MultiChannelDatasetMixin and the Dataset interface.
    It supports various splitting strategies for machine learning workflows, notably
    'intrasession', 'intersubject', and 'intersession' splits.

    Args:
        root: Root directory containing processed OpenNeuro dataset artifacts.
        dataset_dir: Relative dataset directory within the root path.
        recording_ids: Optional explicit recording-id subset to expose from disk.
            If omitted, the dataset uses all available recordings.
        transform: Optional sample transform.
        split_type (SplitType): The split strategy to use, must be one of
            'intrasession', 'intersubject', or 'intersession'.
        recording_ids (Optional[list[str]]): List of recording IDs to include,
            or None to use all available recordings.
    """

    def __init__(
        self,
        root: str,
        dataset_dir: str,
        split_type: SplitType,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        **kwargs,
    ):
        if split_type not in VALID_SPLIT_TYPES:
            raise ValueError(
                f"Invalid split_type '{split_type}'. Must be one of {VALID_SPLIT_TYPES}."
            )
        self.split_type = split_type

        super().__init__(
            dataset_dir=Path(root) / dataset_dir,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "channels.id"],
            **kwargs,
        )

    def get_sampling_intervals(
        self, split_assignment: Optional[SplitAssignmentType] = None
    ) -> dict[str, Interval]:
        """Returns a dictionary of sampling intervals for each recording.
        This represents the intervals that can be sampled from each session.

        If split_assignment is None, all recordings are mapped to their full domain.

        Args:
            split_assignment (Optional[SplitAssignmentType]): One of "train", "valid", "test" to filter intervals
                for that split, or None to retrieve full domains for all recordings.

        Returns:
            dict[str, Interval]: Dict mapping each recording ID to its valid interval (or domain).

        Raises:
            ValueError: If `split_assignment` or `self.split_type` is not a valid option.
            KeyError: If a required split or assignment attribute is missing in a recording.
        """
        if split_assignment is None:
            return super().get_sampling_intervals()

        if split_assignment not in VALID_SPLIT_ASSIGNMENT_TYPES:
            raise ValueError(
                f"Invalid split_assignment '{split_assignment}'. Must be one of {VALID_SPLIT_ASSIGNMENT_TYPES}."
            )

        if self.split_type == "intrasession":
            split_key = f"splits.{split_assignment}"
            intervals = {}
            for rid in self.recording_ids:
                rec = self.get_recording(rid)
                try:
                    intervals[rid] = rec.get_nested_attribute(split_key)
                except (AttributeError, KeyError) as exc:
                    raise KeyError(
                        f"Missing required split attribute for Recording {rid}. "
                        f"Expected {split_key} split attribute in its metadata. "
                    ) from exc
            return intervals

        if self.split_type in ("intersubject", "intersession"):
            assignment_key = f"splits.{self.split_type}_assignment"
            intervals = {}
            for rid in self.recording_ids:
                rec = self.get_recording(rid)
                assignment = None
                try:
                    assignment = str(rec.get_nested_attribute(assignment_key))
                except (AttributeError, KeyError) as exc:
                    raise KeyError(
                        f"Missing required split attribute for Recording {rid}. "
                        f"Expected {assignment_key} split attribute in its metadata. "
                    ) from exc
                if assignment == split_assignment:
                    intervals[rid] = rec.domain
            return intervals

        raise ValueError(
            f"Invalid split_type '{self.split_type}'. Must be one of {VALID_SPLIT_TYPES}."
        )
