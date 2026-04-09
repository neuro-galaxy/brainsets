from pathlib import Path
from typing import Callable, Literal, Optional, get_args

from torch_brain.dataset import MultiChannelDatasetMixin, Dataset

SplitType = Literal["intrasession", "intersubject", "intersession"]
SplitAssignmentType = Literal["train", "valid", "test"]

VALID_SPLIT_TYPES = get_args(SplitType)
VALID_SPLIT_ASSIGNMENT_TYPES = get_args(SplitAssignmentType)

class OpenNeuroDataset(MultiChannelDatasetMixin, Dataset):
    """OpenNeuroDataset base class."""

    def __init__(
        self,
        root: str,
        dataset_dir: str,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        # split args
        fold_num: Optional[int] = None,
        split_type: SplitType = "intrasession",
        task_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            dataset_dir=Path(root) / dataset_dir,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "channels.id"],
            **kwargs,
        )
        
        if not isinstance(fold_num, int):
            raise ValueError(
                f"fold_num must be an integer, got {type(fold_num)}."
            )
        if fold_num < 0:
            raise ValueError(
                f"fold_num must be non-negative, got {fold_num}."
            )
        self.fold_num = fold_num

        if split_type not in VALID_SPLIT_TYPES:
            raise ValueError(
                f"Invalid split_type '{split_type}'. Must be one of {VALID_SPLIT_TYPES}."
            )
        self.split_type = split_type
        
        self.task_type = task_type

    def get_sampling_intervals(self, split: Optional[SplitAssignmentType] = None):
        if split is None:
            return {rid: self.get_recording(rid).domain for rid in self.recording_ids}

        if split not in VALID_SPLIT_ASSIGNMENT_TYPES:
            raise ValueError(
                f"Invalid split '{split}'. Must be one of {VALID_SPLIT_ASSIGNMENT_TYPES}."
            )

        if self.split_type == "intrasession":
            split_key = f"splits.{split}"
            intervals = {}
            for rid in self.recording_ids:
                rec = self.get_recording(rid)
                try:
                    intervals[rid] = rec.get_nested_attribute(split_key)
                    break
                except (AttributeError, KeyError):
                    continue
            return intervals

        assignment_keys = [
            f"splits.{self.fold_type}_assignment",
            f"splits.{self.fold_type}_fold_0_assignment",
            "splits.fold_0_assignment",
        ]
        intervals = {}
        for rid in self.recording_ids:
            rec = self.get_recording(rid)
            assignment = None
            for key in assignment_keys:
                try:
                    assignment = str(rec.get_nested_attribute(key))
                    break
                except (AttributeError, KeyError):
                    continue
            if assignment == split:
                intervals[rid] = rec.domain
        return intervals