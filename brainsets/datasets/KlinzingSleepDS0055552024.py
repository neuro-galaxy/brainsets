from pathlib import Path
from typing import Callable, Literal, Optional, get_args

from torch_brain.dataset import Dataset

FoldType = Literal["intrasession", "intersubject", "intersession"]
VALID_FOLD_TYPES = get_args(FoldType)
SplitType = Literal["train", "valid", "test"]


class KlinzingSleepDS0055552024(Dataset):
    """OpenNeuro DS005555 sleep dataset."""

    def __init__(
        self,
        root: str,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        fold_type: FoldType = "intrasession",
        dirname: str = "klinzing_sleep_ds005555_2024",
        **kwargs,
    ):
        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "channels.id"],
            **kwargs,
        )

        if fold_type not in VALID_FOLD_TYPES:
            raise ValueError(
                f"Invalid fold_type '{fold_type}'. Must be one of {VALID_FOLD_TYPES}."
            )
        self.fold_type = fold_type

    def get_sampling_intervals(self, split: Optional[SplitType] = None):
        if split is None:
            return {rid: self.get_recording(rid).domain for rid in self.recording_ids}

        if split not in ["train", "valid", "test"]:
            raise ValueError(
                f"Invalid split '{split}'. Must be one of ['train', 'valid', 'test']."
            )

        if self.fold_type == "intrasession":
            split_keys = [f"splits.{split}", f"splits.fold_0.{split}"]
            intervals = {}
            for rid in self.recording_ids:
                rec = self.get_recording(rid)
                for key in split_keys:
                    try:
                        intervals[rid] = rec.get_nested_attribute(key)
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
