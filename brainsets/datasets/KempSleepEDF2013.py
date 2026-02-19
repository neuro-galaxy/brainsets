from typing import Callable, Optional, Literal
from pathlib import Path
from torch_brain.utils import np_string_prefix
from temporaldata import Data

from torch_brain.dataset import Dataset


class KempSleepEDF2013(Dataset):
    def __init__(
        self,
        root: str,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        uniquify_channel_ids: bool = True,
        fold_number: Optional[int] = 0,
        fold_type: Literal[
            "intrasession", "intersubject", "intersession"
        ] = "intrasession",
        dirname: str = "kemp_sleep_edf_2013",
        **kwargs,
    ):
        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "channels.id"],
            **kwargs,
        )

        self.uniquify_channel_ids = uniquify_channel_ids
        assert (
            fold_number is not None and fold_number >= 0 and fold_number < 3
        ), f"Fold number must be between 0 and 2, got {fold_number}"
        self.fold_number = fold_number
        self.fold_type = fold_type

        if fold_type not in ["intrasession", "intersubject", "intersession"]:
            raise ValueError(
                f"Invalid fold_type '{fold_type}'. Must be one of ['intrasession', 'intersubject', 'intersession']."
            )

    def get_sampling_intervals(
        self,
        split: Optional[Literal["train", "valid", "test"]] = None,
    ):

        if split is None:
            return {rid: self.get_recording(rid).domain for rid in self.recording_ids}

        if split not in ["train", "valid", "test"]:
            raise ValueError(
                f"Invalid split '{split}'. Must be one of ['train', 'valid', 'test']."
            )

        if self.fold_type == "intrasession":
            key = f"splits.fold_{self.fold_number}.{split}"
            return {
                rid: self.get_recording(rid).get_nested_attribute(key)
                for rid in self.recording_ids
            }
        elif self.fold_type in ("intersubject", "intersession"):
            prefix = "subject" if self.fold_type == "intersubject" else "session"
            key = f"splits.{prefix}_fold_{self.fold_number}_assignment"
            result = {}
            for rid in self.recording_ids:
                rec = self.get_recording(rid)
                assignment = str(rec.get_nested_attribute(key))
                if assignment == split:
                    result[rid] = rec.domain
            return result

    def get_recording_hook(self, data: Data):
        # This dataset does not have unique channel ids across sessions
        # so we prefix the channel ids with the session id to ensure uniqueness
        if self.uniquify_channel_ids:
            data.channels.id = np_string_prefix(f"{data.session.id}/", data.channels.id)

        super().get_recording_hook(data)
