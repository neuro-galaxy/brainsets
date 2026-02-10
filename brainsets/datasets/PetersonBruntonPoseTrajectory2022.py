import numpy as np
from pathlib import Path
from typing import Callable, Literal, Optional
from temporaldata import Interval

from torch_brain.dataset import Dataset


def _empty_interval() -> Interval:
    return Interval(start=np.array([]), end=np.array([]))


class PetersonBruntonPoseTrajectory2022(Dataset):
    def __init__(
        self,
        root: str,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        split_type: Literal[
            "subject_fold_0",
            "subject_fold_1",
            "subject_fold_2",
            "session_fold_0",
            "session_fold_1",
            "session_fold_2",
            "AllActiveBehavior_fold_0",
            "AllActiveBehavior_fold_1",
            "AllActiveBehavior_fold_2",
            "EatVsOther_fold_0",
            "EatVsOther_fold_1",
            "EatVsOther_fold_2",
            "TalkVsOther_fold_0",
            "TalkVsOther_fold_1",
            "TalkVsOther_fold_2",
        ] = "AllActiveBehavior_fold_0",
        dirname: str = "peterson_brunton_pose_trajectory_2022",
        **kwargs,
    ):
        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "channels.id"],
            **kwargs,
        )
        self.split_type = split_type

    def get_sampling_intervals(
        self,
        split: Optional[Literal["train", "valid", "test"]] = None,
    ):
        if split is None:
            return {rid: self.get_recording(rid).domain for rid in self.recording_ids}
        if split not in ("train", "valid", "test"):
            raise ValueError("split must be 'train', 'valid', 'test', or None.")

        st = self.split_type
        if st.startswith("subject_fold_"):
            fold_num = st.replace("subject_fold_", "")
            assignment_key = f"subject_fold_{fold_num}_assignment"
            out = {}
            for rid in self.recording_ids:
                rec = self.get_recording(rid)
                assign = rec.get_nested_attribute(f"splits.{assignment_key}")
                out[rid] = rec.domain if assign == split else _empty_interval()
            return out
        if st.startswith("session_fold_"):
            fold_num = st.replace("session_fold_", "")
            assignment_key = f"session_fold_{fold_num}_assignment"
            out = {}
            for rid in self.recording_ids:
                rec = self.get_recording(rid)
                assign = rec.get_nested_attribute(f"splits.{assignment_key}")
                out[rid] = rec.domain if assign == split else _empty_interval()
            return out
        interval_key = f"{st}_{split}"
        return {
            rid: self.get_recording(rid).get_nested_attribute(f"splits.{interval_key}")
            for rid in self.recording_ids
        }
