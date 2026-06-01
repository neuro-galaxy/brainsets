import logging
import numpy as np
from pathlib import Path
from typing import Callable, Literal, Optional, get_args
from temporaldata import Interval

from torch_brain.dataset import Dataset
from torch_brain.dataset.mixins import MultiChannelDatasetMixin

from ..ajile_behavior_labels import (
    ACTIVE_BEHAVIOR_LABELS,
    ACTIVE_BEHAVIOR_TO_ID,
    ACTIVE_VS_INACTIVE_LABELS,
    ACTIVE_VS_INACTIVE_TO_ID,
    INACTIVE_BEHAVIORS,
)
from ._utils import get_processed_dir

logger = logging.getLogger(__name__)

PetersonBruntonSplitType = Literal["intersubject", "intersession", "intrasession"]
PetersonBruntonTaskType = Literal["active_vs_inactive", "behavior", "pose_estimation"]

VALID_SPLIT_TYPES = get_args(PetersonBruntonSplitType)
VALID_TASK_TYPES = get_args(PetersonBruntonTaskType)
N_FOLDS = 3


def _empty_interval() -> Interval:
    return Interval(start=np.array([]), end=np.array([]))


class PetersonBruntonPoseTrajectory2022(MultiChannelDatasetMixin, Dataset):
    ACTIVE_BEHAVIOR_LABELS = ACTIVE_BEHAVIOR_LABELS
    ACTIVE_BEHAVIOR_TO_ID = ACTIVE_BEHAVIOR_TO_ID
    INACTIVE_BEHAVIORS = INACTIVE_BEHAVIORS
    ACTIVE_VS_INACTIVE_LABELS = ACTIVE_VS_INACTIVE_LABELS
    ACTIVE_VS_INACTIVE_TO_ID = ACTIVE_VS_INACTIVE_TO_ID

    """AJILE12: ECoG and upper body pose trajectories from 12 human subjects
    during naturalistic movements.

    .. admonition:: Preprocessing

        To download and prepare this dataset, run
        ``brainsets prepare peterson_brunton_pose_trajectory_2022``.

    **Tasks:** Free behavior (naturalistic daily activities) with coarse behavior
    labels (Eat, Talk, TV, Computer/Phone, Other Activity, Sleep/Rest, Inactive)
    and upper-body pose trajectories from 9 keypoints.

    **Brain Regions:** Subject-specific ECoG coverage (grids, strips, depths)

    **Dataset Statistics**

    - **Subjects:** 12
    - **Total Sessions:** 55
    - **Neural Modality:** ECoG (intracranial electrophysiology)

    **Links**

    - Paper: `Peterson et al. (2022) <https://doi.org/10.1038/s41597-022-01280-y>`_
    - Dataset: `Dandiset 000055 <https://dandiarchive.org/dandiset/000055>`_

    **Split types**

    - ``"intrasession"``: Stratified epoch-level splits within each session.
    - ``"intersubject"``: Subject-level assignment (all sessions of a subject go
      to the same split).
    - ``"intersession"``: Session-level assignment.

    For ``"intersubject"`` and ``"intersession"``, the returned intervals depend
    on ``task_type``: ``"behavior"`` returns ``active_behavior_trials``,
    ``"active_vs_inactive"`` returns ``active_vs_inactive_trials``, and
    ``"pose_estimation"`` returns ``pose_valid_domain``.

    Args:
        root (str, optional): Root directory for the dataset. Defaults to
            ``processed_dir`` from brainsets config.
        recording_ids (list[str], optional): List of recording IDs to load.
        transform (Callable, optional): Data transformation to apply.
        fold_number (int, optional): Cross-validation fold index (0 to 2).
            Required when ``split`` is not None. Defaults to 0.
        split_type (PetersonBruntonSplitType, optional): Splitting strategy.
            Defaults to ``"intrasession"``.
        task_type (PetersonBruntonTaskType, optional): Which task's intervals
            to return. Defaults to ``"behavior"``.
        dirname (str, optional): Subdirectory for the dataset. Defaults to
            ``"peterson_brunton_pose_trajectory_2022"``.
        uniquify_channel_ids_with_session (bool, optional): Prefix channel IDs
            with session ID. Defaults to True.
        uniquify_channel_ids_with_subject (bool, optional): Prefix channel IDs
            with subject ID. Defaults to False.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        fold_number: int = 0,
        split_type: PetersonBruntonSplitType = "intrasession",
        task_type: PetersonBruntonTaskType = "behavior",
        dirname: str = "peterson_brunton_pose_trajectory_2022",
        uniquify_channel_ids_with_session: bool = True,
        uniquify_channel_ids_with_subject: bool = False,
        **kwargs,
    ):
        if root is None:
            root = get_processed_dir()

        if not (0 <= fold_number < N_FOLDS):
            raise ValueError(
                f"fold_number must be an integer between 0 and {N_FOLDS - 1}, "
                f"got {fold_number}"
            )
        if split_type not in VALID_SPLIT_TYPES:
            raise ValueError(
                f"Invalid split_type '{split_type}'. "
                f"Must be one of {VALID_SPLIT_TYPES}."
            )
        if task_type not in VALID_TASK_TYPES:
            raise ValueError(
                f"Invalid task_type '{task_type}'. "
                f"Must be one of {VALID_TASK_TYPES}."
            )

        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "channels.id"],
            **kwargs,
        )
        self.fold_number = fold_number
        self.split_type = split_type
        self.task_type = task_type
        self.multichannel_dataset_mixin_uniquify_channel_ids_with_session = (
            uniquify_channel_ids_with_session
        )
        self.multichannel_dataset_mixin_uniquify_channel_ids_with_subject = (
            uniquify_channel_ids_with_subject
        )

    def get_sampling_intervals(
        self,
        split: Optional[Literal["train", "valid", "test"]] = None,
    ):
        if split is None:
            return {rid: self.get_recording(rid).domain for rid in self.recording_ids}
        if split not in ("train", "valid", "test"):
            raise ValueError("split must be 'train', 'valid', 'test', or None.")

        if self.split_type == "intrasession":
            return self._get_intrasession_intervals(split)
        if self.split_type in ("intersubject", "intersession"):
            return self._get_intersubject_or_intersession_intervals(split)
        raise ValueError(f"Invalid split_type '{self.split_type}'.")

    def _get_intrasession_intervals(
        self, split: Literal["train", "valid", "test"]
    ) -> dict:
        if self.task_type == "active_vs_inactive":
            key = f"splits.active_vs_inactive_fold_{self.fold_number}_{split}"
        elif self.task_type == "pose_estimation":
            key = f"splits.pose_estimation_fold_{self.fold_number}_{split}"
        elif self.task_type == "behavior":
            key = f"splits.all_active_behavior_fold_{self.fold_number}_{split}"
        else:
            raise ValueError(f"Invalid task_type '{self.task_type}'.")

        result = {}
        for rid in self.recording_ids:
            recording = self.get_recording(rid)
            try:
                result[rid] = recording.get_nested_attribute(key)
            except AttributeError:
                logger.warning(
                    f"Recording {rid} does not have intrasession split "
                    f"'{key}' (likely too few trials for stratification). "
                    f"Returning empty interval."
                )
                result[rid] = _empty_interval()
        return result

    def _get_intersubject_or_intersession_intervals(
        self, split: Literal["train", "valid", "test"]
    ) -> dict:
        if self.split_type == "intersubject":
            assignment_key = f"splits.intersubject_fold_{self.fold_number}_assignment"
        else:
            assignment_key = f"splits.intersession_fold_{self.fold_number}_assignment"

        result = {}
        for rid in self.recording_ids:
            data = self.get_recording(rid)
            # str() guards against h5py returning bytes or numpy.str_
            assignment = str(data.get_nested_attribute(assignment_key))
            if assignment == split:
                if self.task_type == "pose_estimation":
                    result[rid] = data.pose_valid_domain
                elif self.task_type == "behavior":
                    result[rid] = data.active_behavior_trials
                elif self.task_type == "active_vs_inactive":
                    result[rid] = data.active_vs_inactive_trials
                else:
                    raise ValueError(f"Invalid task_type '{self.task_type}'.")
            else:
                result[rid] = _empty_interval()
        return result
