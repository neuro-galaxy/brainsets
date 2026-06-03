from typing import Callable, Optional, Literal, get_args
from pathlib import Path
from torch_brain.utils import np_string_prefix
from temporaldata import Data

from torch_brain.dataset import Dataset

from ._utils import get_processed_dir

SplitCategory = Literal["behavior_agnostic", "behavior_relevant"]
VALID_SPLIT_CATEGORIES = get_args(SplitCategory)

FoldType = Literal["intrasession", "intersubject", "intersession"]
VALID_FOLD_TYPES = get_args(FoldType)

N_FOLDS = 3


class CmiMIPDB2016(Dataset):
    """CMI Multimodal Resource for Studying Information Processing in the
    Developing Brain (MIPDB) — EEG recordings across multiple paradigms from
    126 psychiatric and healthy participants aged 6-44 years.

    .. admonition:: Preprocessing

        To download and prepare this dataset, run
        ``brainsets prepare cmi_mipdb_2016``.

    Args:
        root: Root directory for the dataset.
        recording_ids: List of recording IDs to load.
        transform: Data transformation to apply.
        uniquify_channel_ids: Whether to prefix channel IDs with session ID
            to ensure uniqueness. Defaults to True.
        split_category: Top-level split category. Must be one of:

            - ``"behavior_agnostic"``: No folds, train/valid only. Always available for all recordings.
            - ``"behavior_relevant"``: K-fold with train/valid/test. Only available for recordings that have paradigm annotations.

            Defaults to ``"behavior_agnostic"``.
        fold_type: The splitting strategy. Must be one of:

            - ``"intrasession"``: Epoch-level split within each session.
            - ``"intersubject"``: Subject-level split.
            - ``"intersession"``: Session-level split.

            Defaults to ``"intrasession"``.
        fold_number: Cross-validation fold index (0 to 2). Only used when
            *split_category* is ``"behavior_relevant"``. Defaults to 0.
        dirname: Subdirectory for the dataset. Defaults to ``"cmi_mipdb_2016"``.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        uniquify_channel_ids: bool = True,
        split_category: SplitCategory = "behavior_agnostic",
        fold_type: FoldType = "intrasession",
        fold_number: int = 0,
        dirname: str = "cmi_mipdb_2016",
        **kwargs,
    ):
        if root is None:
            root = get_processed_dir()
        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "channels.id"],
            **kwargs,
        )

        self.uniquify_channel_ids = uniquify_channel_ids

        if split_category not in VALID_SPLIT_CATEGORIES:
            raise ValueError(
                f"Invalid split_category '{split_category}'. "
                f"Must be one of {VALID_SPLIT_CATEGORIES}."
            )
        if fold_type not in VALID_FOLD_TYPES:
            raise ValueError(
                f"Invalid fold_type '{fold_type}'. Must be one of {VALID_FOLD_TYPES}."
            )

        self.split_category = split_category
        self.fold_type = fold_type

        if split_category == "behavior_relevant":
            if not (isinstance(fold_number, int) and not isinstance(fold_number, bool)):
                raise ValueError(
                    f"fold_number must be an int, got "
                    f"{type(fold_number).__name__}: {fold_number!r}"
                )
            if not (0 <= fold_number < N_FOLDS):
                raise ValueError(
                    f"fold_number must be between 0 and {N_FOLDS - 1}, "
                    f"got {fold_number}"
                )
        self.fold_number = fold_number

    def get_sampling_intervals(
        self,
        split: Optional[Literal["train", "valid", "test"]] = None,
    ):
        if split is None:
            return {rid: self.get_recording(rid).domain for rid in self.recording_ids}

        if self.split_category == "behavior_agnostic":
            if split not in ("train", "valid"):
                raise ValueError(
                    f"Behavior-agnostic splits only support 'train' and 'valid', "
                    f"got '{split}'."
                )
            return self._get_default_intervals(split)
        else:
            if split not in ("train", "valid", "test"):
                raise ValueError(
                    f"Invalid split '{split}'. "
                    f"Must be one of ['train', 'valid', 'test']."
                )
            return self._get_behavior_intervals(split)

    def _get_default_intervals(self, split: str) -> dict:
        if self.fold_type == "intrasession":
            key = f"splits.behavior_agnostic.intrasession.{split}"
            return {
                rid: self.get_recording(rid).get_nested_attribute(key)
                for rid in self.recording_ids
            }

        assignment_key = f"splits.behavior_agnostic.{self.fold_type}_assignment"
        result = {}
        for rid in self.recording_ids:
            rec = self.get_recording(rid)
            assignment = str(rec.get_nested_attribute(assignment_key))
            if assignment == split:
                result[rid] = rec.domain
        return result

    def _get_behavior_intervals(self, split: str) -> dict:
        if self.fold_type == "intrasession":
            key = (
                f"splits.behavior_relevant.intrasession"
                f".fold_{self.fold_number}.{split}"
            )
            result = {}
            for rid in self.recording_ids:
                rec = self.get_recording(rid)
                if not hasattr(rec.splits, "behavior_relevant"):
                    continue
                result[rid] = rec.get_nested_attribute(key)
            return result

        assignment_key = (
            f"splits.behavior_relevant"
            f".{self.fold_type}_fold_{self.fold_number}_assignment"
        )
        result = {}
        for rid in self.recording_ids:
            rec = self.get_recording(rid)
            if not hasattr(rec.splits, "behavior_relevant"):
                continue
            assignment = str(rec.get_nested_attribute(assignment_key))
            if assignment == split:
                result[rid] = rec.domain
        return result

    def get_recording_hook(self, data: Data):
        if self.uniquify_channel_ids:
            data.channels.id = np_string_prefix(f"{data.session.id}/", data.channels.id)

        super().get_recording_hook(data)
