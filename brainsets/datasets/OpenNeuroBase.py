from pathlib import Path
from typing import Callable, Literal, Optional, get_args

import numpy as np

from temporaldata import Data, Interval
from torch_brain.dataset import MultiChannelDatasetMixin, Dataset

from brainsets.utils.split import generate_string_kfold_assignment


OpenNeuroSplitType = Literal["intrasession", "intersubject", "intersession"]

VALID_SPLIT_TYPES = get_args(OpenNeuroSplitType)


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
        split_type (SplitType): The split strategy to use, must be one of
            'intrasession', 'intersubject', or 'intersession'.
        recording_ids (Optional[list[str]]): List of recording IDs to include,
            or None to use all available recordings.
        transform (Optional[Callable]): Optional sample transform.
        uniquify_channel_ids_with_subject: Whether to prefix channel IDs with
            ``subject.id`` via ``MultiChannelDatasetMixin``.
            Defaults to ``True``.
        uniquify_channel_ids_with_session: Whether to prefix channel IDs with
            ``session.id`` via ``MultiChannelDatasetMixin``.
            Defaults to ``False``.
        task_paradigm: The task paradigm of the dataset. Depends on the dataset.
            Defaults to None.
        split_ratios: The train/val ratios of the split. Must contain exactly two values (train, val). Sum must between 0 and 1.
            If sum is less than 1, the test ratio is calculated as 1 - sum of train/val ratios. If sum is 1, no test split is generated.
            If sum is less than 0 or greater than 1, an error is raised.
        seed: The seed for the random number generator. Defaults to 42.
    """

    def __init__(
        self,
        root: str,
        dataset_dir: str,
        split_type: OpenNeuroSplitType,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        uniquify_channel_ids_with_subject: bool = False,
        uniquify_channel_ids_with_session: bool = True,
        task_paradigm: str | None = None,
        split_ratios: tuple[float, float] = (0.9, 0.1),
        seed: int = 42,
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

        # Configure subject/session-based channel-id prefixing behavior.
        self.multichannel_dataset_mixin_uniquify_channel_ids_with_subject = (
            uniquify_channel_ids_with_subject
        )
        self.multichannel_dataset_mixin_uniquify_channel_ids_with_session = (
            uniquify_channel_ids_with_session
        )

        self.split_ratios = self._validate_split_ratios(split_ratios)

        self.task_paradigm = task_paradigm

        self.seed = seed

    def _validate_split_ratios(
        self, split_ratios: tuple[float, float]
    ) -> tuple[float, float]:
        """Validate the split ratios."""
        if any(ratio < 0 for ratio in split_ratios):
            raise ValueError("`split_ratios` cannot contain negative values")
        if sum(split_ratios) < 0 or sum(split_ratios) > 1:
            raise ValueError("The sum of `split_ratios` must be between 0 and 1")
        return split_ratios

    def get_sampling_intervals(
        self,
        split: Optional[Literal["train", "val", "test"]] = None,
    ) -> dict[str, (Interval | str)]:
        """
        Retrieve the sampling intervals for each recording according to the specified split.

        If `split` is None, returns the full interval domain for every recording for unrestricted sampling.
        If a split ("train", "val", or "test") is provided, returns only the intervals (within each recording)
        eligible for sampling under the current split type and task paradigm.

        The selection of intervals is determined according to:
        - The current `self.split_type` (intrasession, intersubject, or intersession).
        - Whether a `self.task_paradigm` is specified, which influences the interval extraction.

        Args:
            split: One of "train", "val", or "test" to select intervals corresponding to that split,
                or None to retrieve the entire domain for all recordings.

        Returns:
            Dictionary mapping recording IDs to their valid Interval objects for sampling in the given split
            (or full Interval domain if split is None).

        Raises:
            ValueError: If the requested `split` or the dataset's `split_type` is not recognized/supported.
            KeyError: If a required split or assignment attribute is missing in a recording.

        Notes:
            - For behavioral-agnostic tasks (`task_paradigm is None`), intervals are defined based on recording domains and split logic.
            - For behavioral-relevant tasks, intervals may be further filtered or computed.

        """
        if split is None:
            return super().get_sampling_intervals()

        if self.task_paradigm is None:
            intervals = {}
            for rid in self.recording_ids:
                rec = self.get_recording(rid)
                intervals[rid] = self._get_behavior_agnostic_intervals(rec, split)
            return intervals
        else:
            intervals = {}
            for rid in self.recording_ids:
                rec = self.get_recording(rid)
                intervals[rid] = self._get_behavior_relevant_intervals(rec, split)
            return intervals

    def _get_behavior_agnostic_intervals(
        self,
        recording: Data,
        split: Literal["train", "val", "test"],
    ) -> Interval | str:
        """Get the behavior-agnostic sampling intervals for a given split.
        If the split is "intersubject" or "intersession", returns the assignment string for the current fold.
        """
        if self.split_type == "intrasession":
            # intrasession (causal) split
            starts = np.asarray(recording.domain.start)
            ends = np.asarray(recording.domain.end)
            durations = ends - starts

            train_ends = starts + durations * self.split_ratios[0]
            val_ends = train_ends + durations * self.split_ratios[1]
            test_ends = val_ends + durations * (1 - sum(self.split_ratios))

            if split == "train":
                return Interval(start=starts, end=train_ends)
            elif split == "val":
                return Interval(start=train_ends, end=val_ends)
            elif split == "test":
                return Interval(start=val_ends, end=test_ends)

        elif self.split_type == "intersubject" or self.split_type == "intersession":
            # n_folds determines how many "folds" are used for k-fold assignment in intersubject/intersession splitting.
            # It is based on the test split ratio: e.g., for test_ratio=0.1, assignment_n_folds=10, so each fold has ~10% data.
            # If test_ratio <= 0, only one fold is used.
            test_ratio = 1 - sum(self.split_ratios)
            if test_ratio <= 0:
                n_folds = 1
            else:
                n_folds = max(1, int(round(1.0 / test_ratio)))
            # fold_idx is the (zero-based) index of the current fold to assign as "val".
            # Here, fold 0 is always selected as the validation fold for assignments.
            fold_idx = 0

            if self.split_type == "intersubject":
                string_id = recording.subject.id
            elif self.split_type == "intersession":
                string_id = f"{recording.subject.id}_{recording.session.id}"

            assignments = generate_string_kfold_assignment(
                string_id=string_id,
                n_folds=n_folds,
                # By setting val_ratio to 0, we ensure that all folds
                # are assigned to train, plus the one fold assigned to test.
                val_ratio=0.0,
                seed=self.seed,
            )

            assignment = assignments[fold_idx]
            # By convention, the test fold is assigned to "val".
            if assignment == "test":
                assignment = "val"
            return assignment

        raise ValueError(
            f"Invalid split_type '{self.split_type}'. Must be one of {VALID_SPLIT_TYPES}."
        )

    def get_behavior_relevant_intervals(
        self,
        recording: Data,
        split: Literal["train", "val", "test"],
    ) -> Interval | str:
        """Get the behavior-relevant sampling intervals for a given split."""
        raise NotImplementedError(
            "`get_behavior_relevant_intervals` is not implemented yet."
        )
