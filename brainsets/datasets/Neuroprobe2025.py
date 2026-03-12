from __future__ import annotations

from numbers import Integral
from pathlib import Path
import re
from typing import Callable, Literal, Optional, get_args

import numpy as np
from temporaldata import Data, Interval

from torch_brain.dataset import Dataset, SEEGDatasetMixin


SubsetTier = Literal["full", "lite", "nano"]
LabelMode = Literal["binary", "multiclass"]
Regime = Literal["SS-SM", "SS-DM", "DS-DM"]
Split = Literal["train", "val", "test"]

VALID_SUBSET_TIERS = get_args(SubsetTier)
VALID_LABEL_MODES = get_args(LabelMode)
VALID_REGIMES = get_args(Regime)
VALID_SPLITS = get_args(Split)

# Supported Neuroprobe task labels available in processed H5 splits.
VALID_TASKS = (
    "delta_volume",
    "face_num",
    "frame_brightness",
    "global_flow",
    "gpt2_surprisal",
    "local_flow",
    "onset",
    "pitch",
    "speech",
    "volume",
    "word_gap",
    "word_head_pos",
    "word_index",
    "word_length",
    "word_part_speech",
)

H5_REGIME_BY_REGIME: dict[Regime, str] = {
    "SS-SM": "within_session",
    "SS-DM": "cross_x",
    "DS-DM": "cross_x",
}

# Split interval and channel-mask keys share one selector key:
# <subset_tier>$<label_mode>$<eval_setting>$<task>$fold<k>$<split>

# Neuroprobe benchmark constants (mirrors neuroprobe.config)
# Fixed train subject id for benchmark-default DS-DM configuration.
DS_DM_TRAIN_SUBJECT_ID = 2
# Fixed train trial id for benchmark-default DS-DM configuration.
DS_DM_TRAIN_TRIAL_ID = 4

# Eligible (subject, trial) pairs for Neuroprobe Lite benchmark mode.
NEUROPROBE_LITE_SUBJECT_TRIALS = {
    (1, 1),
    (1, 2),
    (2, 0),
    (2, 4),
    (3, 0),
    (3, 1),
    (4, 0),
    (4, 1),
    (7, 0),
    (7, 1),
    (10, 0),
    (10, 1),
}

# Eligible (subject, trial) pairs for Neuroprobe Nano benchmark mode.
NEUROPROBE_NANO_SUBJECT_TRIALS = {
    (1, 1),
    (2, 4),
    (3, 1),
    (4, 0),
    (7, 1),
    (10, 1),
}

# Per-subject trials ranked by duration, used by full SS-DM train selection.
NEUROPROBE_LONGEST_TRIALS_FOR_SUBJECT: dict[int, list[int]] = {
    1: [0, 1],
    2: [4, 6],
    3: [2, 1],
    4: [2, 1],
    5: [0],
    6: [0, 2],
    7: [1, 0],
    8: [0],
    9: [0],
    10: [1, 0],
}

# Strict parser for canonical recording ids like "sub_1_trial004".
_RECORDING_ID_RE = re.compile(r"^sub_(\d+)_trial(\d{3})$")


def _to_recording_id(subject: int, session: int) -> str:
    # Normalize integer subject/session into the canonical H5 recording id.
    if (
        isinstance(subject, bool)
        or not isinstance(subject, int)
        or subject < 0
        or isinstance(session, bool)
        or not isinstance(session, int)
        or not (0 <= session <= 999)
    ):
        raise ValueError(
            "_to_recording_id received invalid subject/session values: "
            f"subject={subject!r}, session={session!r}. Expected subject to be a "
            "non-negative integer and session to be an integer in 0..999."
        )
    return f"sub_{subject}_trial{session:03d}"


def _from_recording_id(recording_id: str) -> tuple[int, int]:
    # Parse canonical ids like "sub_1_trial004" back into integers.
    match = _RECORDING_ID_RE.match(recording_id)
    if match is None:
        raise ValueError(
            f"Invalid recording_id '{recording_id}'. Expected 'sub_<subject>_trial<session>' "
            "with a zero-padded 3-digit session."
        )
    return int(match.group(1)), int(match.group(2))


class Neuroprobe2025(SEEGDatasetMixin, Dataset):
    """Neuroprobe 2025 iEEG benchmark dataset.

    Each instance operates in exactly one of two mutually-exclusive modes:
    - Split-selection mode (`recording_ids=None`): active recordings are resolved
      from Neuroprobe benchmark selectors.
    - Explicit-recording mode (`recording_ids` provided): active recordings come
      directly from that subset and split selectors must not be provided.

    Args:
        root: Root directory containing processed Neuroprobe artifacts.
        recording_ids: Optional explicit recording-id subset to expose from disk.
            If omitted, the dataset uses benchmark-required recording ids inferred
            from ``subset_tier/test_subject/test_session/split/label_mode/task/regime/fold``.
        transform: Optional sample transform.
        subset_tier: One of ``"full"``, ``"lite"``, ``"nano"``. Required in
            split-selection mode; must be omitted in explicit-recording mode.
        test_subject: Target test subject id (Neuroprobe semantics). Required in
            split-selection mode; must be omitted in explicit-recording mode.
        test_session: Target test trial/session id (Neuroprobe semantics). Required
            in split-selection mode; must be omitted in explicit-recording mode.
        split: One of ``"train"``, ``"val"``, ``"test"``. Required in
            split-selection mode; must be omitted in explicit-recording mode.
        label_mode: One of ``"binary"``, ``"multiclass"``. Defaults to ``"binary"``
            in split-selection mode.
        task: Neuroprobe task name. Defaults to ``"speech"`` in split-selection mode.
        regime: One of ``"SS-SM"``, ``"SS-DM"``, ``"DS-DM"``. Defaults to
            ``"SS-SM"`` in split-selection mode.
        fold: Fold index used only in split-selection mode. Defaults to ``0`` in
            split-selection mode and must be omitted in explicit-recording mode.
            Valid values depend on regime:
            - ``within_session``: valid {0, 1}
            - ``cross_x``: forced to 0
        uniquify_channel_ids: Set of channel-ID components used for prefixing
            via ``SEEGDatasetMixin``. Supported values:
            ``{"subject_id"}``, ``{"session_id"}``, or both.
            Defaults to empty set (no prefixing). This affects IDs returned by
            ``get_recording(...)`` and therefore ``get_channel_ids(...)``.
            With no prefixing, channel IDs are returned as stored and may
            repeat across recordings.
        dirname: Subdirectory under ``root`` containing recording H5 files.
    """

    def __init__(
        self,
        root: str,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        *,
        subset_tier: SubsetTier | None = None,
        test_subject: int | None = None,
        test_session: int | None = None,
        split: Split | None = None,
        label_mode: LabelMode | None = None,
        task: str | None = None,
        regime: Regime | None = None,
        fold: int | None = None,
        uniquify_channel_ids: set[str] | frozenset[str] = frozenset(),
        dirname: str = "neuroprobe_2025",
        **kwargs,
    ):
        # Resolve and validate constructor inputs before touching dataset records.
        self._dataset_dir = Path(root) / dirname

        # XOR recording-source behavior:
        # - no recording_ids => use split-resolved benchmark recordings
        # - recording_ids provided => use the explicit subset as active recordings
        use_split_selection = recording_ids is None
        self._use_split_selection = use_split_selection
        if use_split_selection:
            label_mode = label_mode or "binary"
            task = task or "speech"
            regime = regime or "SS-SM"
            fold = fold or 0

            self.subset_tier = subset_tier
            self.label_mode = label_mode
            self.task = task
            self.regime = regime
            self.test_subject = test_subject
            self.test_session = test_session
            self.split = split

            self._validate_split_args()
            self.h5_regime = H5_REGIME_BY_REGIME[self.regime]
            self.fold = self._resolve_fold(fold=fold)
            active_recording_ids = self._split_recording_ids()
        else:
            unexpected_split_args = [
                name
                for name, value in (
                    ("subset_tier", subset_tier),
                    ("test_subject", test_subject),
                    ("test_session", test_session),
                    ("split", split),
                    ("label_mode", label_mode),
                    ("task", task),
                    ("regime", regime),
                    ("fold", fold),
                )
                if value is not None
            ]
            if unexpected_split_args:
                raise ValueError(
                    "When recording_ids is provided (explicit-recording mode), split-selection args "
                    "must be omitted. Unexpected args: "
                    f"{', '.join(unexpected_split_args)}."
                )
            active_recording_ids = self._resolve_requested_recording_ids(recording_ids)
        if not active_recording_ids:
            raise ValueError(
                "No active recording_ids resolved for Neuroprobe2025 construction."
            )

        super().__init__(
            dataset_dir=self._dataset_dir,
            recording_ids=active_recording_ids,
            transform=transform,
            namespace_attributes=["subject.id", "channels.id"],
            **kwargs,
        )

        # Opt-in hook behavior: keep channel IDs as stored unless caller enables
        # subject/session-based prefixing via `uniquify_channel_ids`.
        if not isinstance(uniquify_channel_ids, (set, frozenset)):
            raise TypeError(
                "uniquify_channel_ids must be a set/frozenset containing "
                "'subject_id' and/or 'session_id'."
            )
        self.seeg_dataset_mixin_uniquify_channel_ids = frozenset(uniquify_channel_ids)
        # Validate components eagerly so config errors fail at construction time.
        self._normalize_channel_uniquify_components()

    def get_sampling_intervals(self) -> dict[str, Interval]:
        """Return split-specific sampling intervals for this dataset instance."""
        if not self._use_split_selection:
            raise RuntimeError(
                "get_sampling_intervals is only available in split-selection mode."
            )
        interval_path = self._interval_attr_path()
        return {
            rid: self.get_recording(rid).get_nested_attribute(interval_path)
            for rid in self.recording_ids
        }

    def get_domain_intervals(self) -> dict[str, Interval]:
        """Return full-domain intervals for active recordings."""
        return {rid: self.get_recording(rid).domain for rid in self.recording_ids}

    def get_sampling_rate(self, recording_id: str | None = None) -> float:
        """Return recording sampling rate in Hz."""
        _ = recording_id
        return 2048.0

    def get_channel_arrays(
        self, recording_id: str, *, included_only: bool = False
    ) -> dict[str, np.ndarray | None]:
        """Return normalized channel metadata arrays for one recording."""
        full_arrays = self._get_full_channel_arrays(recording_id)
        included_mask = full_arrays["included_mask"]
        if included_only:
            indices = np.flatnonzero(included_mask)
            ids = full_arrays["ids"][indices]
            names = full_arrays["names"][indices]
            lip = None if full_arrays["lip"] is None else full_arrays["lip"][indices]
            return {
                "ids": ids,
                "names": names,
                "included_mask": np.ones(len(indices), dtype=bool),
                "lip": lip,
                "indices": indices,
            }

        return {
            "ids": full_arrays["ids"],
            "names": full_arrays["names"],
            "included_mask": included_mask,
            "lip": full_arrays["lip"],
            "indices": np.arange(len(full_arrays["ids"]), dtype=int),
        }

    def _get_full_channel_arrays(
        self, recording_id: str
    ) -> dict[str, np.ndarray | None]:
        rec = self.get_recording(recording_id)
        channels = rec.channels

        ids = np.asarray(channels.id).astype(str)
        names = np.asarray(getattr(channels, "name", ids)).astype(str)
        included_mask = np.asarray(
            getattr(channels, "included", np.ones(len(ids), dtype=bool)),
            dtype=bool,
        )

        if len(names) != len(ids):
            raise ValueError(
                f"Channel name length mismatch for recording '{recording_id}': "
                f"len(names)={len(names)} vs len(ids)={len(ids)}"
            )
        if len(included_mask) != len(ids):
            raise ValueError(
                f"Channel mask length mismatch for recording '{recording_id}': "
                f"len(mask)={len(included_mask)} vs len(ids)={len(ids)}"
            )

        lip: np.ndarray | None = None
        if all(
            hasattr(channels, attr)
            for attr in ("localization_L", "localization_I", "localization_P")
        ):
            lip = np.stack(
                (
                    np.asarray(channels.localization_L, dtype=float),
                    np.asarray(channels.localization_I, dtype=float),
                    np.asarray(channels.localization_P, dtype=float),
                ),
                axis=1,
            )
            if len(lip) != len(ids):
                raise ValueError(
                    f"Channel localization length mismatch for recording '{recording_id}': "
                    f"len(lip)={len(lip)} vs len(ids)={len(ids)}"
                )

        return {
            "ids": ids,
            "names": names,
            "included_mask": included_mask,
            "lip": lip,
        }

    def get_recording_hook(self, data: Data):
        """Apply split-specific channel inclusion mask when available."""
        if not self._use_split_selection:
            super().get_recording_hook(data)
            return

        # Override generic channel masks with the split-specific inclusion mask.
        try:
            data.channels.included = data.get_nested_attribute(
                self._channel_mask_attr_path()
            )
        except (AttributeError, KeyError):
            # If a recording doesn't expose a split-specific mask, keep default mask.
            pass
        super().get_recording_hook(data)

    def describe_selection(self) -> dict[str, object]:
        """Return a compact debug summary of the resolved split selection."""
        summary: dict[str, object] = {
            "uses_split_selection": self._use_split_selection,
            "active_recording_ids": list(self.recording_ids),
        }
        if not self._use_split_selection:
            return summary

        # Expose resolved split internals to make dataset/debug logs self-explanatory.
        summary.update(
            {
                "subset_tier": self.subset_tier,
                "label_mode": self.label_mode,
                "task": self.task,
                "regime": self.regime,
                "h5_regime": self.h5_regime,
                "fold": self.fold,
                "split": self.split,
                "test_subject": self.test_subject,
                "test_session": self.test_session,
                "test_recording_id": _to_recording_id(
                    self.test_subject, self.test_session
                ),
                "interval_path": self._interval_attr_path(),
                "split_key": self._split_key(),
            }
        )
        return summary

    # Path/key builders.
    def _split_key(self) -> str:
        """Return the canonical key shared under both `splits` and `channels`."""
        return (
            f"{self.subset_tier}${self.label_mode}${self.h5_regime}${self.task}$"
            f"fold{self.fold}${self.split}"
        )

    def _interval_attr_path(self) -> str:
        # Primary split interval path under data.splits.
        return f"splits.{self._split_key()}"

    def _channel_mask_attr_path(self) -> str:
        # Return the nested attribute path for the split-specific channel mask.
        return f"channels.{self._split_key()}"

    def _validate_split_args(self) -> None:
        # Keep constructor strict so invalid benchmark configs fail immediately.
        if self.subset_tier not in VALID_SUBSET_TIERS:
            raise ValueError(
                f"Invalid subset_tier '{self.subset_tier}'. Must be one of {VALID_SUBSET_TIERS}."
            )
        if self.label_mode not in VALID_LABEL_MODES:
            raise ValueError(
                f"Invalid label_mode '{self.label_mode}'. Must be one of {VALID_LABEL_MODES}."
            )
        if self.task not in VALID_TASKS:
            raise ValueError(
                f"Invalid task '{self.task}'. Must be one of {VALID_TASKS}."
            )
        if self.regime not in VALID_REGIMES:
            raise ValueError(
                f"Invalid regime '{self.regime}'. Must be one of {VALID_REGIMES}."
            )
        if self.split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{self.split}'. Must be one of {VALID_SPLITS}."
            )

        if not isinstance(self.test_subject, Integral) or isinstance(
            self.test_subject, bool
        ):
            raise TypeError(
                "test_subject must be an int, got "
                f"{type(self.test_subject).__name__}."
            )
        if not isinstance(self.test_session, Integral) or isinstance(
            self.test_session, bool
        ):
            raise TypeError(
                "test_session must be an int, got "
                f"{type(self.test_session).__name__}."
            )

        h5_regime = H5_REGIME_BY_REGIME[self.regime]
        if h5_regime == "cross_x" and self.subset_tier == "nano":
            raise ValueError(
                "subset_tier 'nano' is not compatible with cross_x regimes."
            )

        if self.regime == "DS-DM" and self.test_subject == DS_DM_TRAIN_SUBJECT_ID:
            raise ValueError(
                "DS-DM benchmark-default uses subject 2 as fixed train subject; "
                "test_subject cannot be 2."
            )

        # Enforce benchmark-allowed target subject/session pairs per subset-tier/regime.
        requested_pair = (self.test_subject, self.test_session)
        if (
            self.subset_tier == "lite"
            and requested_pair not in NEUROPROBE_LITE_SUBJECT_TRIALS
        ):
            raise ValueError(
                f"Target pair {requested_pair} is not in NEUROPROBE_LITE_SUBJECT_TRIALS."
            )
        if (
            self.subset_tier == "nano"
            and requested_pair not in NEUROPROBE_NANO_SUBJECT_TRIALS
        ):
            raise ValueError(
                f"Target pair {requested_pair} is not in NEUROPROBE_NANO_SUBJECT_TRIALS."
            )
        if self.regime == "SS-DM" and self.subset_tier == "full":
            longest_trials = NEUROPROBE_LONGEST_TRIALS_FOR_SUBJECT.get(
                self.test_subject, []
            )
            if len(longest_trials) < 2:
                raise ValueError(
                    "SS-DM full benchmark-default requires at least two longest trials "
                    f"for subject {self.test_subject}, found {longest_trials}."
                )
            if self.test_session not in longest_trials:
                raise ValueError(
                    "SS-DM full benchmark-default only supports target sessions present "
                    f"in NEUROPROBE_LONGEST_TRIALS_FOR_SUBJECT for subject {self.test_subject}: "
                    f"{longest_trials}."
                )

    def _resolve_fold(self, fold: int) -> int:
        # SS-SM exposes two folds; SS-DM/DS-DM are fixed to fold0 only.
        regime = self.regime
        if not isinstance(fold, Integral) or isinstance(fold, bool):
            raise TypeError(f"fold must be an int, got {type(fold).__name__}.")

        if regime not in VALID_REGIMES:
            raise ValueError(
                f"Invalid regime '{regime}'. Must be one of {VALID_REGIMES}."
            )

        if regime == "SS-SM":
            if fold not in (0, 1):
                raise ValueError(
                    f"Fold for regime '{regime}' must be 0 or 1, got {fold}."
                )
            return fold

        # SS-DM / DS-DM
        if fold != 0:
            raise ValueError(f"Fold for regime '{regime}' must be 0, got {fold}.")
        return fold

    def _resolve_requested_recording_ids(self, recording_ids: list[str]) -> list[str]:
        # Normalize explicit recording-id subsets to a stable, de-duplicated order.
        if not recording_ids:
            raise ValueError(
                "When using explicit-recording mode, recording_ids must contain at least one id."
            )
        ids = recording_ids
        ids = sorted(set(ids))
        if not ids:
            raise ValueError(
                "When using explicit-recording mode, recording_ids must contain at least one id."
            )

        # Parse each id once so errors are raised consistently at construction.
        for rid in ids:
            _from_recording_id(rid)
        return ids

    def _split_recording_ids(self) -> list[str]:
        """Resolve split-participating recording ids for constructor inputs."""
        test_recording_id = _to_recording_id(self.test_subject, self.test_session)

        if self.regime == "SS-SM":
            # Within-session uses a single target recording for all splits.
            return [test_recording_id]

        if self.regime == "SS-DM":
            # Cross-session trains on a different session from the same subject.
            if self.split == "train":
                return [self._ss_dm_train_recording_id_for_selection()]
            # Val/test evaluate on the requested target recording.
            return [test_recording_id]

        # DS-DM
        if self.split == "train":
            # Cross-subject benchmark-default uses a fixed train anchor recording.
            return [_to_recording_id(DS_DM_TRAIN_SUBJECT_ID, DS_DM_TRAIN_TRIAL_ID)]
        # Val/test evaluate on the requested held-out target recording.
        return [test_recording_id]

    def _ss_dm_train_recording_id_for_selection(
        self,
    ) -> str:
        # Compute SS-DM train recording using benchmark-default selection rules.
        if self.subset_tier == "lite":
            # Lite mode always defines exactly two eligible trials per subject.
            # Training should use "the other lite trial" relative to the test trial.
            subject_trials = sorted(
                trial
                for subject, trial in NEUROPROBE_LITE_SUBJECT_TRIALS
                if subject == self.test_subject
            )
            if len(subject_trials) != 2:
                raise ValueError(
                    "SS-DM lite benchmark-default expects exactly two lite trials "
                    f"for subject {self.test_subject}, found {subject_trials}."
                )
            if self.test_session not in subject_trials:
                raise ValueError(
                    f"Target (test_subject={self.test_subject}, test_session={self.test_session}) "
                    "is not eligible for lite SS-DM benchmark-default."
                )
            # Start with the first lite trial; if that is the test trial, swap to the second.
            train_session = subject_trials[0]
            if train_session == self.test_session:
                train_session = subject_trials[1]
            return _to_recording_id(self.test_subject, train_session)

        if self.subset_tier == "full":
            # Full mode uses the longest-trial ordering table from the benchmark.
            # Normally pick the longest trial for training.
            longest_trials = NEUROPROBE_LONGEST_TRIALS_FOR_SUBJECT.get(
                self.test_subject, []
            )
            if len(longest_trials) < 2:
                raise ValueError(
                    "SS-DM full benchmark-default requires at least two longest trials "
                    f"for subject {self.test_subject}, found {longest_trials}."
                )
            # If the longest trial is already the test target, fall back to second-longest
            # to keep train/test recordings distinct.
            train_session = longest_trials[0]
            if train_session == self.test_session:
                train_session = longest_trials[1]
            return _to_recording_id(self.test_subject, train_session)

        raise ValueError(
            f"subset_tier '{self.subset_tier}' is not supported for SS-DM train selection."
        )
