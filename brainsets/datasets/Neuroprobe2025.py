from __future__ import annotations

from numbers import Integral
from pathlib import Path
import re
from typing import Callable, ClassVar, Literal, Optional, get_args

import h5py
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

# Channel mask keys now mirror split-key semantics:
# included$<subset_tier>$<label_mode>$<eval_setting>$<task>$fold<k>$<split>

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

# Backwards-compatible aliases for callers importing these symbols directly.
ChannelView = SEEGDatasetMixin.ChannelView
RecordingInfo = SEEGDatasetMixin.RecordingInfo


def _to_recording_id(subject: int, session: int) -> str:
    # Normalize integer subject/session into the canonical H5 recording id.
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

    This dataset is split-driven: each instance is configured for one split
    (``train`` / ``val`` / ``test``) and one target test recording
    (``test_subject``, ``test_session``). Sampling intervals always come from
    split-specific H5 intervals.

    Args:
        root: Root directory containing processed Neuroprobe artifacts.
        recording_ids: Optional recording-id subset to expose from disk. If omitted,
            the dataset auto-prunes to the minimal benchmark-required recording ids
            for the selected ``regime/split/test_subject/test_session``.
        transform: Optional sample transform.
        subset_tier: One of ``"full"``, ``"lite"``, ``"nano"``.
        test_subject: Target test subject id (Neuroprobe semantics).
        test_session: Target test trial/session id (Neuroprobe semantics).
        split: One of ``"train"``, ``"val"``, ``"test"``.
        label_mode: One of ``"binary"``, ``"multiclass"``.
        task: Neuroprobe task name.
        regime: One of ``"SS-SM"``, ``"SS-DM"``, ``"DS-DM"``.
        fold: Fold index. Defaults depend on regime:
            - ``within_session``: default 0, valid {0, 1}
            - ``cross_x``: forced to 0
        uniquify_channel_ids: If ``True``, prefix returned channel IDs with
            subject/session context via ``SEEGDatasetMixin``.
        prune_to_split: If ``True``, only split-required recording ids are loaded into
            the dataset at construction time (i.e., ``recording_ids`` is resolved to the
            split selection). If ``False``, keep the caller-provided recording subset.
        dirname: Subdirectory under ``root`` containing recording H5 files.
    """

    # Fixed sampling rate for Neuroprobe2025 processed recordings.
    DEFAULT_SAMPLING_RATE_HZ = 2048.0
    seeg_dataset_mixin_sampling_rate_hz = DEFAULT_SAMPLING_RATE_HZ
    # Process-wide memo cache for recording compatibility checks.
    _RECORDING_COMPAT_ISSUE_CACHE: ClassVar[dict[tuple[object, ...], str | None]] = {}

    def __init__(
        self,
        root: str,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        *,
        subset_tier: SubsetTier,
        test_subject: int,
        test_session: int,
        split: Split,
        label_mode: LabelMode = "binary",
        task: str = "speech",
        regime: Regime = "SS-SM",
        fold: Optional[int] = None,
        uniquify_channel_ids: bool = False,
        prune_to_split: bool = False,
        dirname: str = "neuroprobe_2025",
        **kwargs,
    ):
        # Resolve and validate constructor inputs before touching dataset records.
        self._dataset_dir = Path(root) / dirname
        self._validate_constructor_args(
            subset_tier=subset_tier,
            label_mode=label_mode,
            task=task,
            regime=regime,
            split=split,
            test_subject=test_subject,
            test_session=test_session,
        )
        split_recording_ids = self._recording_ids_for_selection(
            subset_tier=subset_tier,
            regime=regime,
            split=split,
            test_subject=test_subject,
            test_session=test_session,
        )
        if recording_ids is None:
            recording_ids = split_recording_ids
        requested_recording_ids = self._resolve_requested_recording_ids(recording_ids)
        active_recording_ids = (
            split_recording_ids if prune_to_split else requested_recording_ids
        )

        super().__init__(
            dataset_dir=self._dataset_dir,
            recording_ids=active_recording_ids,
            transform=transform,
            namespace_attributes=["subject.id", "channels.id"],
            **kwargs,
        )

        self.subset_tier = subset_tier
        self.label_mode = label_mode
        self.task = task
        self.regime = regime
        self.h5_regime = H5_REGIME_BY_REGIME[regime]
        self.test_subject = test_subject
        self.test_session = test_session
        self.split = split
        self.fold = self._resolve_fold(fold)
        # Opt-in hook behavior: keep default channel IDs unchanged unless a caller
        # explicitly asks for recording-disambiguated IDs in returned recordings.
        self.seeg_dataset_mixin_uniquify_channel_ids = bool(uniquify_channel_ids)
        self.prune_to_split = prune_to_split

        self._validate_selection_compatibility()
        self._initialize_seeg_mixin_caches()

        self._prime_selected_recording_caches()

    def get_sampling_intervals(self) -> dict[str, Interval]:
        """Return split-specific sampling intervals for this dataset instance."""
        interval_path = self._interval_attr_path()
        return {
            rid: self.get_recording(rid).get_nested_attribute(interval_path)
            for rid in self._selected_recording_ids()
        }

    def get_domain_intervals(
        self, recording_ids: Optional[list[str]] = None
    ) -> dict[str, Interval]:
        """Return full-domain intervals."""
        if recording_ids is None:
            recording_ids = self.recording_ids
        return super().get_domain_intervals(recording_ids=recording_ids)

    def get_recording_hook(self, data: Data):
        """Apply split-specific channel inclusion mask when available."""
        # Override generic channel masks with the split-specific inclusion mask.
        try:
            data.channels.included = data.get_nested_attribute(self._channel_mask_attr_path())
        except (AttributeError, KeyError):
            # Compatibility is validated at init for selected split recordings.
            # If users manually access an out-of-split recording, keep default mask.
            pass
        super().get_recording_hook(data)

    def describe_selection(self) -> dict[str, object]:
        """Return a compact debug summary of the resolved split selection."""
        # Expose resolved internals to make dataset/debug logs self-explanatory.
        return {
            "subset_tier": self.subset_tier,
            "label_mode": self.label_mode,
            "task": self.task,
            "regime": self.regime,
            "h5_regime": self.h5_regime,
            "fold": self.fold,
            "split": self.split,
            "test_subject": self.test_subject,
            "test_session": self.test_session,
            "test_recording_id": self._test_recording_id(),
            "selected_recording_ids": list(self._selected_recording_ids()),
            "active_recording_ids": list(self.recording_ids),
            "interval_path": self._interval_attr_path(),
            "channel_mask_key": self._make_channel_mask_key(),
            "prune_to_split": self.prune_to_split,
        }

    def _validate_constructor_args(
        self,
        *,
        subset_tier: str,
        label_mode: str,
        task: str,
        regime: str,
        split: str,
        test_subject: int,
        test_session: int,
    ) -> None:
        # Keep constructor strict so invalid benchmark configs fail immediately.
        if subset_tier not in VALID_SUBSET_TIERS:
            raise ValueError(
                f"Invalid subset_tier '{subset_tier}'. Must be one of {VALID_SUBSET_TIERS}."
            )
        if label_mode not in VALID_LABEL_MODES:
            raise ValueError(
                f"Invalid label_mode '{label_mode}'. Must be one of {VALID_LABEL_MODES}."
            )
        if task not in VALID_TASKS:
            raise ValueError(f"Invalid task '{task}'. Must be one of {VALID_TASKS}.")
        if regime not in VALID_REGIMES:
            raise ValueError(
                f"Invalid regime '{regime}'. Must be one of {VALID_REGIMES}."
            )
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Must be one of {VALID_SPLITS}.")

        if not isinstance(test_subject, Integral) or isinstance(test_subject, bool):
            raise TypeError(
                f"test_subject must be an int, got {type(test_subject).__name__}."
            )
        if not isinstance(test_session, Integral) or isinstance(test_session, bool):
            raise TypeError(
                f"test_session must be an int, got {type(test_session).__name__}."
            )

        if H5_REGIME_BY_REGIME[regime] == "cross_x" and subset_tier == "nano":
            raise ValueError(
                "subset_tier 'nano' is not compatible with cross_x regimes."
            )

        if regime == "DS-DM" and test_subject == DS_DM_TRAIN_SUBJECT_ID:
            raise ValueError(
                "DS-DM benchmark-default uses subject 2 as fixed train subject; "
                "test_subject cannot be 2."
            )

    def _resolve_fold(self, fold: Optional[int]) -> int:
        # within_session exposes two folds; cross_x is fixed to fold0 only.
        if fold is not None and (
            not isinstance(fold, Integral) or isinstance(fold, bool)
        ):
            raise TypeError(
                f"fold must be an int when provided, got {type(fold).__name__}."
            )

        if self.h5_regime == "within_session":
            resolved = 0 if fold is None else fold
            if resolved not in (0, 1):
                raise ValueError(
                    f"Fold for regime '{self.regime}' must be 0 or 1, got {resolved}."
                )
            return resolved

        # cross_x
        resolved = 0 if fold is None else fold
        if resolved != 0:
            raise ValueError(
                f"Fold for regime '{self.regime}' must be 0 (cross_x has only fold0), got {resolved}."
            )
        return resolved

    def _resolve_requested_recording_ids(
        self, recording_ids: Optional[list[str]]
    ) -> list[str]:
        # Default to all H5 files in dataset_dir unless a subset is provided.
        ids = (
            [x.stem for x in self._dataset_dir.glob("*.h5")]
            if recording_ids is None
            else recording_ids
        )
        ids = sorted(set(ids))
        if not ids:
            raise ValueError(f"No recordings found in {self._dataset_dir}")

        # Parse each id once so errors are raised consistently at construction.
        for rid in ids:
            _from_recording_id(rid)
        return ids

    @classmethod
    def _recording_ids_for_selection(
        cls,
        *,
        subset_tier: SubsetTier,
        regime: Regime,
        split: Split,
        test_subject: int,
        test_session: int,
    ) -> list[str]:
        """Resolve split-participating recording ids for constructor inputs."""
        test_recording_id = _to_recording_id(test_subject, test_session)

        if regime == "SS-SM":
            # Within-session uses a single target recording for all splits.
            return [test_recording_id]

        if regime == "SS-DM":
            # Cross-session trains on a different session from the same subject.
            if split == "train":
                return [
                    cls._ss_dm_train_recording_id_for_selection(
                        subset_tier=subset_tier,
                        test_subject=test_subject,
                        test_session=test_session,
                    )
                ]
            # Val/test evaluate on the requested target recording.
            return [test_recording_id]

        # DS-DM
        if split == "train":
            # Cross-subject benchmark-default uses a fixed train anchor recording.
            return [_to_recording_id(DS_DM_TRAIN_SUBJECT_ID, DS_DM_TRAIN_TRIAL_ID)]
        # Val/test evaluate on the requested held-out target recording.
        return [test_recording_id]

    # Path/key builders.
    def _split_selector_key(self) -> str:
        """Return the canonical semantic split selector shared across key types."""
        return (
            f"{self.subset_tier}${self.label_mode}${self.h5_regime}${self.task}$"
            f"fold{self.fold}${self.split}"
        )

    def _interval_flat_key(self) -> str:
        # Match neuroprobe_2025 flat split key format written by pipeline.py.
        return f"{self._split_selector_key()}_intervals"

    def _interval_attr_path(self) -> str:
        # Primary split interval path (flat key under data.splits).
        return f"splits.{self._interval_flat_key()}"

    def _channel_mask_attr_path(self) -> str:
        # Return the nested attribute path for the split-specific channel mask.
        return f"channels.{self._make_channel_mask_key()}"

    def _make_channel_mask_key(self) -> str:
        # Match the key naming convention used by the neuroprobe preprocessing pipeline.
        return f"included${self._split_selector_key()}"

    @classmethod
    def _ss_dm_train_recording_id_for_selection(
        cls,
        *,
        subset_tier: SubsetTier,
        test_subject: int,
        test_session: int,
    ) -> str:
        # Compute SS-DM train recording using benchmark-default selection rules.
        if subset_tier == "lite":
            # Lite mode always defines exactly two eligible trials per subject.
            # Training should use "the other lite trial" relative to the test trial.
            subject_trials = sorted(
                trial
                for subject, trial in NEUROPROBE_LITE_SUBJECT_TRIALS
                if subject == test_subject
            )
            if len(subject_trials) != 2:
                raise ValueError(
                    "SS-DM lite benchmark-default expects exactly two lite trials "
                    f"for subject {test_subject}, found {subject_trials}."
                )
            if test_session not in subject_trials:
                raise ValueError(
                    f"Target (test_subject={test_subject}, test_session={test_session}) "
                    "is not eligible for lite SS-DM benchmark-default."
                )
            # Start with the first lite trial; if that is the test trial, swap to the second.
            train_session = subject_trials[0]
            if train_session == test_session:
                train_session = subject_trials[1]
            return _to_recording_id(test_subject, train_session)

        if subset_tier == "full":
            # Full mode uses the longest-trial ordering table from the benchmark.
            # Normally pick the longest trial for training.
            longest_trials = NEUROPROBE_LONGEST_TRIALS_FOR_SUBJECT.get(test_subject, [])
            if len(longest_trials) < 2:
                raise ValueError(
                    "SS-DM full benchmark-default requires at least two longest trials "
                    f"for subject {test_subject}, found {longest_trials}."
                )
            # If the longest trial is already the test target, fall back to second-longest
            # to keep train/test recordings distinct.
            train_session = longest_trials[0]
            if train_session == test_session:
                train_session = longest_trials[1]
            return _to_recording_id(test_subject, train_session)

        raise ValueError(
            f"subset_tier '{subset_tier}' is not supported for SS-DM train selection."
        )

    def _validate_selection_compatibility(self) -> None:
        # Validate both benchmark eligibility and concrete H5 availability for selected ids.
        # First enforce all high-level selection constraints.
        self._validate_required_recordings_present()
        # Enforce subset-tier/regime-specific benchmark allowlists before touching file internals.
        self._validate_target_pair_eligibility()

        incompatible = self._collect_incompatible_recordings()

        # Aggregate all incompatibilities into one error to reduce fix-iterate cycles.
        if incompatible:
            raise ValueError(
                self._format_compatibility_error(
                    "Selected configuration is incompatible with required recordings.",
                    incompatible,
                )
            )

    def _validate_required_recordings_present(self) -> None:
        # Ensure every recording needed by this split is present in recording_ids.
        missing_recordings = [
            rid
            for rid in self._selected_recording_ids()
            if rid not in self.recording_ids
        ]
        if missing_recordings:
            raise ValueError(
                self._format_compatibility_error(
                    "Selected split requires recordings not present in 'recording_ids'.",
                    {
                        rid: "missing from requested recording subset"
                        for rid in missing_recordings
                    },
                )
            )

    def _collect_incompatible_recordings(self) -> dict[str, str]:
        # Check each selected recording for exact split compatibility and collect failures.
        split_interval_attr_path = self._interval_attr_path()
        split_interval_flat_key = self._interval_flat_key()
        split_channel_mask_key = self._make_channel_mask_key()
        incompatible: dict[str, str] = {}
        for rid in self._selected_recording_ids():
            issue = self._recording_compatibility_issue(
                rid,
                split_interval_attr_path=split_interval_attr_path,
                split_interval_flat_key=split_interval_flat_key,
                split_channel_mask_key=split_channel_mask_key,
            )
            if issue is not None:
                incompatible[rid] = issue
        return incompatible

    def _recording_compatibility_issue(
        self,
        recording_id: str,
        *,
        split_interval_attr_path: str,
        split_interval_flat_key: str,
        split_channel_mask_key: str,
    ) -> str | None:
        # Return None if compatible, otherwise a concise reason string.
        h5_path = self._dataset_dir / f"{recording_id}.h5"
        try:
            stat = h5_path.stat()
            file_fingerprint: tuple[int, int] | tuple[None, None] = (
                stat.st_mtime_ns,
                stat.st_size,
            )
        except FileNotFoundError:
            file_fingerprint = (None, None)

        cache_key = (
            str(self._dataset_dir),
            recording_id,
            file_fingerprint,
            self.subset_tier,
            self.label_mode,
            self.h5_regime,
            self.task,
            self.fold,
            self.split,
            split_interval_flat_key,
            split_channel_mask_key,
        )
        if cache_key in self._RECORDING_COMPAT_ISSUE_CACHE:
            return self._RECORDING_COMPAT_ISSUE_CACHE[cache_key]

        issue: str | None = None
        try:
            with h5py.File(h5_path, "r") as h5:
                if "splits" not in h5:
                    issue = "missing 'splits' group"
                # Flat split key should exist directly under the top-level `splits` group.
                elif split_interval_flat_key not in h5["splits"]:
                    issue = f"missing split interval path '{split_interval_attr_path}'"
                # Channel mask key must exist for this exact split/subset-tier/task setting.
                elif "channels" not in h5 or split_channel_mask_key not in h5["channels"]:
                    issue = f"missing channel mask key 'channels.{split_channel_mask_key}'"
        except (FileNotFoundError, OSError) as exc:
            issue = f"unable to open H5 file: {exc}"

        self._RECORDING_COMPAT_ISSUE_CACHE[cache_key] = issue
        return issue

    def _validate_target_pair_eligibility(self) -> None:
        """Validate benchmark-default subject/session eligibility constraints."""
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

    def _test_recording_id(self) -> str:
        # Centralize canonical test recording-id construction to avoid drift.
        return _to_recording_id(self.test_subject, self.test_session)

    def _format_compatibility_error(
        self, header: str, issues: dict[str, str], max_examples: int = 6
    ) -> str:
        # Keep compatibility failures compact while still showing actionable examples.
        lines = [header]
        lines.append(
            "Selection: "
            f"subset_tier={self.subset_tier}, label_mode={self.label_mode}, task={self.task}, "
            f"regime={self.regime}, fold={self.fold}, split={self.split}, "
            f"test_subject={self.test_subject}, test_session={self.test_session}"
        )
        lines.append("First incompatible recordings:")
        for rid in sorted(issues)[:max_examples]:
            lines.append(f"  - {rid}: {issues[rid]}")
        if len(issues) > max_examples:
            lines.append(f"  ... and {len(issues) - max_examples} more")
        return "\n".join(lines)

    def _build_channel_view_cache(
        self, recording_ids: list[str]
    ) -> dict[str, SEEGDatasetMixin.ChannelView]:
        """Build full channel views for the provided recordings."""
        views: dict[str, SEEGDatasetMixin.ChannelView] = {}
        for rid in recording_ids:
            rec = self.get_recording(rid)
            channels = rec.channels

            ids = np.asarray(channels.id)
            names = np.asarray(getattr(channels, "name", ids))
            included_mask = np.asarray(
                getattr(channels, "included", np.ones(len(ids), dtype=bool)),
                dtype=bool,
            )

            if len(included_mask) != len(ids):
                raise ValueError(
                    f"Channel mask length mismatch for recording '{rid}': "
                    f"len(mask)={len(included_mask)} vs len(ids)={len(ids)}"
                )

            lip: np.ndarray | None = None
            if (
                hasattr(channels, "localization_L")
                and hasattr(channels, "localization_I")
                and hasattr(channels, "localization_P")
            ):
                lip = np.stack(
                    (
                        np.asarray(channels.localization_L, dtype=float),
                        np.asarray(channels.localization_I, dtype=float),
                        np.asarray(channels.localization_P, dtype=float),
                    ),
                    axis=1,
                )

            views[rid] = self.ChannelView(
                ids=ids, names=names, included_mask=included_mask, lip=lip
            )
        return views

    def _initialize_seeg_mixin_caches(self) -> None:
        """Initialize SEEG mixin cache contracts from dataset recordings."""
        self.seeg_dataset_mixin_domain_intervals = {
            rid: self.get_recording(rid).seeg_data.domain for rid in self.recording_ids
        }
        self.seeg_dataset_mixin_channel_views = self._build_channel_view_cache(
            self.recording_ids
        )
        self.seeg_dataset_mixin_recording_infos = self._build_recording_info_cache(
            self.recording_ids
        )

    def _build_recording_info_cache(
        self, recording_ids: list[str]
    ) -> dict[str, SEEGDatasetMixin.RecordingInfo]:
        """Build compact recording metadata for the provided recordings."""
        infos: dict[str, SEEGDatasetMixin.RecordingInfo] = {}
        for rid in recording_ids:
            subject_id, session_id = _from_recording_id(rid)
            channel_view = self.seeg_dataset_mixin_channel_views[rid]
            infos[rid] = self.RecordingInfo(
                recording_id=rid,
                subject_id=subject_id,
                session_id=session_id,
                sampling_rate_hz=self.get_sampling_rate(rid),
                domain=self.seeg_dataset_mixin_domain_intervals[rid],
                n_channels=int(len(channel_view.ids)),
                n_included_channels=int(np.sum(channel_view.included_mask)),
            )
        return infos

    def _prime_selected_recording_caches(self) -> None:
        """Prewarm caches for currently selected split recordings."""
        # Prime hot paths used immediately by samplers/evaluators for this split.
        for rid in self._selected_recording_ids():
            self.get_recording_info(rid)
            self.get_channel_view(rid, included_only=False)
            self.get_channel_view(rid, included_only=True)

    def _selected_recording_ids(self) -> list[str]:
        if self.prune_to_split:
            return list(self.recording_ids)
        return self._recording_ids_for_selection(
            subset_tier=self.subset_tier,
            regime=self.regime,
            split=self.split,
            test_subject=self.test_subject,
            test_session=self.test_session,
        )
