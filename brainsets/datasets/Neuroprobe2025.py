from __future__ import annotations

from pathlib import Path
import re
from typing import Callable, ClassVar, Literal, Optional, get_args

import h5py
import numpy as np
from temporaldata import Data, Interval

from torch_brain.dataset import Dataset, SEEGDatasetMixin


Version = Literal["full", "lite", "nano"]
LabelMode = Literal["binary", "multiclass"]
Regime = Literal["SS-SM", "SS-DM", "DS-DM"]
Split = Literal["train", "val", "test"]

VALID_VERSIONS = get_args(Version)
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

# Channel mask key bits used in generated H5 fields:
# included$lite{0|1}$nano{0|1}$binary_tasks{0|1}$...
VERSION_BITS: dict[Version, tuple[str, str]] = {
    "full": ("0", "0"),
    "lite": ("1", "0"),
    "nano": ("0", "1"),
}
# Map API label mode to the binary_tasks bit used in channel mask keys.
MODE_TO_BINARY_TASKS_BIT: dict[LabelMode, str] = {
    "binary": "1",
    "multiclass": "0",
}

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
_RECORDING_ID_RE = re.compile(r"^sub_(\d+)_trial(\d+)$")

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
            f"Invalid recording_id '{recording_id}'. Expected 'sub_<subject>_trial<session>'."
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
        recording_ids: Optional recording-id subset to expose from disk.
        transform: Optional sample transform.
        version: One of ``"full"``, ``"lite"``, ``"nano"``.
        test_subject: Target test subject id (Neuroprobe semantics).
        test_session: Target test trial/session id (Neuroprobe semantics).
        split: One of ``"train"``, ``"val"``, ``"test"``.
        label_mode: One of ``"binary"``, ``"multiclass"``.
        task: Neuroprobe task name.
        regime: One of ``"SS-SM"``, ``"SS-DM"``, ``"DS-DM"``.
        fold: Fold index. Defaults depend on regime:
            - ``within_session``: default 0, valid {0, 1}
            - ``cross_x``: forced to 0
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
        version: Version,
        test_subject: int,
        test_session: int,
        split: Split,
        label_mode: LabelMode = "binary",
        task: str = "speech",
        regime: Regime = "SS-SM",
        fold: Optional[int] = None,
        dirname: str = "neuroprobe_2025",
        **kwargs,
    ):
        # Resolve and validate constructor inputs before touching dataset records.
        self._dataset_dir = Path(root) / dirname
        requested_recording_ids = self._resolve_requested_recording_ids(recording_ids)

        self._validate_constructor_args(
            version=version,
            label_mode=label_mode,
            task=task,
            regime=regime,
            split=split,
            test_subject=test_subject,
            test_session=test_session,
        )

        super().__init__(
            dataset_dir=self._dataset_dir,
            recording_ids=requested_recording_ids,
            transform=transform,
            namespace_attributes=["subject.id", "channels.id"],
            **kwargs,
        )

        self.version = version
        self.label_mode = label_mode
        self.task = task
        self.regime = regime
        self.h5_regime = H5_REGIME_BY_REGIME[regime]
        self.test_subject = test_subject
        self.test_session = test_session
        self.split = split
        self.fold = self._resolve_fold(fold)

        self._interval_base_path = f"splits.{self.version}.{self.label_mode}.{self.h5_regime}.{self.task}.fold{self.fold}"

        self._split_recording_ids = self._recording_ids_for_split()
        self._validate_selection_compatibility()
        self.seeg_dataset_mixin_domain_intervals = {
            rid: self.get_recording(rid).seeg_data.domain for rid in self.recording_ids
        }
        self.seeg_dataset_mixin_channel_views = self._build_channel_view_cache(
            self.recording_ids
        )
        self.seeg_dataset_mixin_recording_infos = self._build_recording_info_cache(
            self.recording_ids
        )

        self._prime_selected_recording_caches()

    def get_sampling_intervals(self) -> dict[str, Interval]:
        """Return split-specific sampling intervals for this dataset instance."""
        # Sampling is always driven by the split configured at construction time.
        interval_path = self._interval_attr_path(self.split)
        return {
            rid: self.get_recording(rid).get_nested_attribute(interval_path)
            for rid in self._split_recording_ids
        }

    def get_domain_intervals(self) -> dict[str, Interval]:
        """Return full-domain intervals for the selected split recordings."""
        # Restrict domain queries to split-selected recordings for this dataset instance.
        return super().get_domain_intervals(recording_ids=self._split_recording_ids)

    def get_recording_hook(self, data: Data):
        """Apply split-specific channel inclusion mask when available."""
        # Override generic channel masks with the split-specific inclusion mask.
        try:
            data.channels.included = data.get_nested_attribute(
                self._channel_mask_attr_path(self.split)
            )
        except (AttributeError, KeyError):
            # Compatibility is validated at init for selected split recordings.
            # If users manually access an out-of-split recording, keep default mask.
            pass
        super().get_recording_hook(data)

    def describe_selection(self) -> dict[str, object]:
        """Return a compact debug summary of the resolved split selection."""
        # Expose resolved internals to make dataset/debug logs self-explanatory.
        return {
            "version": self.version,
            "label_mode": self.label_mode,
            "task": self.task,
            "regime": self.regime,
            "h5_regime": self.h5_regime,
            "fold": self.fold,
            "split": self.split,
            "test_subject": self.test_subject,
            "test_session": self.test_session,
            "test_recording_id": self._test_recording_id(),
            "selected_recording_ids": list(self._split_recording_ids),
            "interval_path": self._interval_attr_path(self.split),
            "channel_mask_key": self._make_channel_mask_key(self.split),
        }

    def _validate_constructor_args(
        self,
        *,
        version: str,
        label_mode: str,
        task: str,
        regime: str,
        split: str,
        test_subject: int,
        test_session: int,
    ) -> None:
        # Keep constructor strict so invalid benchmark configs fail immediately.
        if version not in VALID_VERSIONS:
            raise ValueError(
                f"Invalid version '{version}'. Must be one of {VALID_VERSIONS}."
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

        if type(test_subject) is not int:
            raise TypeError(
                f"test_subject must be an int, got {type(test_subject).__name__}."
            )
        if type(test_session) is not int:
            raise TypeError(
                f"test_session must be an int, got {type(test_session).__name__}."
            )

        if H5_REGIME_BY_REGIME[regime] == "cross_x" and version == "nano":
            raise ValueError("Version 'nano' is not compatible with cross_x regimes.")

        if regime == "DS-DM" and test_subject == DS_DM_TRAIN_SUBJECT_ID:
            raise ValueError(
                "DS-DM benchmark-default uses subject 2 as fixed train subject; "
                "test_subject cannot be 2."
            )

    def _resolve_fold(self, fold: Optional[int]) -> int:
        # within_session exposes two folds; cross_x is fixed to fold0 only.
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
        if recording_ids is None:
            ids = [x.stem for x in self._dataset_dir.glob("*.h5")]
            if not ids:
                raise ValueError(f"No recordings found in {self._dataset_dir}")
            ids = sorted(ids)
        else:
            ids = sorted(set(recording_ids))

        # Parse each id once so errors are raised consistently at construction.
        for rid in ids:
            _from_recording_id(rid)
        return ids

    # Path/key builders.
    def _interval_attr_path(self, split: Split) -> str:
        # Build dotted nested-attribute path used by temporaldata records.
        return f"{self._interval_base_path}.{split}_intervals"

    def _interval_h5_path(self, split: Split) -> str:
        # Convert dotted attribute path into slash-delimited H5 group path.
        return self._interval_attr_path(split).replace(".", "/")

    def _channel_mask_attr_path(self, split: Split) -> str:
        # Return the nested attribute path for the split-specific channel mask.
        return f"channels.{self._make_channel_mask_key(split)}"

    def _make_channel_mask_key(self, split: Split) -> str:
        # Match the key naming convention used by the neuroprobe preprocessing pipeline.
        lite_bit, nano_bit = VERSION_BITS[self.version]
        binary_tasks_bit = MODE_TO_BINARY_TASKS_BIT[self.label_mode]
        return (
            f"included$lite{lite_bit}$nano{nano_bit}$binary_tasks{binary_tasks_bit}$"
            f"{self.h5_regime}${self.task}$fold{self.fold}${split}"
        )

    def _recording_ids_for_split(self) -> list[str]:
        # Resolve which physical recording(s) participate in the configured split.
        test_recording_id = self._test_recording_id()

        if self.regime == "SS-SM":
            # Within-session uses a single target recording for all splits.
            return [test_recording_id]

        if self.regime == "SS-DM":
            # Cross-session trains on a different session from the same subject.
            if self.split == "train":
                # Train recording is selected by benchmark-default SS-DM rules.
                return [self._ss_dm_train_recording_id()]
            # Val/test evaluate on the requested target recording.
            return [test_recording_id]

        # DS-DM
        if self.split == "train":
            # Cross-subject benchmark-default uses a fixed train anchor recording.
            return [_to_recording_id(DS_DM_TRAIN_SUBJECT_ID, DS_DM_TRAIN_TRIAL_ID)]
        # Val/test evaluate on the requested held-out target recording.
        return [test_recording_id]

    def _ss_dm_train_recording_id(self) -> str:
        # Compute SS-DM train recording using benchmark-default selection rules.
        if self.version == "lite":
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

        if self.version == "full":
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
            f"Version '{self.version}' is not supported for SS-DM train selection."
        )

    def _validate_selection_compatibility(self) -> None:
        # Validate both benchmark eligibility and concrete H5 availability for selected ids.
        # First enforce all high-level selection constraints.
        self._validate_required_recordings_present()
        # Enforce version/regime-specific benchmark allowlists before touching file internals.
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
        # Ensure both target and split-required recordings exist in recording_ids.
        test_recording_id = self._test_recording_id()
        if test_recording_id not in self.recording_ids:
            raise ValueError(
                f"Target recording '{test_recording_id}' is not available in this dataset selection."
            )

        # Ensure every recording needed by this split is present in recording_ids.
        missing_recordings = [
            rid for rid in self._split_recording_ids if rid not in self.recording_ids
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
        split_interval_attr_path = self._interval_attr_path(self.split)
        split_interval_h5_path = self._interval_h5_path(self.split)
        split_channel_mask_key = self._make_channel_mask_key(self.split)
        incompatible: dict[str, str] = {}
        for rid in self._split_recording_ids:
            issue = self._recording_compatibility_issue(
                rid,
                split_interval_attr_path=split_interval_attr_path,
                split_interval_h5_path=split_interval_h5_path,
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
        split_interval_h5_path: str,
        split_channel_mask_key: str,
    ) -> str | None:
        # Return None if compatible, otherwise a concise reason string.
        cache_key = (
            str(self._dataset_dir),
            recording_id,
            self.version,
            self.label_mode,
            self.h5_regime,
            self.task,
            self.fold,
            self.split,
            split_interval_h5_path,
            split_channel_mask_key,
        )
        if cache_key in self._RECORDING_COMPAT_ISSUE_CACHE:
            return self._RECORDING_COMPAT_ISSUE_CACHE[cache_key]

        issue: str | None = None
        with h5py.File(self._dataset_dir / f"{recording_id}.h5", "r") as h5:
            if "splits" not in h5:
                issue = "missing 'splits' group"
            elif self.version not in h5["splits"]:
                issue = f"version '{self.version}' unavailable"
            elif self.label_mode not in h5["splits"][self.version]:
                issue = f"label_mode '{self.label_mode}' unavailable"
            elif self.h5_regime not in h5["splits"][self.version][self.label_mode]:
                issue = f"h5 regime '{self.h5_regime}' unavailable"
            else:
                regime_group = h5["splits"][self.version][self.label_mode][
                    self.h5_regime
                ]
                if self.task not in regime_group:
                    issue = f"task '{self.task}' unavailable"
                elif f"fold{self.fold}" not in regime_group[self.task]:
                    issue = f"fold{self.fold} unavailable"
                # Interval path is checked in H5 slash form, but reported in attribute form.
                elif split_interval_h5_path not in h5:
                    issue = f"missing split interval path '{split_interval_attr_path}'"
                # Channel mask key must exist for this exact split/version/task setting.
                elif (
                    "channels" not in h5 or split_channel_mask_key not in h5["channels"]
                ):
                    issue = (
                        f"missing channel mask key 'channels.{split_channel_mask_key}'"
                    )

        self._RECORDING_COMPAT_ISSUE_CACHE[cache_key] = issue
        return issue

    def _validate_target_pair_eligibility(self) -> None:
        """Validate benchmark-default subject/session eligibility constraints."""
        # Enforce benchmark-allowed target subject/session pairs per version/regime.
        requested_pair = (self.test_subject, self.test_session)
        if (
            self.version == "lite"
            and requested_pair not in NEUROPROBE_LITE_SUBJECT_TRIALS
        ):
            raise ValueError(
                f"Target pair {requested_pair} is not in NEUROPROBE_LITE_SUBJECT_TRIALS."
            )
        if (
            self.version == "nano"
            and requested_pair not in NEUROPROBE_NANO_SUBJECT_TRIALS
        ):
            raise ValueError(
                f"Target pair {requested_pair} is not in NEUROPROBE_NANO_SUBJECT_TRIALS."
            )
        if self.regime == "SS-DM" and self.version == "full":
            longest_trials = NEUROPROBE_LONGEST_TRIALS_FOR_SUBJECT.get(
                self.test_subject, []
            )
            if len(longest_trials) < 2:
                raise ValueError(
                    "SS-DM full benchmark-default requires at least two longest trials "
                    f"for subject {self.test_subject}, found {longest_trials}."
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
            f"version={self.version}, label_mode={self.label_mode}, task={self.task}, "
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
        for rid in self._split_recording_ids:
            self.get_recording_info(rid)
            self.get_channel_view(rid, included_only=False)
            self.get_channel_view(rid, included_only=True)
