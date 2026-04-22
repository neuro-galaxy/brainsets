_functions = [
    "generate_stratified_folds",
    "generate_string_kfold_assignment",
]

__all__ = _functions

import hashlib
import logging
import numpy as np
from typing import List
from temporaldata import Interval, Data


def _create_interval_split(intervals: Interval, indices: np.ndarray) -> Interval:
    """Create an Interval subset from indices and sort it."""
    mask = np.zeros(len(intervals), dtype=bool)
    mask[indices] = True
    split = intervals.select_by_mask(mask)
    split.sort()
    return split


def generate_folds(
    intervals: Interval,
    n_folds: int = 5,
    val_ratio: float = 0.2,
    seed: int = 42,
    stratify_by: Optional[str] = None,
) -> List[Data]:
    """Generate train/valid/test splits using a two-stage splitting process.

    If *stratify_by* is None, splitting uses KFold (outer) and ShuffleSplit (inner).
    If *stratify_by* is an attribute name present on the intervals, splitting uses
    StratifiedKFold and StratifiedShuffleSplit so that each fold preserves the
    class distribution of that attribute.

    The splitting is performed in two stages:
        1. Outer split: The intervals are divided into *n_folds*, where each fold
           uses one partition as the test set and the remaining partitions as train+valid.
        2. Inner split: The train+valid portion of each fold is further split into
           train and valid sets using *val_ratio*.
    When used, stratification ensures each fold maintains the class distribution of the
    original data according to the *stratify_by* attribute.

    Args:
        intervals: The intervals to split.
        n_folds: Number of folds for cross-validation.
        val_ratio: Ratio of validation set relative to train+valid combined.
        seed: Random seed for reproducibility.
        stratify_by: Optional attribute name to stratify by (e.g., "id", "label").
            If provided, intervals must have this attribute.

    Returns:
        List of Data objects, one for each fold.

    Raises:
        ValueError: If there are fewer samples than *n_folds*.
        ValueError: If stratify_by is provided but intervals do not have that attribute.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(
            f"val_ratio must be between 0 and 1 (exclusive), got {val_ratio}"
        )
    if stratify_by is not None and not hasattr(intervals, stratify_by):
        raise ValueError(
            f"Intervals must have a '{stratify_by}' attribute for stratification."
        )

    try:
        from sklearn.model_selection import (
            KFold,
            StratifiedKFold,
            ShuffleSplit,
            StratifiedShuffleSplit,
        )
    except ImportError:
        raise ImportError(
            "This function requires the scikit-learn library which you can install with "
            "`pip install scikit-learn`"
        )

    use_stratify = stratify_by is not None

    if use_stratify:
        class_labels = getattr(intervals, stratify_by)
        if len(class_labels) < n_folds:
            raise ValueError(
                f"Not enough samples ({len(class_labels)}) for {n_folds} folds."
            )
    else:
        if len(intervals) < n_folds:
            raise ValueError(
                f"Not enough samples ({len(intervals)}) for {n_folds} folds."
            )

    folds = []
    sample_indices = np.arange(len(intervals))

    if use_stratify:
        outer_splitter = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=seed
        )
        for fold_idx, (train_val_indices, test_indices) in enumerate(
            outer_splitter.split(sample_indices, class_labels)
        ):
            test_split = _create_interval_split(intervals, test_indices)
            train_val_labels = class_labels[train_val_indices]
            inner_splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=val_ratio, random_state=seed + fold_idx
            )
            for train_indices, val_indices in inner_splitter.split(
                train_val_indices, train_val_labels
            ):
                train_original_indices = train_val_indices[train_indices]
                val_original_indices = train_val_indices[val_indices]
                train_split = _create_interval_split(intervals, train_original_indices)
                val_split = _create_interval_split(intervals, val_original_indices)
                combined_domain = train_split | val_split | test_split
                fold_data = Data(
                    train=train_split,
                    valid=val_split,
                    test=test_split,
                    domain=combined_domain,
                )
                folds.append(fold_data)
    else:
        outer_splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold_idx, (train_val_indices, test_indices) in enumerate(
            outer_splitter.split(sample_indices)
        ):
            test_split = _create_interval_split(intervals, test_indices)
            inner_splitter = ShuffleSplit(
                n_splits=1, test_size=val_ratio, random_state=seed + fold_idx
            )
            for train_indices, val_indices in inner_splitter.split(train_val_indices):
                train_original_indices = train_val_indices[train_indices]
                val_original_indices = train_val_indices[val_indices]
                train_split = _create_interval_split(intervals, train_original_indices)
                val_split = _create_interval_split(intervals, val_original_indices)
                combined_domain = train_split | val_split | test_split
                fold_data = Data(
                    train=train_split,
                    valid=val_split,
                    test=test_split,
                    domain=combined_domain,
                )
                folds.append(fold_data)

    return folds


def generate_string_kfold_assignment(
    string_id: str,
    n_folds: int = 3,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> List[str]:
    """Generate deterministic per-fold train/valid/test assignments for one ID.

    The assignment is independent for each fold index ``k``, but follows a
    deterministic two-step rule:

    1. Compute a global bucket from ``md5(f"{string_id}_{seed}") % n_folds``.
       The fold whose index equals this bucket is labeled ``"test"``.
    2. For every other fold, compute a fold-specific hash
       ``md5(f"{string_id}_{seed}_{k}")`` and map it to ``[0, 1)``.
       If that value is below ``val_ratio``, the fold is ``"valid"``,
       otherwise it is ``"train"``.

    As a result, each ``string_id`` appears in the test split for exactly one
    fold and is never in test for the remaining folds. This makes the output
    reproducible across runs and safe for parallel processing.

    Args
    ----
    string_id : str
        String identifier (e.g., "S001", "sub-01", or "sub-01_ses-01").
    n_folds : int
        Number of folds for cross-validation. Default is 3.
    val_ratio : float
        Ratio of validation set relative to train+valid combined. Default is 0.2.
    seed : int
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    List[str]
        List of fold assignments where index ``k`` corresponds to fold ``k`` and
        each value is one of ``"train"``, ``"valid"``, or ``"test"``.
        Exactly one entry is ``"test"``.

    Examples
    --------
    >>> assignments = generate_string_kfold_assignment("sub-01", n_folds=3)
    >>> assignments
    ['train', 'test', 'train']

    >>> generate_string_kfold_assignment("sub-01_ses-01", n_folds=3)
    ['valid', 'train', 'test']
    """
    if not isinstance(string_id, str) or not string_id:
        raise ValueError("string_id must be a non-empty string")
    if n_folds < 1:
        raise ValueError(f"n_folds must be at least 1, got {n_folds}")
    if not (0.0 <= val_ratio <= 1.0):
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    base_str = f"{string_id}_{seed}"
    base_bytes = base_str.encode("utf-8")
    hash_obj = hashlib.md5(base_bytes)
    hash_int = int(hash_obj.hexdigest(), 16)
    bucket = hash_int % n_folds

    assignments: List[str] = []
    for k in range(n_folds):
        if bucket == k:
            assignments.append("test")
        else:
            fold_str = f"{base_str}_{k}"
            fold_bytes = fold_str.encode("utf-8")
            fold_hash_obj = hashlib.md5(fold_bytes)
            fold_hash_int = int(fold_hash_obj.hexdigest(), 16)
            normalized_hash = (fold_hash_int % 10000) / 10000.0
            if normalized_hash < val_ratio:
                assignments.append("valid")
            else:
                assignments.append("train")
    return assignments
