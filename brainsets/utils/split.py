import logging
import hashlib
import numpy as np
from typing import List, Dict
from collections import Counter
from temporaldata import Interval, Data


def split_one_epoch(epoch, grid, split_ratios=[0.6, 0.1, 0.3]):
    assert len(epoch) == 1
    epoch_start = epoch.start[0]
    epoch_end = epoch.end[0]

    train_val_split_time = epoch_start + split_ratios[0] * (epoch_end - epoch_start)
    val_test_split_time = train_val_split_time + split_ratios[1] * (
        epoch_end - epoch_start
    )

    grid_match = grid.slice(
        train_val_split_time, train_val_split_time, reset_origin=False
    )
    if len(grid_match) > 0:
        if (
            train_val_split_time - grid_match.start[0]
            > grid_match.end[0] - train_val_split_time
        ):
            train_val_split_time = grid_match.end[0]
        else:
            train_val_split_time = grid_match.start[0]

    grid_match = grid.slice(
        val_test_split_time, val_test_split_time, reset_origin=False
    )
    if len(grid_match) > 0:
        if (
            val_test_split_time - grid_match.start[0]
            > grid_match.end[0] - val_test_split_time
        ):
            val_test_split_time = grid_match.end[0]
        else:
            val_test_split_time = grid_match.start[0]

    train_interval = Interval(start=epoch_start, end=train_val_split_time)
    val_interval = Interval(start=train_interval.end[0], end=val_test_split_time)
    test_interval = Interval(start=val_interval.end[0], end=epoch_end)

    return train_interval, val_interval, test_interval


def split_two_epochs(epoch, grid):
    assert len(epoch) == 2
    first_epoch_start = epoch.start[0]
    first_epoch_end = epoch.end[0]

    split_time = first_epoch_start + 0.5 * (first_epoch_end - first_epoch_start)
    grid_match = grid.slice(split_time, split_time, reset_origin=False)
    if len(grid_match) > 0:
        if split_time - grid_match.start[0] > grid_match.end[0] - split_time:
            split_time = grid_match.end[0]
        else:
            split_time = grid_match.start[0]

    train_interval = Interval(
        start=first_epoch_start,
        end=split_time,
    )
    val_interval = Interval(start=train_interval.end[0], end=first_epoch_end)
    test_interval = epoch.select_by_mask(np.array([False, True]))

    return train_interval, val_interval, test_interval


def split_three_epochs(epoch, grid):
    assert len(epoch) == 3

    test_interval = epoch.select_by_mask(np.array([False, False, True]))
    train_interval = epoch.select_by_mask(np.array([True, True, False]))

    split_time = train_interval.end[1] - 0.3 * (
        train_interval.end[1] - train_interval.start[1]
    )
    grid_match = grid.slice(split_time, split_time, reset_origin=False)
    if len(grid_match) > 0:
        if split_time - grid_match.start[0] > grid_match.end[0] - split_time:
            split_time = grid_match.end[0]
        else:
            split_time = grid_match.start[0]

    train_interval.end[1] = split_time
    val_interval = Interval(start=train_interval.end[1], end=epoch.end[1])

    return train_interval, val_interval, test_interval


def split_four_epochs(epoch, grid):
    assert len(epoch) == 4

    test_interval = epoch.select_by_mask(np.array([False, False, False, True]))
    train_interval = epoch.select_by_mask(np.array([True, True, True, False]))
    split_time = train_interval.end[2] - 0.5 * (
        train_interval.end[2] - train_interval.start[2]
    )
    grid_match = grid.slice(split_time, split_time, reset_origin=False)
    if len(grid_match) > 0:
        if split_time - grid_match.start[0] > grid_match.end[0] - split_time:
            split_time = grid_match.end[0]
        else:
            split_time = grid_match.start[0]

    train_interval.end[2] = split_time
    val_interval = Interval(start=train_interval.end[2], end=epoch.end[2])

    return train_interval, val_interval, test_interval


def split_five_epochs(epoch, grid):
    assert len(epoch) == 5

    train_interval = epoch.select_by_mask(np.array([True, True, True, False, False]))
    test_interval = epoch.select_by_mask(np.array([False, False, False, True, True]))

    split_time = train_interval.end[2] - 0.5 * (
        train_interval.end[2] - train_interval.start[2]
    )
    grid_match = grid.slice(split_time, split_time, reset_origin=False)
    if len(grid_match) > 0:
        if split_time - grid_match.start[0] > grid_match.end[0] - split_time:
            split_time = grid_match.end[0]
        else:
            split_time = grid_match.start[0]

    train_interval.end[2] = split_time
    val_interval = Interval(start=train_interval.end[2], end=epoch.end[2])

    return train_interval, val_interval, test_interval


def split_more_than_five_epochs(epoch):
    assert len(epoch) > 5

    train_interval, val_interval, test_interval = epoch.split(
        [0.6, 0.1, 0.3], shuffle=False
    )
    return train_interval, val_interval, test_interval


def generate_train_valid_test_splits(epoch_dict, grid):
    train_intervals = Interval(np.array([]), np.array([]))
    valid_intervals = Interval(np.array([]), np.array([]))
    test_intervals = Interval(np.array([]), np.array([]))

    for name, epoch in epoch_dict.items():
        if name == "invalid_presentation_epochs":
            logging.warning(
                "Found invalid presentation epochs, which will be excluded."
            )
            continue
        if len(epoch) == 1:
            train, valid, test = split_one_epoch(epoch, grid)
        elif len(epoch) == 2:
            train, valid, test = split_two_epochs(epoch, grid)
        elif len(epoch) == 3:
            train, valid, test = split_three_epochs(epoch, grid)
        elif len(epoch) == 4:
            train, valid, test = split_four_epochs(epoch, grid)
        elif len(epoch) == 5:
            train, valid, test = split_five_epochs(epoch, grid)
        else:
            train, valid, test = split_more_than_five_epochs(epoch)

        train_intervals = train_intervals | train
        valid_intervals = valid_intervals | valid
        test_intervals = test_intervals | test

    return train_intervals, valid_intervals, test_intervals


def chop_intervals(
    intervals: Interval, duration: float, check_no_overlap: bool = False
) -> Interval:
    """
    Subdivides intervals into fixed-length epochs using Interval.arange().

    If some intervals are shorter than the duration, keep them as they are.
    If an interval is not a perfect multiple of the duration, the last chunk will be shorter.

    Args:
        intervals: The original intervals to chop.
        duration: The duration of each chopped interval in seconds.
        check_no_overlap: If True, verify the resulting intervals don't overlap.

    Returns:
        Interval: A new Interval object containing the chopped segments.
                  Metadata from the original intervals is preserved and repeated for each segment.

    Raises:
        ValueError: If check_no_overlap is True and intervals overlap.
    """
    if len(intervals) == 0:
        return Interval(start=np.array([]), end=np.array([]))

    chopped_intervals = []
    original_indices = []

    for i, (start, end) in enumerate(zip(intervals.start, intervals.end)):
        if end - start <= duration:
            chopped = Interval(start=start, end=end)
        else:
            chopped = Interval.arange(start, end, step=duration, include_end=True)

        chopped_intervals.append(chopped)
        original_indices.extend([i] * len(chopped))

    all_starts = np.concatenate([c.start for c in chopped_intervals])
    all_ends = np.concatenate([c.end for c in chopped_intervals])

    kwargs = {}
    if hasattr(intervals, "keys"):
        for key in intervals.keys():
            if key in ["start", "end"]:
                continue
            val = getattr(intervals, key)
            if isinstance(val, np.ndarray) and len(val) == len(intervals):
                kwargs[key] = val[original_indices]

    result = Interval(start=all_starts, end=all_ends, **kwargs)

    if check_no_overlap:
        if not result.is_disjoint():
            raise ValueError("Intervals overlap after chopping")

    return result


def _create_interval_split(intervals: Interval, indices: np.ndarray) -> Interval:
    """Create an Interval subset from indices and sort it."""
    mask = np.zeros(len(intervals), dtype=bool)
    mask[indices] = True
    split = intervals.select_by_mask(mask)
    split.sort()
    return split


def generate_trial_folds(
    trials: Interval,
    stratify_by: str,
    n_folds: int = 5,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> List[Data]:
    """
    Generate stratified k-fold train/valid/test splits at the trial level.

    This function performs **intra-session splitting** at the trial level. Individual
    trials are distributed across folds while maintaining the class distribution
    (stratification). Use this when you want to evaluate generalization across trials
    within the same session.

    For **cross-subject splitting** (where entire subjects are held out), use
    :func:`generate_subject_kfold_assignment` instead.

    The splitting is performed in two stages:
        1. Outer split (StratifiedKFold): Trials are divided into n_folds, where each
           fold uses one partition as test and the rest as train+valid. Stratification
           ensures each fold maintains the class distribution.
        2. Inner split (StratifiedShuffleSplit): The train+valid portion is further
           split into train and valid sets using val_ratio, preserving class distribution.

    Args
    ----
    trials : Interval
        The trials to split. Must have an attribute matching `stratify_by`.
    stratify_by : str
        The attribute name to use for stratification (e.g., "label", "class").
    n_folds : int
        Number of folds for cross-validation. Default is 5.
    val_ratio : float
        Ratio of validation set relative to train+valid combined. Default is 0.2.
    seed : int
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    List[Data]
        List of Data objects, one per fold. Each Data object contains:
        - train: Interval with training trials
        - valid: Interval with validation trials
        - test: Interval with test trials
        - domain: Combined interval of all trials in the fold

    Raises
    ------
    ValueError
        If trials don't have the specified stratify_by attribute.
    ValueError
        If there are fewer trials than n_folds.

    Examples
    --------
    >>> folds = generate_trial_folds(
    ...     trials=session.trials,
    ...     stratify_by="label",
    ...     n_folds=5,
    ... )
    >>> for k, fold in enumerate(folds):
    ...     print(f"Fold {k}: train={len(fold.train)}, valid={len(fold.valid)}, test={len(fold.test)}")

    See Also
    --------
    generate_trial_folds_by_task : Higher-level function for multi-task trial splitting.
    generate_subject_kfold_assignment : For cross-subject (leave-subject-out) splitting.
    """
    try:
        from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    except ImportError:
        raise ImportError(
            "This function requires the scikit-learn library which you can install with "
            "`pip install scikit-learn`"
        )

    if not hasattr(trials, stratify_by):
        raise ValueError(
            f"Trials must have a '{stratify_by}' attribute for stratification."
        )

    class_labels = getattr(trials, stratify_by)
    if len(class_labels) < n_folds:
        raise ValueError(
            f"Not enough trials ({len(class_labels)}) for {n_folds} folds."
        )

    outer_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    sample_indices = np.arange(len(trials))

    for fold_idx, (train_val_indices, test_indices) in enumerate(
        outer_splitter.split(sample_indices, class_labels)
    ):
        test_split = _create_interval_split(trials, test_indices)

        train_val_labels = class_labels[train_val_indices]
        inner_splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_ratio, random_state=seed + fold_idx
        )

        for train_indices, val_indices in inner_splitter.split(
            train_val_indices, train_val_labels
        ):
            train_original_indices = train_val_indices[train_indices]
            val_original_indices = train_val_indices[val_indices]

            train_split = _create_interval_split(trials, train_original_indices)
            val_split = _create_interval_split(trials, val_original_indices)

            combined_domain = train_split | val_split | test_split

            fold_data = Data(
                train=train_split,
                valid=val_split,
                test=test_split,
                domain=combined_domain,
            )

            folds.append(fold_data)

    return folds


def generate_trial_folds_by_task(
    trials: Interval,
    task_configs: Dict[str, List[str]],
    label_field: str,
    n_folds: int = 5,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, Interval]:
    """
    Generate trial-level k-fold cross-validation splits within a single session.

    This function performs **intra-session splitting** at the trial level. It takes
    trials from a single recording session and creates stratified k-fold splits,
    ensuring trials from the same session can appear in different folds. Use this
    when you want to evaluate generalization across trials within the same session.

    For **cross-subject splitting** (where entire subjects are held out), use
    :func:`generate_subject_kfold_assignment` instead.

    For each task configuration, creates n_folds stratified train/valid/test splits.
    The splits are returned as Interval objects with names formatted as
    "{task_name}_fold{k}_train", "{task_name}_fold{k}_valid", and
    "{task_name}_fold{k}_test".

    Args
    ----
    trials : Interval
        An Interval object containing trial information from a single session,
        including labels specified by `label_field`.
    task_configs : Dict[str, List[str]]
        Dictionary mapping task names to lists of labels to include for that task.
    label_field : str
        The attribute name in trials that contains the labels for stratification.
    n_folds : int
        Number of folds for cross-validation. Default is 5.
    val_ratio : float
        Ratio of validation set relative to train+valid combined. Default is 0.2.
    seed : int
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    Dict[str, Interval]
        Dictionary mapping split names to Interval objects.

    Examples
    --------
    >>> task_configs = {
    ...     "MotorImagery": ["left_hand", "right_hand", "feet"],
    ...     "RestVsActive": ["rest", "left_hand", "right_hand"],
    ... }
    >>> splits = generate_trial_folds_by_task(
    ...     trials=session.trials,
    ...     task_configs=task_configs,
    ...     label_field="label",
    ...     n_folds=5,
    ... )
    >>> # Returns keys like: "MotorImagery_fold0_train", "MotorImagery_fold0_valid", ...
    >>> train_trials = splits["MotorImagery_fold0_train"]

    See Also
    --------
    generate_subject_kfold_assignment : For cross-subject (leave-subject-out) splitting.
    generate_trial_folds : Lower-level function for stratified fold generation.
    """
    if not hasattr(trials, label_field):
        raise ValueError(
            f"Trials must have a '{label_field}' attribute for task filtering."
        )

    all_labels = getattr(trials, label_field)
    splits_dict = {}

    for task_name, include_labels in task_configs.items():
        logging.info(f"\nGenerating {task_name} k-fold train/valid/test splits")

        task_mask = np.isin(all_labels, include_labels)
        task_trials = trials.select_by_mask(task_mask)

        if len(task_trials) < n_folds:
            logging.warning(
                f"Task {task_name} has only {len(task_trials)} trials, "
                f"skipping (need at least {n_folds})"
            )
            continue

        folds = generate_trial_folds(
            task_trials,
            stratify_by=label_field,
            n_folds=n_folds,
            val_ratio=val_ratio,
            seed=seed,
        )

        for k, fold_data in enumerate(folds):
            task_labels_train = getattr(fold_data.train, label_field)
            task_labels_valid = getattr(fold_data.valid, label_field)
            task_labels_test = getattr(fold_data.test, label_field)

            logging.info(f"Fold {k}:")
            logging.info(f"  Train label counts: {dict(Counter(task_labels_train))}")
            logging.info(f"  Valid label counts: {dict(Counter(task_labels_valid))}")
            logging.info(f"  Test label counts: {dict(Counter(task_labels_test))}")

            splits_dict[f"{task_name}_fold{k}_train"] = fold_data.train
            splits_dict[f"{task_name}_fold{k}_valid"] = fold_data.valid
            splits_dict[f"{task_name}_fold{k}_test"] = fold_data.test

    return splits_dict


def generate_subject_kfold_assignment(
    subject_id: str, n_folds: int = 5, val_ratio: float = 0.2, seed: int = 42
) -> Dict[str, str]:
    """
    Generate cross-subject k-fold train/valid/test assignments for a single subject.

    This function performs **cross-subject splitting** (also known as leave-subject-out
    or between-subject splitting). It deterministically assigns a subject to train,
    valid, or test for each fold using hash-based assignment. Use this when you want
    to evaluate generalization across different subjects, ensuring no data leakage
    between subjects in different splits.

    For **intra-session splitting** (trial-level splits within a session), use
    :func:`generate_trial_folds_by_task` instead.

    The assignment is deterministic: the same subject_id with the same seed will
    always receive the same assignments. This allows processing subjects independently
    (e.g., in parallel) while ensuring consistent fold assignments.

    For each fold k:
    - Subjects hashed to bucket k are assigned to "test"
    - Remaining subjects are assigned to "train" or "valid" based on val_ratio

    Args
    ----
    subject_id : str
        Subject identifier (e.g., "S001", "sub-01").
    n_folds : int
        Number of folds for cross-validation. Default is 5.
    val_ratio : float
        Ratio of validation set relative to train+valid combined. Default is 0.2.
    seed : int
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping "SubjectSplit_fold{k}" to "train", "valid", or "test"
        for each fold k in range(n_folds).

    Examples
    --------
    >>> assignments = generate_subject_kfold_assignment("sub-01", n_folds=5)
    >>> assignments
    {'SubjectSplit_fold0': 'train', 'SubjectSplit_fold1': 'test',
     'SubjectSplit_fold2': 'train', 'SubjectSplit_fold3': 'valid',
     'SubjectSplit_fold4': 'train'}

    >>> # Use in a pipeline to tag sessions with their fold assignments
    >>> for subject_id in subject_ids:
    ...     assignments = generate_subject_kfold_assignment(subject_id)
    ...     session = load_session(subject_id)
    ...     for key, value in assignments.items():
    ...         setattr(session, key, value)

    See Also
    --------
    generate_trial_folds_by_task : For intra-session (trial-level) splitting.
    """
    subject_str = f"{subject_id}_{seed}"
    subject_bytes = subject_str.encode("utf-8")
    hash_obj = hashlib.md5(subject_bytes)
    hash_int = int(hash_obj.hexdigest(), 16)
    bucket = hash_int % n_folds

    assignments = {}

    for k in range(n_folds):
        if bucket == k:
            assignments[f"SubjectSplit_fold{k}"] = "test"
        else:
            fold_str = f"{subject_id}_{seed}_{k}"
            fold_bytes = fold_str.encode("utf-8")
            fold_hash_obj = hashlib.md5(fold_bytes)
            fold_hash_int = int(fold_hash_obj.hexdigest(), 16)
            normalized_hash = (fold_hash_int % 10000) / 10000.0
            if normalized_hash < val_ratio:
                assignments[f"SubjectSplit_fold{k}"] = "valid"
            else:
                assignments[f"SubjectSplit_fold{k}"] = "train"

    return assignments
