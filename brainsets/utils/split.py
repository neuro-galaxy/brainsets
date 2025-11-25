import logging
import numpy as np
from temporaldata import Interval, Data, ArrayDict
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


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
            logging.warn(f"Found invalid presentation epochs, which will be excluded.")
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


def chop_intervals(intervals: Interval, duration: float) -> Interval:
    """
    Subdivides intervals into fixed-length epochs.

    Args:
        intervals: The original intervals to chop.
        duration: The duration of each chopped interval in seconds.

    Returns:
        Interval: A new Interval object containing the chopped segments.
                  Metadata from the original intervals is preserved and repeated for each segment.
    """
    new_starts = []
    new_ends = []
    original_indices = []

    for i, (start, end) in enumerate(zip(intervals.start, intervals.end)):
        n_chunks = int((end - start) // duration)
        if n_chunks == 0:
            continue

        chunk_starts = start + np.arange(n_chunks) * duration
        chunk_ends = chunk_starts + duration

        new_starts.append(chunk_starts)
        new_ends.append(chunk_ends)
        original_indices.extend([i] * n_chunks)

    if not new_starts:
        return Interval(start=np.array([]), end=np.array([]))

    new_starts = np.concatenate(new_starts)
    new_ends = np.concatenate(new_ends)

    kwargs = {}
    if hasattr(intervals, "keys"):
        for key in intervals.keys():
            if key in ["start", "end"]:
                continue
            val = getattr(intervals, key)
            if isinstance(val, (np.ndarray, list)) and len(val) == len(intervals):
                if isinstance(val, np.ndarray):
                    kwargs[key] = val[original_indices]
                else:
                    kwargs[key] = [val[i] for i in original_indices]

    return Interval(start=new_starts, end=new_ends, **kwargs)


def filter_intervals(intervals: Interval, exclude_ids: list) -> Interval:
    """
    Filters out epochs with specified IDs.

    Args:
        intervals: The intervals to filter.
        exclude_ids: A list of IDs to exclude. The intervals must have an 'id' attribute.

    Returns:
        Interval: A new Interval object with excluded IDs removed.
    """
    if not hasattr(intervals, "id"):
        raise ValueError("Intervals must have an 'id' attribute for filtering.")

    mask = ~np.isin(intervals.id, exclude_ids)
    return intervals.select_by_mask(mask)


def _create_interval_split(intervals: Interval, indices: np.ndarray) -> Interval:
    """Create an Interval subset from indices and sort it."""
    mask = np.zeros(len(intervals), dtype=bool)
    mask[indices] = True
    split = intervals.select_by_mask(mask)
    split.sort()
    return split


def generate_stratified_folds(
    intervals: Interval,
    n_folds: int = 5,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Data:
    """
    Generates stratified train/valid/test splits using StratifiedKFold.

    Args:
        intervals: The intervals to split. Must have an 'id' attribute for stratification.
        n_folds: Number of folds for cross-validation.
        val_ratio: Ratio of validation set relative to train+valid combined.
        seed: Random seed.

    Returns:
        Data: A Data object containing the splits structure.
              splits.intrasubject.fold_i.{train, valid, test}
    """
    if not hasattr(intervals, "id"):
        raise ValueError("Intervals must have an 'id' attribute for stratification.")

    if len(intervals.id) < n_folds:
        raise ValueError(
            f"Not enough samples ({len(intervals.id)}) for {n_folds} folds."
        )

    outer_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds_dict = {}
    sample_indices = np.arange(len(intervals))
    class_labels = intervals.id

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

            folds_dict[f"fold_{fold_idx}"] = fold_data

    overall_domain = intervals
    intrasubject = Data(**folds_dict, domain=overall_domain)
    splits = Data(
        intrasubject=intrasubject,
        intersubject=Data(domain=overall_domain),
        domain=overall_domain,
    )

    return splits
