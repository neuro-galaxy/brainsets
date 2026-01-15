import numpy as np
import pytest
from temporaldata import Data, Interval
from brainsets.utils.split import (
    chop_intervals,
    generate_stratified_folds,
    generate_task_kfold_splits,
    compute_subject_kfold_assignments,
)


def test_chop_intervals_exact_multiple():
    start = np.array([0.0, 200.0])
    end = np.array([100.0, 250.0])
    ids = np.array([1, 2])
    intervals = Interval(start=start, end=end, id=ids)

    duration = 10.0
    chopped = chop_intervals(intervals, duration=duration)

    # 10 chunks from first interval (100s), 5 from second (50s)
    assert len(chopped) == 15
    assert np.allclose(chopped.end - chopped.start, duration)

    # Verify gaps are preserved
    sorted_indices = np.argsort(chopped.start)
    sorted_starts = chopped.start[sorted_indices]
    sorted_ends = chopped.end[sorted_indices]

    # Gap between end of chunk 9 (100.0) and start of chunk 10 (200.0)
    assert np.isclose(sorted_ends[9], 100.0)
    assert np.isclose(sorted_starts[10], 200.0)

    # Verify IDs are preserved
    assert np.all(chopped.id[:10] == 1)
    assert np.all(chopped.id[10:] == 2)


def test_chop_intervals_with_remainder():
    start = np.array([0.0])
    end = np.array([25.0])
    ids = np.array([1])
    intervals = Interval(start=start, end=end, id=ids)

    duration = 10.0
    chopped = chop_intervals(intervals, duration=duration)

    # 2 full chunks (0-10, 10-20) + 1 shorter chunk (20-25)
    assert len(chopped) == 3
    assert np.allclose(chopped.start, [0.0, 10.0, 20.0])
    assert np.allclose(chopped.end, [10.0, 20.0, 25.0])
    assert np.all(chopped.id == 1)


def test_chop_intervals_shorter_than_duration():
    start = np.array([0.0, 100.0])
    end = np.array([5.0, 103.0])
    ids = np.array([1, 2])
    intervals = Interval(start=start, end=end, id=ids)

    duration = 10.0
    chopped = chop_intervals(intervals, duration=duration)

    # Both intervals are shorter than duration, kept as-is
    assert len(chopped) == 2
    assert np.allclose(chopped.start, [0.0, 100.0])
    assert np.allclose(chopped.end, [5.0, 103.0])
    assert np.array_equal(chopped.id, [1, 2])


def test_chop_intervals_overlapping_raises():
    start = np.array([0.0, 50.0])
    end = np.array([100.0, 150.0])
    intervals = Interval(start=start, end=end)

    with pytest.raises(ValueError, match="Intervals overlap"):
        chop_intervals(intervals, duration=10.0, check_no_overlap=True)


def test_chop_intervals_overlapping_no_check():
    start = np.array([0.0, 50.0])
    end = np.array([100.0, 150.0])
    intervals = Interval(start=start, end=end)

    chopped = chop_intervals(intervals, duration=10.0, check_no_overlap=False)
    assert len(chopped) == 20


def test_generate_stratified_folds():
    n_samples = 100
    start = np.arange(n_samples, dtype=float)
    end = start + 1.0
    # Imbalanced classes: 60 of class 0, 30 of class 1, 10 of class 2
    ids = np.concatenate(
        [np.zeros(60, dtype=int), np.ones(30, dtype=int), np.full(10, 2, dtype=int)]
    )
    # Shuffle to make it realistic
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_samples)
    ids = ids[perm]
    start = start[perm]
    end = end[perm]

    intervals = Interval(start=start, end=end, id=ids)

    n_folds = 5
    val_ratio = 0.25
    folds = generate_stratified_folds(
        intervals, stratify_by="id", n_folds=n_folds, val_ratio=val_ratio, seed=42
    )

    assert isinstance(folds, list)
    assert len(folds) == n_folds

    test_indices_all = []

    for fold in folds:
        assert isinstance(fold, Data)
        train, valid, test = fold.train, fold.valid, fold.test

        # 1. Verify sizes
        # Test: ~1/5 of 100 = 20 samples (from n_folds)
        assert len(test) == 20
        # Valid: 0.25 of remaining 80 = 20 samples
        assert len(valid) == 20
        # Train: 80 - 20 = 60 samples
        assert len(train) == 60

        # 2. Verify Stratification in Test Set
        # Expected counts for test set (20 samples):
        # Class 0: 0.6 * 20 = 12
        # Class 1: 0.3 * 20 = 6
        # Class 2: 0.1 * 20 = 2
        test_ids = test.id
        unique, counts = np.unique(test_ids, return_counts=True)
        counts_dict = dict(zip(unique, counts))

        # Allow small deviation due to rounding/randomness
        assert counts_dict.get(0, 0) in [11, 12, 13]
        assert counts_dict.get(1, 0) in [5, 6, 7]
        assert counts_dict.get(2, 0) in [1, 2, 3]

        # 3. Verify Stratification in Valid Set
        # Expected for valid (20 samples):
        # Class 0: 0.6 * 20 = 12
        # Class 1: 0.3 * 20 = 6
        # Class 2: 0.1 * 20 = 2
        valid_ids = valid.id
        v_unique, v_counts = np.unique(valid_ids, return_counts=True)
        v_counts_dict = dict(zip(v_unique, v_counts))

        assert v_counts_dict.get(0, 0) in [11, 12, 13]
        assert v_counts_dict.get(1, 0) in [5, 6, 7]
        assert v_counts_dict.get(2, 0) in [1, 2, 3]

        # 4. Collect test indices/IDs to verify full coverage later
        test_indices_all.append(test.start)

    # 5. Verify all samples are used in test sets exactly once across folds
    all_test_starts = np.concatenate(test_indices_all)
    all_test_starts_sorted = np.sort(all_test_starts)
    original_starts_sorted = np.sort(intervals.start)

    assert np.allclose(all_test_starts_sorted, original_starts_sorted)


def test_generate_stratified_folds_custom_attribute():
    n_samples = 50
    start = np.arange(n_samples, dtype=float)
    end = start + 1.0
    # Use "label" instead of "id"
    labels = np.array(["A"] * 25 + ["B"] * 25)

    rng = np.random.default_rng(42)
    perm = rng.permutation(n_samples)
    labels = labels[perm]
    start = start[perm]
    end = end[perm]

    intervals = Interval(start=start, end=end, label=labels)

    folds = generate_stratified_folds(
        intervals, stratify_by="label", n_folds=5, val_ratio=0.25, seed=42
    )

    assert isinstance(folds, list)
    assert len(folds) == 5

    for fold in folds:
        assert isinstance(fold, Data)
        test_labels = fold.test.label
        unique, counts = np.unique(test_labels, return_counts=True)
        assert len(unique) == 2
        assert all(c == 5 for c in counts)


def test_generate_stratified_folds_missing_attribute():
    start = np.arange(10, dtype=float)
    end = start + 1.0
    intervals = Interval(start=start, end=end)

    with pytest.raises(ValueError, match="must have a 'label' attribute"):
        generate_stratified_folds(intervals, stratify_by="label", n_folds=5)


class TestGenerateTaskKfoldSplits:
    def test_basic_functionality(self):
        n_samples = 100
        start = np.arange(n_samples, dtype=float)
        end = start + 1.0
        labels = np.array(
            ["left_hand"] * 30 + ["right_hand"] * 30 + ["feet"] * 20 + ["rest"] * 20
        )

        rng = np.random.default_rng(42)
        perm = rng.permutation(n_samples)
        labels = labels[perm]
        start = start[perm]
        end = end[perm]

        trials = Interval(start=start, end=end, label=labels)

        task_configs = {
            "MotorImagery": ["left_hand", "right_hand", "feet"],
            "BinaryTask": ["left_hand", "rest"],
        }

        splits_dict = generate_task_kfold_splits(
            trials,
            task_configs=task_configs,
            label_field="label",
            n_folds=5,
            val_ratio=0.2,
            seed=42,
        )

        assert isinstance(splits_dict, dict)

        for k in range(5):
            assert f"MotorImagery_fold{k}_train" in splits_dict
            assert f"MotorImagery_fold{k}_valid" in splits_dict
            assert f"MotorImagery_fold{k}_test" in splits_dict

            assert f"BinaryTask_fold{k}_train" in splits_dict
            assert f"BinaryTask_fold{k}_valid" in splits_dict
            assert f"BinaryTask_fold{k}_test" in splits_dict

    def test_labels_preserved_per_task(self):
        n_samples = 60
        start = np.arange(n_samples, dtype=float)
        end = start + 1.0
        labels = np.array(["A"] * 20 + ["B"] * 20 + ["C"] * 20)

        trials = Interval(start=start, end=end, label=labels)

        task_configs = {"TaskAB": ["A", "B"]}

        splits_dict = generate_task_kfold_splits(
            trials,
            task_configs=task_configs,
            label_field="label",
            n_folds=5,
            val_ratio=0.2,
            seed=42,
        )

        for k in range(5):
            train = splits_dict[f"TaskAB_fold{k}_train"]
            valid = splits_dict[f"TaskAB_fold{k}_valid"]
            test = splits_dict[f"TaskAB_fold{k}_test"]

            all_labels_in_fold = np.concatenate([train.label, valid.label, test.label])
            assert set(all_labels_in_fold) == {"A", "B"}
            assert "C" not in all_labels_in_fold

    def test_missing_label_field_raises(self):
        start = np.arange(10, dtype=float)
        end = start + 1.0
        trials = Interval(start=start, end=end)

        with pytest.raises(ValueError, match="must have a 'label' attribute"):
            generate_task_kfold_splits(
                trials,
                task_configs={"Task": ["A"]},
                label_field="label",
            )

    def test_skips_task_with_insufficient_trials(self, caplog):
        start = np.arange(14, dtype=float)
        end = start + 1.0
        labels = np.array(["A"] * 4 + ["B"] * 10)
        trials = Interval(start=start, end=end, label=labels)

        task_configs = {
            "SmallTask": ["A"],
            "LargerTask": ["A", "B"],
        }

        import logging

        with caplog.at_level(logging.WARNING):
            splits_dict = generate_task_kfold_splits(
                trials,
                task_configs=task_configs,
                label_field="label",
                n_folds=5,
                val_ratio=0.2,
                seed=42,
            )

        assert "SmallTask_fold0_train" not in splits_dict
        assert "LargerTask_fold0_train" in splits_dict


class TestComputeSubjectKfoldAssignments:
    def test_basic_output_structure(self):
        assignments = compute_subject_kfold_assignments("S001", n_folds=5)

        assert isinstance(assignments, dict)
        assert len(assignments) == 5

        for k in range(5):
            assert f"SubjectSplit_fold{k}" in assignments
            assert assignments[f"SubjectSplit_fold{k}"] in ["train", "valid", "test"]

    def test_exactly_one_test_assignment(self):
        assignments = compute_subject_kfold_assignments("S001", n_folds=5)

        test_count = sum(1 for v in assignments.values() if v == "test")
        assert test_count == 1

    def test_deterministic_assignment(self):
        assignments1 = compute_subject_kfold_assignments("S001", n_folds=5, seed=42)
        assignments2 = compute_subject_kfold_assignments("S001", n_folds=5, seed=42)

        assert assignments1 == assignments2

    def test_different_subjects_different_assignments(self):
        assignments_s1 = compute_subject_kfold_assignments("S001", n_folds=5, seed=42)
        assignments_s2 = compute_subject_kfold_assignments("S002", n_folds=5, seed=42)

        assert assignments_s1 != assignments_s2

    def test_different_seeds_different_assignments(self):
        assignments_seed1 = compute_subject_kfold_assignments(
            "S001", n_folds=5, seed=42
        )
        assignments_seed2 = compute_subject_kfold_assignments(
            "S001", n_folds=5, seed=99
        )

        assert assignments_seed1 != assignments_seed2

    def test_val_ratio_affects_valid_proportion(self):
        n_subjects = 100
        n_folds = 5

        valid_counts_low = 0
        valid_counts_high = 0

        for i in range(n_subjects):
            assignments_low = compute_subject_kfold_assignments(
                f"S{i:03d}", n_folds=n_folds, val_ratio=0.1, seed=42
            )
            assignments_high = compute_subject_kfold_assignments(
                f"S{i:03d}", n_folds=n_folds, val_ratio=0.4, seed=42
            )

            valid_counts_low += sum(1 for v in assignments_low.values() if v == "valid")
            valid_counts_high += sum(
                1 for v in assignments_high.values() if v == "valid"
            )

        assert valid_counts_high > valid_counts_low

    def test_distribution_across_many_subjects(self):
        n_subjects = 1000
        n_folds = 5
        val_ratio = 0.2

        test_fold_counts = {k: 0 for k in range(n_folds)}
        valid_count = 0
        train_count = 0

        for i in range(n_subjects):
            assignments = compute_subject_kfold_assignments(
                f"subject_{i}", n_folds=n_folds, val_ratio=val_ratio, seed=42
            )

            for k in range(n_folds):
                assignment = assignments[f"SubjectSplit_fold{k}"]
                if assignment == "test":
                    test_fold_counts[k] += 1
                elif assignment == "valid":
                    valid_count += 1
                else:
                    train_count += 1

        expected_per_fold = n_subjects / n_folds
        for k, count in test_fold_counts.items():
            assert abs(count - expected_per_fold) < expected_per_fold * 0.3

        non_test_total = valid_count + train_count
        actual_val_ratio = valid_count / non_test_total
        assert abs(actual_val_ratio - val_ratio) < 0.05
