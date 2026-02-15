from brainsets.processing import register_splits

split_indices = {}
train_intervals = []
test_intervals = []

for eval_name in {"volume", "rms"}:
    split_indices[eval_name] = {}
    for fold_idx in [0, 1]:
        split_indices[eval_name][fold_idx] = {
            "train_intervals": train_intervals,
            "test_intervals": test_intervals,
        }

data = register_splits(split_indices)
print(data.rms.fold0.train)
