from types import SimpleNamespace

def register_splits(split_indices):

    data = SimpleNamespace()

    for split_name, split_index in split_indices.items():

        if not hasattr(data, split_name):
            setattr(data, split_name, SimpleNamespace())
        split_obj = getattr(data, split_name)

        for fold_idx, fold_indices in split_index.items():
            fold_attr = f"fold{fold_idx}"

            if not hasattr(split_obj, fold_attr):
                setattr(split_obj, fold_attr, SimpleNamespace())
            fold_obj = getattr(split_obj, fold_attr)

            fold_obj.train = fold_indices["train_intervals"]
            fold_obj.test  = fold_indices["test_intervals"]
    
    return data
    