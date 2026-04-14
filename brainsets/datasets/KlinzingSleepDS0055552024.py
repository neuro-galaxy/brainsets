from typing import Callable, Optional

from brainsets.datasets.OpenNeuroBase import OpenNeuroDataset, SplitType


class KlinzingSleepDS0055552024(OpenNeuroDataset):
    def __init__(
        self,
        root: str,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        split_type: SplitType = "intrasession",
        **kwargs,
    ):
        dataset_dir = "klinzing_sleep_ds005555_2024"
        super().__init__(
            root, dataset_dir, recording_ids, transform, split_type, **kwargs
        )
