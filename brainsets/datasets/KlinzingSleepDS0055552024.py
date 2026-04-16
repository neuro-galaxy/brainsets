from typing import Callable, Optional

from brainsets.datasets.OpenNeuroBase import OpenNeuroDataset, SplitType


class KlinzingSleepDS0055552024(OpenNeuroDataset):
    def __init__(
        self,
        root: str,
        split_type: SplitType,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        **kwargs,
    ):
        dataset_dir = "klinzing_sleep_ds005555_2024"
        super().__init__(
            root=root,
            dataset_dir=dataset_dir,
            recording_ids=recording_ids,
            transform=transform,
            split_type=split_type,
            **kwargs,
        )
