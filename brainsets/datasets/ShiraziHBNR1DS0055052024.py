from typing import Callable, Optional

from brainsets.datasets.OpenNeuroBase import OpenNeuroDataset, SplitType


class ShiraziHBNR1DS0055052024(OpenNeuroDataset):
    def __init__(
        self,
        root: str,
        split_type: SplitType,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        **kwargs,
    ):
        dataset_dir = "shirazi_hbn_r1_ds005505_2024"
        super().__init__(
            root, dataset_dir, recording_ids, transform, split_type, **kwargs
        )
