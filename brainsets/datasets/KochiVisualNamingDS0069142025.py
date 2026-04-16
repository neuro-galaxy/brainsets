from typing import Callable, Optional

from brainsets.datasets.OpenNeuroBase import OpenNeuroDataset, SplitType


class KochiVisualNamingDS0069142025(OpenNeuroDataset):
    def __init__(
        self,
        root: str,
        split_type: SplitType,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        **kwargs,
    ):
        dataset_dir = "kochi_visualnaming_ds006914_2025"
        super().__init__(
            root, dataset_dir, recording_ids, transform, split_type, **kwargs
        )
