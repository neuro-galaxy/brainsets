from typing import Callable, Optional

from brainsets.datasets.OpenNeuroBase import OpenNeuroDataset, OpenNeuroSplitType


class KochiVisualNamingDS006914(OpenNeuroDataset):
    def __init__(
        self,
        root: str,
        split_type: OpenNeuroSplitType,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        **kwargs,
    ):
        dataset_dir = "kochi_visualnaming_ds006914"
        super().__init__(
            root=root,
            dataset_dir=dataset_dir,
            recording_ids=recording_ids,
            transform=transform,
            split_type=split_type,
            **kwargs,
        )
