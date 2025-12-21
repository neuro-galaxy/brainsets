from typing import Callable, Optional, Literal
from pathlib import Path
from torch_brain.utils import np_string_prefix
from temporaldata import Data

from torch_brain.dataset import Dataset, SpikingDatasetMixin


class PeiPandarinathNLB2021(SpikingDatasetMixin, Dataset):
    def __init__(
        self,
        root: str,
        dirname: str = "pei_pandarinath_nlb_2021",
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "units.id"],
            **kwargs,
        )

    def get_sampling_intervals(self, split: Literal["train", "valid", "test"]):
        domain_key = "domain" if split is None else f"{split}_domain"
        return {
            rid: getattr(self.get_recording(rid), domain_key)
            for rid in self.recording_ids
        }
