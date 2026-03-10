from typing import Callable, Optional, Literal
from pathlib import Path

from torch_brain.dataset import Dataset, SpikingDatasetMixin


class OdohertySabesNonhuman2017(SpikingDatasetMixin, Dataset):
    """
    A dataset class for the O'Doherty and Sabes (2017) recordings,
    sourced from Zenodo 3854034.

    ### Stats
    * Subjects: 2
    * Tasks: Random Target
    * Total Sessions: 47
    * Total Units: 16,566
    * Event Counts: ~105.2M spikes and ~12.4M behavioral timestamps

    ### Links
    * Paper: [O'Doherty and Sabes (2018) - J Neural Eng](https://pubmed.ncbi.nlm.nih.gov/29192609/)
    * Data (Zenodo): (https://zenodo.org/records/3854034)

    ### Reference:
    O'Doherty, J. E., Cardoso, M. M. B., Makin, J. G., & Sabes, P. N. (2020).
    Nonhuman Primate Reaching with Multichannel Sensorimotor Cortex Electrophysiology
    [Data set]. Zenodo. https://doi.org/10.5281/zenodo.788569
    """

    def __init__(
        self,
        root: str,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        split_type: Optional[Literal["cursor_velocity"]] = "cursor_velocity",
        dirname: str = "odoherty_sabes_nonhuman_2017",
        **kwargs,
    ):
        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "units.id"],
            **kwargs,
        )

        self.spiking_dataset_mixin_uniquify_unit_ids = True
        self.split_type = split_type

    def get_sampling_intervals(
        self,
        split: Optional[Literal["train", "valid", "test"]] = None,
    ):
        domain_key = "domain" if split is None else f"{split}_domain"
        ans = {}
        for rid in self.recording_ids:
            data = self.get_recording(rid)
            ans[rid] = getattr(data, domain_key)

            if self.split_type == "cursor_velocity":
                ans[rid] = ans[rid] & data.cursor.domain & data.spikes.domain

        return ans
