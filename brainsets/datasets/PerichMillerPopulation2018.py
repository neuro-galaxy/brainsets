from typing import Callable, Optional, Literal
from pathlib import Path

from torch_brain.dataset import Dataset, SpikingDatasetMixin


class PerichMillerPopulation2018(SpikingDatasetMixin, Dataset):
    """
    A dataset class for the Perich and Miller (2018) recordings,
    sourced from Dandiset 000688.

    ### Stats
    * Subjects: 4
    * Tasks: Center-Out and Random Target
    * Total Sessions: 111 (84 Center-Out, 27 Random Target)
    * Total Units: 10,410
    * Event Counts: ~11.1M spikes and ~15.5M behavioral timestamps

    ### Links
    * Paper: [Perich et al. (2018) - Neuron](https://doi.org/10.1016/j.neuron.2018.09.030)
    * Data (DANDI): [Dandiset 000688](https://dandiarchive.org/dandiset/000688)

    ### Reference:
    Perich, M. G., Miller, L. E., Azabou, M., & Dyer, E. L..
    Long-term recordings of motor and premotor cortical spiking activity during reaching in monkeys (Version 0.250122.1735)
    [Data set]. DANDI Archive. https://doi.org/10.48324/dandi.000688/0.250122.1735
    """

    def __init__(
        self,
        root: str,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        dirname: str = "perich_miller_population_2018",
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

    def get_sampling_intervals(
        self,
        split: Optional[Literal["train", "valid", "test"]] = None,
    ):
        domain_key = "domain" if split is None else f"{split}_domain"
        return {
            rid: getattr(self.get_recording(rid), domain_key)
            for rid in self.recording_ids
        }
