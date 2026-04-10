from typing import Callable, Optional, Literal
from pathlib import Path

from torch_brain.dataset import Dataset, SpikingDatasetMixin


class VollanMoserAlternating2025(SpikingDatasetMixin, Dataset):
    """Neuropixels recordings from MEC and hippocampus in rats during spatial navigation
    and sleep.

    Rats performed various navigation tasks (open field, linear track, M-maze, wagon
    wheel) while neural activity was recorded from medial entorhinal cortex (MEC) and/or
    hippocampus (HC) using Neuropixels probes. Sleep sessions with identified SWS and REM
    epochs are also included. The dataset contains grid cells, head direction cells, and
    other spatially tuned neurons.

    .. admonition:: Preprocessing

        To download and prepare this dataset, run
        ``brainsets prepare vollan_moser_alternating_2025``.

    **Tasks:** Open Field, Linear Track, M-Maze, Wagon Wheel, Sleep

    **Brain Regions:** MEC, Hippocampus

    **Dataset Statistics**

    - **Subjects:** 17
    - **Total Sessions:** 51 (31 Open Field, 7 Linear Track, 2 Wagon Wheel, 1 M-Maze, 1 Novel Open Field, 9 Sleep)
    - **Recording Tech:** Neuropixels

    **References**

    Vollan, A. Z., Gardner, R. J., Moser, M.-B. & Moser, E. I.
    *Left-right-alternating theta sweeps in the entorhinal-hippocampal spatial map.*
    Dataset: `EBRAINS <https://search.kg.ebrains.eu/instances/4080b78d-edc5-4ae4-8144-7f6de79930ea>`_.

    Args:
        root (str): Root directory for the dataset.
        recording_ids (list[str], optional): List of recording IDs to load.
        transform (Callable, optional): Data transformation to apply.
        dirname (str, optional): Subdirectory for the dataset. Defaults to "vollan_moser_alternating_2025".
    """

    def __init__(
        self,
        root: str,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        dirname: str = "vollan_moser_alternating_2025",
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
