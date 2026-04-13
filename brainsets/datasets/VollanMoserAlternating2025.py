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

    - **Subjects:** 19
    - **Total Sessions:** 51 (31 Open Field, 7 Linear Track, 2 Wagon Wheel, 1 M-Maze, 1 Novel Open Field, 9 Sleep)
    - **Recording Tech:** Neuropixels

    **Navigation sessions** (42 sessions) contain:

    ``rec.spikes``
        Spike timestamps and unit indices.

    ``rec.units``
        Per-unit metadata: ``id``, ``location`` (mec/hc), ``probe_id``,
        ``shank_id``, ``shank_pos``, ``mean_rate``, ``ks2_label``, ``is_grid``.

    ``rec.samples``
        All variables on the shared 10 ms timebase (speed-filtered at 5 cm/s)
        in a single flat ``IrregularTimeSeries``:

        *Observed tracking variables:*

        - ``x`` -- head x-position relative to arena centre (m)
        - ``y`` -- head y-position relative to arena centre (m)
        - ``z`` -- head z-position relative to floor (m)
        - ``hd`` -- 2D head direction / azimuth (rad)
        - ``speed`` -- horizontal head speed (m/s)
        - ``theta`` -- instantaneous theta phase (rad)
        - ``id`` -- decoded internal direction (rad, from the LMT model)

        *LMT decoded variables* (for each population ``{pop}`` in
        ``mec``, ``hc``, ``mec_hc``):

        - ``lmt_{pop}_theta`` -- theta phase (fixed; largely duplicates ``theta``)
        - ``lmt_{pop}_hd`` -- head direction (fixed; largely duplicates ``hd``)
        - ``lmt_{pop}_id`` -- internal direction (latent)
        - ``lmt_{pop}_pos_x`` -- decoded x-position (latent)
        - ``lmt_{pop}_pos_y`` -- decoded y-position (latent)

        Populations not present for a given animal are NaN-padded so all
        sessions share the same field schema.

    ``rec.theta_chunks``
        Theta-cycle-binned timeseries (separate timebase, one sample per
        theta cycle): ``id``, ``L`` (log-likelihood, n_cycles x 30),
        ``P`` (probability, n_cycles x 30).

    ``rec.probe_channels``
        Probe channel geometry: ``probe_id``, ``channel_index``,
        ``x_um``, ``y_um``, ``shank_id``, ``connected``.

    **Sleep sessions** (9 sessions) have a fundamentally different structure
    and should be treated independently.  They contain only:

    ``rec.spikes``
        Spike timestamps and unit indices within SWS/REM epochs.

    ``rec.units``
        Minimal unit metadata (``id`` only).

    ``rec.domain``
        Union of SWS and REM epoch intervals.

    None of the navigation fields (``samples``, ``theta_chunks``,
    ``probe_channels``) are present on sleep sessions.

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
