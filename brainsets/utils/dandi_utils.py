from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from pynwb import NWBFile
from temporaldata import ArrayDict, Interval, IrregularTimeSeries, RegularTimeSeries

from brainsets.descriptions import SubjectDescription
from brainsets.taxonomy import (
    Hemisphere,
    RecordingTech,
    Sex,
    Species,
)


def extract_metadata_from_nwb(nwbfile):
    recording_date = nwbfile.session_start_time.strftime("%Y%m%d")
    related_publications = nwbfile.related_publications
    return dict(
        recording_date=recording_date, related_publications=related_publications
    )


def extract_subject_from_nwb(
    nwbfile,
    subject_id: Optional[str] = None,
    species: Optional[Union[Species, str]] = None,
    sex: Optional[Sex] = None,
    age: Optional[float] = None,
):
    r"""DANDI has requirements for metadata included in `subject`. This includes:
    subject_id: A subject identifier must be provided.
    species: either a latin binomial or NCBI taxonomic identifier.
    sex: must be "M", "F", "O" (other), or "U" (unknown).
    date_of_birth or age: this does not appear to be enforced, so will be skipped.
    """
    if subject_id is None:
        subject_id = nwbfile.subject.subject_id.lower()
    if species is None:
        species = nwbfile.subject.species
    elif isinstance(species, str) and "NCBITaxon" in species:
        species = "NCBITaxon_" + species.split("_")[-1]
    if sex is None:
        sex = Sex.from_string(nwbfile.subject.sex)
    if age is None:
        age = nwbfile.subject.age

    return SubjectDescription(
        id=subject_id,
        species=species,
        sex=sex,
        age=age,
    )


def extract_spikes_from_nwbfile(nwbfile, recording_tech):
    # spikes
    timestamps = []
    unit_index = []

    # units
    unit_meta = []

    units = nwbfile.units.spike_times_index[:]
    electrodes = nwbfile.units.electrodes.table

    # all these units are obtained using threshold crossings
    for i in range(len(units)):
        if recording_tech == RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS:
            # label unit
            group_name = electrodes["group_name"][i]
            unit_id = f"group_{group_name}/elec{i}/multiunit_{0}"
        elif recording_tech == RecordingTech.UTAH_ARRAY_SPIKES:
            # label unit
            electrode_id = nwbfile.units[i].electrodes.item().item()
            group_name = electrodes["group_name"][electrode_id]
            unit_id = f"group_{group_name}/elec{electrode_id}/unit_{i}"
        else:
            raise ValueError(f"Recording tech {recording_tech} not supported")

        # extract spikes
        spiketimes = units[i]
        timestamps.append(spiketimes)

        if len(spiketimes) > 0:
            unit_index.append([i] * len(spiketimes))

        # extract unit metadata
        unit_meta.append(
            {
                "id": unit_id,
                "unit_number": i,
                "count": len(spiketimes),
                "type": int(recording_tech),
            }
        )

    # convert unit metadata to a Data object
    unit_meta_df = pd.DataFrame(unit_meta)  # list of dicts to dataframe
    units = ArrayDict.from_dataframe(
        unit_meta_df,
        unsigned_to_long=True,
    )

    # concatenate spikes
    timestamps = np.concatenate(timestamps)
    unit_index = np.concatenate(unit_index)

    # create spikes object
    spikes = IrregularTimeSeries(
        timestamps=timestamps,
        unit_index=unit_index,
        domain="auto",
    )

    # make sure to sort the spikes
    spikes.sort()

    return spikes, units


def download_file(path, url, raw_dir, overwrite=False) -> Path:
    try:
        import dandi.download
    except ImportError:
        raise ImportError("dandi package not present, and is required")

    asset_path = Path(path)
    download_dir = raw_dir / asset_path.parent
    download_dir.mkdir(exist_ok=True, parents=True)
    dandi.download.download(
        url,
        download_dir,
        existing=(
            dandi.download.DownloadExisting.REFRESH
            if not overwrite
            else dandi.download.DownloadExisting.OVERWRITE
        ),
    )
    return raw_dir / asset_path


def get_nwb_asset_list(dandiset_id: str):
    try:
        from dandi import dandiarchive
    except ImportError:
        raise ImportError("dandi package not present, and is required")

    parsed_url = dandiarchive.parse_dandi_url(dandiset_id)
    with parsed_url.navigate() as (client, dandiset, assets):
        asset_list = [x for x in assets if x.path.endswith(".nwb")]
    return asset_list


def _identify_elecs(electrodes_table) -> Tuple[np.ndarray, np.ndarray]:
    if not hasattr(electrodes_table, "group_name") or "group_name" not in getattr(
        electrodes_table, "colnames", []
    ):
        n = len(electrodes_table.id) if hasattr(electrodes_table, "id") else 0
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    group_names = np.asarray(electrodes_table["group_name"][:]).astype(str)
    surface = np.array(
        ["surface" in g.lower() or "ecog" in g.lower() for g in group_names]
    )
    depth = np.array(["depth" in g.lower() or "seeg" in g.lower() for g in group_names])
    return surface, depth


def _identify_surface_elecs(group_names: np.ndarray) -> np.ndarray:
    """Surface vs depth ECoG electrodes (AJILE12 / Peterson-Brunton convention)."""
    group_names = np.asarray(group_names).astype(str)
    has_phd = np.any(np.char.upper(group_names) == "PHD")
    is_surf = []
    for label in group_names:
        g = label.lower()
        if "grid" in g:
            is_surf.append(True)
        elif g in ("mhd", "latd", "lmtd", "ltpd"):
            is_surf.append(True)
        elif g == "ahd" and not has_phd:
            is_surf.append(True)
        elif "d" in g:
            is_surf.append(False)
        else:
            is_surf.append(True)
    return np.array(is_surf, dtype=bool)


def _hemisphere_from_reach_events(nwbfile: NWBFile) -> Optional[Hemisphere]:
    behavior = nwbfile.processing.get("behavior") if nwbfile.processing else None
    if behavior is None:
        return None
    reach = behavior.data_interfaces.get("ReachEvents")
    if reach is None or not getattr(reach, "description", None):
        return None
    desc = reach.description[:]
    if desc is None or len(desc) == 0:
        return None
    c_wrist = str(desc[0]).strip().lower()
    if c_wrist == "r":
        return Hemisphere.LEFT
    if c_wrist == "l":
        return Hemisphere.RIGHT
    return None


def _resolve_hemisphere(
    subject_hemisphere: Optional[Union[Hemisphere, str]], nwbfile: NWBFile
) -> Hemisphere:
    if subject_hemisphere is not None:
        if isinstance(subject_hemisphere, Hemisphere):
            return subject_hemisphere
        s = subject_hemisphere.strip().upper()
        if s == "L":
            return Hemisphere.LEFT
        if s == "R":
            return Hemisphere.RIGHT
        try:
            return Hemisphere.from_string(subject_hemisphere)
        except ValueError:
            pass
    inferred = _hemisphere_from_reach_events(nwbfile)
    return inferred if inferred is not None else Hemisphere.UNKNOWN


def extract_ecog_from_nwb(
    nwbfile: NWBFile,
    resample_rate: float = 250.0,
    apply_filter: bool = True,
    subject_hemisphere: Optional[Union[Hemisphere, str]] = None,
    iir_filter: bool = False,
    chunk_duration: float = 60.0,
) -> Tuple[RegularTimeSeries, ArrayDict]:
    """
    Extract ECoG data from NWB file with memory-efficient chunked loading.

    Args:
        nwbfile: NWB file object
        resample_rate: Target sampling rate in Hz
        apply_filter: Legacy parameter, filtering is not supported in chunked mode
        subject_hemisphere: Subject hemisphere (left/right)
        iir_filter: Legacy parameter, not used
        chunk_duration: Duration in seconds to process at once (tune based on memory)

    Returns:
        Tuple of (RegularTimeSeries with ecog data, ArrayDict with channel metadata)
    """
    try:
        from scipy import signal
    except ImportError:
        raise ImportError("extract_ecog_from_nwb requires scipy")

    if "ElectricalSeries" not in nwbfile.acquisition:
        raise KeyError("NWB file has no acquisition['ElectricalSeries']")

    if apply_filter:
        raise NotImplementedError(
            "Filtering is not supported in the optimized chunked implementation. "
            "Set apply_filter=False and apply filtering after extraction if needed."
        )

    electrical_series = nwbfile.acquisition["ElectricalSeries"]
    ecog_sampling_rate = float(electrical_series.rate)
    electrodes = nwbfile.electrodes
    n_samples = electrical_series.data.shape[0]
    n_channels = electrical_series.data.shape[1]

    # Calculate downsampling parameters
    downsample_factor = int(ecog_sampling_rate / resample_rate)
    chunk_samples = int(chunk_duration * ecog_sampling_rate)

    # Process in chunks to avoid memory issues with long recordings
    downsampled_chunks = []

    for start_idx in range(0, n_samples, chunk_samples):
        end_idx = min(start_idx + chunk_samples, n_samples)

        # Lazy load only this chunk from HDF5
        chunk = np.asarray(
            electrical_series.data[start_idx:end_idx, :], dtype=np.float64
        )

        # Downsample with built-in anti-aliasing filter
        if downsample_factor > 1:
            chunk = signal.decimate(
                chunk, downsample_factor, axis=0, ftype="iir", zero_phase=True
            )

        downsampled_chunks.append(chunk)
        del chunk  # Explicit memory cleanup

    # Concatenate all downsampled chunks
    data_out = np.concatenate(downsampled_chunks, axis=0)
    del downsampled_chunks

    # Create time array
    times_out = np.arange(data_out.shape[0]) / resample_rate

    # Extract channel metadata
    good = np.ones(n_channels, dtype=bool)
    if hasattr(electrodes, "good") and electrodes.good is not None:
        good = np.asarray(electrodes["good"][:]).astype(bool)
    bad_channels = ~good

    colnames = getattr(electrodes, "colnames", [])
    if "group_name" in colnames:
        group_names = np.asarray(electrodes["group_name"][:])
        is_surface = _identify_surface_elecs(group_names)
    else:
        group_names = np.array([""] * n_channels)
        is_surface = np.zeros(n_channels, dtype=bool)

    hemisphere = _resolve_hemisphere(subject_hemisphere, nwbfile)
    channel_meta = []
    for i in range(n_channels):
        grp = str(group_names[i]) if i < len(group_names) else ""
        channel_meta.append(
            {
                "id": f"group_ECoGArray/channel_{i}",
                "unit_number": i,
                "hemisphere": int(hemisphere),
                "group": grp,
                "surface": bool(is_surface[i]),
                "count": -1,
                "type": int(RecordingTech.ECOG_ARRAY_ECOGS),
                "bad": bool(bad_channels[i]),
            }
        )
    channels = ArrayDict.from_dataframe(
        pd.DataFrame(channel_meta), unsigned_to_long=True
    )

    domain = Interval(start=np.array([times_out[0]]), end=np.array([times_out[-1]]))

    ecog_rts = RegularTimeSeries(
        ecogs=data_out,
        sampling_rate=resample_rate,
        domain=domain,
    )
    return ecog_rts, channels


def extract_pose_from_nwb(nwbfile: NWBFile) -> Tuple[Interval, RegularTimeSeries]:
    """
    Extract coarse active behavior label as intervals, and extract both wrist movement
    trajectories and contralateral reach movement events

    Returns:
        behavior_trials: Interval containing behavior trial information with active event masking
        wrist_trajectories: RegularTimeSeries containing R_Wrist and L_Wrist position data
    """
    # Extract wrist position data
    r_wrist = nwbfile.processing["behavior"].data_interfaces["Position"]["R_Wrist"]
    l_wrist = nwbfile.processing["behavior"].data_interfaces["Position"]["L_Wrist"]

    behavior_sampling_rate = r_wrist.rate
    assert r_wrist.rate == l_wrist.rate

    wrist_trajectories = RegularTimeSeries(
        r_wrist=r_wrist.data[:],  # dim (total time length x 2)
        l_wrist=l_wrist.data[:],
        sampling_rate=behavior_sampling_rate,
        domain=Interval(
            start=np.array([0.0]),
            end=np.array([(len(r_wrist.data) - 1) / behavior_sampling_rate]),
        ),
    )

    # Extract coarse behavior labels
    coarse_behaviors = nwbfile.intervals["epochs"]
    coarse_behaviors_labels = coarse_behaviors.labels.data[:].tolist()

    # Define active events and compute active event mask
    active_events = ["Eat", "Talk", "TV", "Computer/phone", "Other activity"]
    active_event_mask = np.zeros(len(coarse_behaviors_labels)).astype(bool)
    for k in range(len(coarse_behaviors_labels)):
        is_active = True
        for single_event in coarse_behaviors_labels[k].split(", "):
            if not (single_event in active_events):
                is_active = False
        active_event_mask[k] = is_active

    behavior_trials = Interval(
        start=coarse_behaviors.start_time.data[:],
        end=coarse_behaviors.stop_time.data[:],
        behavior_labels=coarse_behaviors.labels.data[:],
        active=active_event_mask,
    )

    # Print unique behavior labels and which are active
    unique_behavior_labels = np.unique(coarse_behaviors_labels)
    unique_active_behavior_mask = np.zeros(unique_behavior_labels.shape[0]).astype(bool)
    for k in range(unique_behavior_labels.shape[0]):
        is_active = True
        for single_event in unique_behavior_labels[k].split(", "):
            if not (single_event in active_events):
                is_active = False
        unique_active_behavior_mask[k] = is_active

    print(
        "unique behavior labels",
        unique_behavior_labels,
        "in which",
        unique_behavior_labels[unique_active_behavior_mask],
        "are active.",
    )

    return behavior_trials, wrist_trajectories


def extract_behavior_intervals_from_nwb(nwbfile: NWBFile) -> Interval:
    if "epochs" not in nwbfile.intervals:
        return Interval(
            start=np.array([]),
            end=np.array([]),
            behavior_labels=np.array([]),
            active=np.array([]),
        )

    epochs = nwbfile.intervals["epochs"]
    starts = np.asarray(epochs.start_time.data[:])
    ends = np.asarray(epochs.stop_time.data[:])
    labels = epochs.labels.data[:].tolist()

    active_events = ["Eat", "Talk", "TV", "Computer/phone", "Other activity"]
    active_event_mask = np.zeros(len(labels)).astype(bool)
    for k in range(len(labels)):
        is_active = True
        for single_event in labels[k].split(", "):
            if not (single_event in active_events):
                is_active = False
        active_event_mask[k] = is_active

    behavior_trials = Interval(
        start=starts,
        end=ends,
        behavior_labels=np.asarray(labels),
        active=active_event_mask,
    )

    # Print unique behavior labels and which are active
    unique_behavior_labels = np.unique(labels)
    unique_active_behavior_mask = np.zeros(unique_behavior_labels.shape[0]).astype(bool)
    for k in range(unique_behavior_labels.shape[0]):
        is_active = True
        for single_event in unique_behavior_labels[k].split(", "):
            if not (single_event in active_events):
                is_active = False
        unique_active_behavior_mask[k] = is_active

    print(
        "unique behavior labels",
        unique_behavior_labels,
        "in which",
        unique_behavior_labels[unique_active_behavior_mask],
        "are active.",
    )

    return behavior_trials


def extract_reach_events_from_nwb(
    nwbfile: NWBFile,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    behavior = nwbfile.processing.get("behavior")
    if behavior is None:
        return None, None
    reach_events = behavior.data_interfaces.get("ReachEvents")
    if reach_events is None:
        return None, None
    timestamps = (
        reach_events.timestamps[:] if hasattr(reach_events, "timestamps") else None
    )
    hemisphere = None
    if hasattr(reach_events, "data") and reach_events.data is not None:
        data = reach_events.data[:]
        if data is not None and data.size > 0:
            hemisphere = np.asarray(data).flatten()
    return np.asarray(timestamps) if timestamps is not None else None, hemisphere
