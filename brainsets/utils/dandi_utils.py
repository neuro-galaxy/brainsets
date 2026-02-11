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


def _normalize_hemisphere_input(value: Union[Hemisphere, str]) -> Hemisphere:
    if isinstance(value, Hemisphere):
        return value
    s = str(value).strip().upper()
    if s == "L":
        return Hemisphere.LEFT
    if s == "R":
        return Hemisphere.RIGHT
    try:
        return Hemisphere.from_string(value)
    except ValueError:
        return Hemisphere.UNKNOWN


def _hemisphere_from_nwb(nwbfile: NWBFile, n_channels: int) -> Hemisphere:
    colnames = getattr(nwbfile.electrodes, "colnames", [])
    for col in ("hemisphere", "location"):
        if col not in colnames:
            continue
        vals = nwbfile.electrodes[col][:]
        if vals is None or len(vals) == 0:
            continue
        texts = np.asarray([str(v).strip().lower() for v in vals])
        left = np.any(
            (texts == "l") | (texts == "left") | np.char.find(texts.astype(str), "left")
            >= 0
        )
        right = np.any(
            (texts == "r")
            | (texts == "right")
            | np.char.find(texts.astype(str), "right")
            >= 0
        )
        if left and not right:
            return Hemisphere.LEFT
        if right and not left:
            return Hemisphere.RIGHT
        break
    if hasattr(nwbfile, "subject") and nwbfile.subject is not None:
        subj = nwbfile.subject
        for attr in ("hemisphere", "location"):
            if not hasattr(subj, attr):
                continue
            val = getattr(subj, attr)
            if val is None:
                continue
            v = str(val).strip().lower()
            if v in ("l", "left"):
                return Hemisphere.LEFT
            if v in ("r", "right"):
                return Hemisphere.RIGHT
    return Hemisphere.UNKNOWN


def extract_ecog_from_nwb(
    nwbfile: NWBFile,
    subject_hemisphere: Optional[Union[Hemisphere, str]] = None,
) -> Tuple[RegularTimeSeries, ArrayDict]:
    """
    Extract ECoG data from NWB file at native sampling rate.

    Hemisphere is taken from subject_hemisphere if provided, otherwise from the
    NWB file (electrodes table "location"/"hemisphere" or subject metadata).
    """
    if "ElectricalSeries" not in nwbfile.acquisition:
        raise KeyError("NWB file has no acquisition['ElectricalSeries']")

    electrical_series = nwbfile.acquisition["ElectricalSeries"]
    sampling_rate = float(electrical_series.rate)
    electrodes = nwbfile.electrodes
    n_channels = electrical_series.data.shape[1]

    data_out = np.asarray(electrical_series.data, dtype=np.float64)
    n_samples = data_out.shape[0]
    times_out = np.arange(n_samples) / sampling_rate

    good = np.ones(n_channels, dtype=bool)
    if hasattr(electrodes, "good") and electrodes.good is not None:
        good = np.asarray(electrodes["good"][:]).astype(bool)
    bad_channels = ~good

    if subject_hemisphere is not None:
        hemisphere = _normalize_hemisphere_input(subject_hemisphere)
    else:
        hemisphere = _hemisphere_from_nwb(nwbfile, n_channels)

    colnames = getattr(electrodes, "colnames", [])
    if "group_name" in colnames:
        group_names = np.asarray(electrodes["group_name"][:])
    else:
        group_names = np.array([""] * n_channels)

    channel_meta = []
    for i in range(n_channels):
        grp = str(group_names[i]) if i < len(group_names) else ""
        channel_meta.append(
            {
                "id": f"channel_{i}",
                "unit_number": i,
                "hemisphere": int(hemisphere),
                "group": grp,
                "surface": False,
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
        sampling_rate=sampling_rate,
        domain=domain,
    )
    return ecog_rts, channels
