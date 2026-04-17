_functions = [
    "extract_subject_from_nwb",
    "extract_spikes_from_nwbfile",
    "download_file",
    "get_nwb_asset_list",
]

__all__ = _functions


from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from pynwb import NWBFile
from temporaldata import (
    ArrayDict,
    Interval,
    IrregularTimeSeries,
    RegularTimeSeries,
    Data,
)

from brainsets.descriptions import SubjectDescription
from brainsets.taxonomy import (
    Hemisphere,
    RecordingTech,
    Sex,
    Species,
)

try:
    import dandi

    DANDI_AVAILABLE = True
except ImportError:
    DANDI_AVAILABLE = False


def _check_dandi_available(func_name: str) -> None:
    """Raise ImportError if DANDI is not available."""
    if not DANDI_AVAILABLE:
        raise ImportError(
            f"{func_name} requires the dandi library which is not installed. "
            "Install it with `pip install dandi`"
        )


def _normalize_subject_species(nwbfile: NWBFile) -> str | Species:
    subject = getattr(nwbfile, "subject", None)
    raw_species = getattr(subject, "species", None) if subject is not None else None
    if raw_species is None:
        return Species.UNKNOWN

    normalized_species = str(raw_species).strip()
    if not normalized_species:
        return Species.UNKNOWN
    if "NCBITaxon" in normalized_species:
        normalized_species = "NCBITaxon_" + normalized_species.split("_")[-1]
    return normalized_species


def _normalize_subject_sex(nwbfile: NWBFile) -> str | Sex:
    subject = getattr(nwbfile, "subject", None)
    raw_sex = getattr(subject, "sex", None) if subject is not None else None
    if raw_sex is None:
        return Sex.UNKNOWN

    normalized_sex = str(raw_sex).strip()
    if not normalized_sex:
        return Sex.UNKNOWN
    return normalized_sex


def extract_subject_from_nwb(nwbfile: NWBFile):
    r"""Extract a :obj:`SubjectDescription <brainsets.descriptions.SubjectDescription>` from an NWBFile

    The resultant description includes ``id``, ``species``, and ``sex``.
    This helper assumes ``subject_id`` exists in the source NWB file. When
    ``species`` or ``sex`` is missing/blank, it uses UNKNOWN placeholders so
    downstream processing can continue.

    Args:
        nwbfile: An open NWB file handle

    Returns:
        A :obj:`SubjectDescription <brainsets.descriptions.SubjectDescription>`
    """

    # Some files in the wild omit optional subject fields even when the
    # recording itself is valid, so we preserve processability with UNKNOWNs.
    return SubjectDescription(
        id=str(nwbfile.subject.subject_id).strip().lower(),
        species=_normalize_subject_species(nwbfile),
        sex=_normalize_subject_sex(nwbfile),
    )


def extract_spikes_from_nwbfile(nwbfile: NWBFile, recording_tech: RecordingTech):
    r"""Extract spikes and unit metadata from an NWBFile

    Args:
        nwbfile: An open NWB file handle
        recording_tech: Only supports
            :obj:`RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS` and
            :obj:`RecordingTech.UTAH_ARRAY_SPIKES`
    """
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


def download_file(
    path: str | Path,
    url: str,
    raw_dir: str | Path,
    overwrite: bool = False,
    skip_existing: bool = False,
) -> Path:
    r"""Download a file from DANDI

    Full path of the downloaded path will be ``raw_dir / path``.

    Args:
        path: path of the downloaded file within :obj:`raw_dir`
        url: URL of the DANDI asset
        raw_dir: root directory where the file will be downloaded
        overwrite: Will overwrite existing file if :obj:`True`
            (default :obj:`False`)

    """
    _check_dandi_available("download_file")
    import dandi.download

    raw_dir = Path(raw_dir)
    asset_path = Path(path)
    download_dir = raw_dir / asset_path.parent
    download_dir.mkdir(exist_ok=True, parents=True)
    existing_mode = (
        dandi.download.DownloadExisting.OVERWRITE
        if overwrite
        else (
            dandi.download.DownloadExisting.SKIP
            if skip_existing
            else dandi.download.DownloadExisting.REFRESH
        )
    )
    dandi.download.download(
        url,
        download_dir,
        existing=existing_mode,
    )
    return raw_dir / asset_path


def get_nwb_asset_list(dandiset_id: str) -> list:
    r"""Get a list of all remote NWB assets in the given dandiset

    Args:
        dandiset_id: The dandiset ID (e.g. 'DANDI:000688/draft')

    Returns:
        A list of all remote NWB assets (``dandi.dandiapi.RemoteBlobAsset``) within this dandiset
    """
    _check_dandi_available("get_nwb_asset_list")
    from dandi import dandiarchive

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
) -> Tuple[RegularTimeSeries, Data]:
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
                "id": f"electrode_{i}",
                "index": i,
                "hemisphere": int(hemisphere),
                "group": grp,
                "surface": False,
                "type": "ECOG",
                "bad": bool(bad_channels[i]),
            }
        )
    channels = ArrayDict.from_dataframe(
        pd.DataFrame(channel_meta), unsigned_to_long=True
    )

    domain = Interval(start=np.array([times_out[0]]), end=np.array([times_out[-1]]))
    ecog_rts = RegularTimeSeries(
        signal=data_out,
        sampling_rate=sampling_rate,
        domain=domain,
    )
    return ecog_rts, channels
