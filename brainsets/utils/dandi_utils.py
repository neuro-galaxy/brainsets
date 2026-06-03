__all__ = [
    "extract_subject_from_nwb",
    "extract_spikes_from_nwbfile",
    "extract_ecog_from_nwb",
    "download_file",
    "get_nwb_asset_list",
]

# Drives the generated API reference; see docs/source/api_reference.py.
__api_ref__ = {
    "description": None,
    "sections": [{"autosummary": __all__}],
}


from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pynwb import NWBFile
from temporaldata import ArrayDict, IrregularTimeSeries, RegularTimeSeries

from brainsets.descriptions import SubjectDescription

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


def extract_subject_from_nwb(nwbfile: NWBFile):
    r"""Extract a :obj:`SubjectDescription <brainsets.descriptions.SubjectDescription>` from an NWBFile

    The resultant description will include ``id``, ``species``, and ``sex``

    Args:
        nwbfile: An open NWB file handle

    Returns:
        A :obj:`SubjectDescription <brainsets.descriptions.SubjectDescription>`
    """

    # DANDI has requirements for metadata included in `subject`
    # - subject_id: A subject identifier must be provided.
    # - species: either a latin binomial or NCBI taxonomic identifier.
    # - sex: must be "M", "F", "O" (other), or "U" (unknown).
    # - date_of_birth or age: this does not appear to be enforced, so will be skipped.
    species = nwbfile.subject.species

    if "NCBITaxon" in species:
        species = "NCBITaxon_" + species.split("_")[-1]

    return SubjectDescription(
        id=nwbfile.subject.subject_id.lower(),
        species=species,
        sex=nwbfile.subject.sex,
    )


def extract_spikes_from_nwbfile(
    nwbfile: NWBFile,
    recording_tech: Literal["UTAH_ARRAY_THRESHOLD_CROSSINGS", "UTAH_ARRAY_SPIKES"],
):
    r"""Extract spikes and unit metadata from an NWBFile

    Args:
        nwbfile: An open NWB file handle
        recording_tech: One of ``"UTAH_ARRAY_THRESHOLD_CROSSINGS"``
            or ``"UTAH_ARRAY_SPIKES"``
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
        if recording_tech == "UTAH_ARRAY_THRESHOLD_CROSSINGS":
            # label unit
            group_name = electrodes["group_name"][i]
            unit_id = f"group_{group_name}/elec{i}/multiunit_{0}"
        elif recording_tech == "UTAH_ARRAY_SPIKES":
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
    download_policy: Literal["skip", "overwrite", "error"] = "error",
) -> Path:
    r"""Download a file from DANDI

    Full path of the downloaded path will be ``raw_dir / path``.

    Download policy controls behavior when a target file already exists.
    ``"overwrite"`` always re-downloads, ``"skip"`` keeps existing files,
    and ``"error"`` asks DANDI to raise.

    Args:
        path: path of the downloaded file within :obj:`raw_dir`
        url: URL of the DANDI asset
        raw_dir: root directory where the file will be downloaded
        download_policy: One of ``"skip"``, ``"overwrite"``, or ``"error"``
            (default ``"error"``)

    """
    _check_dandi_available("download_file")
    import dandi.download

    _EXISTING_POLICY = {
        "overwrite": dandi.download.DownloadExisting.OVERWRITE,
        "skip": dandi.download.DownloadExisting.SKIP,
        "refresh": dandi.download.DownloadExisting.REFRESH,
    }

    raw_dir = Path(raw_dir)
    asset_path = Path(path)
    target_path = raw_dir / asset_path
    download_dir = raw_dir / asset_path.parent
    download_dir.mkdir(exist_ok=True, parents=True)

    if download_policy == "error" and target_path.exists():
        raise FileExistsError(f"Target file already exists: {target_path}")

    existing_mode_map = {
        "overwrite": dandi.download.DownloadExisting.OVERWRITE,
        "skip": dandi.download.DownloadExisting.SKIP,
        # For "error", we pre-check file existence and then use REFRESH for download.
        "error": dandi.download.DownloadExisting.REFRESH,
    }
    try:
        existing_mode = existing_mode_map[download_policy]
    except KeyError as exc:
        raise ValueError(
            "download_policy must be one of: 'skip', 'overwrite', 'error'"
        ) from exc
    dandi.download.download(
        url,
        download_dir,
        existing=_EXISTING_POLICY[existing_mode],
    )
    return target_path


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


def _hemisphere_str_from_nwb(nwbfile: NWBFile) -> str:
    """Infer hemisphere as ``"L"``, ``"R"``, or ``"U"`` from NWB metadata."""
    colnames = getattr(nwbfile.electrodes, "colnames", [])
    for col in ("hemisphere", "location"):
        if col not in colnames:
            continue
        vals = nwbfile.electrodes[col][:]
        if vals is None or len(vals) == 0:
            continue
        texts = np.asarray([str(v).strip().lower() for v in vals])
        left = np.any(
            (texts == "l")
            | (texts == "left")
            | (np.char.find(texts.astype(str), "left") >= 0)
        )
        right = np.any(
            (texts == "r")
            | (texts == "right")
            | (np.char.find(texts.astype(str), "right") >= 0)
        )
        if left and not right:
            return "L"
        if right and not left:
            return "R"
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
                return "L"
            if v in ("r", "right"):
                return "R"
    return "U"


def extract_ecog_from_nwb(
    nwbfile: NWBFile,
    subject_hemisphere: str | None = None,
) -> Tuple[RegularTimeSeries, ArrayDict]:
    """Extract ECoG data from NWB file at native sampling rate.

    Hemisphere is taken from subject_hemisphere if provided (``"L"`` or ``"R"``),
    otherwise from the NWB file (electrodes table ``"location"``/``"hemisphere"`` or
    subject metadata). Channel metadata stores ``"L"``, ``"R"``, or ``"U"``.

    Warning: The entire ECoG signal is loaded into RAM as float64. For long
    recordings with many channels this can require tens of GB of memory.
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
        s = str(subject_hemisphere).strip().upper()
        if s in ("L", "LEFT"):
            hemisphere = "L"
        elif s in ("R", "RIGHT"):
            hemisphere = "R"
        else:
            hemisphere = "U"
    else:
        hemisphere = _hemisphere_str_from_nwb(nwbfile)

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
                "hemisphere": hemisphere,
                "group": grp,
                "surface": False,
                "type": "ECOG",
                "bad": bool(bad_channels[i]),
            }
        )
    channels = ArrayDict.from_dataframe(
        pd.DataFrame(channel_meta), unsigned_to_long=True
    )

    ecog_rts = RegularTimeSeries(
        signal=data_out,
        sampling_rate=sampling_rate,
        domain_start=float(times_out[0]),
    )
    return ecog_rts, channels
