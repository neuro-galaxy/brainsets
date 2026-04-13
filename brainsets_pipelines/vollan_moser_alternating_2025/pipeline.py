# /// brainset-pipeline
# python-version = "3.11"
# dependencies = ["scipy==1.10.1"]
# ///

import datetime
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import requests
from scipy.io import loadmat
from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries

from brainsets import serialize_fn_map
from brainsets.descriptions import (
    BrainsetDescription,
    DeviceDescription,
    SessionDescription,
    SubjectDescription,
)
from brainsets.pipeline import BrainsetPipeline
from brainsets.taxonomy import RecordingTech, Sex, Species, Task


parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")


BUCKET_URL = "https://data-proxy.ebrains.eu/api/v1/buckets/d-4080b78d-edc5-4ae4-8144-7f6de79930ea"

# Hardcoded file list — this is a fixed published dataset.
MANIFEST_FILES = [
    "navigation/lt/26648_1.mat",
    "navigation/lt/27764_1.mat",
    "navigation/lt/27765_2.mat",
    "navigation/lt/28063_1.mat",
    "navigation/lt/28229_3.mat",
    "navigation/lt/28304_1.mat",
    "navigation/lt/29502_3.mat",
    "navigation/mmaze/29502_1.mat",
    "navigation/of/24365_2.mat",
    "navigation/of/24666_1.mat",
    "navigation/of/25127_1.mat",
    "navigation/of/25691_1.mat",
    "navigation/of/25691_2.mat",
    "navigation/of/25843_1.mat",
    "navigation/of/25843_2.mat",
    "navigation/of/25843_5.mat",
    "navigation/of/25953_4.mat",
    "navigation/of/25953_5.mat",
    "navigation/of/25954_1.mat",
    "navigation/of/26018_2.mat",
    "navigation/of/26034_3.mat",
    "navigation/of/26035_1.mat",
    "navigation/of/26648_1.mat",
    "navigation/of/26648_2.mat",
    "navigation/of/26820_2.mat",
    "navigation/of/27764_1.mat",
    "navigation/of/27765_1.mat",
    "navigation/of/27765_2.mat",
    "navigation/of/27765_3.mat",
    "navigation/of/28063_1.mat",
    "navigation/of/28063_4.mat",
    "navigation/of/28063_5.mat",
    "navigation/of/28229_2.mat",
    "navigation/of/28229_3.mat",
    "navigation/of/28258_4.mat",
    "navigation/of/28304_1.mat",
    "navigation/of/28304_2.mat",
    "navigation/of/29502_1.mat",
    "navigation/of/29502_3.mat",
    "navigation/of_novel/25843_5.mat",
    "navigation/ww/25691_1.mat",
    "navigation/ww/25843_1.mat",
    "sleep/25691_1.mat",
    "sleep/25843_2.mat",
    "sleep/25953_5.mat",
    "sleep/26034_3.mat",
    "sleep/26648_2.mat",
    "sleep/27765_3.mat",
    "sleep/28063_5.mat",
    "sleep/28229_2.mat",
    "sleep/28304_2.mat",
]

SESSION_TYPE_TO_TASK = {
    "of": Task.NAVIGATION_OPEN_FIELD,
    "mmaze": Task.NAVIGATION_MMAZE,
    "lt": Task.NAVIGATION_LINEAR_TRACK,
    "ww": Task.NAVIGATION_WAGON_WHEEL,
    "of_novel": Task.NAVIGATION_OPEN_FIELD_NOVEL,
    "sleep": Task.SLEEP,
}

BRAINSET_DESCRIPTION = BrainsetDescription(
    id="vollan_moser_alternating_2025",
    origin_version="sharing_v4",
    derived_version="1.0.0",
    source="https://doi.org/10.25493/R5FR-EDG",
    description="Neuropixels recordings from MEC and hippocampus in rats "
    "performing spatial navigation tasks (open field, linear track, M-maze, "
    "wagon wheel) and during sleep. Includes grid cells, head direction cells, "
    "and other spatially tuned neurons.",
)


class Pipeline(BrainsetPipeline):
    brainset_id = "vollan_moser_alternating_2025"
    parser = parser

    @classmethod
    def get_manifest(
        cls,
        raw_dir: Path,
        args: Optional[Namespace],
    ) -> pd.DataFrame:
        manifest_list = []
        for fname in MANIFEST_FILES:
            parts = Path(fname).parts
            if parts[0] == "navigation":
                session_type = parts[1]
                animal_rec = Path(fname).stem
                session_id = f"{session_type}_{animal_rec}"
                data_category = "navigation"
            else:
                animal_rec = Path(fname).stem
                session_id = f"sleep_{animal_rec}"
                session_type = "sleep"
                data_category = "sleep"

            manifest_list.append(
                {
                    "session_id": session_id,
                    "session_type": session_type,
                    "data_category": data_category,
                    "fname": fname,
                }
            )

        manifest = pd.DataFrame(manifest_list).set_index("session_id")
        return manifest

    def download(self, manifest_item):
        # Each file is downloaded individually from the ebrains data-proxy API
        fpath = self.raw_dir / manifest_item.fname
        if fpath.exists() and not self.args.redownload:
            return fpath

        fpath.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BUCKET_URL}/sharing_v4/{manifest_item.fname}"

        self.update_status("DOWNLOADING")
        logging.info(f"Downloading {manifest_item.fname}")
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            with open(fpath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        print(
                            f"\r  {manifest_item.fname}: "
                            f"{downloaded / 1e6:.0f} / {total / 1e6:.0f} MB ({pct}%)",
                            end="",
                            flush=True,
                        )
            if total:
                print()

        return fpath

    def process(self, fpath):
        fpath = Path(fpath)
        # Infer data category and session type from file path
        # Navigation: .../navigation/<session_type>/<animal>_<rec>.mat
        # Sleep: .../sleep/<animal>_<rec>.mat
        if fpath.parent.parent.name == "navigation":
            session_type = fpath.parent.name  # of, mmaze, lt, ww, of_novel
            session_id = f"{session_type}_{fpath.stem}"
            self._process_navigation(fpath, session_id, session_type)
        elif fpath.parent.name == "sleep":
            session_id = f"sleep_{fpath.stem}"
            self._process_sleep(fpath, session_id)
        else:
            raise ValueError(f"Cannot infer session type from path: {fpath}")

    def _process_navigation(self, fpath, session_id, session_type):
        store_path = self.processed_dir / f"{session_id}.h5"
        if store_path.exists() and not self.args.reprocess:
            self.update_status("Skipped Processing")
            return

        self.update_status("Loading mat")
        mat = loadmat(fpath, simplify_cells=True)
        ds = mat["Dsession"]

        # Extract animal ID from filename (e.g. "29502" from "29502_1.mat")
        animal_id = fpath.stem.split("_")[0]

        subject = SubjectDescription(
            id=animal_id,
            species=Species.RATTUS_NORVEGICUS,
            sex=Sex.UNKNOWN,
        )

        task = SESSION_TYPE_TO_TASK[session_type]

        session_description = SessionDescription(
            id=session_id,
            recording_date=datetime.datetime(
                1970, 1, 1, tzinfo=datetime.timezone.utc
            ),  # placeholder date since actual recording dates are not provided
            task=task,
        )

        device_description = DeviceDescription(
            id=f"{animal_id}_neuropixels",
            recording_tech=RecordingTech.NEUROPIXELS_SPIKES,
        )

        # Build domain from contiguous segments of the speed-filtered timeseries
        self.update_status("Building domain")
        domain = build_domain_from_timestamps(ds["t"])

        # Extract behavior timeseries
        self.update_status("Extracting behavior")
        behavior = extract_navigation_behavior(ds, domain)

        # Extract units and spikes
        self.update_status("Extracting units and spikes")
        units, spikes = extract_navigation_units_and_spikes(ds, domain)

        # Extract probe channel maps as metadata.
        # Note: the raw data does not provide a direct unit→channel mapping.
        # Units have probe_id and shank_id/shank_pos; channels have probe_id,
        # shank, and x/y coords. Join on probe_id + shank_id and nearest
        # shank_pos ↔ y_um to associate units with channels if needed.
        probe_channels = extract_probe_channel_maps(ds)

        data = Data(
            brainset=BRAINSET_DESCRIPTION,
            subject=subject,
            session=session_description,
            device=device_description,
            # neural activity
            spikes=spikes,
            units=units,
            # behavior
            behavior=behavior,
            # metadata
            probe_channels=probe_channels,
            # domain
            domain=domain,
        )

        # Split domain intervals into train/valid/test
        self.update_status("Creating splits")
        train_domain, valid_domain, test_domain = domain.split(
            [0.7, 0.1, 0.2], shuffle=True, random_seed=42
        )
        data.set_train_domain(train_domain)
        data.set_valid_domain(valid_domain)
        data.set_test_domain(test_domain)

        self.update_status("Storing")
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

    def _process_sleep(self, fpath, session_id):
        """Process a sleep session from ``Dsleep``.

        Sleep sessions have a fundamentally different structure to navigation
        sessions and should be treated as independent.  The raw ``Dsleep``
        struct contains only spike times within identified SWS/REM epochs and
        minimal unit identifiers — there is no shared 10 ms timebase, no
        tracking data, no LMT results, and no probe channel maps.  None of
        the fields on navigation sessions (``samples``, ``theta_chunks``,
        ``probe_channels``) are present here.

        Stored fields:
            ``spikes``   spike timestamps and unit indices
            ``units``    unit metadata (id only)
            ``domain``   union of SWS and REM epoch intervals
        """
        store_path = self.processed_dir / f"{session_id}.h5"
        if store_path.exists() and not self.args.reprocess:
            self.update_status("Skipped Processing")
            return

        self.update_status("Loading mat")
        mat = loadmat(fpath, simplify_cells=True)
        ds = mat["Dsleep"]

        animal_id = fpath.stem.split("_")[0]

        subject = SubjectDescription(
            id=animal_id,
            species=Species.RATTUS_NORVEGICUS,
            sex=Sex.UNKNOWN,
        )

        session_description = SessionDescription(
            id=session_id,
            recording_date=datetime.datetime(
                1970, 1, 1, tzinfo=datetime.timezone.utc
            ),  # placeholder date since actual recording dates are not provided
            task=Task.SLEEP,
        )

        device_description = DeviceDescription(
            id=f"{animal_id}_neuropixels",
            recording_tech=RecordingTech.NEUROPIXELS_SPIKES,
        )

        # Domain: union of SWS and REM epochs
        self.update_status("Building domain")
        domain = build_sleep_domain(ds)

        # Extract spikes
        self.update_status("Extracting spikes")
        units, spikes = extract_sleep_units_and_spikes(ds, domain)

        data = Data(
            brainset=BRAINSET_DESCRIPTION,
            subject=subject,
            session=session_description,
            device=device_description,
            spikes=spikes,
            units=units,
            domain=domain,
        )

        self.update_status("Creating splits")
        train_domain, valid_domain, test_domain = domain.split(
            [0.7, 0.1, 0.2], shuffle=True, random_seed=42
        )
        data.set_train_domain(train_domain)
        data.set_valid_domain(valid_domain)
        data.set_test_domain(test_domain)

        self.update_status("Storing")
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def build_domain_from_timestamps(t):
    """Build an Interval domain from speed-filtered timestamps.

    In this dataset, ``ds["t"]`` contains timestamps of position samples that
    have already been speed-filtered (stationary periods removed). This means
    the array can have large gaps wherever the animal was below the speed
    threshold. We detect those gaps (where dt > 2x the nominal sampling
    interval) and create separate Interval segments for each contiguous block
    of movement.
    """
    t = t.flatten().astype(np.float64)
    if len(t) == 0:
        return Interval(
            start=np.array([], dtype=np.float64),
            end=np.array([], dtype=np.float64),
        )
    if len(t) == 1:
        return Interval(start=t[:1], end=t[:1] + 0.01)

    dt_nominal = np.median(np.diff(t))
    gap_threshold = 2.0 * dt_nominal

    dt = np.diff(t)
    gap_mask = dt > gap_threshold

    # Segment boundaries: each gap splits the timeseries
    gap_indices = np.where(gap_mask)[0]

    starts = np.empty(len(gap_indices) + 1, dtype=np.float64)
    ends = np.empty(len(gap_indices) + 1, dtype=np.float64)

    starts[0] = t[0]
    ends[-1] = t[-1] + dt_nominal  # extend last segment by one sample

    for i, gap_idx in enumerate(gap_indices):
        ends[i] = t[gap_idx] + dt_nominal  # end of segment before gap
        starts[i + 1] = t[gap_idx + 1]  # start of segment after gap

    domain = Interval(start=starts, end=ends)
    assert domain.is_sorted()
    assert domain.is_disjoint()
    return domain


def extract_navigation_behavior(ds, domain):
    """Extract behavior timeseries (x, y, z, hd, theta, id) from Dsession."""
    t = ds["t"].flatten().astype(np.float64)
    n = len(t)

    # id (decoded internal direction) may be empty if LMT model was not fitted
    id_raw = ds["id"].flatten()
    id_arr = (
        id_raw.astype(np.float32)
        if len(id_raw) == n
        else np.full(n, np.nan, dtype=np.float32)
    )

    # theta may also be empty in some sessions
    theta_raw = ds["theta"].flatten()
    theta_arr = (
        theta_raw.astype(np.float32)
        if len(theta_raw) == n
        else np.full(n, np.nan, dtype=np.float32)
    )

    behavior = IrregularTimeSeries(
        timestamps=t,
        x=ds["x"].flatten().astype(np.float32),
        y=ds["y"].flatten().astype(np.float32),
        z=ds["z"].flatten().astype(np.float32),
        hd=ds["hd"].flatten().astype(np.float32),
        theta=theta_arr,
        id=id_arr,
        domain=domain,
    )
    return behavior


def extract_navigation_units_and_spikes(ds, domain):
    """Extract units metadata and spikes from Dsession.units (MEC + HC)."""
    units_struct = ds["units"]
    unit_meta = []
    spike_timestamps_list = []
    spike_unit_index_list = []
    unit_idx = 0

    # units_struct contains only "mec" and "hc" keys (medial entorhinal
    # cortex and hippocampus) — these are the two recorded brain regions.
    for location in ["mec", "hc"]:
        region_units = units_struct[location]

        # Handle empty regions (numpy array with size 0 or empty list)
        if isinstance(region_units, np.ndarray) and region_units.size == 0:
            continue
        if isinstance(region_units, list) and len(region_units) == 0:
            continue

        # With simplify_cells=True, this is usually a list of dicts, but a
        # single MATLAB struct element may be returned as a bare dict.
        if isinstance(region_units, dict):
            region_units = [region_units]
        elif not isinstance(region_units, list):
            continue

        for i, u in enumerate(region_units):
            unit_meta.append(
                {
                    "id": f"{location}_{i}",
                    "location": location,
                    "probe_id": int(u["probeId"]),
                    "shank_id": int(u["shank"]),
                    "shank_pos": float(u["shankPos"]),
                    "mean_rate": float(u["meanRate"]),
                    "ks2_label": str(u.get("ks2Label", "")),
                    "is_grid": int(u.get("isGrid", 0)),
                }
            )

            spike_times = u["spikeTimes"].flatten().astype(np.float64)
            if len(spike_times) > 0:
                spike_timestamps_list.append(spike_times)
                spike_unit_index_list.append(
                    np.full(len(spike_times), unit_idx, dtype=np.int64)
                )

            unit_idx += 1

    units = ArrayDict.from_dataframe(pd.DataFrame(unit_meta))

    if spike_timestamps_list:
        all_timestamps = np.concatenate(spike_timestamps_list)
        all_unit_indices = np.concatenate(spike_unit_index_list)
    else:
        all_timestamps = np.array([], dtype=np.float64)
        all_unit_indices = np.array([], dtype=np.int64)

    spikes = IrregularTimeSeries(
        timestamps=all_timestamps,
        unit_index=all_unit_indices,
        domain=domain,
    )
    spikes.sort()

    return units, spikes


def extract_probe_channel_maps(ds):
    """Extract probe channel locations as an ArrayDict."""
    probe_maps = ds["probeChannelMaps"]
    if not isinstance(probe_maps, list):
        probe_maps = [probe_maps]

    channel_meta = []
    for probe_idx, pm in enumerate(probe_maps):
        n_channels = len(pm["xcoords"])
        for ch in range(n_channels):
            channel_meta.append(
                {
                    "probe_id": probe_idx + 1,
                    "channel_index": ch,
                    "x_um": float(pm["xcoords"][ch]),
                    "y_um": float(pm["ycoords"][ch]),
                    "shank_id": int(pm["shankInd"][ch]),
                    "connected": bool(pm["connected"][ch]),
                }
            )

    return ArrayDict.from_dataframe(pd.DataFrame(channel_meta))


def build_sleep_domain(ds):
    """Build domain from SWS and REM epoch times."""
    times = ds["times"]
    sws_times = times["sws"]  # (N, 2) array
    rem_times = times["rem"]  # (M, 2) array

    all_starts = []
    all_ends = []

    if sws_times.size > 0:
        sws_times = np.atleast_2d(sws_times)
        all_starts.append(sws_times[:, 0])
        all_ends.append(sws_times[:, 1])

    if rem_times.size > 0:
        rem_times = np.atleast_2d(rem_times)
        all_starts.append(rem_times[:, 0])
        all_ends.append(rem_times[:, 1])

    starts = np.concatenate(all_starts).astype(np.float64)
    ends = np.concatenate(all_ends).astype(np.float64)

    # Sort by start time
    sort_idx = np.argsort(starts)
    starts = starts[sort_idx]
    ends = ends[sort_idx]

    domain = Interval(start=starts, end=ends)
    assert domain.is_sorted()
    assert domain.is_disjoint()
    return domain


def extract_sleep_units_and_spikes(ds, domain):
    """Extract units and spikes from Dsleep.units."""
    sleep_units = ds["units"]

    # With simplify_cells, this is a list of dicts
    if not isinstance(sleep_units, list):
        sleep_units = [sleep_units] if isinstance(sleep_units, dict) else []

    unit_meta = []
    spike_timestamps_list = []
    spike_unit_index_list = []

    for i, u in enumerate(sleep_units):
        unit_meta.append(
            {
                "id": f"unit_{i}",
            }
        )

        spike_times = u["spikeTimes"]
        if isinstance(spike_times, np.ndarray) and spike_times.size > 0:
            spike_times = spike_times.flatten().astype(np.float64)
            spike_timestamps_list.append(spike_times)
            spike_unit_index_list.append(np.full(len(spike_times), i, dtype=np.int64))

    units = ArrayDict.from_dataframe(pd.DataFrame(unit_meta))

    if spike_timestamps_list:
        all_timestamps = np.concatenate(spike_timestamps_list)
        all_unit_indices = np.concatenate(spike_unit_index_list)
    else:
        all_timestamps = np.array([], dtype=np.float64)
        all_unit_indices = np.array([], dtype=np.int64)

    spikes = IrregularTimeSeries(
        timestamps=all_timestamps,
        unit_index=all_unit_indices,
        domain=domain,
    )
    spikes.sort()

    return units, spikes
