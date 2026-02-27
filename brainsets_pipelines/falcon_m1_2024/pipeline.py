# /// brainset-pipeline
# python-version = "3.11"
# dependencies = ["dandi>=0.71.3"]
# ///
"""FALCON M1: Monkey reach-to-grasp with EMG decoding.

Neural activity recorded from 64-channel Utah array in motor cortex.
Task involves controlled finger movements with 16-channel EMG recordings
from upper limb muscles. Part of the FALCON Benchmark.

Reference:
    Karpowicz et al. (2024). FALCON: Few-shot Adaptive Learning of
    neural COdecodersN. https://dandiarchive.org/dandiset/000941

Split Strategy:
    - held-in sessions: Used for calibration/training
    - held-out sessions: Used for validation/few-shot adaptation
    - minival: Small validation set (used as test)

    Held-in sessions: 20120924, 20120926, 20120927, 20120928
    Held-out sessions: 20121004, 20121017, 20121024
"""

import datetime
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from temporaldata import (
    ArrayDict,
    Data,
    Interval,
    IrregularTimeSeries,
    RegularTimeSeries,
)

from brainsets import serialize_fn_map
from brainsets.descriptions import (
    BrainsetDescription,
    DeviceDescription,
    SessionDescription,
    SubjectDescription,
)
from brainsets.pipeline import BrainsetPipeline
from brainsets.taxonomy import RecordingTech, Sex, Species, Task
from brainsets.utils.dandi_utils import (
    download_file,
    extract_spikes_from_nwbfile,
    get_nwb_asset_list,
)


parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")
parser.add_argument(
    "--bin-size-ms",
    type=float,
    default=None,
    help="Bin size in milliseconds. If provided, spikes will be binned into "
    "RegularTimeSeries. If None (default), spike times are preserved as "
    "IrregularTimeSeries. FALCON benchmark uses 20ms bins.",
)

# FALCON M1 held-in/held-out split definition
HELD_IN_SESSIONS = ["20120924", "20120926", "20120927", "20120928"]
HELD_OUT_SESSIONS = ["20121004", "20121017", "20121024"]


class Pipeline(BrainsetPipeline):
    """Pipeline for processing FALCON M1 dataset from DANDI."""

    brainset_id = "falcon_m1_2024"
    dandiset_id = "DANDI:000941"
    parser = parser

    @classmethod
    def get_manifest(
        cls,
        raw_dir: Path,
        args: Optional[Namespace],
    ) -> pd.DataFrame:
        """Get manifest of NWB files from DANDI archive.

        Returns:
            DataFrame with columns: path, url, session_date, split_type
            Index: session_id (e.g., "20120924_held_in")
        """
        asset_list = get_nwb_asset_list(cls.dandiset_id)
        manifest_list = []

        for asset in asset_list:
            basename = asset.path.split("/")[-1]

            # Parse session date from filename
            # Format: sub-MonkeyL-held-in-calib_ses-20120924_behavior+ecephys.nwb
            date_match = re.search(r"ses-(\d{8})", basename)
            if not date_match:
                continue
            session_date = date_match.group(1)

            # Determine split type
            split_type = determine_split_type(basename)

            session_id = f"{session_date}_{split_type}"
            manifest_list.append(
                {
                    "session_id": session_id,
                    "path": asset.path,
                    "url": asset.download_url,
                    "session_date": session_date,
                    "split_type": split_type,
                }
            )

        manifest = pd.DataFrame(manifest_list).set_index("session_id")
        return manifest

    def download(self, manifest_item) -> Path:
        """Download a single NWB file from DANDI.

        Args:
            manifest_item: Row from manifest DataFrame.

        Returns:
            Path to downloaded NWB file.
        """
        self.update_status("DOWNLOADING")
        self.raw_dir.mkdir(exist_ok=True, parents=True)

        fpath = download_file(
            manifest_item.path,
            manifest_item.url,
            self.raw_dir,
            overwrite=self.args.redownload if self.args else False,
        )
        return fpath

    def process(self, fpath: Path) -> None:
        """Process NWB file and save as HDF5.

        Args:
            fpath: Path to downloaded NWB file.
        """
        self.processed_dir.mkdir(exist_ok=True, parents=True)

        # Parse session info from filename
        basename = fpath.name
        date_match = re.search(r"ses-(\d{8})", basename)
        if not date_match:
            raise ValueError(f"Could not parse date from {fpath}")
        session_date = date_match.group(1)
        split_type = determine_split_type(basename)
        session_id = f"{session_date}_{split_type}"

        # Generate output filename
        bin_size_ms = getattr(self.args, "bin_size_ms", None) if self.args else None
        if bin_size_ms is not None:
            store_path = self.processed_dir / f"{session_id}_bin{int(bin_size_ms)}ms.h5"
        else:
            store_path = self.processed_dir / f"{session_id}.h5"

        if store_path.exists() and not (self.args and self.args.reprocess):
            self.update_status(f"Skipped {session_id}")
            return

        self.update_status(f"Processing {session_id}")

        brainset_description = BrainsetDescription(
            id=self.brainset_id,
            origin_version="dandi/000941/draft",
            derived_version="1.0.0",
            source="https://dandiarchive.org/dandiset/000941",
            description=(
                "FALCON M1 dataset: Monkey reach-to-grasp task. "
                "Neural activity recorded from 64-channel Utah array in motor cortex. "
                "Task involves controlled finger movements with 16-channel EMG recordings "
                "from upper limb muscles. Part of the FALCON Benchmark."
            ),
        )

        # Open NWB file
        self.update_status("Loading NWB")
        with NWBHDF5IO(fpath, "r") as io:
            nwbfile = io.read()

            # Extract subject metadata (Monkey L)
            subject = SubjectDescription(
                id="monkey_l",
                species=Species.MACACA_MULATTA,
                sex=Sex.UNKNOWN,
            )

            # Extract session metadata
            recording_date = datetime.datetime.strptime(session_date, "%Y%m%d")
            session_description = SessionDescription(
                id=session_id,
                recording_date=recording_date,
                task=Task.REACHING,
            )

            # Device description
            device_description = DeviceDescription(
                id=f"monkey_l_{session_id}",
                recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,
            )

            # Extract EMG data (needed for timestamps and behavior target)
            self.update_status("Extracting EMG")
            emg = extract_emg_data(nwbfile)

            # Extract neural activity (raw spike times first)
            self.update_status("Extracting Spikes")
            spikes_raw, units = extract_spikes_from_nwbfile(
                nwbfile, recording_tech=RecordingTech.UTAH_ARRAY_SPIKES
            )

            # Conditionally bin spikes
            if bin_size_ms is not None:
                bin_size_s = bin_size_ms / 1000.0
                self.update_status(f"Binning spikes at {bin_size_ms}ms")
                binned_counts, bin_timestamps = bin_spikes(
                    spikes_raw, units, bin_size_s, emg.timestamps[:]
                )
                spikes = RegularTimeSeries(
                    counts=binned_counts,
                    sampling_rate=1.0 / bin_size_s,
                    domain="auto",
                    domain_start=bin_timestamps[0],
                )
            else:
                spikes = spikes_raw

            # Extract evaluation mask (FALCON-specific)
            self.update_status("Extracting Eval Mask")
            eval_intervals = extract_eval_mask(nwbfile, emg)

            # Extract trials
            self.update_status("Extracting Trials")
            trials = extract_trials(nwbfile)

        # Create Data object
        data = Data(
            brainset=brainset_description,
            subject=subject,
            session=session_description,
            device=device_description,
            spikes=spikes,
            units=units,
            emg=emg,
            trials=trials,
            eval_intervals=eval_intervals,
            domain=emg.domain,
        )

        # Add metadata
        data.falcon_split = split_type
        data.falcon_session_group = (
            "held_in" if session_date in HELD_IN_SESSIONS else "held_out"
        )
        if bin_size_ms is not None:
            data.spike_bin_size_ms = bin_size_ms
            data.spike_format = "binned"
        else:
            data.spike_format = "spike_times"

        # Assign train/valid/test based on FALCON split
        self.update_status("Creating Splits")
        assign_falcon_split(data, split_type, trials)

        # Save to HDF5
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


def determine_split_type(filename: str) -> str:
    """Determine if file belongs to held-in, held-out, or minival split."""
    basename = filename.lower()

    if "minival" in basename:
        return "minival"
    elif "held_out" in basename or "held-out" in basename:
        return "held_out"
    elif "held_in" in basename or "held-in" in basename:
        return "held_in"
    else:
        return "unknown"


def extract_emg_data(nwbfile) -> IrregularTimeSeries:
    """Extract 16D EMG from NWB file.

    Args:
        nwbfile: Open NWB file object.

    Returns:
        IrregularTimeSeries with data attribute of shape (time, 16).
    """
    emg_container = nwbfile.acquisition["preprocessed_emg"]
    muscles = [ts for ts in emg_container.time_series]

    emg_data = []
    emg_timestamps = None
    for muscle in muscles:
        ts_data = emg_container.get_timeseries(muscle)
        emg_data.append(ts_data.data[:])
        if emg_timestamps is None:
            emg_timestamps = ts_data.timestamps[:]
        else:
            assert np.array_equal(
                emg_timestamps, ts_data.timestamps[:]
            ), f"Timestamp mismatch for muscle {muscle}"

    emg_data = np.vstack(emg_data).T  # (time, 16)

    emg = IrregularTimeSeries(
        timestamps=emg_timestamps,
        data=emg_data.astype(np.float32),
        domain="auto",
    )

    return emg


def extract_trials(nwbfile) -> Interval:
    """Extract trial information from NWB file.

    Args:
        nwbfile: Open NWB file object.

    Returns:
        Interval with trial start/end times.
    """
    trial_table = nwbfile.trials.to_dataframe().reset_index()

    # Rename columns to match temporaldata conventions
    trial_table = trial_table.rename(
        columns={
            "start_time": "start",
            "stop_time": "end",
        }
    )

    # Drop string columns that cause HDF5 serialization issues
    columns_to_drop = []
    for col in trial_table.columns:
        if trial_table[col].dtype == "object":
            first_val = trial_table[col].iloc[0] if len(trial_table) > 0 else None
            if isinstance(first_val, str):
                columns_to_drop.append(col)

    if columns_to_drop:
        trial_table = trial_table.drop(columns=columns_to_drop)

    trials = Interval.from_dataframe(trial_table)
    trials.is_valid = np.ones(len(trials), dtype=bool)

    return trials


def extract_eval_mask(nwbfile, emg: IrregularTimeSeries) -> Interval:
    """Extract evaluation mask and convert to Interval.

    Args:
        nwbfile: Open NWB file object.
        emg: EMG time series (for timestamps).

    Returns:
        Interval with evaluation periods.
    """
    eval_mask = nwbfile.acquisition["eval_mask"].data[:].astype(bool)
    emg_ts = emg.timestamps[:]
    assert len(eval_mask) == len(
        emg_ts
    ), f"eval_mask length ({len(eval_mask)}) != emg timestamps ({len(emg_ts)})"

    # Convert eval_mask to Interval (periods where eval_mask is True)
    # End timestamps use end-exclusive convention (first sample after eval period)
    eval_mask_starts = []
    eval_mask_ends = []
    in_eval_period = False

    for is_eval, timestamp in zip(eval_mask, emg_ts):
        if is_eval and not in_eval_period:
            eval_mask_starts.append(timestamp)
            in_eval_period = True
        elif not is_eval and in_eval_period:
            eval_mask_ends.append(timestamp)
            in_eval_period = False

    # Handle case where eval period extends to end
    if in_eval_period:
        dt = np.median(np.diff(emg_ts[:100]))
        eval_mask_ends.append(emg_ts[-1] + dt)

    if len(eval_mask_starts) > 0:
        eval_intervals = Interval(
            start=np.array(eval_mask_starts), end=np.array(eval_mask_ends)
        )
    else:
        eval_intervals = Interval(start=np.array([]), end=np.array([]))

    return eval_intervals


def bin_spikes(
    spikes: IrregularTimeSeries,
    units: ArrayDict,
    bin_size_s: float,
    reference_timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin spike times into regular time bins aligned with reference timestamps.

    Args:
        spikes: IrregularTimeSeries containing spike times and unit indices.
        units: ArrayDict containing unit metadata.
        bin_size_s: Bin size in seconds (e.g., 0.02 for 20ms).
        reference_timestamps: Timestamps to align bins with (e.g., from EMG).

    Returns:
        Tuple of (binned_counts, bin_timestamps):
            binned_counts: (n_bins, n_units) array of spike counts per bin.
            bin_timestamps: (n_bins,) array of bin center timestamps.
    """
    n_units = len(units.id)
    n_bins = len(reference_timestamps)

    # Initialize output array
    binned_counts = np.zeros((n_bins, n_units), dtype=np.float32)

    # Generate uniform bin edges
    start = reference_timestamps[0] - bin_size_s
    bin_edges = start + np.arange(n_bins + 1) * bin_size_s

    # Bin spikes for each unit
    spike_times = spikes.timestamps[:]
    spike_units = spikes.unit_index[:]

    for unit_idx in range(n_units):
        unit_mask = spike_units == unit_idx
        unit_spike_times = spike_times[unit_mask]
        counts, _ = np.histogram(unit_spike_times, bins=bin_edges)
        binned_counts[:, unit_idx] = counts

    # Bin center timestamps (midpoints of uniform edges)
    bin_timestamps = bin_edges[:-1] + bin_size_s / 2.0

    return binned_counts, bin_timestamps


def assign_falcon_split(data: Data, split_type: str, trials: Interval) -> None:
    """Assign train/valid/test split based on FALCON split type.

    - held_in: Training data
    - held_out: Validation data (for few-shot adaptation)
    - minival: Test data

    Args:
        data: Data object to modify.
        split_type: One of "held_in", "held_out", "minival".
        trials: Trial intervals.
    """
    valid_trials = trials.select_by_mask(trials.is_valid)
    empty_interval = Interval(start=np.array([]), end=np.array([]))

    if "held_in" in split_type:
        data.set_train_domain(valid_trials)
        data.set_valid_domain(empty_interval)
        data.set_test_domain(empty_interval)
    elif "held_out" in split_type:
        data.set_train_domain(empty_interval)
        data.set_valid_domain(valid_trials)
        data.set_test_domain(empty_interval)
    elif split_type == "minival":
        data.set_train_domain(empty_interval)
        data.set_valid_domain(empty_interval)
        data.set_test_domain(valid_trials)
    else:
        # Unknown split, default to train
        data.set_train_domain(valid_trials)
        data.set_valid_domain(empty_interval)
        data.set_test_domain(empty_interval)
