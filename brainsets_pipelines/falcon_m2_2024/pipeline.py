# /// brainset-pipeline
# python-version = "3.11"
# dependencies = ["dandi>=0.71.3"]
# ///
"""FALCON M2: Monkey 2D finger velocity decoding.

Neural activity recorded from 96-channel Utah array in motor cortex.
Task involves 2D finger velocity tracking.
Part of the FALCON Benchmark.

Reference:
    Karpowicz et al. (2024). FALCON: Few-shot Adaptive Learning of
    neural COdecodersN. https://dandiarchive.org/dandiset/000953

Split Strategy:
    - held-in sessions: Used for calibration/training
    - held-out sessions: Used for validation/few-shot adaptation
    - minival: Small validation set (used as test)

    Held-in sessions: Run1_20201019, Run2_20201019, Run1_20201020, Run2_20201020,
                      Run1_20201027, Run2_20201027, Run1_20201028
    Held-out sessions: Run1_20201030, Run2_20201030, Run1_20201118,
                       Run1_20201119, Run1_20201124, Run2_20201124
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

# FALCON M2 held-in/held-out split definition
HELD_IN_SESSIONS = [
    "Run1_20201019",
    "Run2_20201019",
    "Run1_20201020",
    "Run2_20201020",
    "Run1_20201027",
    "Run2_20201027",
    "Run1_20201028",
]
HELD_OUT_SESSIONS = [
    "Run1_20201030",
    "Run2_20201030",
    "Run1_20201118",
    "Run1_20201119",
    "Run1_20201124",
    "Run2_20201124",
]


class Pipeline(BrainsetPipeline):
    """Pipeline for processing FALCON M2 dataset from DANDI."""

    brainset_id = "falcon_m2_2024"
    dandiset_id = "DANDI:000953"
    parser = parser

    @classmethod
    def get_manifest(
        cls,
        raw_dir: Path,
        args: Optional[Namespace],
    ) -> pd.DataFrame:
        """Get manifest of NWB files from DANDI archive.

        Returns:
            DataFrame with columns: path, url, session_date, run, split_type
            Index: session_id (e.g., "Run1_20201019_held_in")
        """
        asset_list = get_nwb_asset_list(cls.dandiset_id)
        manifest_list = []

        for asset in asset_list:
            basename = asset.path.split("/")[-1]

            # Parse session info from filename
            # Format: sub-MonkeyN-held-in-calib_ses-2020-10-19-Run1_behavior+ecephys.nwb
            parsed = parse_m2_filename(basename)
            if parsed is None:
                continue

            run_name, date_str, split_type = parsed
            session_id = f"{run_name}_{date_str}_{split_type}"

            manifest_list.append(
                {
                    "session_id": session_id,
                    "path": asset.path,
                    "url": asset.download_url,
                    "session_date": date_str,
                    "run_name": run_name,
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
        parsed = parse_m2_filename(basename)
        if parsed is None:
            raise ValueError(f"Could not parse session info from {fpath}")

        run_name, date_str, split_type = parsed
        full_session_id = f"{run_name}_{date_str}"  # e.g., "Run1_20201019"
        session_id = f"{full_session_id}_{split_type}"

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
            origin_version="dandi/000953/draft",
            derived_version="1.0.0",
            source="https://dandiarchive.org/dandiset/000953",
            description=(
                "FALCON M2 dataset: Monkey 2D finger velocity task. "
                "Neural activity recorded from 96-channel Utah array in motor cortex. "
                "Task involves 2D finger velocity tracking. "
                "Part of the FALCON Benchmark."
            ),
        )

        # Open NWB file
        self.update_status("Loading NWB")
        io = NWBHDF5IO(fpath, "r")
        nwbfile = io.read()

        # Extract subject metadata (Monkey N)
        subject = SubjectDescription(
            id="monkey_n",
            species=Species.MACACA_MULATTA,
            sex=Sex.UNKNOWN,
        )

        # Extract session metadata
        recording_date = datetime.datetime.strptime(date_str, "%Y%m%d")
        session_description = SessionDescription(
            id=session_id,
            recording_date=recording_date,
            task=Task.REACHING,
        )

        # Device description
        device_description = DeviceDescription(
            id=f"monkey_n_{session_id}",
            recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,
        )

        # Extract finger velocity (needed for timestamps and behavior target)
        self.update_status("Extracting Finger Velocity")
        finger = extract_finger_velocity(nwbfile)

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
                spikes_raw, units, bin_size_s, finger.timestamps[:]
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
        eval_intervals = extract_eval_mask(nwbfile, finger)

        # Extract trials
        self.update_status("Extracting Trials")
        trials = extract_trials(nwbfile)

        io.close()

        # Create Data object
        data = Data(
            brainset=brainset_description,
            subject=subject,
            session=session_description,
            device=device_description,
            spikes=spikes,
            units=units,
            finger=finger,
            trials=trials,
            eval_intervals=eval_intervals,
            domain=finger.domain,
        )

        # Add metadata
        data.falcon_split = split_type
        data.falcon_session_group = (
            "held_in" if full_session_id in HELD_IN_SESSIONS else "held_out"
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


def parse_m2_filename(filename: str) -> Optional[tuple[str, str, str]]:
    """Parse FALCON M2 NWB filename to extract session info.

    Args:
        filename: NWB filename.

    Returns:
        Tuple of (run_name, date_str, split_type) or None if parsing fails.

    Examples:
        - sub-MonkeyN-held-in-calib_ses-2020-10-19-Run1_behavior+ecephys.nwb
          -> ("Run1", "20201019", "held_in")
    """
    basename = filename.split("/")[-1]

    # Determine split type
    split_type = determine_split_type(basename)

    if "behavior+ecephys" in basename:
        # DANDI format: ses-2020-10-19-Run1
        ses_match = re.search(r"ses-(\d{4})-(\d{2})-(\d{2})-(Run\d+)", basename)
        if ses_match:
            year, month, day, run_name = ses_match.groups()
            date_str = f"{year}{month}{day}"
            return (run_name, date_str, split_type)
    else:
        # Evaluation format: sub-MonkeyNRun1_20201019
        eval_match = re.search(r"MonkeyN(Run\d+)_(\d{8})", basename)
        if eval_match:
            run_name, date_str = eval_match.groups()
            return (run_name, date_str, split_type)

    return None


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


def extract_finger_velocity(nwbfile) -> IrregularTimeSeries:
    """Extract 2D finger velocity from NWB file.

    Args:
        nwbfile: Open NWB file object.

    Returns:
        IrregularTimeSeries with vel attribute of shape (time, 2).
    """
    vel_container = nwbfile.acquisition["finger_vel"]
    labels = [ts for ts in vel_container.time_series]

    vel_data = []
    vel_timestamps = None
    for ts in labels:
        ts_data = vel_container.get_timeseries(ts)
        vel_data.append(ts_data.data[:])
        vel_timestamps = ts_data.timestamps[:]

    vel_data = np.vstack(vel_data).T  # (time, 2)

    finger = IrregularTimeSeries(
        timestamps=vel_timestamps,
        vel=vel_data.astype(np.float32),
        domain="auto",
    )

    return finger


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


def extract_eval_mask(nwbfile, finger: IrregularTimeSeries) -> Interval:
    """Extract evaluation mask and convert to Interval.

    Args:
        nwbfile: Open NWB file object.
        finger: Finger velocity time series (for timestamps).

    Returns:
        Interval with evaluation periods.
    """
    eval_mask = nwbfile.acquisition["eval_mask"].data[:].astype(bool)

    # Convert eval_mask to Interval (periods where eval_mask is True)
    eval_mask_starts = []
    eval_mask_ends = []
    in_eval_period = False

    for i, (is_eval, timestamp) in enumerate(zip(eval_mask, finger.timestamps)):
        if is_eval and not in_eval_period:
            eval_mask_starts.append(timestamp)
            in_eval_period = True
        elif not is_eval and in_eval_period:
            eval_mask_ends.append(timestamp)
            in_eval_period = False

    # Handle case where eval period extends to end
    if in_eval_period:
        eval_mask_ends.append(finger.timestamps[-1])

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
        reference_timestamps: Timestamps to align bins with.

    Returns:
        Tuple of (binned_counts, bin_timestamps):
            binned_counts: (n_bins, n_units) array of spike counts per bin.
            bin_timestamps: (n_bins,) array of bin center timestamps.
    """
    n_units = len(units.id)

    # Use reference timestamps as bin end times
    bin_end_timestamps = reference_timestamps
    n_bins = len(bin_end_timestamps)

    # Initialize output array
    binned_counts = np.zeros((n_bins, n_units), dtype=np.float32)

    # Create bin edges
    bin_edges = np.concatenate(
        [np.array([bin_end_timestamps[0] - bin_size_s]), bin_end_timestamps]
    )

    # Bin spikes for each unit
    spike_times = spikes.timestamps[:]
    spike_units = spikes.unit_index[:]

    for unit_idx in range(n_units):
        unit_mask = spike_units == unit_idx
        unit_spike_times = spike_times[unit_mask]
        counts, _ = np.histogram(unit_spike_times, bins=bin_edges)
        binned_counts[:, unit_idx] = counts

    # Bin center timestamps
    bin_timestamps = bin_end_timestamps - (bin_size_s / 2.0)

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
