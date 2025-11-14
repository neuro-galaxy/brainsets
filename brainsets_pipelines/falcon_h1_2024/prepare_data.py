import argparse
import datetime
import logging
import h5py
import os

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from temporaldata import (
    Data,
    IrregularTimeSeries,
    RegularTimeSeries,
    Interval,
    ArrayDict,
)
from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    SubjectDescription,
    DeviceDescription,
)
from brainsets.taxonomy import RecordingTech, Task, Species, Sex
from brainsets import serialize_fn_map

logging.basicConfig(level=logging.INFO)


# FALCON H1 held-in/held-out split definition
# Based on DANDI 000954 file structure (held-in-calib + held-out-calib)
# Session IDs are timestamp-based (YYYYMMDDTHHMMSS format)
HELD_IN_SESSIONS = [
    "19250101T111740",
    "19250101T112404",
    "19250108T110520",
    "19250108T111022",
    "19250108T111455",
    "19250113T120811",
    "19250113T121303",
    "19250115T110633",
    "19250115T111328",
    "19250119T113543",
    "19250119T114045",
    "19250120T115044",
    "19250120T115537",
]
HELD_OUT_SESSIONS = [
    "19250126T113454",
    "19250126T114029",
    "19250127T120333",
    "19250127T120826",
    "19250129T112555",
    "19250129T113059",
    "19250202T113958",
    "19250202T114452",
    "19250203T113515",
    "19250203T114018",
    "19250206T112219",
    "19250206T112712",
    "19250209T111826",
    "19250209T112327",
]


def parse_session_id_from_filename(filename):
    """Extract session ID from FALCON H1 NWB filename.

    H1 uses timestamp-based session IDs.

    Examples:
    - sub-HumanPitt-held-in-minival_ses-19250101T111740.nwb -> 19250101T111740
    - sub-HumanPitt-held-out-calib_ses-19250203T114018.nwb -> 19250203T114018
    """
    basename = os.path.basename(filename)

    if "behavior+ecephys" in basename or "ses-" in basename:
        # Extract from ses-19250101T111740
        return basename.split("_ses-")[-1].split(".")[0]
    else:
        # Fallback: extract timestamp-like pattern
        # Format: YYYYMMDDTHHMMSS
        parts = basename.split("_")
        for part in parts:
            if "T" in part and len(part) >= 15:
                return part.split(".")[0]
        raise ValueError(f"Could not parse session ID from: {filename}")


def determine_split_type(filename):
    """Determine if file belongs to held-in, held-out, or minival split."""
    basename = os.path.basename(filename)

    if "minival" in basename:
        return "minival"
    elif "held_out" in basename or "held-out" in basename:
        return "held_out"
    elif "held_in" in basename or "held-in" in basename:
        return "held_in"
    else:
        return "unknown"


def extract_kinematics(nwbfile):
    """Extract 7D kinematics from NWB file.

    H1 has 7D output: 3D translation + 4D rotation/gripper.
    Dimensions: [tx, ty, tz, rx, g1, g2, g3]
    """
    # Get velocity data directly (time, 7)
    kin_vel = nwbfile.acquisition["OpenLoopKinematicsVelocity"].data[:]

    # Get timestamps from position data
    kin_pos = nwbfile.acquisition["OpenLoopKinematics"]
    rate = kin_pos.rate  # Sampling rate (e.g., 50 Hz)
    offset = kin_pos.offset  # Time offset
    timestamps = offset + np.arange(kin_vel.shape[0]) / rate

    kinematics = IrregularTimeSeries(
        timestamps=timestamps,
        vel=kin_vel,  # (time, 7)
        domain="auto",
    )

    return kinematics


def extract_trials_from_trialnum(nwbfile, kinematics):
    """Construct trial Intervals from TrialNum array.

    H1 does not have nwb.trials table. Instead, use TrialNum array
    which indicates trial number for each timestep.
    """
    trial_num = nwbfile.acquisition["TrialNum"].data[:]

    # Find trial boundaries (where TrialNum changes)
    trial_changes = np.concatenate([[True], np.diff(trial_num) != 0])
    change_indices = np.where(trial_changes)[0]

    # Create trial start/end times
    trial_starts = kinematics.timestamps[change_indices]
    trial_ends = np.concatenate(
        [kinematics.timestamps[change_indices[1:] - 1], [kinematics.timestamps[-1]]]
    )

    # Create DataFrame for Interval
    trial_data = pd.DataFrame(
        {
            "start": trial_starts,
            "end": trial_ends,
            "trial_num": trial_num[change_indices],
        }
    )

    trials = Interval.from_dataframe(trial_data)
    trials.is_valid = np.ones(len(trials), dtype=bool)

    return trials


def extract_spikes_h1(nwbfile):
    """Extract spike data from H1 NWB file.

    H1 units table doesn't have electrodes linkage like other datasets.
    Extract spikes directly from units table.
    """
    units_df = nwbfile.units.to_dataframe()

    # Extract spike times for all units
    all_spike_times = []
    all_spike_units = []

    for unit_id, row in units_df.iterrows():
        spike_times = row["spike_times"]
        all_spike_times.extend(spike_times)
        all_spike_units.extend([unit_id] * len(spike_times))

    # Sort by time
    sort_idx = np.argsort(all_spike_times)
    all_spike_times = np.array(all_spike_times)[sort_idx]
    all_spike_units = np.array(all_spike_units)[sort_idx]

    spikes = IrregularTimeSeries(
        timestamps=all_spike_times,
        unit_index=all_spike_units,
        domain="auto",
    )

    units = ArrayDict(
        id=np.array(units_df.index),
    )

    return spikes, units


def extract_eval_mask(nwbfile, kinematics):
    """Extract evaluation mask from NWB file.

    CRITICAL: H1 eval_mask has INVERTED semantics compared to M1/M2!
    - M1/M2: eval_mask True = include in evaluation
    - H1: eval_mask True = EXCLUDE from evaluation

    We invert the mask to maintain consistency across datasets.
    """
    eval_mask_raw = nwbfile.acquisition["eval_mask"].data[:].astype(bool)
    eval_mask = ~eval_mask_raw  # INVERT for consistency!

    # Convert eval_mask to Interval (periods where eval_mask is True)
    eval_mask_starts = []
    eval_mask_ends = []
    in_eval_period = False

    for i, (is_eval, timestamp) in enumerate(zip(eval_mask, kinematics.timestamps)):
        if is_eval and not in_eval_period:
            eval_mask_starts.append(timestamp)
            in_eval_period = True
        elif not is_eval and in_eval_period:
            eval_mask_ends.append(timestamp)
            in_eval_period = False

    # Handle case where eval period extends to end
    if in_eval_period:
        eval_mask_ends.append(kinematics.timestamps[-1])

    if len(eval_mask_starts) > 0:
        eval_intervals = Interval(
            start=np.array(eval_mask_starts), end=np.array(eval_mask_ends)
        )
    else:
        # Empty interval if no eval periods
        eval_intervals = Interval(start=np.array([]), end=np.array([]))

    return eval_intervals


def bin_spikes(spikes, units, bin_size_s, reference_timestamps):
    """Bin spike times into regular time bins aligned with reference timestamps.

    Args:
        spikes: IrregularTimeSeries containing spike times and unit indices
        units: ArrayDict containing unit metadata
        bin_size_s: Bin size in seconds (e.g., 0.02 for 20ms)
        reference_timestamps: Timestamps to align bins with (e.g., from kinematics)

    Returns:
        binned_counts: (n_bins, n_units) array of spike counts per bin
        bin_timestamps: (n_bins,) array of bin center timestamps
    """
    n_units = len(units.id)

    # Create bin edges aligned with reference timestamps
    # Use reference timestamps as bin end times (like FALCON does)
    bin_end_timestamps = reference_timestamps
    n_bins = len(bin_end_timestamps)

    # Initialize output array
    binned_counts = np.zeros((n_bins, n_units), dtype=np.float32)

    # Create bin edges (one extra edge at the start)
    bin_edges = np.concatenate(
        [np.array([bin_end_timestamps[0] - bin_size_s]), bin_end_timestamps]
    )

    # Bin spikes for each unit
    spike_times = spikes.timestamps[:]
    spike_units = spikes.unit_index[:]

    for unit_idx in range(n_units):
        # Get spike times for this unit
        unit_mask = spike_units == unit_idx
        unit_spike_times = spike_times[unit_mask]

        # Histogram spike times into bins
        counts, _ = np.histogram(unit_spike_times, bins=bin_edges)
        binned_counts[:, unit_idx] = counts

    # Bin center timestamps (for RegularTimeSeries domain_start)
    bin_timestamps = bin_end_timestamps - (bin_size_s / 2.0)

    return binned_counts, bin_timestamps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./processed")
    parser.add_argument(
        "--bin-size-ms",
        type=float,
        default=None,
        help="Bin size in milliseconds. If provided, spikes will be binned into "
        "RegularTimeSeries. If None (default), spike times are preserved as "
        "IrregularTimeSeries. FALCON benchmark uses 20ms bins.",
    )

    args = parser.parse_args()

    brainset_description = BrainsetDescription(
        id="falcon_h1_2024",
        origin_version="dandi/000954/draft",
        derived_version="1.0.0",
        source="https://dandiarchive.org/dandiset/000954",
        description="FALCON H1 dataset: Human 7-DoF arm control task. "
        "Neural activity recorded from 176-channel intracortical brain-computer interface. "
        "Task involves 3D translation and rotation movements with gripper control. "
        "Part of the FALCON (Few-shot Algorithms for Consistent Neural Decoding) Benchmark.",
    )

    logging.info(f"Processing file: {args.input_file}")

    # Parse session information from filename
    session_id = parse_session_id_from_filename(args.input_file)
    split_type = determine_split_type(args.input_file)

    logging.info(f"Session ID: {session_id}, Split: {split_type}")

    # Open NWB file
    io = NWBHDF5IO(args.input_file, "r")
    nwbfile = io.read()

    # Extract subject metadata
    # FALCON H1 is from human participant
    subject = SubjectDescription(
        id="HumanPitt",
        species=Species.HOMO_SAPIENS,
        sex=Sex.UNKNOWN,
    )

    # Extract session metadata
    # Parse date from timestamp format (YYYYMMDDTHHMMSS)
    recording_date = datetime.datetime.strptime(session_id[:8], "%Y%m%d")

    session_description = SessionDescription(
        id=session_id,
        recording_date=recording_date,
        task=Task.REACHING,  # 7-DoF arm control is a type of reaching
    )

    # Register device
    device_description = DeviceDescription(
        id=f"HumanPitt_{session_id}",
        recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,
    )

    # Extract 7D kinematics (needed for timestamps)
    kinematics = extract_kinematics(nwbfile)

    # Extract neural activity (raw spike times first)
    spikes_raw, units = extract_spikes_h1(nwbfile)

    # Conditionally bin spikes based on --bin-size-ms flag
    if args.bin_size_ms is not None:
        bin_size_s = args.bin_size_ms / 1000.0  # Convert ms to seconds
        logging.info(f"Binning spikes at {args.bin_size_ms}ms ({bin_size_s}s)")

        # Bin spikes aligned with kinematics timestamps
        binned_counts, bin_timestamps = bin_spikes(
            spikes_raw, units, bin_size_s, kinematics.timestamps
        )

        logging.info(
            f"Binned spikes: {binned_counts.shape} "
            f"({binned_counts.shape[0]} bins × {binned_counts.shape[1]} units)"
        )

        # Create RegularTimeSeries for binned spikes
        spikes = RegularTimeSeries(
            counts=binned_counts,  # (n_bins, n_units)
            sampling_rate=1.0 / bin_size_s,  # Hz
            domain="auto",
            domain_start=bin_timestamps[0],
        )
    else:
        # Default: preserve spike times as IrregularTimeSeries
        logging.info("Preserving spike times (no binning)")
        spikes = spikes_raw

    # Extract trials (from TrialNum array, not trials table!)
    trials = extract_trials_from_trialnum(nwbfile, kinematics)

    # Extract evaluation mask (INVERTED semantics!)
    eval_intervals = extract_eval_mask(nwbfile, kinematics)

    io.close()

    # Create Data object
    data = Data(
        brainset=brainset_description,
        subject=subject,
        session=session_description,
        device=device_description,
        spikes=spikes,
        units=units,
        kinematics=kinematics,
        trials=trials,
        eval_intervals=eval_intervals,
        domain=kinematics.domain,
    )

    # Add metadata about FALCON split type
    data.falcon_split = split_type
    data.falcon_session_group = (
        "held_in" if session_id in HELD_IN_SESSIONS else "held_out"
    )

    # Add metadata about binning
    if args.bin_size_ms is not None:
        data.spike_bin_size_ms = args.bin_size_ms
        data.spike_format = "binned"
    else:
        data.spike_format = "spike_times"

    # Set up splits based on FALCON's pre-defined file-level splits
    # CRITICAL: Use FALCON's split_type to assign data appropriately
    # - held_in_calib  → train
    # - held_out_calib → valid (minival too small, use held-out instead)
    # - minival        → ignore (too small for meaningful validation)

    valid_trials = trials.select_by_mask(trials.is_valid)

    # Map file type to appropriate split
    if "held_in" in split_type or "held-in" in split_type:
        # Held-in calibration data → training set
        data.set_train_domain(valid_trials)
        data.set_valid_domain(Interval(start=np.array([]), end=np.array([])))
        data.set_test_domain(Interval(start=np.array([]), end=np.array([])))
        logging.info(f"Assigned {len(valid_trials)} trials to TRAIN (held-in-calib)")
    elif "held_out" in split_type or "held-out" in split_type:
        # Held-out calibration data → validation set
        data.set_train_domain(Interval(start=np.array([]), end=np.array([])))
        data.set_valid_domain(valid_trials)
        data.set_test_domain(Interval(start=np.array([]), end=np.array([])))
        logging.info(f"Assigned {len(valid_trials)} trials to VALID (held-out-calib)")
    elif split_type == "minival":
        # Minival data → test set (too small for validation)
        data.set_train_domain(Interval(start=np.array([]), end=np.array([])))
        data.set_valid_domain(Interval(start=np.array([]), end=np.array([])))
        data.set_test_domain(valid_trials)
        logging.info(f"Assigned {len(valid_trials)} trials to TEST (minival - small)")
    else:
        logging.warning(f"Unknown split_type: {split_type}, defaulting to train")
        data.set_train_domain(valid_trials)
        data.set_valid_domain(Interval(start=np.array([]), end=np.array([])))
        data.set_test_domain(Interval(start=np.array([]), end=np.array([])))

    # Generate filename with bin size info if binned
    if args.bin_size_ms is not None:
        filename = f"{session_id}_{split_type}_bin{int(args.bin_size_ms)}ms.h5"
    else:
        filename = f"{session_id}_{split_type}.h5"

    path = os.path.join(args.output_dir, filename)
    logging.info(f"Saving to: {path}")

    with h5py.File(path, "w") as file:
        data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

    logging.info(f"Successfully processed {session_id}")


if __name__ == "__main__":
    main()
