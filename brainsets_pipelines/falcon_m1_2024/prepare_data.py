import argparse
import datetime
import logging
import h5py
import os

import numpy as np
from pynwb import NWBHDF5IO

from temporaldata import (
    Data,
    IrregularTimeSeries,
    RegularTimeSeries,
    Interval,
)
from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    SubjectDescription,
    DeviceDescription,
)
from brainsets.utils.dandi_utils import extract_spikes_from_nwbfile
from brainsets.taxonomy import RecordingTech, Task, Species, Sex
from brainsets import serialize_fn_map

logging.basicConfig(level=logging.INFO)


# FALCON M1 held-in/held-out split definition
HELD_IN_SESSIONS = ["20120924", "20120926", "20120927", "20120928"]
HELD_OUT_SESSIONS = ["20121004", "20121017", "20121024"]


def parse_session_id_from_filename(filename):
    """Extract session ID from FALCON M1 NWB filename.

    Examples:
    - sub-MonkeyL-held-in-minival_ses-20120924_behavior+ecephys.nwb -> 20120924
    - L_20120924_held_in_eval.nwb -> 20120924
    """
    basename = os.path.basename(filename)

    if "behavior+ecephys" in basename:  # DANDI public format
        # Extract from ses-20120924
        return basename.split("_ses-")[-1].split("_")[0]
    else:  # Evaluation format
        # Extract from L_20120924
        return basename.split("_")[1][:8]


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


def extract_emg_data(nwbfile):
    """Extract 16D EMG from NWB file."""
    emg_container = nwbfile.acquisition["preprocessed_emg"]
    muscles = [ts for ts in emg_container.time_series]  # 16 muscles

    emg_data = []
    emg_timestamps = None
    for muscle in muscles:
        ts_data = emg_container.get_timeseries(muscle)
        emg_data.append(ts_data.data[:])
        emg_timestamps = ts_data.timestamps[:]  # Same for all muscles

    emg_data = np.vstack(emg_data).T  # (time, 16)

    emg = IrregularTimeSeries(
        timestamps=emg_timestamps,
        data=emg_data,
        domain="auto",
    )

    return emg


def extract_trials(nwbfile, emg):
    """Extract trial information from NWB file."""
    trial_table = nwbfile.trials.to_dataframe().reset_index()

    # Rename columns to match temporaldata conventions
    trial_table = trial_table.rename(
        columns={
            "start_time": "start",
            "stop_time": "end",
        }
    )

    # Drop string columns that cause HDF5 serialization issues
    # Keep numeric columns and arrays
    columns_to_drop = []
    for col in trial_table.columns:
        if trial_table[col].dtype == "object":
            # Check if it's a string or numpy array
            first_val = trial_table[col].iloc[0]
            if isinstance(first_val, str):
                columns_to_drop.append(col)

    if columns_to_drop:
        trial_table = trial_table.drop(columns=columns_to_drop)

    trials = Interval.from_dataframe(trial_table)
    trials.is_valid = np.ones(len(trials), dtype=bool)

    return trials


def bin_spikes(spikes, units, bin_size_s, reference_timestamps):
    """Bin spike times into regular time bins aligned with reference timestamps.

    Args:
        spikes: IrregularTimeSeries containing spike times and unit indices
        units: ArrayDict containing unit metadata
        bin_size_s: Bin size in seconds (e.g., 0.02 for 20ms)
        reference_timestamps: Timestamps to align bins with (e.g., from EMG)

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
        id="falcon_m1_2024",
        origin_version="dandi/000941/draft",
        derived_version="1.0.0",
        source="https://dandiarchive.org/dandiset/000941",
        description="FALCON M1 dataset: Monkey reach-to-grasp task. "
        "Neural activity recorded from 64-channel Utah array in motor cortex. "
        "Task involves controlled finger movements with 16-channel EMG recordings from upper limb muscles. "
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
    # FALCON M1 is from Macaque monkey (Monkey L)
    subject = SubjectDescription(
        id="MonkeyL",
        species=Species.MACACA_MULATTA,
        sex=Sex.UNKNOWN,
    )

    # Extract session metadata
    recording_date = datetime.datetime.strptime(session_id, "%Y%m%d")

    session_description = SessionDescription(
        id=session_id,
        recording_date=recording_date,
        task=Task.REACHING,
    )

    # Register device
    device_description = DeviceDescription(
        id=f"MonkeyL_{session_id}",
        recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,
    )

    # Extract EMG data (needed for timestamps)
    emg = extract_emg_data(nwbfile)

    # Extract neural activity (raw spike times first)
    spikes_raw, units = extract_spikes_from_nwbfile(
        nwbfile, recording_tech=RecordingTech.UTAH_ARRAY_SPIKES
    )

    # Conditionally bin spikes based on --bin-size-ms flag
    if args.bin_size_ms is not None:
        bin_size_s = args.bin_size_ms / 1000.0  # Convert ms to seconds
        logging.info(f"Binning spikes at {args.bin_size_ms}ms ({bin_size_s}s)")

        # Bin spikes aligned with EMG timestamps
        binned_counts, bin_timestamps = bin_spikes(
            spikes_raw, units, bin_size_s, emg.timestamps[:]
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

    # Extract evaluation mask (FALCON-specific)
    eval_mask = nwbfile.acquisition["eval_mask"].data[:].astype(bool)

    # Convert eval_mask to Interval (periods where eval_mask is True)
    eval_mask_starts = []
    eval_mask_ends = []
    in_eval_period = False

    for i, (is_eval, timestamp) in enumerate(zip(eval_mask, emg.timestamps)):
        if is_eval and not in_eval_period:
            eval_mask_starts.append(timestamp)
            in_eval_period = True
        elif not is_eval and in_eval_period:
            eval_mask_ends.append(timestamp)
            in_eval_period = False

    # Handle case where eval period extends to end
    if in_eval_period:
        eval_mask_ends.append(emg.timestamps[-1])

    if len(eval_mask_starts) > 0:
        eval_intervals = Interval(
            start=np.array(eval_mask_starts), end=np.array(eval_mask_ends)
        )
    else:
        # Empty interval if no eval periods
        eval_intervals = Interval(start=np.array([]), end=np.array([]))

    # Extract trials
    trials = extract_trials(nwbfile, emg)

    io.close()

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
