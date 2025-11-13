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

    # All trials are marked as valid by default
    # (FALCON data is already curated)
    trials.is_valid = np.ones(len(trials), dtype=bool)

    return trials

    # Removed sklearn-based split in favor of Interval.split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./processed")

    args = parser.parse_args()

    brainset_description = BrainsetDescription(
        id="falcon_m1_2024",
        origin_version="dandi/000941/draft",
        derived_version="1.0.0",
        source="https://dandiarchive.org/dandiset/000941",
        description="FALCON M1 dataset: Monkey reach-to-grasp task. "
        "Neural activity recorded from 64-channel Utah array in motor cortex. "
        "Task involves controlled finger movements with 16-channel EMG recordings from upper lim muscles. "
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
        task=Task.REACHING,  # Finger movement is a type of reaching
    )

    # Register device
    device_description = DeviceDescription(
        id=f"MonkeyL_{session_id}",
        recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,
    )

    # Extract neural activity
    spikes, units = extract_spikes_from_nwbfile(
        nwbfile, recording_tech=RecordingTech.UTAH_ARRAY_SPIKES
    )

    # Extract EMG data
    emg = extract_emg_data(nwbfile)

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

    # Set up splits
    valid_trials = trials.select_by_mask(trials.is_valid)

    # Standard splits: random split of trials using Interval.split
    num_trials = len(valid_trials)
    if num_trials < 10:
        logging.warning(
            f"Only {num_trials} trials available. Using all for training, none for validation/test."
        )
        train_trials = valid_trials
        valid_trials_split = valid_trials.select_by_mask(
            np.zeros(num_trials, dtype=bool)
        )
        test_trials = valid_trials.select_by_mask(np.zeros(num_trials, dtype=bool))
    else:
        train_trials, valid_trials_split, test_trials = valid_trials.split(
            [0.7, 0.1, 0.2],
            shuffle=True,
            random_seed=42,
        )

    # Handle split domain setting (accounting for empty intervals)
    if len(valid_trials_split) > 0 or len(test_trials) > 0:
        # Create combined valid+test interval
        if len(valid_trials_split) > 0 and len(test_trials) > 0:
            excluded = (valid_trials_split | test_trials).dilate(1.0)
        elif len(valid_trials_split) > 0:
            excluded = valid_trials_split.dilate(1.0)
        else:
            excluded = test_trials.dilate(1.0)

        train_sampling_intervals = data.domain.difference(excluded)
    else:
        # All data is training
        train_sampling_intervals = data.domain

    data.set_train_domain(train_sampling_intervals)
    data.set_valid_domain(valid_trials_split)
    data.set_test_domain(test_trials)

    # Save data to disk
    path = os.path.join(args.output_dir, f"{session_id}_{split_type}.h5")
    logging.info(f"Saving to: {path}")

    with h5py.File(path, "w") as file:
        data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

    logging.info(f"Successfully processed {session_id}")


if __name__ == "__main__":
    main()
