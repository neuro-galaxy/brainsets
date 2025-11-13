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
    ArrayDict,
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


def parse_session_id_from_filename(filename):
    """Extract session ID from FALCON M2 NWB filename.

    Examples:
    - sub-MonkeyN-held-in-calib_ses-2020-10-19-Run1_behavior+ecephys.nwb -> Run1_20201019
    - sub-MonkeyNRun1_20201019_held_in_eval.nwb -> Run1_20201019
    """
    basename = os.path.basename(filename)

    if "behavior+ecephys" in basename:  # DANDI public format
        # Extract from ses-2020-10-19-Run1
        ses_part = basename.split("_ses-")[-1].split("_")[0]  # 2020-10-19-Run1
        parts = ses_part.split("-")
        run_str = parts[-1]  # Run1
        date_str = "".join(parts[:-1])  # 20201019
        session_id = f"{run_str}_{date_str}"
    else:  # Evaluation format
        # Extract from sub-MonkeyNRun1_20201019
        parts = basename.split("_")
        run_str = parts[0].split("MonkeyN")[-1]  # Run1
        date_str = parts[1][:8]  # 20201019
        session_id = f"{run_str}_{date_str}"

    return session_id


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


def extract_finger_velocity(nwbfile):
    """Extract 2D finger velocity from NWB file."""
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
        vel=vel_data,
        domain="auto",
    )

    return finger


def extract_trials(nwbfile, finger):
    """Extract trial information from NWB file."""
    trial_table = nwbfile.trials.to_dataframe().reset_index()

    # Rename columns to match temporaldata conventions
    trial_table = trial_table.rename(
        columns={
            "start_time": "start",
            "stop_time": "end",
        }
    )

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
        id="falcon_m2_2024",
        origin_version="dandi/000953/draft",
        derived_version="1.0.0",
        source="https://dandiarchive.org/dandiset/000953",
        description="FALCON M2 dataset: Monkey 2D finger velocity task. "
        "Neural activity recorded from 96-channel Utah array in motor cortex. "
        "Task involves 2D finger velocity tracking. "
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
    # FALCON M2 is from Macaque monkey (Monkey N)
    subject = SubjectDescription(
        id="MonkeyN",
        species=Species.MACACA_MULATTA,
        sex=Sex.UNKNOWN,
    )

    # Extract session metadata
    recording_date_str = session_id.split("_")[1]  # e.g., 20201019
    recording_date = datetime.datetime.strptime(recording_date_str, "%Y%m%d")

    session_description = SessionDescription(
        id=session_id,
        recording_date=recording_date,
        task=Task.REACHING,
    )

    # Register device
    device_description = DeviceDescription(
        id=f"MonkeyN_{recording_date_str}",
        recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,
    )

    # Extract neural activity
    spikes, units = extract_spikes_from_nwbfile(
        nwbfile, recording_tech=RecordingTech.UTAH_ARRAY_SPIKES
    )

    # Extract finger velocity
    finger = extract_finger_velocity(nwbfile)

    # Extract evaluation mask (FALCON-specific)
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
        # Empty interval if no eval periods
        eval_intervals = Interval(start=np.array([]), end=np.array([]))

    # Extract trials
    trials = extract_trials(nwbfile, finger)

    # Close NWB file
    io.close()

    # Create Data object
    data = Data(
        brainset=brainset_description,
        subject=subject,
        session=session_description,
        device=device_description,
        # Neural activity
        spikes=spikes,
        units=units,
        # Behavior
        finger=finger,
        trials=trials,
        # FALCON-specific: evaluation mask
        eval_intervals=eval_intervals,
        # Domain
        domain=finger.domain,
    )

    # Add metadata about FALCON split type
    data.falcon_split = split_type
    data.falcon_session_group = (
        "held_in" if session_id in HELD_IN_SESSIONS else "held_out"
    )

    # Set up splits
    # We create two split configurations:
    # 1. Standard splits: random train/valid/test
    # 2. FALCON splits: preserve held-in/held-out structure

    valid_trials = trials.select_by_mask(trials.is_valid)

    # Standard splits: random split of trials using Interval.split
    num_trials = len(valid_trials)
    if num_trials < 10:
        logging.warning(
            f"Only {num_trials} trials available. Using all for training, none for validation/test."
        )
        train_trials = valid_trials
        valid_trials_split = valid_trials.select_by_mask(np.zeros(num_trials, dtype=bool))
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

    path = os.path.join(args.output_dir, f"{session_id}_{split_type}.h5")
    logging.info(f"Saving to: {path}")

    with h5py.File(path, "w") as file:
        data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

    logging.info(f"Successfully processed {session_id}")


if __name__ == "__main__":
    main()
