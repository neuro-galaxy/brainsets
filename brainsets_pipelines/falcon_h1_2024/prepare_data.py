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

    # All trials are marked as valid by default
    # (FALCON data is already curated)
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

    # Create IrregularTimeSeries for spikes
    spikes = IrregularTimeSeries(
        timestamps=all_spike_times,
        unit_index=all_spike_units,
        domain="auto",
    )

    # Create units ArrayDict
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./processed")

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

    # Extract neural activity
    # H1 has no electrodes table, must extract spikes manually
    spikes, units = extract_spikes_h1(nwbfile)

    # Extract 7D kinematics
    kinematics = extract_kinematics(nwbfile)

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

    # Set up splits
    valid_trials = trials.select_by_mask(trials.is_valid)

    # Standard splits: random split of trials using Interval.split()
    # Handle case where there are very few trials
    num_trials = len(valid_trials)

    if num_trials < 10:
        logging.warning(
            f"Only {num_trials} trials available. "
            f"Using all for training, none for validation/test."
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
