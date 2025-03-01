import argparse
import datetime
import logging
import h5py
import os

import numpy as np
from pynwb import NWBHDF5IO
from scipy.ndimage import binary_dilation

from temporaldata import (
    Data,
    RegularTimeSeries,
    IrregularTimeSeries,
    Interval,
    ArrayDict,
)
from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    DeviceDescription,
)
from brainsets.taxonomy import RecordingTech, Task
from brainsets.utils.dandi_utils import (
    extract_spikes_from_nwbfile,
    extract_subject_from_nwb,
)
from brainsets import serialize_fn_map

logging.basicConfig(level=logging.INFO)


def extract_trials(nwbfile):
    r"""Extract trial information from the NWB file. Trials that are flagged as
    "to discard" or where the monkey failed are marked as invalid."""
    trial_table = nwbfile.trials.to_dataframe()

    # rename start and end time columns
    trial_table = trial_table.rename(
        columns={
            "start_time": "start",
            "stop_time": "end",
            "target_presentation_time": "target_on_time",
        }
    )

    # some sessions seem to have an incorrect trial table
    # TODO investigate
    index = np.where(
        (trial_table.start.to_numpy()[1:] - trial_table.start.to_numpy()[:-1]) < 0
    )[0]
    if len(index) > 0:
        logging.warning(
            f"Found {len(index) + 1} non contiguous blocks of trials in the "
            f"trial table. Truncating the table to the first contiguous block."
        )
        trial_table = trial_table.iloc[: index[0] + 1]

    trials = Interval.from_dataframe(trial_table)
    is_valid = np.logical_and(trials.discard_trial == 0.0, trials.task_success == 1.0)

    # for some reason some trials are overlapping, we will flag them as invalid
    overlapping_mask = np.zeros_like(is_valid, dtype=bool)
    overlapping_mask[1:] = trials.start[1:] < trials.end[:-1]
    overlapping_mask[:-1] = np.logical_or(overlapping_mask[:-1], overlapping_mask[1:])
    if np.any(overlapping_mask):
        logging.warning(
            f"Found {np.sum(overlapping_mask)} overlapping trials. "
            f"Marking them as invalid."
        )
    is_valid[overlapping_mask] = False

    trials.is_valid = is_valid

    valid_trials = trials.select_by_mask(trials.is_valid)

    movement_phases = Data(
        hold_period=Interval(
            start=valid_trials.target_on_time, end=valid_trials.go_cue_time
        ),
        reach_period=Interval(
            start=valid_trials.move_begins_time, end=valid_trials.move_ends_time
        ),
        return_period=Interval(start=valid_trials.move_ends_time, end=valid_trials.end),
        domain="auto",
    )

    return trials, movement_phases


def extract_behavior(nwbfile):
    """Extract behavior from the NWB file.

    ..note::
        Cursor position and target position are in the same frame of reference.
        They are both of size (sequence_len, 2). Finger position can be either 3d or 6d,
        depending on the sequence. # todo investigate more
    """
    # cursor, hand and eye share the same timestamps (verified)
    # TODO what's the difference between hand and cursor?
    timestamps = nwbfile.processing["behavior"]["Position"]["Cursor"].timestamps[:]
    cursor_pos = nwbfile.processing["behavior"]["Position"]["Cursor"].data[:]  # 2d
    hand_pos = nwbfile.processing["behavior"]["Position"]["Hand"].data[:]
    eye_pos = nwbfile.processing["behavior"]["Position"]["Eye"].data[:]  # 2d

    # derive the velocity of the cursor
    cursor_vel = np.gradient(cursor_pos, timestamps, edge_order=1, axis=0)
    # derive the velocity and acceleration of the hand
    hand_vel = np.gradient(hand_pos, timestamps, edge_order=1, axis=0)
    hand_acc = np.gradient(hand_vel, timestamps, edge_order=1, axis=0)

    cursor = IrregularTimeSeries(
        timestamps=timestamps,
        pos=cursor_pos,
        vel=cursor_vel,
        domain="auto",
    )

    hand = IrregularTimeSeries(
        timestamps=timestamps,
        pos_2d=hand_pos,
        vel_2d=hand_vel,
        acc_2d=hand_acc,
        domain="auto",
    )

    eye = IrregularTimeSeries(
        timestamps=timestamps,
        pos=eye_pos,
        domain="auto",
    )

    return cursor, hand, eye


def detect_outliers(hand):
    # sometimes monkeys get angry, we want to identify the segments where the hand is
    # moving too fast, and mark them as outliers
    # we use the norm of the acceleration to identify outliers
    hand_acc_norm = np.linalg.norm(hand.acc_2d, axis=1)
    mask = hand_acc_norm > 100.0
    # we dilate the mask to make sure we are not missing any outliers
    structure = np.ones(50, dtype=bool)
    mask = binary_dilation(mask, structure)

    # convert to interval, you need to find the start and end of the outlier segments
    start = hand.timestamps[np.where(np.diff(mask.astype(int)) == 1)[0]]
    if mask[0]:
        start = np.insert(start, 0, hand.timestamps[0])

    end = hand.timestamps[np.where(np.diff(mask.astype(int)) == -1)[0]]
    if mask[-1]:
        end = np.insert(end, 0, hand.timestamps[-1])

    hand_outlier_segments = Interval(start=start, end=end)

    return hand_outlier_segments


def main():
    # use argparse to get arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_dir", type=str, default="./processed")

    args = parser.parse_args()

    # intiantiate a DatasetBuilder which provides utilities for processing data
    brainset_description = BrainsetDescription(
        id="churchland_shenoy_neural_2012",
        origin_version="dandi/000070/draft",
        derived_version="1.0.0",
        source="https://dandiarchive.org/dandiset/000070",
        description="Monkeys recordings of Motor Cortex (M1) and dorsal Premotor Cortex"
        " (PMd) using two 96 channel high density Utah Arrays (Blackrock Microsystems) "
        "while performing reaching tasks with right hand.",
    )

    logging.info(f"Processing file: {args.input_file}")

    # open file
    io = NWBHDF5IO(args.input_file, "r")
    nwbfile = io.read()

    # extract subject metadata
    # this dataset is from dandi, which has structured subject metadata, so we
    # can use the helper function extract_subject_from_nwb
    subject = extract_subject_from_nwb(nwbfile)

    # extract experiment metadata
    recording_date = nwbfile.session_start_time.strftime("%Y%m%d")
    subject_id = subject.id
    device_id = f"{subject_id}_{recording_date}"
    session_id = f"{device_id}_center_out_reaching"

    # register session
    session_description = SessionDescription(
        id=session_id,
        recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
        task=Task.REACHING,
    )

    device_description = DeviceDescription(
        id=device_id,
        recording_tech=RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS,
    )

    # extract spiking activity
    # this data is from dandi, we can use our helper function
    spikes, units = extract_spikes_from_nwbfile(
        nwbfile, recording_tech=RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS
    )

    # extract data about trial structure
    trials, movement_phases = extract_trials(nwbfile)

    # extract behavior
    cursor, hand, eye = extract_behavior(nwbfile)
    hand_outlier_segments = detect_outliers(hand)
    # close file
    io.close()

    data = Data(
        brainset=brainset_description,
        subject=subject,
        session=session_description,
        device=device_description,
        # neural activity
        spikes=spikes,
        units=units,
        # stimuli and behavior
        trials=trials,
        movement_phases=movement_phases,
        cursor=cursor,
        hand=hand,
        eye=eye,
        hand_outlier_segments=hand_outlier_segments,
        domain=hand.domain,
    )

    # split trials into train, validation and test
    successful_trials = trials.select_by_mask(trials.is_valid)
    assert successful_trials.is_disjoint()

    _, valid_trials, test_trials = successful_trials.split(
        [0.7, 0.1, 0.2], shuffle=True, random_seed=42
    )

    train_sampling_intervals = data.domain.difference(
        (valid_trials | test_trials).dilate(3.0)
    )

    data.set_train_domain(train_sampling_intervals)
    data.set_valid_domain(valid_trials)
    data.set_test_domain(test_trials)

    # save data to disk
    path = os.path.join(args.output_dir, f"{session_id}.h5")
    with h5py.File(path, "w") as file:
        data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


if __name__ == "__main__":
    main()
