import argparse
import datetime
import logging
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

from brainsets import serialize_fn_map
from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    SubjectDescription,
    DeviceDescription,
)

from brainsets.taxonomy import (
    RecordingTech,
    Species,
    Task,
)
from temporaldata import (
    ArrayDict,
    Data,
    Interval,
    IrregularTimeSeries,
)

logging.basicConfig(level=logging.INFO)


def extract_behavior(h5file):
    """Extract the behavior from the h5 file.

    ..note::
        Cursor position and target position are in the same frame of reference.
        They are both of size (sequence_len, 2). Finger position can be either 3d or 6d,
        depending on the sequence.
    """

    cursor_pos = h5file["cursor_pos"][:].T
    finger_pos = h5file["finger_pos"][:].T
    target_pos = h5file["target_pos"][:].T
    timestamps = h5file["t"][:][0]

    expected_period = 0.001
    assert np.all(np.abs(np.diff(timestamps) - expected_period) < 1e-4)

    # calculate the velocity of the cursor
    cursor_vel = np.gradient(cursor_pos, timestamps, edge_order=1, axis=0)
    cursor_acc = np.gradient(cursor_vel, timestamps, edge_order=1, axis=0)
    finger_vel = np.gradient(finger_pos, timestamps, edge_order=1, axis=0)

    cursor = IrregularTimeSeries(
        timestamps=timestamps,
        pos=cursor_pos,
        vel=cursor_vel,
        acc=cursor_acc,
        domain="auto",
    )

    # The position of the working fingertip in Cartesian coordinates (z, -x, -y), as
    # reported by the hand tracker in cm. Thus the cursor position is an affine
    # transformation of fingertip position.
    finger = IrregularTimeSeries(
        timestamps=timestamps,
        pos_3d=finger_pos[:, :3],
        vel_3d=finger_vel[:, :3],
        domain="auto",
    )
    if finger_pos.shape[1] == 6:
        finger.orientation = finger_pos[:, 3:]
        finger.angular_vel = finger_vel[:, 3:]

    assert cursor.is_sorted()
    assert finger.is_sorted()

    return cursor, finger


def detect_outliers(cursor):
    """
    Helper to identify outliers in the behavior data.
    Outliers are defined as points where the hand acceleration is greater than a
    threshold. This is a simple heuristic to identify when the monkey is moving
    the hand quickly when angry or frustrated.
    An additional step is to dilate the binary mask to flag the surrounding points.
    """
    hand_acc_norm = np.linalg.norm(cursor.acc, axis=1)
    mask = hand_acc_norm > 6000.0
    structure = np.ones(100, dtype=bool)
    # Dilate the binary mask
    outlier_mask = binary_dilation(mask, structure)

    # convert to interval, you need to find the start and end of the outlier segments
    start = cursor.timestamps[np.where(np.diff(outlier_mask.astype(int)) == 1)[0]]
    if outlier_mask[0]:
        start = np.insert(start, 0, cursor.timestamps[0])

    end = cursor.timestamps[np.where(np.diff(outlier_mask.astype(int)) == -1)[0]]
    if outlier_mask[-1]:
        end = np.append(end, cursor.timestamps[-1])

    cursor_outlier_segments = Interval(start=start, end=end)
    assert cursor_outlier_segments.is_disjoint()

    return cursor_outlier_segments


def extract_spikes(h5file: h5py.File):
    r"""This dataset has a mixture of sorted and unsorted (threshold crossings)
    units.
    """

    # helpers specific to spike extraction
    def _to_ascii(vector):
        return ["".join(chr(char) for char in row) for row in vector]

    def _load_references_2d(h5file, ref_name):
        return [[h5file[ref] for ref in ref_row] for ref_row in h5file[ref_name][:]]

    spikesvec = _load_references_2d(h5file, "spikes")
    waveformsvec = _load_references_2d(h5file, "wf")

    # this is slightly silly but we can convert channel names back to an ascii token
    # this way.
    chan_names = _to_ascii(
        np.array(_load_references_2d(h5file, "chan_names")).squeeze()
    )

    spikes = []
    unit_index = []
    types = []
    waveforms = []
    unit_meta = []

    # The 0'th spikesvec corresponds to unsorted thresholded units, the rest are sorted.
    suffixes = ["unsorted"] + [f"sorted_{i:02}" for i in range(1, 11)]
    type_map = [int(RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS)] + [
        int(RecordingTech.UTAH_ARRAY_SPIKES)
    ] * 10

    encountered = set()

    unit_index_delta = 0
    for j in range(len(spikesvec)):
        crossings = spikesvec[j]
        for i in range(len(crossings)):
            spiketimes = crossings[i][:][0]
            if spiketimes.ndim == 0:
                continue

            spikes.append(spiketimes)
            area, channel_number = chan_names[i].split(" ")

            unit_name = f"{chan_names[i]}/{suffixes[j]}"

            unit_index.append([unit_index_delta] * len(spiketimes))
            types.append(np.ones_like(spiketimes, dtype=np.int64) * type_map[j])

            if unit_name in encountered:
                raise ValueError(f"Duplicate unit name: {unit_name}")
            encountered.add(unit_name)

            wf = np.array(waveformsvec[j][i][:])
            unit_meta.append(
                {
                    "count": len(spiketimes),
                    "channel_name": chan_names[i],
                    "id": unit_name,
                    "area_name": area,
                    "channel_number": channel_number,
                    "unit_number": j,
                    "type": type_map[j],
                    "average_waveform": wf.mean(axis=1)[:48],
                }
            )
            waveforms.append(wf.T)
            unit_index_delta += 1

    spikes = np.concatenate(spikes)
    waveforms = np.concatenate(waveforms)
    unit_index = np.concatenate(unit_index)

    spikes = IrregularTimeSeries(
        timestamps=spikes,
        unit_index=unit_index,
        waveforms=waveforms,
        domain="auto",
    )
    spikes.sort()

    units = ArrayDict.from_dataframe(pd.DataFrame(unit_meta))
    return spikes, units


def main():
    # use argparse to get arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_dir", type=str, default="./processed")

    args = parser.parse_args()

    brainset_description = BrainsetDescription(
        id="odoherty_sabes_nonhuman_2017",
        origin_version="583331",  # Zenodo version
        derived_version="1.0.0",
        source="https://zenodo.org/record/583331",
        description="The behavioral task was to make self-paced reaches to targets "
        "arranged in a grid (e.g. 8x8) without gaps or pre-movement delay intervals. "
        "One monkey reached with the right arm (recordings made in the left hemisphere)"
        "The other reached with the left arm (right hemisphere). In some sessions "
        "recordings were made from both M1 and S1 arrays (192 channels); "
        "in most sessions M1 recordings were made alone (96 channels).",
    )

    logging.info(f"Processing file: {args.input_file}")

    # open file
    h5file = h5py.File(args.input_file, "r")

    # extract experiment metadata
    # determine session_id and sortset_id
    session_id = Path(args.input_file).stem  # type: ignore
    device_id = session_id[:-3]
    assert device_id.count("_") == 1, f"Unexpected file name: {device_id}"

    animal, recording_date = device_id.split("_")
    subject = SubjectDescription(
        id=animal,
        species=Species.MACACA_MULATTA,
    )

    session_description = SessionDescription(
        id=session_id,
        recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
        task=Task.REACHING,
    )

    device_description = DeviceDescription(
        id=device_id,
        recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,
    )

    # extract spiking activity, unit metadata and channel names info
    spikes, units = extract_spikes(h5file)

    # extract behavior
    cursor, finger = extract_behavior(h5file)
    cursor_outlier_segments = detect_outliers(cursor)

    # close file
    h5file.close()

    # register session
    data = Data(
        brainset=brainset_description,
        subject=subject,
        session=session_description,
        device=device_description,
        # neural activity
        spikes=spikes,
        units=units,
        # stimuli and behavior
        cursor=cursor,
        finger=finger,
        cursor_outlier_segments=cursor_outlier_segments,
        domain=cursor.domain,
    )

    # slice the session into 10 blocks then randomly split them into train,
    # validation and test sets, using a 8/1/1 ratio.
    intervals = Interval.linspace(data.domain.start[0], data.domain.end[-1], 10)
    [
        train_sampling_intervals,
        valid_sampling_intervals,
        test_sampling_intervals,
    ] = intervals.split([8, 1, 1], shuffle=True, random_seed=42)

    data.set_train_domain(train_sampling_intervals)
    data.set_valid_domain(valid_sampling_intervals)
    data.set_test_domain(test_sampling_intervals)

    # save data to disk
    path = os.path.join(args.output_dir, f"{session_id}.h5")
    with h5py.File(path, "w") as file:
        data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


if __name__ == "__main__":
    main()
