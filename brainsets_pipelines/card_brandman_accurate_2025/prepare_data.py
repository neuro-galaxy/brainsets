import argparse
import datetime
import logging
import h5py
import os
import re
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.io as sio

from temporaldata import Data, IrregularTimeSeries, Interval, ArrayDict

from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    SubjectDescription,
    DeviceDescription,
)
from brainsets.taxonomy import RecordingTech, Sex, Species, Task
from brainsets import serialize_fn_map

logging.basicConfig(level=logging.INFO)

# 20 ms bins
FREQ = 50


def get_unit_metadata():
    recording_tech = RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS
    unit_meta = []
    for i in range(256):
        unit_id = f"group_0/elec{i:03d}/multiunit_0"
        unit_meta.append(
            {"id": unit_id, "unit_number": i, "count": -1, "type": int(recording_tech)}
        )
    unit_meta_df = pd.DataFrame(unit_meta)
    units = ArrayDict.from_dataframe(unit_meta_df, unsigned_to_long=True)

    return units


def stack_trials(h5_data, test=False):
    """Stack all the trial data into a single array."""
    activity = [x["input_features"] for x in h5_data]
    trial_times = [x.shape[0] / FREQ for x in activity]
    trial_bounds = np.cumsum([0] + trial_times)

    # Create the corresponding intervals for that dataset
    trials = Interval(start=trial_bounds[:-1], end=trial_bounds[1:])

    units = get_unit_metadata()

    timestamps = np.concatenate(
        [np.arange(1/FREQ, 1/FREQ * x.shape[0] + 1e-6, 1/FREQ) + trial_bounds[i] for i, x in enumerate(activity)]
    )

    spikes = IrregularTimeSeries(
        timestamps=np.array(timestamps),
        activity=np.concatenate(activity, axis=0),
        domain=trials,
    )

    if not test:
        transcripts = [x["transcript"] for x in h5_data]
        transcript = ["".join(chr(c) for c in s if c != 0).replace(u"\u2019", "'").encode("ascii", errors="ignore") for s in transcripts]
        # We assign the labels to the start of each trial.
        sentences = IrregularTimeSeries(
            timestamps=trial_bounds[:-1] + 1e-5,
            transcript=np.array(transcript),
            domain="auto",
        )
    else:
        sentences = None

    return spikes, units, trials, sentences


def get_subject():
    return SubjectDescription(
        id="T15",
        species=Species.from_string("HUMAN"),
        sex=Sex.from_string("M"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_dir", type=str, default="./processed")

    args = parser.parse_args()

    brainset_description = BrainsetDescription(
        id="card_brandman_accurate_2025",
        origin_version="1.0.0",
        derived_version="1.0.0",
        source="",
        description="",
    )

    logging.info(f"Processing file: {args.input_file}")

    subject = get_subject()  # Participant pseudonym

    h5_data = []
    with h5py.File(args.input_file, "r") as f:
        for trial in f.keys():
            trial_data = {}
            for key in f[trial].keys():
                trial_data[key] = f[trial][key][:]
            h5_data.append(trial_data)

    session_date = args.input_file.split("/")[-2]
    _, year, month, day = session_date.split(".")
    recording_date = datetime.datetime(int(year), int(month), int(day))
    device_id = str.replace(session_date, ".", "_")
    session_id = device_id

    session_description = SessionDescription(
        id=session_id,
        recording_date=recording_date,
        task=Task.CONTINUOUS_SPEAKING_SENTENCE,
    )

    device_description = DeviceDescription(
        id=device_id,
        recording_tech=RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS,
    )

    spikes, units, trials, sentences = stack_trials(h5_data, test=("test" in args.input_file))

    data = Data(
        brainset=brainset_description,
        subject=subject,
        session=session_description,
        device=device_description,
        spikes=spikes,
        units=units,
        trials=trials,
        speech=sentences,
        domain=spikes.domain,
    )

    if "train" in args.input_file:
        train_split = data.domain
        valid_split = Interval(start=np.array([]), end=np.array([]))
        test_split = Interval(start=np.array([]), end=np.array([]))
    elif "val" in args.input_file:
        train_split = Interval(start=np.array([]), end=np.array([]))
        valid_split = data.domain
        test_split = Interval(start=np.array([]), end=np.array([]))
    else:
        train_split = Interval(start=np.array([]), end=np.array([]))
        valid_split = Interval(start=np.array([]), end=np.array([]))
        test_split = data.domain

    data.set_train_domain(train_split)
    data.set_valid_domain(valid_split)
    data.set_test_domain(test_split)

    path = os.path.join(args.output_dir, f"{session_id}.h5")
    with h5py.File(path, "w") as f:
        data.to_hdf5(f, serialize_fn_map=serialize_fn_map)


if __name__ == "__main__":
    main()
