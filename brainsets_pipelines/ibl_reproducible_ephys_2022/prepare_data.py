import argparse
import datetime
import logging
import h5py
import os

import numpy as np
import pandas as pd
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader, SessionLoader

from temporaldata import Data, IrregularTimeSeries, Interval, ArrayDict
from brainsets.descriptions import (
    BrainsetDescription,
    SubjectDescription,
    SessionDescription,
    DeviceDescription,
)

from brainsets.taxonomy import RecordingTech, Sex, Species
from brainsets import serialize_fn_map

logging.basicConfig(level=logging.INFO)


def extract_spikes(one, eid):
    # get probes
    pids, probe_names = one.eid2pid(eid)

    spikes_list = []
    units_list = []
    unit_ptr = 0

    # iterate over probes and load spikes and clusters
    for pid, probe_name in zip(pids, probe_names):
        spike_loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=probe_name)
        spikes, clusters, channels = spike_loader.load_spike_sorting()

        clusters = SpikeSortingLoader.merge_clusters(
            spikes, clusters, channels, compute_metrics=False
        )

        clusters["pid"] = pid

        spikes["clusters"] += unit_ptr

        assert len(clusters["cluster_id"]) == len(
            np.unique(clusters["cluster_id"])
        ), f"There are duplicate units in {eid}"

        assert len(clusters["cluster_id"]) == len(
            np.unique(spikes["clusters"])
        ), f"There are units that have no spikes in {eid}"

        num_units = len(clusters["cluster_id"])
        unit_ptr += num_units

        spikes_list.append(spikes)
        units = pd.DataFrame(clusters).rename(columns={"uuids": "id"})
        units_list.append(units)

    spikes = IrregularTimeSeries(
        timestamps=np.concatenate([s["times"] for s in spikes_list]),
        unit_index=np.concatenate([s["clusters"] for s in spikes_list]),
        amplitude=np.concatenate([s["amps"] for s in spikes_list]),
        depth=np.concatenate([s["depths"] for s in spikes_list]),
        domain="auto",
    )
    spikes.sort()

    units = ArrayDict.from_dataframe(
        pd.concat(units_list, ignore_index=True),
        unsigned_to_long=True,
    )

    # check that id is unique
    assert len(units.id) == len(np.unique(units.id)), "uuids is not unique"

    return spikes, units


def extract_wheel(session_loader):
    session_loader.load_wheel()

    wheel = IrregularTimeSeries(
        timestamps=session_loader.wheel["times"].to_numpy(),
        pos=session_loader.wheel["position"].to_numpy()[:, None],
        vel=session_loader.wheel["velocity"].to_numpy()[:, None],
        domain="auto",
    )

    # get wheel speed
    wheel.speed = np.abs(wheel.vel)
    return wheel


def extract_whisker_motion_energy(session_loader):
    session_loader.load_motion_energy(views=["right"])

    whisker = IrregularTimeSeries(
        timestamps=session_loader.motion_energy["rightCamera"]["times"].to_numpy(),
        motion_energy=session_loader.motion_energy["rightCamera"][
            "whiskerMotionEnergy"
        ].to_numpy()[:, None],
        domain="auto",
    )
    return whisker


def load_trials(
    session_loader,
    min_rt=0.0,
    max_rt=10.0,
    nan_exclude="default",
    min_trial_len=None,
    max_trial_len=10,
    exclude_unbiased=False,
    exclude_nochoice=True,
):
    session_loader.load_trials()
    trials = session_loader.trials

    trials = trials.rename(columns={"intervals_0": "start", "intervals_1": "end"})
    trials = Interval.from_dataframe(
        trials,
        timekeys=[
            "start",
            "end",
            "stimOn_times",
            "feedback_times",
            "goCueTrigger_times",
            "stimOff_times",
            "response_times",
            "firstMovement_times",
            "goCue_times",
        ],
    )

    # choice is -1, 0 or 1, we will map it to 0, 2 and 1 respectively
    trials.choice = np.array([0, 2, 1], dtype=np.int64)[(trials.choice + 1).astype(int)]

    # block is 0.2, 0.5 or 0.8, we will map it to 0, 1 and 2 respectively
    def map_block(data):
        block_map = {"0.2": 0, "0.5": 1, "0.8": 2}
        return block_map[str(data)]

    trials.block = np.array([map_block(p) for p in trials.probabilityLeft])
    trials.timestamps = np.ones_like(trials.start)

    # filter trials
    if nan_exclude == "default":
        nan_exclude = [
            "stimOn_times",
            "choice",
            "feedback_times",
            "probabilityLeft",
            "firstMovement_times",
            "feedbackType",
        ]

    query = ""
    if min_rt is not None:
        query += f" | (firstMovement_times - stimOn_times < {min_rt})"
    if max_rt is not None:
        query += f" | (firstMovement_times - stimOn_times > {max_rt})"
    if min_trial_len is not None:
        query += f" | (feedback_times - goCue_times < {min_trial_len})"
    if max_trial_len is not None:
        query += f" | (feedback_times - goCue_times > {max_trial_len})"
    for event in nan_exclude:
        query += f" | {event}.isnull()"
    if exclude_unbiased:
        query += " | (probabilityLeft == 0.5)"
    if exclude_nochoice:
        query += " | (choice == 0)"
    query = query.lstrip(" |")

    trials.successful = ~session_loader.trials.eval(query).values

    return trials


def create_and_split_sampling_intervals(trials, mask):
    sampling_intervals = trials.select_by_mask(mask)
    sampling_intervals.start = sampling_intervals.stimOn_times - 0.5
    sampling_intervals.end = sampling_intervals.stimOn_times + 1.5

    num_sampling_intervals = len(sampling_intervals)

    rng = np.random.default_rng(42)
    trial_ids = rng.choice(
        np.arange(num_sampling_intervals), num_sampling_intervals, replace=False
    )
    train_ids = trial_ids[: int(0.7 * num_sampling_intervals)]
    train_mask = np.zeros(num_sampling_intervals, dtype=bool)
    train_mask[train_ids] = True

    val_ids = trial_ids[
        int(0.7 * num_sampling_intervals) : int(0.8 * num_sampling_intervals)
    ]
    val_mask = np.zeros(num_sampling_intervals, dtype=bool)
    val_mask[val_ids] = True

    test_ids = trial_ids[int(0.8 * num_sampling_intervals) :]
    test_mask = np.zeros(num_sampling_intervals, dtype=bool)
    test_mask[test_ids] = True

    train_intervals = sampling_intervals.select_by_mask(train_mask)
    val_intervals = sampling_intervals.select_by_mask(val_mask)
    test_intervals = sampling_intervals.select_by_mask(test_mask)

    return train_intervals, val_intervals, test_intervals


def load_predefined_split_intervals(eid, split_path):
    trial_start_end = np.load(split_path, allow_pickle=True)[()][eid]

    num_trials = len(trial_start_end)

    train_intervals = Interval(
        start=trial_start_end[: int(0.7 * num_trials), 0],
        end=trial_start_end[: int(0.7 * num_trials), 1],
    )

    val_intervals = Interval(
        start=trial_start_end[int(0.7 * num_trials) : int(0.8 * num_trials), 0],
        end=trial_start_end[int(0.7 * num_trials) : int(0.8 * num_trials), 1],
    )

    test_intervals = Interval(
        start=trial_start_end[int(0.8 * num_trials) :, 0],
        end=trial_start_end[int(0.8 * num_trials) :, 1],
    )

    train_intervals.sort()
    val_intervals.sort()
    test_intervals.sort()

    return train_intervals, val_intervals, test_intervals


def interpolate_wheel_and_whisker(data, train_intervals, val_intervals, test_intervals):
    """
    Interpolate the wheel and whisker motion energy to match the preprocessing done in NEDS.
    """
    # to match the preprocessing done in NEDS, we need to interpolate the whisker motion energy and wheel speed
    from scipy.interpolate import interp1d

    wheel_timestamps_list, wheel_speed_list = [], []
    motion_energy_timestamps_list, motion_energy_list = [], []

    n_bins = 100
    bin_size = 0.02

    for sampling_intervals in [train_intervals, val_intervals, test_intervals]:
        for start, end in zip(sampling_intervals.start, sampling_intervals.end):
            assert (
                np.abs((end - start) - 2.0) <= 0.01
            ), f"Interval length is not 2.0: {end - start} in eid {data.session.id}"
            end = start + 2.0
            sliced_data = data.slice(start, end)
            wheel_timestamps, wheel_speed = (
                sliced_data.wheel.timestamps,
                sliced_data.wheel.speed.squeeze(),
            )
            motion_energy_timestamps, motion_energy = (
                sliced_data.whisker.timestamps,
                sliced_data.whisker.motion_energy.squeeze(),
            )

            x_interp = np.linspace(bin_size, 2.0, n_bins)
            y_interp = interp1d(
                wheel_timestamps, wheel_speed, kind="linear", fill_value="extrapolate"
            )(x_interp)
            wheel_timestamps_list.append(x_interp + start)
            wheel_speed_list.append(y_interp)

            y_interp = interp1d(
                motion_energy_timestamps,
                motion_energy,
                kind="linear",
                fill_value="extrapolate",
            )(x_interp)
            motion_energy_timestamps_list.append(x_interp + start)
            motion_energy_list.append(y_interp)

    wheel_timestamps = np.concatenate(wheel_timestamps_list)
    wheel_speed = np.concatenate(wheel_speed_list)
    motion_energy_timestamps = np.concatenate(motion_energy_timestamps_list)
    motion_energy = np.concatenate(motion_energy_list)

    wheel_interpolated = IrregularTimeSeries(
        timestamps=wheel_timestamps,
        speed=wheel_speed[:, None],
        domain="auto",
    )

    whisker_interpolated = IrregularTimeSeries(
        timestamps=motion_energy_timestamps,
        motion_energy=motion_energy[:, None],
        domain="auto",
    )

    return wheel_interpolated, whisker_interpolated


def compute_trial_aligned_firing_rate(data):
    # precompute firing rate
    # the firing rate is estimated based on trial-aligned data only
    mask = ~np.isnan(data.trials.stimOn_times)
    if np.any(~mask):
        logging.warning(
            f"There are {np.sum(~mask)} nan values (out of {len(data.trials)}) in the trials.stimOn_times for session with eid {data.session.id}"
        )
    trial_aligned_intervals = Interval(
        start=data.trials.stimOn_times[mask] - 0.5,
        end=data.trials.stimOn_times[mask] + 1.5,
    )
    trial_aligned_spikes = data.spikes.select_by_interval(trial_aligned_intervals)
    assert trial_aligned_spikes.domain.is_disjoint()

    # this is a bug in the original code used in NEDS, we should exclude the trials
    # for which stimOn_times is nan, but they were used in the denominator of the firing rate
    recording_duration = len(data.trials) * 2.0
    firing_rate = []
    for unit_index, _ in enumerate(data.units.id):
        unit_spikes = trial_aligned_spikes.timestamps[
            trial_aligned_spikes.unit_index == unit_index
        ]
        fr = len(unit_spikes) / recording_duration
        firing_rate.append(fr)

    firing_rate = np.array(firing_rate)

    return firing_rate


def main():
    # use argparse to get arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--eid", type=str)
    parser.add_argument("--split_path", type=str, default="./splits.npy")
    parser.add_argument("--output_dir", type=str, default="./processed")
    args = parser.parse_args()

    # intiantiate a DatasetBuilder which provides utilities for processing data
    brainset_description = BrainsetDescription(
        id="ibl_reproducible_ephys_2022",
        origin_version="",
        derived_version="1.0.0",
        source="https://int-brain-lab.github.io/iblenv/notebooks_external/data_release_repro_ephys.html",
        description="Repeatedly inserted Neuropixels multi-electrode probes targeting "
        "the same brain locations (called the repeated site, including posterior "
        "parietal cortex, hippocampus, and thalamus) in mice performing a behavioral "
        "task.",
    )

    logging.info(f"Processing session eid: {args.eid}")

    # connect to sdk
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
    )

    # get subject metadata
    subject_id = one.get_details(args.eid)["subject"]

    subject = SubjectDescription(
        id=subject_id,
        species=Species.MUS_MUSCULUS,
        sex=Sex.MALE,
    )

    # extract experiment metadata
    recording_date = datetime.datetime.fromisoformat(
        one.get_details(args.eid)["start_time"]
    ).strftime("%Y%m%d")
    device_id = f"{subject.id}_{recording_date}"
    session_id = args.eid

    # register session
    session_description = SessionDescription(
        id=session_id,
        recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
    )

    # register device
    device_description = DeviceDescription(
        id=device_id,  # TODO: update
        recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,  # TODO: update
    )

    # extract spiking activity
    spikes, units = extract_spikes(one, args.eid)

    session_loader = SessionLoader(
        one, session_path=one.get_details(args.eid)["local_path"], eid=args.eid
    )
    wheel = extract_wheel(session_loader)
    whisker = extract_whisker_motion_energy(session_loader)

    trials = load_trials(session_loader)

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
        trials=trials,
        wheel=wheel,
        whisker=whisker,
        domain=spikes.domain,
    )

    train_intervals, val_intervals, test_intervals = load_predefined_split_intervals(
        args.eid, args.split_path
    )

    data.wheel_interpolated, data.whisker_interpolated = interpolate_wheel_and_whisker(
        data, train_intervals, val_intervals, test_intervals
    )
    data.units.trial_aligned_firing_rate = compute_trial_aligned_firing_rate(data)

    # set train, validation and test domains
    data.set_train_domain(train_intervals)
    data.set_valid_domain(val_intervals)
    data.set_test_domain(test_intervals)

    # save data to disk
    path = os.path.join(args.output_dir, f"{session_id}.h5")
    with h5py.File(path, "w") as file:
        data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


if __name__ == "__main__":
    main()
