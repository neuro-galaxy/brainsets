# /// brainset-pipeline
# python-version = "3.11"
# dependencies = ["dandi==0.74.0"]
# ///

from argparse import ArgumentParser
import datetime

import h5py
from pynwb import NWBHDF5IO
from temporaldata import Data, IrregularTimeSeries, Interval
import pandas as pd
import numpy as np

from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    DeviceDescription,
)
from brainsets.utils.dandi_utils import (
    extract_spikes_from_nwbfile,
    extract_subject_from_nwb,
    get_nwb_asset_list,
    download_file,
)
from brainsets.taxonomy import RecordingTech, Task
from brainsets import serialize_fn_map

from brainsets.pipeline import BrainsetPipeline

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")


class Pipeline(BrainsetPipeline):
    brainset_id = "pei_pandarinath_nlb_2021"
    dandiset_id = {
        "jenkins_maze": "DANDI:000140/0.220113.0408",
        "indy_RTT": "DANDI:000129/0.241017.1444",
    }
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir, args) -> pd.DataFrame:
        manifest_list = []
        for task, dandiset_id in cls.dandiset_id.items():
            asset_list = get_nwb_asset_list(dandiset_id)
            manifest_item = [
                {"path": x.path, "url": x.download_url} for x in asset_list
            ]

            for m in manifest_item:
                path = m["path"]
                m["id"] = f"{task}_test" if "test" in path else f"{task}_train"
                m["dandiset_id"] = dandiset_id
                m["task"] = task.split("_")[1]
            manifest_list.extend(manifest_item)

        return pd.DataFrame(manifest_list).set_index("id")

    def download(self, manifest_item):
        self.update_status("DOWNLOADING")
        self.raw_dir.mkdir(exist_ok=True, parents=True)
        fpath = download_file(
            manifest_item.path,
            manifest_item.url,
            self.raw_dir,
            overwrite=self.args.redownload,
        )
        return {"fpath": fpath, "manifest_item": manifest_item}

    def process(self, download_output):
        self.processed_dir.mkdir(exist_ok=True, parents=True)

        # intiantiate a DatasetBuilder which provides utilities for processing data
        brainset_description = BrainsetDescription(
            id=self.brainset_id,
            origin_version=download_output["manifest_item"].dandiset_id,
            derived_version="2.0.0",
            source=f"https://dandiarchive.org/dandiset/{download_output['manifest_item'].dandiset_id}",
            description="This dataset contains sorted unit spiking times and behavioral"
            " data from a macaque performing a delayed reaching task. The experimental task"
            " was a center-out reaching task with obstructing barriers forming a maze,"
            " resulting in a variety of straight and curved reaches. The dataset also has RTT ie random target reaching task.",
        )
        fpath = download_output["fpath"]
        task = download_output["manifest_item"].task
        # open file
        self.update_status("Loading NWB")
        io = NWBHDF5IO(fpath, "r")
        nwbfile = io.read()

        self.update_status("Extracting Metadata")
        # extract subject metadata
        # this dataset is from dandi, which has structured subject metadata, so we
        # can use the helper function extract_subject_from_nwb
        subject = extract_subject_from_nwb(nwbfile)

        # extract experiment metadata
        recording_date = nwbfile.session_start_time.strftime("%Y%m%d")
        device_id = f"{subject.id}_{recording_date}"
        session_id = f"{subject.id}_{task}"
        if "test" in str(fpath):
            session_id += "_test"
        else:
            session_id += "_train"

        store_path = self.processed_dir / f"{session_id}.h5"
        if store_path.exists() and not self.args.reprocess:
            self.update_status("Skipped Processing")
            return

        # register session
        session_description = SessionDescription(
            id=session_id,
            recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
            task=Task.REACHING,
        )

        # register device
        device_description = DeviceDescription(
            id=device_id,
            recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,
        )

        # extract spiking activity
        # this data is from dandi, we can use our helper function
        self.update_status("Extracting Spikes")
        spikes, units = extract_spikes_from_nwbfile(
            nwbfile,
            recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,
        )

        # extract data about trial structure
        self.update_status("Extracting Trials")
        trials = extract_trials(nwbfile)

        data = Data(
            brainset=brainset_description,
            session=session_description,
            device=device_description,
            # neural activity
            spikes=spikes,
            units=units,
            # stimuli and behavior
            trials=trials,
            # domain
            domain="auto",
        )

        if not "test" in str(fpath):
            self.update_status("Creating Splits")
            # extract behavior
            if task == "maze":
                data.hand, data.eye = extract_behavior_maze(nwbfile, trials)
                # report accuracy only on the evaluation intervals
                data.nlb_eval_intervals = Interval(
                    start=trials.move_onset_time - 0.05,
                    end=trials.move_onset_time + 0.65,
                )
                # split and register trials into train, validation and test
                train_trials, valid_trials = trials.select_by_mask(
                    trials.train_mask_nwb
                ).split([0.8, 0.2], shuffle=True, random_seed=42)
                test_trials = trials.select_by_mask(trials.test_mask_nwb)

                data.set_train_domain(train_trials)
                data.set_valid_domain(valid_trials)
                data.set_test_domain(test_trials)

            elif task == "RTT":
                data.cursor, data.finger, data.target = extract_behavior_rtt(
                    nwbfile, trials
                )

        # close file
        io.close()

        # save data to disk
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


def extract_trials(nwbfile):
    r"""Extract trial information from the NWB file. Trials that are flagged as
    "to discard" or where the monkey failed are marked as invalid."""
    trial_table = nwbfile.trials.to_dataframe()

    # rename start and end time columns
    trial_table = trial_table.rename(
        columns={
            "start_time": "start",
            "stop_time": "end",
            "split": "split_indicator",
        }
    )

    trials = Interval.from_dataframe(trial_table)

    # For RTT, trial boundaries have small floating-point inconsistencies
    # which can make the end of a trial slightly less than the start of the next making the interval not disjoint.
    if not trials.is_disjoint():
        trials.end = np.round(trials.end, 1)
        trials.start = np.round(trials.start, 1)

    # the dataset has pre-defined train/valid splits, we will use the valid split
    # as our test
    train_mask_nwb = trial_table.split_indicator.to_numpy() == "train"
    test_mask_nwb = trial_table.split_indicator.to_numpy() == "val"

    trials.train_mask_nwb = (
        train_mask_nwb  # Naming with "_" since train_mask is reserved
    )
    trials.test_mask_nwb = test_mask_nwb  # Naming with "_" since test_mask is reserved

    return trials


def extract_behavior_maze(nwbfile, trials):
    timestamps = nwbfile.processing["behavior"]["hand_vel"].timestamps[:]
    hand_pos = nwbfile.processing["behavior"]["hand_pos"].data[:]
    hand_vel = nwbfile.processing["behavior"]["hand_vel"].data[:]
    eye_pos = nwbfile.processing["behavior"]["eye_pos"].data[:]

    hand = IrregularTimeSeries(
        timestamps=timestamps,
        pos=hand_pos,
        vel=hand_vel,
        domain="auto",
    )

    eye = IrregularTimeSeries(
        timestamps=timestamps,
        pos=eye_pos,
        domain="auto",
    )

    return hand, eye


def extract_behavior_rtt(nwbfile, trials):

    cursor_pos = nwbfile.processing["behavior"]["cursor_pos"].data[:]
    n = cursor_pos.shape[0]
    rate = nwbfile.processing["behavior"]["cursor_pos"].rate
    start = nwbfile.processing["behavior"]["cursor_pos"].starting_time
    timestamps = start + np.arange(n) / rate
    finger_pos = nwbfile.processing["behavior"]["finger_pos"].data[:]
    finger_vel = nwbfile.processing["behavior"]["finger_vel"].data[:]
    target_pos = nwbfile.processing["behavior"]["target_pos"].data[:]

    cursor = IrregularTimeSeries(
        timestamps=timestamps,
        pos=cursor_pos,
        domain="auto",
    )

    finger = IrregularTimeSeries(
        timestamps=timestamps,
        pos=finger_pos,
        vel=finger_vel,
        domain="auto",
    )

    target = IrregularTimeSeries(
        timestamps=timestamps,
        pos=target_pos,
        domain="auto",
    )

    return cursor, finger, target
