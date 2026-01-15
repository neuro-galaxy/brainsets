# /// brainset-pipeline
# python-version = "3.12"
# dependencies = [
#   "dandi==0.61.2",
#   "pynwb==2.8.2",
# ]
# ///

## TODO - add brain region information to units
## TODO - add licking events
## TODO - implement download from DANDI

from argparse import ArgumentParser
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO, NWBFile
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
    DeviceDescription,
)
from brainsets.utils.dandi_utils import extract_subject_from_nwb
from brainsets.taxonomy import RecordingTech
from brainsets import serialize_fn_map
from brainsets.pipeline import BrainsetPipeline


parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")


class Pipeline(BrainsetPipeline):
    brainset_id = "ibl_brain_wide_map_2026"
    dandiset_id = "DANDI:000409/draft"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir, args) -> pd.DataFrame:
        # TODO: Implement manifest generation from DANDI
        # For now, we keep just a simple manifest for local testing
        manifest_list = [
            {
                "session_id": "sub-NYU-11_ses-6713a4a7-faed-4df2-acab-ee4e63326f8d",
                "filename": "sub-NYU-11_ses-6713a4a7-faed-4df2-acab-ee4e63326f8d_desc-processed_behavior+ecephys.nwb",
            },
            {
                "session_id": "sub-NYU-11_ses-56956777-dca5-468c-87cb-78150432cc57",
                "filename": "sub-NYU-11_ses-56956777-dca5-468c-87cb-78150432cc57_desc-processed_behavior+ecephys.nwb",
            },
            {
                "session_id": "sub-NYU-12_ses-4364a246-f8d7-4ce7-ba23-a098104b96e4",
                "filename": "sub-NYU-12_ses-4364a246-f8d7-4ce7-ba23-a098104b96e4_desc-processed_behavior+ecephys.nwb",
            },
            {
                "session_id": "sub-NYU-12_ses-b182b754-3c3e-4942-8144-6ee790926b58",
                "filename": "sub-NYU-12_ses-b182b754-3c3e-4942-8144-6ee790926b58_desc-processed_behavior+ecephys.nwb",
            },
            {
                "session_id": "sub-NYU-39_ses-6ed57216-498d-48a6-b48b-a243a34710ea",
                "filename": "sub-NYU-39_ses-6ed57216-498d-48a6-b48b-a243a34710ea_desc-processed_behavior+ecephys.nwb",
            },
            {
                "session_id": "sub-NYU-39_ses-35ed605c-1a1a-47b1-86ff-2b56144f55af",
                "filename": "sub-NYU-39_ses-35ed605c-1a1a-47b1-86ff-2b56144f55af_desc-processed_behavior+ecephys.nwb",
            },
            {
                "session_id": "sub-NYU-46_ses-64e3fb86-928c-4079-865c-b364205b502e",
                "filename": "sub-NYU-46_ses-64e3fb86-928c-4079-865c-b364205b502e_desc-processed_behavior+ecephys.nwb",
            },
        ]
        manifest = pd.DataFrame(manifest_list).set_index("session_id")
        return manifest

    def download(self, manifest_item):
        # Placeholder for download functionality
        # For local testing, assume the file is already in raw_dir
        self.update_status("DOWNLOADING")
        fpath = self.raw_dir / manifest_item.filename

        if not fpath.exists():
            print(f"Warning: File not found at {fpath}")
            print("For local testing, please place the NWB file in the raw directory")

        return fpath

    def process(self, fpath):
        self.update_status("Loading NWB")

        # Check if already processed
        session_id = fpath.stem
        store_path = self.processed_dir / f"{session_id}.h5"

        if store_path.exists() and not self.args.reprocess:
            self.update_status("Skipped Processing")
            return

        # Open NWB file
        io = NWBHDF5IO(str(fpath), "r")
        nwbfile = io.read()

        self.processed_dir.mkdir(exist_ok=True, parents=True)

        # Extract metadata
        self.update_status("Extracting Metadata")
        brainset_description = BrainsetDescription(
            id="ibl_brain_wide_map_2026",
            origin_version="draft",
            derived_version="1.0.0",
            source="https://dandiarchive.org/dandiset/000409/draft",
            description="International Brain Laboratory Brain Wide Map dataset containing "
            "Neuropixels recordings across multiple brain regions in mice performing "
            "a visual discrimination task.",
        )

        session_description = SessionDescription(
            id=session_id,
            recording_date=nwbfile.session_start_time,
        )

        subject = extract_subject_from_nwb(nwbfile)

        # Device description
        device_description = DeviceDescription(
            id=session_id,
            recording_tech=RecordingTech.NEUROPIXELS_SPIKES,
        )

        # Get last timestamp from valid epoch
        # Only "task" epoch contains valid behavioral data
        if nwbfile.epochs is not None:
            epoch_df = nwbfile.epochs.to_dataframe()
            max_time = epoch_df[epoch_df["protocol_type"] == "task"]["stop_time"].item()
        else:
            max_time = np.inf

        # Extract neural data
        self.update_status("Extracting Neural Data")
        units, spikes = extract_units_and_spikes(nwbfile=nwbfile, max_time=max_time)

        # Extract behavioral data - wheel
        if "wheel" in nwbfile.processing:
            self.update_status("Extracting Behavioral Data - wheel")
            wheel_data = extract_wheel_data(nwbfile=nwbfile, max_time=max_time)
        else:
            self.update_status(f"No wheel data found for session {session_id}")
            wheel_data = dict()

        # Extract pose estimation data
        if "pose_estimation" in nwbfile.processing:
            self.update_status("Extracting Behavioral Data - pose estimation")
            pose_estimation_data = extract_pose_estimation_data(
                nwbfile=nwbfile, max_time=max_time
            )
        else:
            self.update_status(
                f"No pose estimation data found for session {session_id}"
            )
            pose_estimation_data = dict()

        # Extract trials
        self.update_status("Extracting Trials")
        trials = extract_trials(nwbfile=nwbfile, max_time=max_time)

        # Close NWB file
        io.close()

        # Create Data object
        data = Data(
            # Metadata
            brainset=brainset_description,
            subject=subject,
            session=session_description,
            device=device_description,
            # Neural activity
            units=units,
            spikes=spikes,
            domain="auto",
            # Trials
            trials=trials,
            # Behavior
            **wheel_data,
            **pose_estimation_data,
        )

        # Create train/validation/test splits
        self.update_status("Creating Splits")
        train_trials, valid_trials, test_trials = trials.split(
            [0.7, 0.1, 0.2],  # proportions for train/valid/test
            shuffle=True,  # randomly shuffle trials
            random_seed=42,  # for reproducibility
        )

        data.set_train_domain(train_trials)
        data.set_valid_domain(valid_trials)
        data.set_test_domain(test_trials)

        # Save to HDF5
        self.update_status("Storing")
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


def extract_units_and_spikes(nwbfile: NWBFile, max_time: float):
    """Extract unit information and spike times from NWB file."""
    units_df = nwbfile.units.to_dataframe(
        exclude=set(
            [
                "spike_times",
                "waveform_mean",
                "spike_amplitudes_uV",
                "spike_distances_from_probe_tip_um",
            ]
        )
    )

    # Extract unit metadata
    spike_train_list = nwbfile.units.spike_times_index[:]
    unit_ids = []
    for i in range(len(spike_train_list)):
        unit_ids.append(f"unit_{i}")

    # Collect all extra metadata, floats, strings, etc. except object types
    extra_metadata = {
        col: units_df[col].values
        for col in units_df.select_dtypes(exclude="object").columns
    }

    units = ArrayDict(
        id=np.array(unit_ids),
        unit_name=units_df.unit_name.values,
        **extra_metadata,
    )

    # Extract spike times
    spike_timestamps = np.array([])
    spike_unit_index = np.array([], dtype=np.int64)

    for i in range(len(spike_train_list)):
        spike_train = spike_train_list[i]
        valid_mask = spike_train < max_time
        filtered_spike_train = spike_train[valid_mask]
        spike_timestamps = np.concatenate([spike_timestamps, filtered_spike_train])
        spike_unit_index = np.concatenate(
            [
                spike_unit_index,
                np.full_like(filtered_spike_train, fill_value=i, dtype=np.int64),
            ]
        )

    spikes = IrregularTimeSeries(
        timestamps=spike_timestamps,
        unit_index=spike_unit_index,
        domain="auto",
    )
    spikes.sort()

    return units, spikes


def extract_wheel_data(nwbfile: NWBFile, max_time: float):
    """Extract wheel velocity and acceleration data."""
    wheel_rate = float(nwbfile.processing["wheel"]["TimeSeriesWheelAcceleration"].rate)
    wheel_start = float(
        nwbfile.processing["wheel"]["TimeSeriesWheelAcceleration"].starting_time
    )

    if np.isinf(max_time):
        num_samples = None
    else:
        num_samples = int((max_time - wheel_start) * wheel_rate)

    wheel_acc = RegularTimeSeries(
        values=nwbfile.processing["wheel"]["TimeSeriesWheelAcceleration"].data[
            :num_samples
        ],
        sampling_rate=wheel_rate,
        domain_start=wheel_start,
        domain="auto",
    )

    wheel_vel = RegularTimeSeries(
        values=nwbfile.processing["wheel"]["TimeSeriesWheelVelocity"].data[
            :num_samples
        ],
        sampling_rate=wheel_rate,
        domain_start=wheel_start,
        domain="auto",
    )

    # Filter IrregularTimeSeries by timestamp
    wheel_pos_timestamps = nwbfile.processing["wheel"][
        "SpatialSeriesWheelPosition"
    ].timestamps[:]
    wheel_pos_values = nwbfile.processing["wheel"]["SpatialSeriesWheelPosition"].data[:]
    valid_pos_mask = wheel_pos_timestamps < max_time

    wheel_pos = IrregularTimeSeries(
        values=wheel_pos_values[valid_pos_mask],
        timestamps=wheel_pos_timestamps[valid_pos_mask],
        domain="auto",
    )

    # Filter intervals - keep only complete intervals within valid epoch
    wheel_movement_intervals_df = nwbfile.processing["wheel"][
        "TimeIntervalsWheelMovement"
    ].to_dataframe()
    valid_intervals = wheel_movement_intervals_df[
        wheel_movement_intervals_df["stop_time"] < max_time
    ]

    wheel_movement_intervals = Interval(
        start=valid_intervals["start_time"].values,
        end=valid_intervals["stop_time"].values,
        peak_amplitude=valid_intervals["peak_amplitude"].values,
        timekeys=["start", "end"],
    )

    wheel_data = {
        "wheel_acceleration": wheel_acc,
        "wheel_velocity": wheel_vel,
        "wheel_position": wheel_pos,
        "wheel_movement_intervals": wheel_movement_intervals,
    }

    return wheel_data


def extract_pose_estimation_data(nwbfile: NWBFile, max_time: float):
    """Extract pose estimation data from left, right and body cameras."""
    pose_estimation = {}

    # Left Camera
    if "LeftCamera" in nwbfile.processing["pose_estimation"].data_interfaces:
        pes_left = nwbfile.processing["pose_estimation"][
            "LeftCamera"
        ].pose_estimation_series
        timestamps_left = list(pes_left.values())[0].timestamps[:]
        valid_mask_left = timestamps_left < max_time
        pose_estimation_left_dict = dict()
        for k, v in pes_left.items():
            pose_estimation_left_dict[k] = v.data[valid_mask_left]
        pose_estimation_left_camera = IrregularTimeSeries(
            timestamps=timestamps_left[valid_mask_left],
            domain="auto",
            **pose_estimation_left_dict,
        )
        pose_estimation["pose_estimation_left_camera"] = pose_estimation_left_camera

    # Right Camera
    if "RightCamera" in nwbfile.processing["pose_estimation"].data_interfaces:
        pes_right = nwbfile.processing["pose_estimation"][
            "RightCamera"
        ].pose_estimation_series
        timestamps_right = list(pes_right.values())[0].timestamps[:]
        valid_mask_right = timestamps_right < max_time
        pose_estimation_right_dict = dict()
        for k, v in pes_right.items():
            pose_estimation_right_dict[k] = v.data[valid_mask_right]
        pose_estimation_right_camera = IrregularTimeSeries(
            timestamps=timestamps_right[valid_mask_right],
            domain="auto",
            **pose_estimation_right_dict,
        )
        pose_estimation["pose_estimation_right_camera"] = pose_estimation_right_camera

    # Body Camera
    if "BodyCamera" in nwbfile.processing["pose_estimation"].data_interfaces:
        pes_body = nwbfile.processing["pose_estimation"][
            "BodyCamera"
        ].pose_estimation_series
        timestamps_body = list(pes_body.values())[0].timestamps[:]
        valid_mask_body = timestamps_body < max_time
        pose_estimation_body_dict = dict()
        for k, v in pes_body.items():
            pose_estimation_body_dict[k] = v.data[valid_mask_body]
        pose_estimation_body_camera = IrregularTimeSeries(
            timestamps=timestamps_body[valid_mask_body],
            domain="auto",
            **pose_estimation_body_dict,
        )
        pose_estimation["pose_estimation_body_camera"] = pose_estimation_body_camera

    return pose_estimation


def extract_trials(nwbfile: NWBFile, max_time: float):
    """Extract trial information."""
    df = nwbfile.trials.to_dataframe()
    df.rename(columns={"start_time": "start", "stop_time": "end"}, inplace=True)
    df = df[df["end"] < max_time]
    time_columns = ["start", "end"] + [c for c in df.columns if "_time" in c]
    trials = Interval.from_dataframe(
        df,
        timekeys=time_columns,
    )
    return trials
