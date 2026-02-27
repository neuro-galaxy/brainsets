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
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import pynwb
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
parser.add_argument("--split_ref_time", type=str, default="start_time")
parser.add_argument("--split_max_duration", type=float, default=None)


key_name_mapping = {
    # wheel_position
    "SpatialSeriesWheelPosition": "wheel_position",
    "WheelPositionSmoothed": "wheel_position",
    "WheelPosition": "wheel_position",
    # wheel_velocity
    "TimeSeriesWheelVelocity": "wheel_velocity",
    "WheelSmoothedVelocity": "wheel_velocity",
    "WheelVelocitySmoothed": "wheel_velocity",
    # wheel_acceleration
    "TimeSeriesWheelAcceleration": "wheel_acceleration",
    "WheelSmoothedAcceleration": "wheel_acceleration",
    "WheelAccelerationSmoothed": "wheel_acceleration",
    # wheel_movement_intervals
    "TimeIntervalsWheelMovement": "wheel_movement_intervals",
    "WheelMovement": "wheel_movement_intervals",
    "WheelMovementIntervals": "wheel_movement_intervals",
}


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
                "session_id": "sub-NYU-21_ses-8c33abef-3d3e-4d42-9f27-445e9def08f9",
                "filename": "sub-NYU-21_ses-8c33abef-3d3e-4d42-9f27-445e9def08f9_desc-processed_behavior+ecephys.nwb",
            },
            {
                "session_id": "sub-CSHL059_ses-d2f5a130-b981-4546-8858-c94ae1da75ff",
                "filename": "sub-CSHL059_ses-d2f5a130-b981-4546-8858-c94ae1da75ff_desc-processed_behavior+ecephys.nwb",
            },
            {
                "session_id": "sub-UCLA035_ses-6f36868f-5cc1-450c-82fa-6b9829ce0cfe",
                "filename": "sub-UCLA035_ses-6f36868f-5cc1-450c-82fa-6b9829ce0cfe_desc-processed_behavior+ecephys.nwb",
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
        train_trials, valid_trials, test_trials = extract_splits(
            nwbfile=nwbfile,
            max_time=max_time,
            split_ref_time=self.args.split_ref_time,
            split_max_duration=self.args.split_max_duration,
        )
        data.set_train_domain(train_trials)
        data.set_valid_domain(valid_trials)
        data.set_test_domain(test_trials)

        # Save to HDF5
        self.update_status("Storing")
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

        # Close NWB file
        io.close()


def extract_units_and_spikes(nwbfile: NWBFile, max_time: float):
    """Extract unit information and spike times from NWB file."""
    units_df = nwbfile.units.to_dataframe(
        exclude=set(
            [
                "spike_times",
                # "waveform_mean",
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

    # Extract location names based on peak waveform electrodes
    location_names = list()
    for i, r in units_df.iterrows():
        # Get electrodes with peak waveform values
        waveform_peaks = np.max(np.abs(r["waveform_mean"]), axis=0)
        sorted_peaks = np.argsort(waveform_peaks)[::-1]
        top_sorted_peak = sorted_peaks[0]
        # Get the most common locations for the peak signal electrodes
        top_location = r["electrodes"]["location"].values[top_sorted_peak]
        location_names.append(str(top_location))
    extra_metadata["location_names"] = np.array(location_names)

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
    """Extract wheel data in all available forms: RegularTimeSeries, IrregularTimeSeries, and Intervals."""
    wheel_data = dict()
    for k, v in nwbfile.processing["wheel"].data_interfaces.items():
        key_name = key_name_mapping.get(k, None)
        if key_name is not None:
            if hasattr(v, "rate") and v.rate is not None:
                wheel_data[key_name] = extract_timeseries_with_rate(obj=v, max_time=max_time)
            elif hasattr(v, "timestamps") and v.timestamps is not None:
                wheel_data[key_name] = extract_timeseries_with_timestamps(obj=v, max_time=max_time)
            elif isinstance(v, pynwb.epoch.TimeIntervals):
                wheel_data[key_name] = extract_timeintervals_table(obj=v, max_time=max_time)
        else:
            print(f"Warning: Unrecognized wheel data interface '{k}' found. Skipping.")

    return wheel_data


def extract_timeseries_with_rate(obj, max_time: float):
    """Extract RegularTimeSeries."""
    sampling_rate = float(obj.rate)
    starting_time = float(obj.starting_time)
    if np.isinf(max_time):
        num_samples = None
    else:
        num_samples = int((max_time - starting_time) * sampling_rate)

    return RegularTimeSeries(
        values=obj.data[:num_samples],
        sampling_rate=sampling_rate,
        domain_start=starting_time,
        domain="auto",
    )


def extract_timeseries_with_timestamps(obj, max_time: float):
    """Extract IrregularTimeSeries."""
    timestamps = obj.timestamps[:]
    data_values = obj.data[:]
    valid_pos_mask = timestamps < max_time

    return IrregularTimeSeries(
        values=data_values[valid_pos_mask],
        timestamps=timestamps[valid_pos_mask],
        domain="auto",
    )


def extract_timeintervals_table(obj, max_time: float):
    """Extract Intervals from TimeIntervals table."""
    intervals_df = obj.to_dataframe()
    valid_intervals = intervals_df[intervals_df["stop_time"] < max_time]

    return Interval(
        start=valid_intervals["start_time"].values,
        end=valid_intervals["stop_time"].values,
        peak_amplitude=valid_intervals["peak_amplitude"].values,
        timekeys=["start", "end"],
    )


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


def extract_splits(
    nwbfile: NWBFile,
    max_time: float,
    split_ref_time: str = "start_time",
    split_max_duration: float | None = None,
):
    """Extract splits for torch_brain."""
    # Filter trials based on max_time, split_ref_time, and optional max_duration
    df = nwbfile.trials.to_dataframe()    
    df.rename(columns={split_ref_time: "start", "stop_time": "end"}, inplace=True)
    df = df[df["end"] < max_time]
    df = df[["start", "end"]]
    if split_max_duration is not None:
        df = df[(df["end"] - df["start"]) <= split_max_duration]

    # Create intervals and split into train/valid/test
    selected_intervals = Interval.from_dataframe(
        df,
        timekeys=["start", "end"],
    )
    train_trials, valid_trials, test_trials = selected_intervals.split(
        [0.7, 0.1, 0.2],  # proportions for train/valid/test
        shuffle=True,     # randomly shuffle trials
        random_seed=42,   # for reproducibility
    )
    return train_trials, valid_trials, test_trials