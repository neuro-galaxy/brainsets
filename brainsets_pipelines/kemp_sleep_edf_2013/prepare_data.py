import argparse
import logging
import os
from pathlib import Path

import h5py
import mne
import numpy as np

from brainsets import serialize_fn_map
from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    SubjectDescription,
    DeviceDescription,
)
from brainsets.taxonomy import RecordingTech, Species, Sex
from temporaldata import Data, IrregularTimeSeries

logging.basicConfig(level=logging.INFO)


def parse_subject_metadata(raw):
    """Extract subject metadata from EDF header."""
    info = raw.info
    subject_info = info.get("subject_info", {})

    age = subject_info.get("age")
    sex_str = subject_info.get("sex")

    if sex_str is not None:
        sex = Sex.MALE if sex_str == 1 else Sex.FEMALE if sex_str == 2 else Sex.UNKNOWN
    else:
        sex = Sex.UNKNOWN

    return age, sex


def extract_signals(raw_psg):
    """Extract physiological signals from PSG EDF file as IrregularTimeSeries."""
    data, times = raw_psg.get_data(return_times=True)
    ch_names = raw_psg.ch_names

    signals = {}

    for idx, ch_name in enumerate(ch_names):
        ch_name_lower = ch_name.lower()
        signal_data = data[idx, :]

        if (
            "eeg" in ch_name_lower
            or "fpz-cz" in ch_name_lower
            or "pz-oz" in ch_name_lower
        ):
            key = ch_name.replace(" ", "_").replace("-", "_")
            signals[f"eeg_{key}"] = IrregularTimeSeries(
                timestamps=times,
                signal=signal_data.astype(np.float32).reshape(-1, 1),
                domain="auto",
            )

        elif "eog" in ch_name_lower:
            signals["eog"] = IrregularTimeSeries(
                timestamps=times,
                signal=signal_data.astype(np.float32).reshape(-1, 1),
                domain="auto",
            )

        elif "emg" in ch_name_lower:
            signals["emg"] = IrregularTimeSeries(
                timestamps=times,
                signal=signal_data.astype(np.float32).reshape(-1, 1),
                domain="auto",
            )

        elif "resp" in ch_name_lower:
            signals["respiration"] = IrregularTimeSeries(
                timestamps=times,
                signal=signal_data.astype(np.float32).reshape(-1, 1),
                domain="auto",
            )

        elif "temp" in ch_name_lower:
            signals["temperature"] = IrregularTimeSeries(
                timestamps=times,
                signal=signal_data.astype(np.float32).reshape(-1, 1),
                domain="auto",
            )

    return signals


def extract_sleep_stages(hypnogram_file, psg_times):
    """Extract sleep stage annotations from hypnogram EDF+ file."""
    annotations = mne.read_annotations(hypnogram_file)

    sleep_stage_map = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 4,
        "Sleep stage R": 5,
        "Sleep stage ?": 6,
        "Movement time": 7,
    }

    stage_times = []
    stage_labels = []

    for annot_onset, annot_duration, annot_description in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        if annot_description in sleep_stage_map:
            stage_times.append(annot_onset + annot_duration / 2)
            stage_labels.append(sleep_stage_map[annot_description])

    if len(stage_times) == 0:
        logging.warning("No sleep stage annotations found in hypnogram")
        return None

    stage_times = np.array(stage_times)
    stage_labels = np.array(stage_labels, dtype=np.int64)

    sleep_stages = IrregularTimeSeries(
        timestamps=stage_times,
        stage=stage_labels.reshape(-1, 1),
        domain="auto",
    )

    return sleep_stages


def main():
    parser = argparse.ArgumentParser(
        description="Process Sleep-EDF data into brainsets format"
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to PSG EDF file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./processed", help="Output directory"
    )

    args = parser.parse_args()

    brainset_description = BrainsetDescription(
        id="kemp_sleep_edf_2013",
        origin_version="1.0.0",
        derived_version="1.0.0",
        source="https://www.physionet.org/content/sleep-edfx/1.0.0/",
        description="Sleep-EDF Database Expanded containing 197 whole-night "
        "polysomnographic sleep recordings with EEG, EOG, EMG, and sleep stage annotations.",
    )

    logging.info(f"Processing file: {args.input_file}")

    input_path = Path(args.input_file)
    base_name = input_path.stem

    if "PSG" not in base_name:
        logging.error(f"Expected PSG file, got: {base_name}")
        return

    hypnogram_name = base_name.replace("PSG", "Hypnogram")
    hypnogram_file = input_path.parent / f"{hypnogram_name}.edf"

    if not hypnogram_file.exists():
        logging.error(f"Hypnogram file not found: {hypnogram_file}")
        return

    raw_psg = mne.io.read_raw_edf(args.input_file, preload=True, verbose=False)

    age, sex = parse_subject_metadata(raw_psg)

    if "SC4" in base_name:
        subject_id = base_name[3:5]
        study_type = "sleep_cassette"
    elif "ST7" in base_name:
        subject_id = base_name[3:5]
        study_type = "sleep_telemetry"
    else:
        subject_id = base_name
        study_type = "unknown"

    subject = SubjectDescription(
        id=f"{study_type}_{subject_id}",
        species=Species.HOMO_SAPIENS,
        age=age,
        sex=sex,
    )

    recording_date = raw_psg.info.get("meas_date")
    if recording_date is not None:
        recording_date = recording_date.strftime("%Y-%m-%d")

    session_description = SessionDescription(
        id=base_name,
        recording_date=recording_date,
    )

    device_description = DeviceDescription(
        id=base_name,
        recording_tech=RecordingTech.POLYSOMNOGRAPHY_EEG,
    )

    signals = extract_signals(raw_psg)

    if len(signals) == 0:
        logging.error("No signals extracted from PSG file")
        return

    sleep_stages = extract_sleep_stages(str(hypnogram_file), raw_psg.times)

    data_dict = {
        "brainset": brainset_description,
        "subject": subject,
        "session": session_description,
        "device": device_description,
    }

    data_dict.update(signals)

    if sleep_stages is not None:
        data_dict["sleep_stages"] = sleep_stages

    first_signal = list(signals.values())[0]
    data_dict["domain"] = first_signal.domain

    data = Data(**data_dict)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{base_name}.h5")

    with h5py.File(output_path, "w") as file:
        data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

    logging.info(f"Saved processed data to: {output_path}")


if __name__ == "__main__":
    main()
