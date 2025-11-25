# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#     "mne",
#     "brainsets @ git+https://github.com/neuro-galaxy/brainsets@main",
#     "h5py",
#     "numpy",
#     "temporaldata"
# ]
# ///

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
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
from brainsets.core import StringIntEnum
from brainsets.utils.split import (
    chop_intervals,
    filter_intervals,
    generate_stratified_folds,
)
from temporaldata import Data, Interval, RegularTimeSeries, ArrayDict


class Modality(StringIntEnum):
    EEG = "EEG"
    EMG = "EMG"
    EOG = "EOG"
    RESP = "RESP"
    TEMP = "TEMP"


logging.basicConfig(level=logging.INFO)


def parse_subject_metadata(raw):
    """Extract subject metadata from EDF header."""
    info = raw.info
    subject_info = info.get("subject_info", {})

    # Try to get age from "age" key, otherwise look in "last_name" (e.g. "54yr")
    age = subject_info.get("age")
    if age is None:
        age_str = subject_info.get("last_name")
        if age_str and isinstance(age_str, str) and "yr" in age_str:
            try:
                age = int(age_str.replace("yr", ""))
            except ValueError:
                logging.warning(f"Could not parse age from last_name: {age_str}")
                age = None

    sex_str = subject_info.get("sex")

    if sex_str is not None:
        sex = Sex.MALE if sex_str == 1 else Sex.FEMALE if sex_str == 2 else Sex.UNKNOWN
    else:
        sex = Sex.UNKNOWN

    return age, sex


def extract_signals(raw_psg):
    """Extract physiological signals from PSG EDF file as a single IrregularTimeSeries."""
    data, times = raw_psg.get_data(return_times=True)
    ch_names = raw_psg.ch_names

    signal_list = []
    unit_meta = []

    for idx, ch_name in enumerate(ch_names):
        ch_name_lower = ch_name.lower()
        signal_data = data[idx, :]

        modality = None
        if (
            "eeg" in ch_name_lower
            or "fpz-cz" in ch_name_lower
            or "pz-oz" in ch_name_lower
        ):
            modality = Modality.EEG
        elif "eog" in ch_name_lower:
            modality = Modality.EOG
        elif "emg" in ch_name_lower:
            modality = Modality.EMG
        elif "resp" in ch_name_lower:
            modality = Modality.RESP
        elif "temp" in ch_name_lower:
            modality = Modality.TEMP
        else:
            # Skip channels that don't match our interest
            continue

        signal_list.append(signal_data)

        unit_meta.append(
            {
                "id": str(ch_name),
                "modality": str(modality.value),
            }
        )

    if not signal_list:
        return None, None

    stacked_signals = np.stack(signal_list, axis=1)

    signals = RegularTimeSeries(
        signal=stacked_signals,
        sampling_rate=raw_psg.info["sfreq"],
        domain=Interval(start=times[0], end=times[-1]),
    )

    # Create units ArrayDict
    units_df = pd.DataFrame(unit_meta)
    units = ArrayDict.from_dataframe(units_df)

    return signals, units


def extract_sleep_stages(hypnogram_file, psg_times):
    """Extract sleep stage annotations from hypnogram EDF+ file as an Interval object."""
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

    starts = []
    ends = []
    stage_names = []
    stage_ids = []

    for annot_onset, annot_duration, annot_description in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        if annot_description in sleep_stage_map:
            starts.append(annot_onset)
            ends.append(annot_onset + annot_duration)
            stage_names.append(annot_description)
            stage_ids.append(sleep_stage_map[annot_description])

    if len(starts) == 0:
        logging.warning("No sleep stage annotations found in hypnogram")
        return None

    stages = Interval(
        start=np.array(starts),
        end=np.array(ends),
        names=np.array(stage_names),
        id=np.array(stage_ids, dtype=np.int64),
    )

    return stages


def create_splits(stages, epoch_duration=30.0, n_folds=5, seed=42):
    """
    Generate train/valid/test splits from sleep stage intervals.

    Args:
        stages: Interval object containing sleep stages with 'id' attribute
        epoch_duration: Duration in seconds to chop intervals into
        n_folds: Number of cross-validation folds
        seed: Random seed for reproducibility

    Returns:
        Data object containing splits structure with intrasubject folds
    """
    if stages is None or len(stages) == 0:
        logging.warning("No stages provided for splitting")
        return None

    chopped = chop_intervals(stages, duration=epoch_duration)
    logging.info(f"Chopped {len(stages)} stages into {len(chopped)} epochs")

    UNKNOWN_STAGE_ID = 6
    filtered = filter_intervals(chopped, exclude_ids=[UNKNOWN_STAGE_ID])
    logging.info(f"Filtered out unknown stages, {len(filtered)} epochs remaining")

    if len(filtered) == 0:
        logging.warning("No valid epochs after filtering")
        return None

    splits = generate_stratified_folds(
        filtered,
        n_folds=n_folds,
        val_ratio=0.2,
        seed=seed,
    )
    logging.info(f"Generated {n_folds} stratified folds")

    return splits


def main():
    parser = argparse.ArgumentParser(
        description="Process Sleep-EDF data into brainsets format"
    )
    parser.add_argument(
        "--input-file", type=str, required=True, help="Path to PSG EDF file"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./processed", help="Output directory"
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

    hypnogram_file = None
    # We try matching the first 7 characters first (SC4ssNE), then 6 (SC4ssN).
    for prefix_len in [7, 6]:
        prefix = base_name[:prefix_len]
        candidates = list(input_path.parent.glob(f"{prefix}*Hypnogram.edf"))
        if len(candidates) == 1:
            hypnogram_file = candidates[0]
            logging.info(f"Found hypnogram: {hypnogram_file.name}")
            break
        elif len(candidates) > 1:
            logging.warning(
                f"Multiple hypnogram candidates found for prefix {prefix}: "
                f"{[c.name for c in candidates]}. Skipping ambiguity."
            )
            break

    if hypnogram_file is None or not hypnogram_file.exists():
        logging.error(f"Hypnogram file not found for base name: {base_name}")
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
        recording_tech=RecordingTech.POLYSOMNOGRAPHY,
    )

    signals, units = extract_signals(raw_psg)

    if signals is None:
        logging.error("No signals extracted from PSG file")
        return

    stages = extract_sleep_stages(str(hypnogram_file), raw_psg.times)

    splits = create_splits(stages, n_folds=3, seed=42) if stages is not None else None

    data_dict = {
        "brainset": brainset_description,
        "subject": subject,
        "session": session_description,
        "device": device_description,
        "eeg": signals,
        "units": units,
        "domain": signals.domain,
    }

    if stages is not None:
        data_dict["stages"] = stages

    if splits is not None:
        data_dict["splits"] = splits

    data = Data(**data_dict)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{base_name}.h5")

    with h5py.File(output_path, "w") as file:
        data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

    logging.info(f"Saved processed data to: {output_path}")


if __name__ == "__main__":
    main()
