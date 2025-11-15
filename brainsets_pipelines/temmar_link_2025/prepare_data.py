import argparse
import datetime
import logging
import h5py
import os
import re

import numpy as np
from pynwb import NWBHDF5IO

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
    SubjectDescription,
    DeviceDescription,
)
from brainsets.taxonomy import RecordingTech, Task, Species, Sex
from brainsets import serialize_fn_map

logging.basicConfig(level=logging.INFO)


# ============================================================================
# SPLIT STRATEGY CONFIGURATION
# ============================================================================
# LINK has 312 sessions over 3.5 years (303 days). We assign entire sessions
# to train/valid splits (no test set for foundation model training).
#
# AVAILABLE STRATEGIES:
# ---------------------
# 1. "temporal": Chronological 60/40 train/valid split
#    - First 187 sessions → train, next 125 → valid
#    - Tests long-term generalization (early → late sessions)
#
# 2. "temporal_stratified": Chronological + balanced by target style
#    - Separate CO and RD sessions, take 60/40 from each group
#    - Ensures both train and valid have balanced CO/RD representation
#
# 3. "within_session": Paper's original 300 train / 75 valid per session
#    - First 300 trials → train, last 75 → valid (per session)
#    - Not compatible with brainsets pattern (mixes splits within sessions)
#
# CURRENT STRATEGY: temporal_stratified (configurable below)
# ============================================================================

SPLIT_STRATEGY = "temporal_stratified"  # Change to "temporal" or "within_session"
TRAIN_FRACTION = 0.60  # 60% train, 40% valid

# Session manifest will be populated after first full processing run
# Format: {"CO": ["20201019_CO", ...], "RD": ["20201020_RD", ...]}
# Sorted chronologically within each target style
SESSION_MANIFEST = {
    "CO": [],  # Will be populated from processed files
    "RD": [],  # Will be populated from processed files
}


def determine_split_assignment(session_id, target_style, strategy=SPLIT_STRATEGY):
    """Determine which split (train/valid) a session belongs to.

    Args:
        session_id: Session identifier (e.g., "20201023_CO")
        target_style: Target presentation style ("CO" or "RD")
        strategy: Split strategy to use (see module-level docs)

    Returns:
        str: "train" or "valid"

    Notes:
        - For foundation model training, we only use train/valid (no test)
        - "temporal_stratified" ensures balanced CO/RD in both splits
        - Update SESSION_MANIFEST above after processing all files
    """
    if strategy == "temporal_stratified":
        # TODO: Implement after generating session manifest
        # For now, use simple temporal split as placeholder
        # This will be updated once we have the full session list
        logging.warning(
            f"temporal_stratified not yet implemented, using temporal fallback"
        )
        strategy = "temporal"

    if strategy == "temporal":
        # Simple chronological split (placeholder until we have manifest)
        # Extract date from session_id for rough ordering
        date_str = session_id.split("_")[0]
        date = datetime.datetime.strptime(date_str, "%Y%m%d")

        # Placeholder cutoff (will be replaced with manifest-based logic)
        # Approximate: sessions before mid-2021 → train, after → valid
        cutoff_date = datetime.datetime(2021, 6, 1)

        if date < cutoff_date:
            return "train"
        else:
            return "valid"

    elif strategy == "within_session":
        # Paper's original approach: not used for cross-session training
        # All sessions assigned to train; within-session split done per trial
        logging.warning(
            "within_session strategy assigns all sessions to train; "
            "use trial-level filtering for validation"
        )
        return "train"

    else:
        raise ValueError(f"Unknown split strategy: {strategy}")


def parse_date_from_filename(filename):
    """Extract date from LINK NWB filename.

    LINK follows BIDS format:
    - sub-Monkey-N_ses-20201023_ecephys.nwb -> 20201023

    Note: Each file contains BOTH CO and RD trials - we split them during processing.
    """
    basename = os.path.basename(filename)

    # Extract date from ses-YYYYMMDD
    date_match = re.search(r"ses-(\d{8})", basename)
    if not date_match:
        raise ValueError(f"Could not parse date from {filename}")

    return date_match.group(1)


def extract_threshold_crossings(nwbfile):
    """Extract threshold crossings (spike counts) from NWB file.

    LINK provides pre-binned spike counts (no raw spike times).
    Returns RegularTimeSeries with spike counts for 96 channels.
    """
    tcfr_ts = nwbfile.analysis["ThresholdCrossings"]
    tcfr_data = tcfr_ts.data[:].astype(np.int16)  # (time, 96)
    tcfr_timestamps = tcfr_ts.timestamps[:]

    # Check if regular sampling
    dt = np.diff(tcfr_timestamps)
    is_regular = np.allclose(dt, dt[0], rtol=1e-5)

    if is_regular:
        sampling_rate = 1.0 / dt[0]
        logging.info(
            f"TCFR sampling rate: {sampling_rate:.2f} Hz (bin size: {dt[0]*1000:.1f} ms)"
        )

        tcfr = RegularTimeSeries(
            counts=tcfr_data,
            sampling_rate=sampling_rate,
            domain="auto",
            domain_start=tcfr_timestamps[0],
        )
    else:
        logging.warning("TCFR timestamps are irregular, using IrregularTimeSeries")
        tcfr = IrregularTimeSeries(
            timestamps=tcfr_timestamps,
            data=tcfr_data,
            domain="auto",
        )

    return tcfr


def extract_finger_velocity(nwbfile):
    """Extract 2D finger velocity from NWB file.

    Returns IrregularTimeSeries with:
    - finger.vel: (time, 2) [index_vel, mrs_vel]
    """
    # Velocity data
    index_vel_ts = nwbfile.analysis["index_velocity"]
    mrs_vel_ts = nwbfile.analysis["mrs_velocity"]

    # Get timestamps (should be same for both)
    timestamps = index_vel_ts.timestamps[:]

    # Stack into (time, 2) array
    vel_data = np.column_stack(
        [
            index_vel_ts.data[:].ravel(),
            mrs_vel_ts.data[:].ravel(),
        ]
    ).astype(np.float64)

    finger = IrregularTimeSeries(
        timestamps=timestamps,
        vel=vel_data,  # (time, 2)
        domain="auto",
    )

    return finger


def extract_trials(nwbfile, finger):
    """Extract trial information from NWB file.

    LINK trials contain:
    - start_time, stop_time
    - trial_number, trial_count
    - index_target_position, mrs_target_position
    - target_style, trial_timeout

    Note: String columns (like target_style) are dropped to avoid HDF5
    serialization issues. Metadata is preserved in data.target_style instead.
    """
    trial_table = nwbfile.trials.to_dataframe().reset_index()

    # Rename columns to match temporaldata conventions
    trial_table = trial_table.rename(
        columns={
            "start_time": "start",
            "stop_time": "end",
        }
    )

    # Drop string columns that cause HDF5 serialization issues
    # Keep numeric columns and arrays
    columns_to_drop = []
    for col in trial_table.columns:
        if trial_table[col].dtype == "object":
            # Check if it's a string or numpy array
            first_val = trial_table[col].iloc[0]
            if isinstance(first_val, str):
                columns_to_drop.append(col)

    if columns_to_drop:
        logging.info(f"Dropping string columns from trials: {columns_to_drop}")
        trial_table = trial_table.drop(columns=columns_to_drop)

    trials = Interval.from_dataframe(trial_table)
    trials.is_valid = np.ones(len(trials), dtype=bool)

    return trials


def extract_units_metadata(nwbfile):
    """Extract electrode/channel metadata as units.

    LINK has 96 channels (2 Utah arrays, 8x8 each).
    """
    electrodes_df = nwbfile.electrodes.to_dataframe()

    # Create unit IDs (0-95 for 96 channels)
    unit_ids = np.arange(len(electrodes_df), dtype=np.int32)

    # Extract relevant metadata
    units = ArrayDict(
        id=unit_ids,
        array_name=electrodes_df["array_name"].values,
        bank=electrodes_df["bank"].values,
        row=electrodes_df["row"].values.astype(np.int32),
        col=electrodes_df["col"].values.astype(np.int32),
        # Note: LINK dataset doesn't include impedance data
    )

    return units


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./processed")
    args = parser.parse_args()

    brainset_description = BrainsetDescription(
        id="temmar_link_2025",
        origin_version="dandi/001201/draft",
        derived_version="1.0.0",
        source="https://dandiarchive.org/dandiset/001201",
        description="LINK: Long-term Intracortical Neural activity and Kinematics. "
        "312 sessions over 3.5 years from Monkey N performing self-paced finger movements. "
        "96-channel Utah arrays (M1) with threshold crossings and 2-finger kinematics.",
    )

    logging.info(f"Processing file: {args.input_file}")

    # Parse date from filename
    date_str = parse_date_from_filename(args.input_file)
    recording_date = datetime.datetime.strptime(date_str, "%Y%m%d")

    # Open NWB file
    io = NWBHDF5IO(args.input_file, "r")
    nwbfile = io.read()

    # Extract neural activity (pre-binned threshold crossings)
    tcfr = extract_threshold_crossings(nwbfile)

    # Extract units metadata (96 channels)
    units = extract_units_metadata(nwbfile)

    logging.info(
        f"TCFR (threshold crossings, pre-binned): "
        f"{tcfr.counts.shape if hasattr(tcfr, 'counts') else tcfr.data.shape} "
        f"({len(units.id)} channels)"
    )

    # Extract finger velocity
    finger = extract_finger_velocity(nwbfile)
    logging.info(f"Finger velocity: {finger.vel.shape}")

    # Extract all trials (will be filtered by target_style later)
    trials = extract_trials(nwbfile, finger)

    # Get trials DataFrame to check target styles
    trials_df = nwbfile.trials.to_dataframe()
    target_styles = trials_df["target_style"].unique()
    logging.info(f"Found target styles: {target_styles}")

    io.close()

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each target style separately (CO and RD)
    for target_style in target_styles:
        session_id = f"{date_str}_{target_style}"
        logging.info(f"Processing session: {session_id}")

        # Filter trials to this target style
        style_mask = trials_df["target_style"] == target_style

        # Create filtered trial intervals
        filtered_trials = Interval(
            start=trials.start[style_mask.values], end=trials.end[style_mask.values]
        )
        # Copy other attributes
        for key in trials.__dict__.keys():
            if key not in ["start", "end", "domain"]:
                attr_val = getattr(trials, key)
                if hasattr(attr_val, "__len__") and len(attr_val) == len(trials):
                    setattr(filtered_trials, key, attr_val[style_mask.values])

        filtered_trials.is_valid = np.ones(len(filtered_trials), dtype=bool)

        logging.info(f"  Trials for {target_style}: {len(filtered_trials)}")

        # Extract subject metadata
        subject = SubjectDescription(
            id="MonkeyN",
            species=Species.MACACA_MULATTA,
            sex=Sex.MALE,
        )

        # Extract session metadata
        session_description = SessionDescription(
            id=session_id,
            recording_date=recording_date,
            task=Task.REACHING,  # Finger movements
        )

        # Register device (2 Utah arrays)
        device_description = DeviceDescription(
            id=f"MonkeyN_{session_id}",
            recording_tech=RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS,
        )

        # Create Data object
        data = Data(
            brainset=brainset_description,
            subject=subject,
            session=session_description,
            device=device_description,
            tcfr=tcfr,  # Pre-binned spike counts
            units=units,  # 96 channels
            finger=finger,  # Velocity only for now
            trials=filtered_trials,
            domain=finger.domain,
        )

        # Add metadata
        data.target_style = target_style
        data.spike_format = "threshold_crossings"  # Pre-binned counts (not spike times)
        if hasattr(tcfr, "sampling_rate"):
            data.tcfr_sampling_rate = float(tcfr.sampling_rate)
        data.split_strategy = SPLIT_STRATEGY  # Document which strategy was used

        # Assign session to train or valid split
        split_assignment = determine_split_assignment(
            session_id, target_style, strategy=SPLIT_STRATEGY
        )

        valid_trials = filtered_trials.select_by_mask(filtered_trials.is_valid)

        if split_assignment == "train":
            data.set_train_domain(valid_trials)
            data.set_valid_domain(Interval(start=np.array([]), end=np.array([])))
            logging.info(f"  Assigned {len(valid_trials)} trials to TRAIN")
        elif split_assignment == "valid":
            data.set_train_domain(Interval(start=np.array([]), end=np.array([])))
            data.set_valid_domain(valid_trials)
            logging.info(f"  Assigned {len(valid_trials)} trials to VALID")
        else:
            raise ValueError(f"Unknown split: {split_assignment}")

        # No test set for foundation model training
        data.set_test_domain(Interval(start=np.array([]), end=np.array([])))

        # Optional: Store within-session split indices (matches paper's 300/75)
        n_trials = len(filtered_trials)
        train_cutoff = min(300, int(0.8 * n_trials))  # Adjust if fewer trials
        data.within_session_train_trials = np.arange(0, train_cutoff)
        data.within_session_valid_trials = np.arange(train_cutoff, n_trials)

        # Generate filename
        filename = f"{session_id}.h5"
        path = os.path.join(args.output_dir, filename)

        logging.info(f"  Saving to: {path}")

        with h5py.File(path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

        logging.info(f"  Successfully processed {session_id}")


if __name__ == "__main__":
    main()
