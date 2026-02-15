# /// brainset-pipeline
# python-version = "3.11"
# dependencies = ["dandi>=0.71.3"]
# ///
"""LINK: Long-term Intracortical Neural activity and Kinematics.

312 sessions over 3.5 years from Monkey N performing self-paced finger movements.
96-channel Utah arrays (M1) with threshold crossings and 2-finger kinematics.

Reference:
    Temmar et al. (2025). LINK: Long-term Intracortical Neural activity
    and Kinematics. https://dandiarchive.org/dandiset/001201

Split Strategies (selectable via --split-strategy):
    - temporal (default): Cross-session split based on recording date.
      Sessions before June 2021 go to train, after to valid.
      Better for foundation model pretraining (tests long-term generalization).

    - within_session: Per-session split matching the original paper.
      First 300 trials -> train, remaining -> valid (up to 75).
      Matches the paper's BCI decoder evaluation approach.

Each NWB file contains both CO (center-out) and RD (random) target styles,
which are processed as separate sessions.
"""

import datetime
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from temporaldata import (
    ArrayDict,
    Data,
    Interval,
    IrregularTimeSeries,
    RegularTimeSeries,
)

from brainsets import serialize_fn_map
from brainsets.descriptions import (
    BrainsetDescription,
    DeviceDescription,
    SessionDescription,
    SubjectDescription,
)
from brainsets.pipeline import BrainsetPipeline
from brainsets.taxonomy import RecordingTech, Sex, Species, Task
from brainsets.utils.dandi_utils import download_file, get_nwb_asset_list

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")
parser.add_argument(
    "--split-strategy",
    choices=["temporal", "within_session"],
    default="temporal",
    help="Split strategy: 'temporal' (cross-session by date) or 'within_session' (300/75 per session)",
)

# Split configuration
SPLIT_CUTOFF_DATE = datetime.datetime(2021, 6, 1)  # Sessions before this -> train
WITHIN_SESSION_TRAIN_TRIALS = 300  # Paper uses 300 train trials per session
WITHIN_SESSION_VALID_TRIALS = 75  # Paper uses up to 75 valid trials


class Pipeline(BrainsetPipeline):
    """Pipeline for processing LINK dataset from DANDI."""

    brainset_id = "temmar_link_2025"
    dandiset_id = "DANDI:001201"
    parser = parser

    @classmethod
    def get_manifest(
        cls,
        raw_dir: Path,
        args: Optional[Namespace],
    ) -> pd.DataFrame:
        """Get manifest of NWB files from DANDI archive.

        Returns:
            DataFrame with columns: path, url, session_date
            Index: session_id (e.g., "20201023")
        """
        asset_list = get_nwb_asset_list(cls.dandiset_id)
        manifest_list = []

        for asset in asset_list:
            # Parse date from filename: sub-Monkey-N_ses-20201023_ecephys.nwb
            date_match = re.search(r"ses-(\d{8})", asset.path)
            if date_match:
                session_date = date_match.group(1)
                manifest_list.append(
                    {
                        "session_id": session_date,
                        "path": asset.path,
                        "url": asset.download_url,
                        "session_date": session_date,
                    }
                )

        manifest = pd.DataFrame(manifest_list).set_index("session_id")
        return manifest

    def download(self, manifest_item) -> Path:
        """Download a single NWB file from DANDI.

        Args:
            manifest_item: Row from manifest DataFrame.

        Returns:
            Path to downloaded NWB file.
        """
        self.update_status("DOWNLOADING")
        self.raw_dir.mkdir(exist_ok=True, parents=True)

        fpath = download_file(
            manifest_item.path,
            manifest_item.url,
            self.raw_dir,
            overwrite=self.args.redownload if self.args else False,
        )
        return fpath

    def process(self, fpath: Path) -> None:
        """Process NWB file and save as HDF5.

        Each NWB file contains both CO and RD target styles, which are
        processed as separate sessions (e.g., 20201023_CO, 20201023_RD).

        Args:
            fpath: Path to downloaded NWB file.
        """
        self.processed_dir.mkdir(exist_ok=True, parents=True)

        brainset_description = BrainsetDescription(
            id=self.brainset_id,
            origin_version="dandi/001201/draft",
            derived_version="1.0.0",
            source="https://dandiarchive.org/dandiset/001201",
            description=(
                "LINK: Long-term Intracortical Neural activity and Kinematics. "
                "312 sessions over 3.5 years from Monkey N performing self-paced "
                "finger movements. 96-channel Utah arrays (M1) with threshold "
                "crossings and 2-finger kinematics."
            ),
        )

        # Parse date from filename
        date_match = re.search(r"ses-(\d{8})", str(fpath))
        if not date_match:
            raise ValueError(f"Could not parse date from {fpath}")
        date_str = date_match.group(1)
        recording_date = datetime.datetime.strptime(date_str, "%Y%m%d")

        # Open NWB file
        self.update_status("Loading NWB")
        io = NWBHDF5IO(fpath, "r")
        nwbfile = io.read()

        # Extract neural activity (pre-binned threshold crossings)
        self.update_status("Extracting Neural Data")
        tcfr = extract_threshold_crossings(nwbfile)
        units = extract_units_metadata(nwbfile)

        # Extract finger velocity
        self.update_status("Extracting Behavior")
        finger = extract_finger_velocity(nwbfile)

        # Get trial information to identify target styles
        trials_df = nwbfile.trials.to_dataframe()
        target_styles = trials_df["target_style"].unique()

        io.close()

        # Process each target style as a separate session
        for target_style in target_styles:
            session_id = f"{date_str}_{target_style}"
            store_path = self.processed_dir / f"{session_id}.h5"

            if store_path.exists() and not (self.args and self.args.reprocess):
                self.update_status(f"Skipped {session_id}")
                continue

            self.update_status(f"Processing {session_id}")

            # Filter trials to this target style
            style_mask = trials_df["target_style"] == target_style
            filtered_trials = create_filtered_trials(trials_df, style_mask)

            # Create subject description
            subject = SubjectDescription(
                id="monkey_n",
                species=Species.MACACA_MULATTA,
                sex=Sex.MALE,
            )

            # Create session description
            session_description = SessionDescription(
                id=session_id,
                recording_date=recording_date,
                task=Task.REACHING,
            )

            # Create device description
            device_description = DeviceDescription(
                id=f"monkey_n_{session_id}",
                recording_tech=RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS,
            )

            # Create Data object
            data = Data(
                brainset=brainset_description,
                subject=subject,
                session=session_description,
                device=device_description,
                tcfr=tcfr,
                units=units,
                finger=finger,
                trials=filtered_trials,
                domain=finger.domain,
            )

            # Add metadata
            data.target_style = target_style
            data.spike_format = "threshold_crossings"
            if hasattr(tcfr, "sampling_rate"):
                data.tcfr_sampling_rate = float(tcfr.sampling_rate)

            # Assign train/valid split based on selected strategy
            self.update_status("Creating Splits")
            split_strategy = (
                getattr(self.args, "split_strategy", "temporal")
                if self.args
                else "temporal"
            )
            data.split_strategy = split_strategy

            if split_strategy == "temporal":
                assign_temporal_split(data, recording_date, filtered_trials)
            else:
                assign_within_session_split(data, filtered_trials)

            # Save to HDF5
            with h5py.File(store_path, "w") as file:
                data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


def extract_threshold_crossings(nwbfile):
    """Extract threshold crossings (spike counts) from NWB file.

    LINK provides pre-binned spike counts (no raw spike times).

    Args:
        nwbfile: Open NWB file object.

    Returns:
        RegularTimeSeries or IrregularTimeSeries with spike counts.
    """
    tcfr_ts = nwbfile.analysis["ThresholdCrossings"]
    tcfr_data = tcfr_ts.data[:].astype(np.int16)  # (time, 96)
    tcfr_timestamps = tcfr_ts.timestamps[:]

    # Check if regular sampling
    dt = np.diff(tcfr_timestamps)
    is_regular = np.allclose(dt, dt[0], rtol=1e-5)

    if is_regular:
        sampling_rate = 1.0 / dt[0]
        tcfr = RegularTimeSeries(
            counts=tcfr_data,
            sampling_rate=sampling_rate,
            domain="auto",
            domain_start=tcfr_timestamps[0],
        )
    else:
        tcfr = IrregularTimeSeries(
            timestamps=tcfr_timestamps,
            data=tcfr_data,
            domain="auto",
        )

    return tcfr


def extract_finger_velocity(nwbfile):
    """Extract 2D finger velocity from NWB file.

    Args:
        nwbfile: Open NWB file object.

    Returns:
        IrregularTimeSeries with vel attribute of shape (time, 2).
    """
    index_vel_ts = nwbfile.analysis["index_velocity"]
    mrs_vel_ts = nwbfile.analysis["mrs_velocity"]

    timestamps = index_vel_ts.timestamps[:]
    vel_data = np.column_stack(
        [
            index_vel_ts.data[:].ravel(),
            mrs_vel_ts.data[:].ravel(),
        ]
    ).astype(np.float64)

    finger = IrregularTimeSeries(
        timestamps=timestamps,
        vel=vel_data,
        domain="auto",
    )

    return finger


def extract_units_metadata(nwbfile):
    """Extract electrode/channel metadata as units.

    LINK has 96 channels (2 Utah arrays, 8x8 each).

    Args:
        nwbfile: Open NWB file object.

    Returns:
        ArrayDict with unit metadata.
    """
    electrodes_df = nwbfile.electrodes.to_dataframe()

    unit_ids = np.arange(len(electrodes_df), dtype=np.int32)

    units = ArrayDict(
        id=unit_ids,
        array_name=electrodes_df["array_name"].values,
        bank=electrodes_df["bank"].values,
        row=electrodes_df["row"].values.astype(np.int32),
        col=electrodes_df["col"].values.astype(np.int32),
    )

    return units


def create_filtered_trials(trials_df: pd.DataFrame, mask: pd.Series) -> Interval:
    """Create filtered trial Interval from DataFrame and boolean mask.

    Args:
        trials_df: Trials DataFrame from NWB.
        mask: Boolean mask for filtering.

    Returns:
        Interval with filtered trials.
    """
    filtered_df = trials_df[mask].copy()
    filtered_df = filtered_df.rename(
        columns={
            "start_time": "start",
            "stop_time": "end",
        }
    )

    # Drop string columns that cause HDF5 serialization issues
    for col in list(filtered_df.columns):
        if filtered_df[col].dtype == "object":
            first_val = filtered_df[col].iloc[0] if len(filtered_df) > 0 else None
            if isinstance(first_val, str):
                filtered_df = filtered_df.drop(columns=[col])

    trials = Interval.from_dataframe(filtered_df)
    trials.is_valid = np.ones(len(trials), dtype=bool)

    return trials


def assign_temporal_split(
    data: Data,
    recording_date: datetime.datetime,
    trials: Interval,
) -> None:
    """Assign train/valid split based on recording date.

    Sessions before SPLIT_CUTOFF_DATE go to train, after to valid.

    Args:
        data: Data object to modify.
        recording_date: Session recording date.
        trials: Trial intervals to use as domain.
    """
    valid_trials = trials.select_by_mask(trials.is_valid)
    empty_interval = Interval(start=np.array([]), end=np.array([]))

    if recording_date < SPLIT_CUTOFF_DATE:
        data.set_train_domain(valid_trials)
        data.set_valid_domain(empty_interval)
    else:
        data.set_train_domain(empty_interval)
        data.set_valid_domain(valid_trials)

    # No test set for foundation model training
    data.set_test_domain(empty_interval)


def assign_within_session_split(data: Data, trials: Interval) -> None:
    """Assign train/valid split within session (matching paper's approach).

    First 300 trials go to train, remaining (up to 75) go to valid.
    This matches the split used in Temmar et al. (2025) for BCI decoder evaluation.

    Args:
        data: Data object to modify.
        trials: Trial intervals to split.
    """
    n_trials = len(trials)
    empty_interval = Interval(start=np.array([]), end=np.array([]))

    # Create train/valid masks
    train_cutoff = min(WITHIN_SESSION_TRAIN_TRIALS, n_trials)
    valid_end = min(train_cutoff + WITHIN_SESSION_VALID_TRIALS, n_trials)

    train_mask = np.zeros(n_trials, dtype=bool)
    train_mask[:train_cutoff] = True

    valid_mask = np.zeros(n_trials, dtype=bool)
    valid_mask[train_cutoff:valid_end] = True

    # Apply masks
    train_trials = trials.select_by_mask(train_mask)
    valid_trials = trials.select_by_mask(valid_mask)

    data.set_train_domain(train_trials)
    data.set_valid_domain(valid_trials)
    data.set_test_domain(empty_interval)
