# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne~=1.11.0",
#   "dandi==0.61.2",
#   "pynwb==3.1.3",
# ]
# ///

from argparse import ArgumentParser, Namespace
import datetime
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from pynwb import NWBFile, NWBHDF5IO
from temporaldata import Data, Interval, RegularTimeSeries
from tqdm.auto import tqdm

from brainsets import serialize_fn_map
from brainsets.descriptions import (
    BrainsetDescription,
    DeviceDescription,
    SessionDescription,
)
from brainsets.pipeline import BrainsetPipeline
from brainsets.taxonomy import Hemisphere
from brainsets.taxonomy import RecordingTech, Sex, Species, Task
from brainsets.utils.dandi_utils import (
    download_file,
    extract_ecog_from_nwb,
    extract_subject_from_nwb,
    get_nwb_asset_list,
)
from brainsets.utils.split import (
    generate_subject_kfold_assignment,
    generate_trial_folds_by_task,
)

logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")
parser.add_argument(
    "--resample_rate",
    type=float,
    default=500.0,
    help="ECoG resample rate in Hz",
)
parser.add_argument(
    "--filter_ecog",
    action="store_true",
    help="Apply 70–150 Hz bandpass and Hilbert envelope",
)

# Some sessions contain very few "Other activity" trials (<2) so we can't use them for splitting.
# We exclude them from the AllActiveBehavior task.
BEHAVIOR_TASK_CONFIGS = {
    "AllActiveBehavior": ["Eat", "Talk", "TV", "Computer/phone"],
}


class Pipeline(BrainsetPipeline):
    brainset_id = "peterson_brunton_pose_trajectory_2022"
    dandiset_id = "DANDI:000055/0.220127.0436"
    parser = parser

    @classmethod
    def get_manifest(
        cls,
        raw_dir: Path,
        args: Optional[Namespace],
    ) -> pd.DataFrame:
        # Old code excluded: sub-06_ses-4, sub-06_ses-5, sub-06_ses-6, sub-06_ses-7 (minimal active behavior)
        asset_list = get_nwb_asset_list(cls.dandiset_id)
        rows = []
        for asset in asset_list:
            path = asset.path
            if not path.endswith(".nwb"):
                continue
            parts = path.replace("\\", "/").split("/")
            sub_part = next(
                (p for p in parts if p.startswith("sub-") and "_ses-" in p), None
            )
            if sub_part is None:
                sub_part = path
            subject = sub_part.split("sub-")[1].split("_")[0].strip()
            session = ""
            if "_ses-" in sub_part:
                session = sub_part.split("_ses-")[1].split("_")[0].split(".")[0].strip()
            session_id = f"sub-{subject}_ses-{session}"
            rows.append(
                {
                    "session_id": session_id,
                    "path": path,
                    "url": asset.download_url,
                    "subject_id": f"sub-{subject}",
                    "session_num": session,
                }
            )
        manifest = pd.DataFrame(rows).set_index("session_id")
        return manifest

    def download(self, manifest_item: pd.Series) -> Path:
        self.update_status("DOWNLOADING")
        self.raw_dir.mkdir(exist_ok=True, parents=True)
        fpath = download_file(
            manifest_item.path,
            manifest_item.url,
            self.raw_dir,
            overwrite=getattr(self.args, "redownload", False),
        )
        return fpath

    def _generate_splits(
        self,
        subject_id: str,
        session_id: str,
        behavior_trials: Interval,
    ) -> Data:
        subject_assign = generate_subject_kfold_assignment(
            subject_id, n_folds=3, val_ratio=0.2, seed=42
        )
        session_assign = generate_subject_kfold_assignment(
            subject_id, session_id=session_id, n_folds=3, val_ratio=0.2, seed=42
        )
        behavior_splits = {}
        # The trials main contain multiple behavior labels.
        # If a trial contains Eat and any other behavior label, we want to consider it as just Eat.
        # For example, if a trial contains "Eat, Talk" or "Eat, TV", we want to consider it as just "Eat".
        for index, label in enumerate(behavior_trials.behavior_labels):
            if "Eat" in label:
                behavior_trials.behavior_labels[index] = "Eat"
            elif "Talk" in label:
                behavior_trials.behavior_labels[index] = "Talk"
            elif "TV" in label:
                behavior_trials.behavior_labels[index] = "TV"
            elif "Computer/phone" in label:
                behavior_trials.behavior_labels[index] = "Computer/phone"
            elif "Other activity" in label:
                behavior_trials.behavior_labels[index] = "Other activity"

        if len(behavior_trials) > 0 and hasattr(behavior_trials, "behavior_labels"):
            behavior_splits = generate_trial_folds_by_task(
                behavior_trials,
                BEHAVIOR_TASK_CONFIGS,
                "behavior_labels",
                n_folds=3,
                val_ratio=0.2,
                seed=42,
            )
        return Data(
            **subject_assign,
            **session_assign,
            **behavior_splits,
            domain=behavior_trials,
        )

    def process(self, fpath: Path) -> None:
        self.update_status("Loading NWB")
        io = NWBHDF5IO(str(fpath), "r")
        nwbfile = io.read()

        self.processed_dir.mkdir(exist_ok=True, parents=True)

        participants_path = Path(__file__).parent / "participants.json"
        participants = {}
        if participants_path.exists():
            with open(participants_path) as f:
                participants = json.load(f)
        par = str(nwbfile.subject.subject_id).replace("sub-", "").strip()
        part = participants.get(par)
        if part is not None:
            subject = extract_subject_from_nwb(
                nwbfile,
                subject_id=f"AJILE12_P{par}",
                species=Species.HOMO_SAPIENS,
                sex=Sex.from_string(part["sex"]),
                age=part["age"] * 365.25,
            )
        else:
            subject = extract_subject_from_nwb(nwbfile)

        recording_date = nwbfile.session_start_time.strftime("%Y%m%d")
        subject_num = (
            subject.id.replace("AJILE12_P", "").replace("sub-", "").strip()
            or subject.id
        )
        stem = Path(fpath).stem
        if "_ses-" in stem:
            session_num = stem.split("_ses-")[1].split("_")[0]
        else:
            session_num = "0"
        session_id = f"AJILE12_P{subject_num}_{recording_date}_ses{session_num}_pose_trajectories"

        output_path = self.processed_dir / f"{session_id}.h5"
        if output_path.exists() and not getattr(self.args, "reprocess", False):
            io.close()
            self.update_status("Skipped Processing")
            return

        brainset_description = BrainsetDescription(
            id=self.brainset_id,
            origin_version="dandi/000055/0.220127.0436",
            derived_version="1.0.0",
            source="https://dandiarchive.org/dandiset/000055",
            description="AJILE12: ECoG and upper body pose trajectories from 12 human subjects during naturalistic movements.",
        )

        self.update_status("Extracting ECoG")
        resample_rate = getattr(self.args, "resample_rate")
        subject_hemisphere = resolve_hemisphere_ajile(
            part.get("hemi") if part else None, nwbfile
        )
        ecog, channels = extract_ecog_from_nwb(
            nwbfile, subject_hemisphere=subject_hemisphere
        )
        self.update_status("Resampling ECoG")
        if resample_rate < ecog.sampling_rate:
            ecog = resample_ecog_ajile(ecog, resample_rate, chunk_duration_sec=60.0)
        else:
            self.update_status("ECoG already at desired sampling rate")

        self.update_status("Extracting pose trajectories")
        pose = ajile_extract_pose_from_nwb(nwbfile)

        # self.update_status("Extracting behavior intervals")
        behavior_trials = ajile_extract_behavior_intervals_from_nwb(nwbfile)

        self.update_status("Generating splits")
        splits = self._generate_splits(subject.id, session_id, behavior_trials)

        session_description = SessionDescription(
            id=session_id,
            recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
            task=Task.FREE_BEHAVIOR,
        )
        device_id = f"{subject.id}_{recording_date}"
        device_description = DeviceDescription(
            id=device_id,
            recording_tech=RecordingTech.ECOG_ARRAY_ECOGS,
        )

        data = Data(
            brainset=brainset_description,
            subject=subject,
            session=session_description,
            device=device_description,
            ecog=ecog,
            channels=channels,
            pose=pose,
            behavior_trials=behavior_trials,
            splits=splits,
            domain=ecog.domain,
        )

        self.update_status("Storing")
        with h5py.File(output_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)
        logging.info("Saved processed data to %s", output_path)
        io.close()


def ajile_surface_mask_from_group_names(group_names: np.ndarray) -> np.ndarray:
    """Surface vs depth ECoG electrodes (AJILE12 / Peterson-Brunton convention)."""
    group_names = np.asarray(group_names).astype(str)
    has_phd = np.any(np.char.upper(group_names) == "PHD")
    is_surf = []
    for label in group_names:
        g = label.lower()
        if "grid" in g:
            is_surf.append(True)
        elif g in ("mhd", "latd", "lmtd", "ltpd"):
            is_surf.append(True)
        elif g == "ahd" and not has_phd:
            is_surf.append(True)
        elif "d" in g:
            is_surf.append(False)
        else:
            is_surf.append(True)
    return np.array(is_surf, dtype=bool)


def ajile_hemisphere_from_reach_events(nwbfile: NWBFile) -> Optional[Hemisphere]:
    behavior = nwbfile.processing.get("behavior") if nwbfile.processing else None
    if behavior is None:
        return None
    reach = behavior.data_interfaces.get("ReachEvents")
    if reach is None or not getattr(reach, "description", None):
        return None
    desc = reach.description[:]
    if desc is None or len(desc) == 0:
        return None
    c_wrist = str(desc[0]).strip().lower()
    if c_wrist == "r":
        return Hemisphere.LEFT
    if c_wrist == "l":
        return Hemisphere.RIGHT
    return None


def resolve_hemisphere_ajile(
    subject_hemisphere: Optional[Union[Hemisphere, str]], nwbfile: NWBFile
) -> Hemisphere:
    if subject_hemisphere is not None:
        if isinstance(subject_hemisphere, Hemisphere):
            return subject_hemisphere
        s = subject_hemisphere.strip().upper()
        if s == "L":
            return Hemisphere.LEFT
        if s == "R":
            return Hemisphere.RIGHT
        try:
            return Hemisphere.from_string(subject_hemisphere)
        except ValueError:
            pass
    inferred = ajile_hemisphere_from_reach_events(nwbfile)
    return inferred if inferred is not None else Hemisphere.UNKNOWN


def resample_ecog_ajile(
    ecog_rts: RegularTimeSeries,
    resample_rate_hz: float,
    chunk_duration_sec: float = 60.0,
) -> RegularTimeSeries:
    try:
        from scipy import signal
    except ImportError:
        raise ImportError("resample_ecog_ajile requires scipy")

    current_rate = float(ecog_rts.sampling_rate)
    data = np.asarray(ecog_rts.ecogs[:], dtype=np.float64)
    n_samples, _ = data.shape
    downsample_factor = int(current_rate / resample_rate_hz)
    if downsample_factor < 1:
        raise ValueError("resample_rate_hz must be <= native rate for decimation")
    chunk_samples = int(chunk_duration_sec * current_rate)

    downsampled_chunks = []
    for start_idx in tqdm(
        range(0, n_samples, chunk_samples), desc="Resampling ECoG chunks"
    ):
        end_idx = min(start_idx + chunk_samples, n_samples)
        chunk = data[start_idx:end_idx, :]
        if downsample_factor > 1:
            chunk = signal.decimate(
                chunk, downsample_factor, axis=0, ftype="iir", zero_phase=True
            )
        downsampled_chunks.append(chunk)

    data_out = np.concatenate(downsampled_chunks, axis=0)
    n_out = data_out.shape[0]
    times_out = np.arange(n_out) / resample_rate_hz
    domain = Interval(start=np.array([times_out[0]]), end=np.array([times_out[-1]]))
    return RegularTimeSeries(
        ecogs=data_out,
        sampling_rate=resample_rate_hz,
        domain=domain,
    )


def ajile_extract_pose_from_nwb(
    nwbfile: NWBFile,
) -> Tuple[Interval, RegularTimeSeries]:
    r_wrist = nwbfile.processing["behavior"].data_interfaces["Position"]["R_Wrist"]
    l_wrist = nwbfile.processing["behavior"].data_interfaces["Position"]["L_Wrist"]
    l_ear = nwbfile.processing["behavior"].data_interfaces["Position"]["L_Ear"]
    l_elbow = nwbfile.processing["behavior"].data_interfaces["Position"]["L_Elbow"]
    l_shoulder = nwbfile.processing["behavior"].data_interfaces["Position"][
        "L_Shoulder"
    ]
    nose = nwbfile.processing["behavior"].data_interfaces["Position"]["Nose"]
    r_ear = nwbfile.processing["behavior"].data_interfaces["Position"]["R_Ear"]
    r_elbow = nwbfile.processing["behavior"].data_interfaces["Position"]["R_Elbow"]
    r_shoulder = nwbfile.processing["behavior"].data_interfaces["Position"][
        "R_Shoulder"
    ]

    behavior_sampling_rate = r_wrist.rate

    pose_trajectories = RegularTimeSeries(
        r_wrist=r_wrist.data[:],
        l_wrist=l_wrist.data[:],
        l_ear=l_ear.data[:],
        l_elbow=l_elbow.data[:],
        l_shoulder=l_shoulder.data[:],
        nose=nose.data[:],
        r_ear=r_ear.data[:],
        r_elbow=r_elbow.data[:],
        r_shoulder=r_shoulder.data[:],
        sampling_rate=behavior_sampling_rate,
        domain=Interval(
            start=np.array([0.0]),
            end=np.array([(len(r_wrist.data) - 1) / behavior_sampling_rate]),
        ),
    )

    return pose_trajectories


def ajile_extract_behavior_intervals_from_nwb(nwbfile: NWBFile) -> Interval:
    if "epochs" not in nwbfile.intervals:
        return Interval(
            start=np.array([]),
            end=np.array([]),
            behavior_labels=np.array([]),
            active=np.array([]),
        )

    coarse_behaviors = nwbfile.intervals["epochs"]
    coarse_behaviors_labels = coarse_behaviors.labels.data[:].tolist()

    active_events = set(["Eat", "Talk", "TV", "Computer/phone", "Other activity"])
    labels_arr = np.array(coarse_behaviors_labels, dtype=object)

    # Vectorized: mark True if all split events for a row are in active_events
    def is_all_active(label):
        return all(event in active_events for event in label.split(", "))

    active_event_mask = np.vectorize(is_all_active)(labels_arr)

    behavior_trials = Interval(
        start=np.round(coarse_behaviors.start_time.data[:], 3),
        end=np.round(coarse_behaviors.stop_time.data[:], 3),
        behavior_labels=coarse_behaviors.labels.data[:],
        active=active_event_mask,
    )

    unique_behavior_labels = np.unique(coarse_behaviors_labels)
    unique_active_behavior_mask = np.zeros(unique_behavior_labels.shape[0]).astype(bool)
    for k in range(unique_behavior_labels.shape[0]):
        is_active = True
        for single_event in unique_behavior_labels[k].split(", "):
            if single_event not in active_events:
                is_active = False
        unique_active_behavior_mask[k] = is_active

    print(
        "unique behavior labels",
        unique_behavior_labels,
        "in which",
        unique_behavior_labels[unique_active_behavior_mask],
        "are active.",
    )

    return behavior_trials
