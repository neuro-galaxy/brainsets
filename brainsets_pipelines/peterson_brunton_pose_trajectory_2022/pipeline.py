# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne~=1.11.0",
#   "dandi==0.61.2",
#   "pynwb==3.1.3",
# ]
# ///

import datetime
import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

import h5py
import pandas as pd
from pynwb import NWBHDF5IO
from temporaldata import Data, Interval

from brainsets import serialize_fn_map
from brainsets.descriptions import (
    BrainsetDescription,
    DeviceDescription,
    SessionDescription,
)
from brainsets.pipeline import BrainsetPipeline
from brainsets.taxonomy import RecordingTech, Sex, Species, Task
from brainsets.utils.dandi_utils import (
    download_file,
    extract_subject_from_nwb,
    get_nwb_asset_list,
)
from brainsets.utils.dandi_utils import (
    extract_behavior_intervals_from_nwb,
    extract_ecog_from_nwb,
    extract_pose_from_nwb,
    extract_reach_events_from_nwb,
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
    default=250.0,
    help="ECoG resample rate in Hz",
)
parser.add_argument(
    "--filter_ecog",
    action="store_true",
    help="Apply 70–150 Hz bandpass and Hilbert envelope",
)

BEHAVIOR_TASK_CONFIGS = {
    "AllActiveBehavior": ["Eat", "Talk", "TV", "Computer/phone", "Other activity"],
    "EatVsOther": ["Eat", "Talk", "TV", "Computer/phone", "Other activity"],
    "TalkVsOther": ["Eat", "Talk", "TV", "Computer/phone", "Other activity"],
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
        resample_rate = getattr(self.args, "resample_rate", 250.0)
        filter_ecog = getattr(self.args, "filter_ecog", False)
        subject_hemisphere = part.get("hemi") if part else None
        ecog, channels = extract_ecog_from_nwb(
            nwbfile,
            resample_rate=resample_rate,
            apply_filter=filter_ecog,
            subject_hemisphere=subject_hemisphere,
        )

        self.update_status("Extracting pose")
        wrist = extract_pose_from_nwb(nwbfile)

        self.update_status("Extracting behavior intervals")
        behavior_trials = extract_behavior_intervals_from_nwb(nwbfile)
        _reach_ts, _reach_hemi = extract_reach_events_from_nwb(nwbfile)

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
            wrist=wrist,
            behavior_trials=behavior_trials,
            splits=splits,
            domain=ecog.domain,
        )

        self.update_status("Storing")
        with h5py.File(output_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)
        logging.info("Saved processed data to %s", output_path)
        io.close()
