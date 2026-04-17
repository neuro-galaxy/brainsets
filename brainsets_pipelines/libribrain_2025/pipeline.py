# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne~=1.11.0",
#   "huggingface-hub>=0.20.0",
#   "pnpl~=0.0.8",
#   "temporaldata@git+https://github.com/neuro-galaxy/temporaldata@main",
# ]
# ///

from argparse import ArgumentParser
from pathlib import Path

import h5py
import json
import logging
import mne
import numpy as np
import pandas as pd

from huggingface_hub import HfApi, hf_hub_download

from brainsets import serialize_fn_map
from brainsets.descriptions import (
    BrainsetDescription,
    SubjectDescription,
    SessionDescription,
    DeviceDescription,
)
from brainsets.taxonomy import RecordingTech, Species
from brainsets.pipeline import BrainsetPipeline
from brainsets.utils.mne_utils import (
    extract_measurement_date,
    extract_signal,
    extract_channels,
)
from temporaldata import Data, Interval

logging.basicConfig(level=logging.INFO)

REPO_ID = "pnpl/LibriBrain"
SUBDATASETS = [f"Sherlock{i}" for i in range(1, 8)]
# TODO: could add a validation for the subdataset, and accept sherlock1 or 1 or sherlock_1, etc
MEG_SUFFIX = "_meg.fif"

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")
parser.add_argument(
    "--subdataset",
    type=str,
    nargs="+",
    choices=SUBDATASETS + ["all"],
    default=["all"],
    help="Which sub-dataset(s) to process (e.g. Sherlock1 Sherlock3). Default: all.",
)


def _resolve_subdatasets(subdataset_arg: list[str]) -> list[str]:
    if "all" in subdataset_arg:
        return SUBDATASETS
    return subdataset_arg


class Pipeline(BrainsetPipeline):
    brainset_id = "libribrain_2025"
    origin_version = "1.0.0"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir: Path, args) -> pd.DataFrame:
        api = HfApi()
        subdatasets = _resolve_subdatasets(args.subdataset)

        manifest_rows = []
        for subdataset in subdatasets:
            all_paths = [
                info.rfilename
                for info in api.list_repo_tree(
                    REPO_ID,
                    path_in_repo=subdataset,
                    repo_type="dataset",
                    recursive=True,
                )
                if hasattr(info, "rfilename")
            ]

            # Select only the raw meg data (excluding preprocessed and serialized data)
            raw_meg_prefix = f"{subdataset}/sub-"
            fif_paths = [
                p
                for p in all_paths
                if p.startswith(raw_meg_prefix)
                and "/ses-" in p
                and "/meg/" in p
                and p.endswith(MEG_SUFFIX)
            ]

            for fif_path in fif_paths:
                prefix = fif_path.removesuffix(MEG_SUFFIX)
                stem = Path(prefix).name
                parts = _parse_bids_prefix(stem)

                manifest_rows.append(
                    {
                        "recording_id": f"{subdataset}_{stem}",
                        "subdataset": subdataset,
                        "subject_id": parts["sub"],
                        "session_id": parts["ses"],
                        "task": parts["task"],
                        "run": parts.get("run", "1"),
                        "hf_prefix": prefix,
                    }
                )

        if not manifest_rows:
            raise ValueError(f"No MEG recordings found for subdatasets: {subdatasets}")

        manifest = pd.DataFrame(manifest_rows).set_index("recording_id")
        logging.info(f"Manifest contains {len(manifest)} recordings")
        return manifest

    def download(self, manifest_item) -> dict[str, Path]:
        self.update_status("DOWNLOADING")

        hf_prefix = manifest_item.hf_prefix
        suffixes = ["_meg.fif", "_meg.json", "_channels.tsv", "_events.tsv"]

        local_paths = {}
        for suffix in suffixes:
            remote_path = hf_prefix + suffix
            key = suffix.lstrip("_").replace(".", "_")  # e.g. meg_fif, meg_json

            local_path = self.raw_dir / remote_path
            if local_path.exists() and not self.args.redownload:
                logging.info(f"Skipping download, file exists: {local_path}")
                local_paths[key] = local_path
                continue

            logging.info(f"Downloading: {remote_path}")
            downloaded = hf_hub_download(
                repo_id=REPO_ID,
                filename=remote_path,
                repo_type="dataset",
                local_dir=str(self.raw_dir),
            )
            local_paths[key] = Path(downloaded)

        return local_paths

    def process(self, download_output: dict[str, Path]) -> None:
        self.update_status("PROCESSING")

        fif_path = download_output["meg_fif"]
        json_path = download_output["meg_json"]
        events_path = download_output["events_tsv"]

        stem = fif_path.stem.removesuffix("_meg")
        parts = _parse_bids_prefix(stem)
        subdataset = parts["task"]

        output_path = self.processed_dir / f"{subdataset}_{stem}.h5"
        if output_path.exists() and not self.args.reprocess:
            self.update_status("Skipped Processing")
            logging.info(f"Skipping processing, file exists: {output_path}")
            return

        brainset_description = BrainsetDescription(
            id="libribrain_2025",
            origin_version="1.0.0",
            derived_version="1.0.0",
            source="https://huggingface.co/datasets/pnpl/LibriBrain",
            description="LibriBrain MEG dataset: MEG recordings of a single subject "
            "listening to Sherlock Holmes audiobooks, with word- and phoneme-level annotations.",
        )

        self.update_status("Loading FIF")
        raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)

        self.update_status("Reading sidecar JSON")
        sidecar = _load_json_sidecar(json_path)

        subject_id = f"sub-{parts['sub']}"
        subject = SubjectDescription(
            id=subject_id,
            species=Species.HOMO_SAPIENS,
        )

        recording_date = extract_measurement_date(raw)

        session_description = SessionDescription(
            id=f"{subject_id}_ses-{parts['ses']}_task-{parts['task']}_run-{parts.get('run', '1')}",
            recording_date=recording_date,
        )

        # TODO: add device type or model
        device_description = DeviceDescription(
            id=sidecar.get("InstitutionName", "unknown_meg_device"),
            recording_tech=RecordingTech.MEG,
        )

        self.update_status("Extracting Signals")
        signals = extract_signal(raw)

        self.update_status("Extracting Channels")
        channels = extract_channels(raw)
        # FIXME Reverif MEG channel types

        self.update_status("Extracting Events")
        events = _extract_events(events_path)
        # TODO add all annotations of an event, in addition to the onset and duration
        # TODO add event type to the events object (word, phoneme, silence)

        data = Data(
            brainset=brainset_description,
            subject=subject,
            session=session_description,
            device=device_description,
            meg=signals,
            channels=channels,
            events=events,
            domain=signals.domain,
        )
        # TODO: add splits once the splitting strategy is defined
        # splits = _create_splits(...)
        # data.splits = splits

        self.update_status("Storing")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

        logging.info(f"Saved processed data to: {output_path}")


def _parse_bids_prefix(stem: str) -> dict[str, str]:
    """Parse BIDS key-value pairs from a filename stem like 'sub-0_ses-1_task-Sherlock2_run-1'."""
    parts = {}
    for token in stem.split("_"):
        if "-" in token:
            key, value = token.split("-", 1)
            parts[key] = value
    return parts


def _load_json_sidecar(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_events(events_path: Path) -> Interval:
    """Extract events from a BIDS _events.tsv file into an Interval object."""
    df = pd.read_csv(events_path, sep="\t")

    if "timemeg" not in df.columns or "duration" not in df.columns:
        raise ValueError(
            f"Events file missing required 'timemeg'/'duration' columns: {events_path}"
        )

    starts = df["timemeg"].values.astype(np.float64)
    durations = df["duration"].values.astype(np.float64)
    ends = starts + durations

    fields = {
        "start": starts,
        "end": ends,
    }

    if "trial_type" in df.columns:
        fields["trial_type"] = df["trial_type"].values.astype("U")
    if "value" in df.columns:
        fields["value"] = df["value"].values.astype(np.float64)
    if "sample" in df.columns:
        fields["sample"] = df["sample"].values.astype(np.int64)

    return Interval(**fields)


def _create_splits():
    raise NotImplementedError(
        "Splits for libribrain_2025 are not yet defined. "
        "This will be implemented in a future iteration."
    )
