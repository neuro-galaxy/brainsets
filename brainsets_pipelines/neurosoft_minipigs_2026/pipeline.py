# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne==1.11.0",
#   "mne-bids==0.18",
# ]
# ///

from argparse import ArgumentParser, Namespace

from typing import NamedTuple, Optional

import h5py

import pandas as pd
from mne_bids import read_raw_bids

from pathlib import Path

from brainsets.utils.bids_utils import (
    fetch_ieeg_recordings,
    check_recording_files_exist,
    build_bids_path,
    get_subject_info,
)
from brainsets.utils.mne_utils import (
    extract_measurement_date,
    extract_channels,
    extract_eeg_signal,
)

from temporaldata import Data
from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    DeviceDescription,
    SubjectDescription,
    Species,
)
from brainsets import serialize_fn_map

from brainsets.pipeline import BrainsetPipeline

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")
parser.add_argument(
    "--bids-root", type=Path, required=False, help="Path to BIDS root directory"
)


class Pipeline(BrainsetPipeline):
    brainset_id = "neurosoft_minipigs_2026"
    modality = "ieeg"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir: Path, args: Optional[Namespace]) -> pd.DataFrame:
        """Generates a manifest DataFrame of recordings discovered within a BIDS root directory.

        Args:
            raw_dir: Raw data directory assigned to this brainset
            args: Pipeline-specific arguments parsed from the command line. Expects 'bids_root'
            (path to the BIDS dataset root) as a required argument.

        Returns:
            DataFrame indexed by 'recording_id' with columns:
                - subject_id  : (str) BIDS subject identifier, e.g., 'sub-01'
                - session_id  : (str or None) BIDS session identifier, e.g., 'ses-01', or None if not present
                - task_id     : (str) BIDS task identifier (e.g., 'Sleep')
                - fpath       : (pathlib.Path) Local file full path to the iEEG recording file (e.g., .edf, .vhdr, etc.)
        Raises:
            ValueError: If no iEEG recordings are found within the provided BIDS root directory.
        """

        bids_root = args.bids_root
        if bids_root is None:
            raise ValueError(
                f"{cls.brainset_id.upper()} is a private dataset. "
                "Request access from Neurosoft to download the data and place it in the appropriate BIDS root raw_dir/brainset_id directory. "
                f"Please set the --bids-root argument when running the pipeline, e.g. --bids-root /path/to/raw_dir/{cls.brainset_id}"
            )

        if not bids_root.exists():
            raise FileNotFoundError(
                f"BIDS root directory '{bids_root}' does not exist. "
            )

        cls.bids_root = bids_root

        recordings = fetch_ieeg_recordings(bids_root)
        manifest_list = []
        for rec in recordings:
            manifest_list.append(
                {
                    "recording_id": rec["recording_id"],
                    "subject_id": rec["subject_id"],
                    "session_id": rec["session_id"],
                    "task_id": rec["task_id"],
                    "fpath": rec["ieeg_file"],
                }
            )

        if not manifest_list:
            raise ValueError(f"No iEEG recordings found in BODS root {bids_root}")

        manifest = pd.DataFrame(manifest_list).set_index("recording_id")
        # manifest.to_csv("manifest.csv")
        return manifest

    def download(self, manifest_item):
        """Download data for a single recording from OpenNeuro S3.

        Args:
            manifest_item: A single row of the manifest

        Returns:
            Dictionary with keys:
                - recording_id: Recording identifier
                - subject_id: Subject identifier
                - session_id: Session identifier or None
                - fpath: Local file path where data was downloaded
        """
        self.update_status("DOWNLOADING")

        recording_id = manifest_item.Index
        subject_id = manifest_item.subject_id
        session_id = getattr(manifest_item, "session_id", None)

        target_dir = self.raw_dir

        subject_dir = target_dir / subject_id

        if not getattr(self.args, "redownload", False):
            if check_recording_files_exist(recording_id, subject_dir):
                self.update_status("Already Downloaded")

        return {
            "recording_id": recording_id,
            "subject_id": subject_id,
            "session_id": session_id,
        }

    def _process_common(self, download_output: dict) -> Optional[tuple[Data, Path]]:
        """Process data files and create a Data object.

        This method handles common OpenNeuro processing tasks:
        1. Loads data files using MNE
        2. Extracts metadata (subject, session, device, brainset descriptions)
        3. Extracts signal and channel information
        4. Applies modality-specific channel handling via _build_channels()
        5. Creates a Data object

        Args:
            download_output: Dictionary returned by download()

        Returns:
            Tuple of (Data object, store_path), or None if already processed and skipped
        """
        self.processed_dir.mkdir(exist_ok=True, parents=True)

        recording_id = download_output.get("recording_id")
        subject_id = download_output["subject_id"]

        store_path = self.processed_dir / f"{recording_id}.h5"

        if not getattr(self.args, "reprocess", False):
            if store_path.exists():
                self.update_status("Already Processed")
                return None

        self.update_status(f"Loading {self.modality.upper()} file")
        bids_path = build_bids_path(self.bids_root, recording_id, self.modality)
        raw = read_raw_bids(bids_path, verbose=False)
        # TODO test getting BIDS path from fpath
        # fpath = download_output["fpath"]
        # bids_path = mne_bids.get_bids_path_from_fname(fpath, self.modality)

        self.update_status("Extracting Metadata")
        source = "NeurosoftBioelectronics"
        dataset_description = (
            "This dataset contains electrophysiology data from minipigs undergoing acoustic stimulation at various frequencies. "
            "Each trial consists of 1 second: 0.5 seconds of stimulation followed by 0.5 seconds of rest."
        )

        brainset_description = BrainsetDescription(
            id=self.brainset_id,
            origin_version="0.0.1",
            derived_version="1.0.0",
            source=source,
            description=dataset_description,
        )

        subject_info = get_subject_info(bids_root=self.bids_root, subject_id=subject_id)
        subject_description = SubjectDescription(
            id=subject_id,
            species=Species.UNKNOWN,
            age=subject_info["age"],
            sex=subject_info["sex"],
        )

        meas_date = extract_measurement_date(raw)

        session_description = SessionDescription(
            id=recording_id, recording_date=meas_date
        )

        device_description = DeviceDescription(id=recording_id)

        self.update_status(f"Extracting {self.modality.upper()} Signal")
        signal = extract_eeg_signal(raw)

        self.update_status("Building Channels")
        channels = extract_channels(raw)

        self.update_status("Creating Data Object")
        data_kwargs = {
            "brainset": brainset_description,
            "subject": subject_description,
            "session": session_description,
            "device": device_description,
            "channels": channels,
            "domain": signal.domain,
        }
        data_kwargs[self.modality] = signal

        data = Data(**data_kwargs)

        return data, store_path

    def process(self, download_output: dict) -> None:
        """Process and save the dataset.

        Default implementation calls _process_common() and saves the result.
        Subclasses can override to add dataset-specific processing.

        Args:
            download_output: Dictionary returned by download()
        """
        result = self._process_common(download_output)

        if result is None:
            return

        data, store_path = result

        self.update_status("Storing")
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)
