# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "boto3~=1.41.0",
#   "mne~=1.11.0",
# ]
# ///

from argparse import ArgumentParser
from pathlib import Path
import ssl
from urllib.request import urlopen

import mne
import h5py
import pandas as pd
from temporaldata import Data
from brainsets import serialize_fn_map
from brainsets.pipeline import BrainsetPipeline
from brainsets.taxonomy import RecordingTech
from brainsets.utils.s3_utils import get_cached_s3_client, get_object_list
from brainsets.utils.mne_utils import (
    extract_measurement_date,
    extract_eeg_signal,
    extract_channels,
    extract_annotations,
)
from brainsets.descriptions import (
    BrainsetDescription,
    SubjectDescription,
    SessionDescription,
    DeviceDescription,
)

PARADIGM_MAP = {
    # "annotation_code": "Paradigm Name",
    90: "Resting Paradigm",
    91: "Sequence Learning Paradigm",
    92: "Symbol Search Paradigm",
    93: "Surround Suppression Paradigm Block 1",
    94: "Contrast Change Paradigm Block 1",
    95: "Contrast Change Paradigm Block 2",
    96: "Contrast Change Paradigm Block 3",
    97: "Surround Suppression Paradigm Block 2",
    81: "Naturalistic Viewing Paradigm Video 1",
    82: "Naturalistic Viewing Paradigm Video 2",
    83: "Naturalistic Viewing Paradigm Video 3",
    84: "Naturalistic Viewing Paradigm Video 4",
    85: "Naturalistic Viewing Paradigm Video 5",
    86: "Naturalistic Viewing Paradigm Video 6",
}

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")


class Pipeline(BrainsetPipeline):
    brainset_id = "kelly_CMI_developingbrain_2016"
    bucket = "fcp-indi"
    prefix = "data/Projects/EEG_Eyetracking_CMI_data/"
    parser = parser

    def _download_subject_metadata(self) -> Path:
        """Download the MIPDB public metadata CSV if not already cached locally.

        Returns the local path to the downloaded CSV file.
        """
        # the metadata file is not hosted in the same s3 bucket
        local_path = self.raw_dir / "MIPDB_PublicFile.csv"
        metadata_url = "https://fcon_1000.projects.nitrc.org/indi/cmi_eeg/_static/MIPDB_PublicFile.csv"

        if not local_path.exists() or self.args.redownload:
            self.update_status("Downloading subject metadata CSV")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            # NITRC's certificate has a hostname mismatch; skip verification
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urlopen(metadata_url, context=ctx) as response:
                local_path.write_bytes(response.read())
        else:
            self.update_status("Subject metadata CSV already cached")

        return local_path

    def _get_subject_metadata(self, participant_id: str) -> dict:
        """Look up subject metadata (age, sex) from the MIPDB public metadata.

        Args:
            participant_id: Subject identifier (e.g., "A00054400").

        Returns:
            Dict with 'age' and 'sex' keys.
        """
        csv_path = self._download_subject_metadata()
        df = pd.read_csv(csv_path, index_col="ID")

        if participant_id not in df.index:
            self.update_status(
                f"Warning: no metadata found for subject {participant_id}"
            )
            return {"age": 0.0, "sex": 0}

        subject = df.loc[participant_id]

        age = subject.get("Age", None)
        sex = subject.get("Sex", None)
        sex = int(sex) if pd.notna(sex) else 0

        return {"age": age, "sex": sex}

    @classmethod
    def get_manifest(cls, raw_dir: Path, args) -> pd.DataFrame:
        s3 = get_cached_s3_client()

        manifest_rows = []

        rel_keys = get_object_list(bucket=cls.bucket, prefix=cls.prefix, s3_client=s3)

        for rel_key in rel_keys:
            rel_path = Path(rel_key)
            filename = rel_path.name

            if filename.startswith("."):
                continue

            rel_parts = rel_path.parts

            # Pattern should be: participant_id/EEG/raw/raw_format/filename
            if (
                len(rel_parts) >= 4
                and rel_parts[1] == "EEG"
                and rel_parts[2] == "raw"
                and rel_parts[3] == "raw_format"
            ) and filename.endswith(".raw"):
                participant_id = rel_parts[0]
                session_id = f"{Path(filename).stem}"

                manifest_rows.append(
                    {
                        "session_id": session_id,
                        "participant_id": participant_id,
                        "s3_key": cls.prefix + rel_key,
                    }
                )

        manifest = pd.DataFrame(manifest_rows).set_index("session_id")
        return manifest

    def download(self, manifest_item) -> Path:
        self.update_status("DOWNLOADING")
        s3 = get_cached_s3_client()

        s3_key = manifest_item.s3_key

        local_path = self.raw_dir / Path(s3_key).relative_to(self.prefix)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if not local_path.exists() or self.args.redownload:
            self.update_status(f"Downloading: {Path(s3_key).name}")
            s3.download_file(self.bucket, s3_key, str(local_path))
        else:
            self.update_status(f"Skipping download, file exists: {local_path}")

        return local_path

    def process(self, download_output: Path) -> None:
        raw_path = download_output

        self.update_status("PROCESSING")

        recording_id = raw_path.stem

        output_path = self.processed_dir / "EEG" / f"{recording_id}.h5"
        if output_path.exists() and not self.args.reprocess:
            self.update_status(f"Skipping processing, file exists: {output_path}")
            return

        brainset_description = BrainsetDescription(
            id="kelly_CMI_developingbrain_2016",
            origin_version="1.0.0",
            derived_version="1.0.0",
            source="https://fcon_1000.projects.nitrc.org/indi/cmi_eeg/",
            description="The Child Mind Institute - MIPDB provides EEG,"
            "eye-tracking, and behavioral data across multiple paradigms from"
            "126 psychiatric and healthy participants aged 6 - 44 years old.",
        )

        self.update_status("Loading EEG file")
        raw = mne.io.read_raw_egi(str(download_output), preload=True, verbose=True)

        meas_date = extract_measurement_date(raw)

        session_description = SessionDescription(
            id=recording_id, recording_date=meas_date
        )

        device_description = DeviceDescription(
            id=recording_id,
            recording_tech=RecordingTech.SCALP_EEG,
        )

        subject_id = recording_id[:9]
        subject_info = self._get_subject_metadata(subject_id)

        subject_description = SubjectDescription(
            id=subject_id,
            species="HUMAN",
            age=subject_info["age"],
            sex=subject_info["sex"],
        )

        self.update_status("Extracting EEG signal")
        eeg_signal = extract_eeg_signal(raw)
        channels = extract_channels(raw)

        self.update_status("Extracting annotations")
        annotations = extract_annotations(raw)
        if len(annotations) > 0:
            first_code = int(annotations.description[0])
            paradigm_name = PARADIGM_MAP.get(first_code, first_code)

        self.update_status("Creating Data Object")
        data_kwargs = {
            "brainset": brainset_description,
            "subject": subject_description,
            "session": session_description,
            "device": device_description,
            "eeg": eeg_signal,
            "channels": channels,
            "domain": eeg_signal.domain,
        }
        if len(annotations) > 0:
            data_kwargs["paradigm"] = paradigm_name
            data_kwargs["annotations"] = annotations

        data = Data(**data_kwargs)

        self.update_status("Storing processed data to disk")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)
