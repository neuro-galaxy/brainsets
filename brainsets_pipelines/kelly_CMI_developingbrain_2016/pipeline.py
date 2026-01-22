# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "boto3~=1.41.0",
#   "pandas~=2.0.0",
#   "h5py~=3.10.0",
#   "mne~=1.11.0",
# ]
# ///

from argparse import ArgumentParser
from pathlib import Path

import mne
import h5py
import logging
import datetime
import pandas as pd
import numpy as np
from temporaldata import Data, RegularTimeSeries, Interval, ArrayDict

from brainsets.descriptions import (
    BrainsetDescription,
    SubjectDescription,
    SessionDescription,
    DeviceDescription,
)

from brainsets.taxonomy import RecordingTech, Species, Sex

from brainsets import serialize_fn_map
from brainsets.pipeline import BrainsetPipeline
from brainsets.utils.s3_utils import get_s3_client_for_download

logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")


class Pipeline(BrainsetPipeline):
    brainset_id = "kelly_CMI_developingbrain_2016"
    bucket = "fcp-indi"
    prefix = "data/Projects/EEG_Eyetracking_CMI_data/"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir: Path, args) -> pd.DataFrame:
        s3 = get_s3_client_for_download()

        manifest_rows = []

        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=cls.bucket, Prefix=cls.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue

                filename = Path(key).name
                if filename.startswith("."):
                    continue

                rel_path = Path(key).relative_to(cls.prefix)
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
                            "s3_key": key,
                        }
                    )

        manifest = pd.DataFrame(manifest_rows).set_index("session_id")
        return manifest

    def download(self, manifest_item) -> Path:
        self.update_status("DOWNLOADING")
        s3 = get_s3_client_for_download()

        s3_key = manifest_item.s3_key

        local_path = self.raw_dir / Path(s3_key).relative_to(self.prefix)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if not local_path.exists() or self.args.redownload:
            logging.info(f"Downloading: {Path(s3_key).name}")
            s3.download_file(self.bucket, s3_key, str(local_path))
        else:
            logging.info(f"Skipping download, file exists: {local_path}")

        return local_path

    def process(self, download_output: Path) -> None:
        raw_path = download_output

        self.update_status("PROCESSING")

        recording_id = raw_path.stem

        output_path = self.processed_dir / "EEG" / f"{recording_id}.h5"
        if output_path.exists() and not self.args.reprocess:
            self.update_status("Skipped Processing")
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

        # TODO: add the data_extraction utils back in once they are merged
        # meas_date = extract_meas_date(raw)
        meas_date = raw.info["meas_date"]
        if meas_date is None:
            logging.warning(f"No measurement date found for {download_output}")
            meas_date = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)

        # session_description = extract_session_description(
        #     session_id=recording_id, recording_date=meas_date
        # )
        session_description = SessionDescription(
            id=recording_id, recording_date=meas_date
        )

        # device_description = extract_device_description(device_id=recording_id)
        device_description = DeviceDescription(id=recording_id)

        # subject_description = extract_subject_description(subject_id=subject_id)
        subject_description = SubjectDescription(
            id=recording_id[:9],
            species=Species.HOMO_SAPIENS,
            age=0.0,
            sex=Sex.UNKNOWN,
        )

        self.update_status("Extracting EEG signal")
        # eeg_signal = extract_signal(raw)
        sfreq = raw.info["sfreq"]
        eeg_signal = raw.get_data().T
        eeg_signal = RegularTimeSeries(
            signal=eeg_signal,
            sampling_rate=sfreq,
            domain=Interval(
                start=np.array([0.0]),
                end=np.array([(len(eeg_signal) - 1) / sfreq]),
            ),
        )
        # channels = extract_channels(raw)
        channels = ArrayDict(
            id=np.array(raw.ch_names, dtype="U"),
            types=np.array(raw.get_channel_types(), dtype="U"),
        )

        self.update_status("Creating Data Object")
        data = Data(
            brainset=brainset_description,
            subject=subject_description,
            session=session_description,
            device=device_description,
            eeg=eeg_signal,
            channels=channels,
            domain=eeg_signal.domain,
        )

        self.update_status("Storing processed data to disk")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)
