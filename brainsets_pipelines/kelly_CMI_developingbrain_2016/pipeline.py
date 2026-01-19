# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "boto3~=1.41.0",
#   "pandas~=2.0.0",
#   "h5py~=3.10.0",
# ]
# ///

from argparse import ArgumentParser
from pathlib import Path

import h5py
import logging
import pandas as pd

from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    DeviceDescription,
)

from brainsets.taxonomy import RecordingTech, Species, Sex

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
                ):
                    participant_id = rel_parts[0]
                    session_id = filename.replace(".", "_")
                    session_id = f"eeg_{session_id}"

                    manifest_rows.append(
                        {
                            "session_id": session_id,
                            "participant_id": participant_id,
                            "s3_key": key,
                        }
                    )

        manifest = pd.DataFrame(manifest_rows).set_index("session_id")
        return manifest
