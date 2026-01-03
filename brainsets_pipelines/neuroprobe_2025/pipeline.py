# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "neuroprobe==0.1.5",
#   "requests==2.32.5",
#   "bs4==0.0.2",
# ]
# ///

from argparse import ArgumentParser, BooleanOptionalAction
import logging
from pathlib import Path
import shutil
import urllib.request
import zipfile

import pandas as pd
import os 

# Allow environment override for BrainTreeBank root directory
os.environ.setdefault(
    "ROOT_DIR_BRAINTREEBANK",
    "/home/geeling/Projects/tb_buildathon/data/raw_2/neuroprobe_2025"
)

from brainsets.pipeline import BrainsetPipeline
from brainsets_pipelines.neuroprobe_2025.prepare_data import process_file

BASE_URL = "https://braintreebank.dev"
PIPELINE_DIR = Path(__file__).resolve().parent

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")
parser.add_argument(
    "--hrefs-file",
    default="download_hrefs.txt",
    help="Which href list to use (relative to pipeline dir).",
)
parser.add_argument(
    "--lite",
    action=BooleanOptionalAction,
    default=True,
    help="Use Neuroprobe-Lite trial/electrode subsets (default: true).",
)
parser.add_argument(
    "--nano",
    action=BooleanOptionalAction,
    default=False,
    help="Use Neuroprobe-Nano trial/electrode subsets (default: false).",
)
parser.add_argument(
    "--no_splits",
    action="store_true",
    help="Skip trialization/split extraction; write only processed data.",
)


def _read_hrefs(hrefs_path: Path) -> list[str]:
    hrefs = []
    with hrefs_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            href = line.strip()
            if not href or href.startswith("#"):
                continue
            hrefs.append(href)
    return hrefs


def _download_file(url: str, dest: Path, *, overwrite: bool) -> None:
    if dest.exists() and not overwrite:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _download_and_extract(raw_dir: Path, href: str, *, overwrite: bool) -> Path:
    url = f"{BASE_URL}/{href}"
    basename = Path(href).name
    zip_path = raw_dir / basename

    if href.endswith(".zip"):
        extracted_path = raw_dir / Path(href).stem
        if extracted_path.exists() and not overwrite:
            return extracted_path

        _download_file(url, zip_path, overwrite=overwrite)
        with zipfile.ZipFile(zip_path, "r") as zip_handle:
            zip_handle.extractall(raw_dir)
        zip_path.unlink()
        return extracted_path

    extracted_path = raw_dir / basename
    _download_file(url, extracted_path, overwrite=overwrite)
    return extracted_path


def _ensure_common_assets(raw_dir: Path, hrefs: list[str], *, overwrite: bool) -> None:
    for href in hrefs:
        if href.endswith(".h5.zip"):
            continue
        _download_and_extract(raw_dir, href, overwrite=overwrite)


class Pipeline(BrainsetPipeline):
    brainset_id = "neuroprobe_2025"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir, args):
        raw_dir.mkdir(exist_ok=True, parents=True)
        hrefs_path = PIPELINE_DIR / (
            args.hrefs_file if args is not None else "download_hrefs.txt"
        )
        hrefs = _read_hrefs(hrefs_path)

        _ensure_common_assets(raw_dir, hrefs, overwrite=bool(args and args.redownload))

        manifest_list = []
        for href in hrefs:
            if not href.endswith(".h5.zip"):
                continue
            h5_relpath = href[:-4]
            manifest_list.append(
                {
                    "href": href,
                    "h5_relpath": h5_relpath,
                }
            )

        manifest = pd.DataFrame(manifest_list).set_index("h5_relpath")
        return manifest

    def download(self, manifest_item):
        self.update_status("DOWNLOADING")
        extracted_path = _download_and_extract(
            self.raw_dir,
            manifest_item.href,
            overwrite=bool(self.args and self.args.redownload),
        )
        return extracted_path

    def process(self, fpath):
        self.processed_dir.mkdir(exist_ok=True, parents=True)
        output_path = self.processed_dir / Path(fpath).name
        if output_path.exists() and not (self.args and self.args.reprocess):
            logging.info(f"Skipping processing for {output_path} because it exists")
            self.update_status("Skipped Processing")
            return
        
        logging.info(f"Processing {fpath} to {self.processed_dir}")
        logging.info(f"  lite={self.args.lite if self.args is not None else True}, "
                     f"  nano={self.args.nano if self.args is not None else False}, "
                     f"  no_splits={self.args.no_splits if self.args is not None else False}")

        process_file(
            str(fpath),
            str(self.processed_dir),
            lite=self.args.lite if self.args is not None else True,
            nano=self.args.nano if self.args is not None else False,
            no_splits=self.args.no_splits if self.args is not None else False,
            root_dir=str(self.raw_dir),
        )
