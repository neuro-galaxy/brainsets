# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne==1.11.0",
#   "mne-bids==0.18",
#   "boto3>=1.42.32",
#   "requests==2.32.5",
# ]
# ///

from argparse import ArgumentParser
import h5py
import pandas as pd
from brainsets import serialize_fn_map
from brainsets.utils.openneuro import OpenNeuroEEGPipeline
from brainsets.utils.openneuro.pipeline import _openneuro_parser
from brainsets.utils.openneuro.dataset import fetch_participants_tsv

RELEASES = {
    1: "ds005505",
    2: "ds005506",
    3: "ds005507",
    4: "ds005508",
    5: "ds005509",
    6: "ds005510",
    7: "ds005511",
    8: "ds005512",
    9: "ds005514",
    10: "ds005515",
    11: "ds005516",
}

MODALITY_CHANNELS = {"EEG": [f"E{i}" for i in range(1, 129)] + ["Cz"]}

parser = ArgumentParser(parents=[_openneuro_parser], add_help=False)
parser.add_argument(
    "--release",
    type=int,
    default=None,
    help="Prepare only this release (1-11). Omit to prepare all releases.",
)


class Pipeline(OpenNeuroEEGPipeline):
    brainset_id = "cmi_hbn_2025"
    dataset_id = "ds005505"
    description = (
        "Healthy Brain Network (HBN) EEG dataset combining all 11 data "
        "releases. Contains recordings from participants performing various "
        "passive and active tasks including resting state, movie watching, "
        "and cognitive tasks."
    )
    MODALITY_CHANNELS = MODALITY_CHANNELS
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir, args):
        if args and args.release is not None:
            if args.release not in RELEASES:
                raise ValueError(
                    f"Invalid choice: release must be one of 1-11, got {args.release}."
                )
            releases = {args.release: RELEASES[args.release]}
        else:
            releases = RELEASES

        all_manifests = []
        for release_id, dataset_id in releases.items():
            cls.dataset_id = dataset_id
            manifest = super().get_manifest(raw_dir, args)
            manifest["release_id"] = release_id
            manifest["release_dataset_id"] = dataset_id
            all_manifests.append(manifest)

        return pd.concat(all_manifests)

    def _run_item(self, manifest_item):

        release_id = manifest_item.release_id
        original_raw_dir = self.raw_dir
        original_processed_dir = self.processed_dir

        self.dataset_id = manifest_item.release_dataset_id
        self.brainset_id = f"cmi_hbn_r{release_id}_2025"
        self.raw_dir = original_raw_dir / f"R{release_id}"
        self.processed_dir = original_processed_dir.with_name(
            f"{original_processed_dir.name}_r{release_id}"
        )
        self.__dict__.pop("_participants_data", None)

        try:
            super()._run_item(manifest_item)
        finally:
            self.raw_dir = original_raw_dir
            self.processed_dir = original_processed_dir

    def process(self, download_output: dict) -> None:

        result = super()._process_common(download_output)

        if result is None:
            return

        data, store_path = result

        row = self._participants_data.loc[data.subject.id]
        if row is not None:
            data.subject.ehq_total = row.get("ehq_total", None)
            data.subject.commercial_use = row.get("commercial_use", None)
            data.subject.full_pheno = row.get("full_pheno", None)
            data.subject.p_factor = row.get("p_factor", None)
            data.subject.attention = row.get("attention", None)
            data.subject.internalizing = row.get("internalizing", None)
            data.subject.externalizing = row.get("externalizing", None)

        self.update_status("Storing")
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)
