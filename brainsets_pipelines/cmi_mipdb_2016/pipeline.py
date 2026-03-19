# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "boto3~=1.41.0",
#   "mne~=1.11.0",
#   "scikit-learn~=1.6.0",
#   "temporaldata@git+https://github.com/neuro-galaxy/temporaldata@main",
# ]
# ///

from argparse import ArgumentParser
from pathlib import Path
import ssl
import sys
from urllib.request import urlopen

import json
import re
import logging

import mne
import h5py
import numpy as np
import pandas as pd
from temporaldata import ArrayDict, Data, Interval
from brainsets import serialize_fn_map
from brainsets.pipeline import BrainsetPipeline
from brainsets.taxonomy import RecordingTech
from brainsets.utils.s3_utils import get_cached_s3_client, get_object_list
from brainsets.utils.split import generate_folds, generate_string_kfold_assignment
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eyetracking  # noqa: E402
import paradigm  # noqa: E402

# Metadata files are not hosted in the same s3 bucket as the public data
# Public data on the 126 subjects
SUBJECTS_METADATA_URL = (
    "https://fcon_1000.projects.nitrc.org/indi/cmi_eeg/_static/MIPDB_PublicFile.csv"
)
# EEG Channel location file
CHANNEL_LOCATION_URL = (
    "https://fcon_1000.projects.nitrc.org/indi/cmi_eeg/_static/GSN_HydroCel_129.sfp"
)

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")


class Pipeline(BrainsetPipeline):
    brainset_id = "cmi_mipdb_2016"
    bucket = "fcp-indi"
    prefix = "data/Projects/EEG_Eyetracking_CMI_data/"
    parser = parser

    def _download_subject_metadata(self) -> Path:
        """Download the MIPDB public metadata CSV if not already cached locally.

        Returns the local path to the downloaded CSV file.
        """
        local_path = self.raw_dir / "MIPDB_PublicFile.csv"

        if not local_path.exists() or self.args.redownload:
            self.update_status("Downloading subject metadata CSV")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            # NITRC's certificate has a hostname mismatch; skip verification
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urlopen(SUBJECTS_METADATA_URL, context=ctx) as response:
                local_path.write_bytes(response.read())
        else:
            self.update_status("Subject metadata CSV already cached")

        return local_path

    def _download_channel_location(self) -> Path:
        """Download the GSN HydroCel 129 channel location SFP file.

        Returns the local path to the downloaded file.
        """
        local_path = self.raw_dir / "GSN_HydroCel_129.sfp"

        if not local_path.exists() or self.args.redownload:
            self.update_status("Downloading channel location file")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urlopen(CHANNEL_LOCATION_URL, context=ctx) as response:
                local_path.write_bytes(response.read())
        else:
            self.update_status("Channel location file already cached")

        return local_path

    @classmethod
    def get_manifest(cls, raw_dir: Path, args) -> pd.DataFrame:
        s3 = get_cached_s3_client()

        manifest_rows = []
        et_keys_by_subject: dict[str, list[str]] = {}

        rel_keys = get_object_list(bucket=cls.bucket, prefix=cls.prefix, s3_client=s3)

        for rel_key in rel_keys:
            rel_path = Path(rel_key)
            filename = rel_path.name

            if filename.startswith("."):
                continue

            rel_parts = rel_path.parts

            # EEG: subject_id/EEG/raw/raw_format/*.raw
            if (
                len(rel_parts) >= 4
                and rel_parts[1] == "EEG"
                and rel_parts[2] == "raw"
                and rel_parts[3] == "raw_format"
            ) and filename.endswith(".raw"):
                subject_id = rel_parts[0]
                session_id = f"{Path(filename).stem}"

                manifest_rows.append(
                    {
                        "session_id": session_id,
                        "subject_id": subject_id,
                        "s3_key": cls.prefix + rel_key,
                    }
                )

            # ET: subject_id/Eyetracking/txt/*.txt
            if (
                len(rel_parts) >= 3
                and rel_parts[1] == "Eyetracking"
                and rel_parts[2] == "txt"
                and filename.endswith(".txt")
            ):
                subject_id = rel_parts[0]
                et_keys_by_subject.setdefault(subject_id, []).append(
                    cls.prefix + rel_key
                )

        for row in manifest_rows:
            et_keys = et_keys_by_subject.get(row["subject_id"], [])
            row["et_s3_keys"] = json.dumps(et_keys)

        manifest = pd.DataFrame(
            manifest_rows,
            columns=["session_id", "subject_id", "s3_key", "et_s3_keys"],
        ).set_index("session_id")
        return manifest

    def download(self, manifest_item) -> dict:
        self.update_status("DOWNLOADING")
        s3 = get_cached_s3_client()

        # Ensure dataset-level metadata is cached (downloaded once per dataset)
        self._download_subject_metadata()
        self._download_channel_location()

        # --- EEG file ---
        s3_key = manifest_item.s3_key
        eeg_path = self.raw_dir / Path(s3_key).relative_to(self.prefix)
        eeg_path.parent.mkdir(parents=True, exist_ok=True)

        if not eeg_path.exists() or self.args.redownload:
            self.update_status(f"Downloading EEG: {Path(s3_key).name}")
            s3.download_file(self.bucket, s3_key, str(eeg_path))
        else:
            self.update_status(f"Skipping EEG download, exists: {eeg_path}")

        # --- ET files (shared across sessions for a subject, cached) ---
        et_dir = None
        et_s3_keys = json.loads(manifest_item.et_s3_keys)
        if et_s3_keys:
            subject_id = manifest_item.subject_id
            et_dir = self.raw_dir / subject_id / "Eyetracking" / "txt"
            for et_key in et_s3_keys:
                et_path = self.raw_dir / Path(et_key).relative_to(self.prefix)
                et_path.parent.mkdir(parents=True, exist_ok=True)
                if not et_path.exists() or self.args.redownload:
                    self.update_status(f"Downloading ET: {Path(et_key).name}")
                    s3.download_file(self.bucket, et_key, str(et_path))
        else:
            logging.warning(
                f"No ET files available for subject " f"{manifest_item.subject_id}"
            )

        return {"eeg": eeg_path, "et_dir": et_dir}

    def process(self, download_output: dict) -> None:
        raw_path = download_output["eeg"]
        et_dir = download_output.get("et_dir")

        self.update_status("PROCESSING")

        recording_id = raw_path.stem

        output_path = self.processed_dir / f"{recording_id}.h5"
        if output_path.exists() and not self.args.reprocess:
            self.update_status(f"Skipping processing, file exists: {output_path}")
            return

        brainset_description = BrainsetDescription(
            id="cmi_mipdb_2016",
            origin_version="1.0.0",
            derived_version="1.0.0",
            source="https://fcon_1000.projects.nitrc.org/indi/cmi_eeg/",
            description="The Child Mind Institute - MIPDB provides EEG,"
            "eye-tracking, and behavioral data across multiple paradigms from"
            "126 psychiatric and healthy subjects aged 6 - 44 years old.",
        )

        self.update_status("Loading EEG file")
        raw = mne.io.read_raw_egi(str(raw_path), preload=True, verbose=True)
        meas_date = extract_measurement_date(raw)

        session_description = SessionDescription(
            id=recording_id, recording_date=meas_date
        )

        subject_id = recording_id[:9]
        validate_subject_id(subject_id)
        device_description = DeviceDescription(
            id=f"GSN_HydroCel_129_{subject_id}",
            recording_tech=RecordingTech.SCALP_EEG,
        )

        csv_path = self.raw_dir / "MIPDB_PublicFile.csv"
        subject_info = parse_subject_metadata(csv_path, subject_id)

        subject_description = SubjectDescription(
            id=subject_id,
            species="HUMAN",
            age=subject_info["age"],
            sex=subject_info["sex"],
        )

        self.update_status("Extracting EEG signal and channels")
        eeg_signal = extract_eeg_signal(raw)
        channels = extract_channels(raw)
        sfp_path = self.raw_dir / "GSN_HydroCel_129.sfp"
        channels.locations = extract_channel_locations(sfp_path, channels)

        self.update_status("Extracting annotations")
        annotations = extract_annotations(raw)
        paradigm_intervals = paradigm.get_all_paradigm_intervals(annotations)

        # --- Eyetracking  ---
        et_data = None
        if et_dir is not None and len(paradigm_intervals) > 0:
            self.update_status("Processing eyetracking")
            et_data = eyetracking.process_session(
                et_dir,
                paradigm_intervals,
                float(eeg_signal.domain.start[0]),
                float(eeg_signal.domain.end[0]),
            )
            if et_data is None:
                logging.warning(f"No ET data aligned for session {recording_id}")
        elif et_dir is None:
            logging.warning(f"No ET directory for session {recording_id}")

        self.update_status("Creating splits")
        splits = create_splits(
            eeg_domain=eeg_signal.domain,
            annotations=annotations,
            subject_id=subject_description.id,
            session_id=session_description.id,
            paradigm_intervals=paradigm_intervals,
        )

        self.update_status("Creating Data Object")
        data_kwargs = {
            "brainset": brainset_description,
            "subject": subject_description,
            "session": session_description,
            "device": device_description,
            "eeg": eeg_signal,
            "channels": channels,
            "domain": eeg_signal.domain,
            "splits": splits,
        }
        if len(annotations) > 0:
            data_kwargs["annotations"] = annotations
        if len(paradigm_intervals) > 0:
            data_kwargs["paradigms"] = paradigm_intervals
        if et_data is not None:
            data_kwargs["eyetracking"] = et_data

        data = Data(**data_kwargs)

        self.update_status("Storing processed data to disk")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


def validate_subject_id(subject_id: str) -> None:
    """Validate that subject_id matches the CMI MIPDB convention: A000xxxxx.

    Raises:
        ValueError: If subject_id does not match A000xxxxx format.
    """
    SUBJECT_ID_PATTERN = re.compile(r"^A000\d{5}$")
    if not SUBJECT_ID_PATTERN.match(subject_id):
        raise ValueError(
            f"Invalid subject_id for CMI MIPDB: {subject_id!r}. "
            "Expected format: A000xxxxx where x are digits (e.g. A00054400)."
        )


def parse_subject_metadata(csv_path: Path, subject_id: str) -> dict:
    """Look up subject metadata (age, sex) from the MIPDB public metadata.

    Args:
        csv_path: Path to the MIPDB public metadata CSV file.
        subject_id: Subject identifier (e.g., "A00054400").

    Returns:
        Dict with 'age' and 'sex' keys.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Subject metadata not found at {csv_path}. Run download() first."
        )
    df = pd.read_csv(csv_path, index_col="ID")

    if subject_id not in df.index:
        logging.warning("No metadata found for subject %s", subject_id)
        return {"age": 0.0, "sex": 0}

    subject = df.loc[subject_id]

    age = subject.get("Age", None)
    sex = subject.get("Sex", None)
    sex = int(sex) if pd.notna(sex) else 0

    return {"age": age, "sex": sex}


def extract_channel_locations(sfp_path: Path, channels: ArrayDict) -> np.ndarray:
    """Build 3D EEG electrode positions aligned with the given channel list.

    Reads the GSN HydroCel 129 SFP file (label, X, Y, Z in Cartesian
    head coordinates) and returns an ``(N, 3)`` float array of positions
    in the same order as ``channels.id``. Only channels whose type is
    ``"eeg"`` are looked up in the SFP file; all others receive NaN
    coordinates.

    Args:
        sfp_path: Path to the GSN HydroCel 129 SFP file.
        channels: ArrayDict with ``id`` and ``type``.

    Returns:
        Array of shape ``(N, 3)`` with XYZ coordinates in head space.
        Non-EEG channels and missing labels are filled with NaN.

    Raises:
        FileNotFoundError: If ``sfp_path`` does not exist.
        ValueError: If the SFP file does not have the expected format
            (four whitespace-separated columns: label, x, y, z).
    """
    if not sfp_path.exists():
        raise FileNotFoundError(
            f"Channel location file not found at {sfp_path}. Run download() first."
        )
    try:
        loc_df = pd.read_csv(
            sfp_path,
            sep=r"\s+",
            header=None,
            names=["label", "x", "y", "z"],
        )
    except pd.errors.ParserError as e:
        raise ValueError(
            f"Failed to parse channel location file at {sfp_path}: {e}"
        ) from e

    try:
        loc_df[["x", "y", "z"]] = loc_df[["x", "y", "z"]].astype(float)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Channel location file at {sfp_path} must have four "
            f"whitespace-separated columns (label, x, y, z) with numeric "
            f"coordinates: {e}"
        ) from e

    loc_map = {row.label: (row.x, row.y, row.z) for _, row in loc_df.iterrows()}

    locations = np.full((len(channels.id), 3), np.nan)
    for i, (ch, ch_type) in enumerate(zip(channels.id, channels.type)):
        if ch_type == "eeg" and ch in loc_map:
            locations[i] = loc_map[ch]

    return locations


def create_splits(
    eeg_domain: Interval,
    annotations: Interval,
    subject_id: str,
    session_id: str,
    paradigm_intervals: Interval | None = None,
    epoch_duration: float = 30.0,
    n_folds: int = 3,
    seed: int = 42,
) -> Data:
    """Generate train/valid/test splits for one recording.

    Generates three types of splits:
    - Intrasession (epoch-level): k-fold within each session
    - Intersubject (session-level): subject assigned to train/valid/test per fold
    - Intersession (session-level): subject-session assigned per fold

    Args:
        eeg_domain: Full time domain of the EEG recording.
        annotations: Raw annotations extracted from the recording.
        subject_id: Subject identifier for cross-subject splits.
        session_id: Session identifier for cross-session splits.
        paradigm_intervals: Optional extracted paradigm intervals.
        epoch_duration: Duration of each epoch in seconds.
        n_folds: Number of cross-validation folds.
        seed: Random seed for reproducibility.
    """
    if paradigm_intervals is not None and len(paradigm_intervals) > 0:
        crop_start = paradigm_intervals.start[0]
    elif len(annotations) > 0:
        crop_start = annotations.start[0]
    else:
        crop_start = eeg_domain.start[0]

    cropped_domain = Interval(
        start=np.array([crop_start]),
        end=eeg_domain.end.copy(),
    )

    epochs = cropped_domain.subdivide(step=epoch_duration, drop_short=True)
    logging.info(f"Subdivided domain into {len(epochs)} epochs of {epoch_duration}s")

    if len(epochs) < n_folds:
        raise ValueError(
            f"Not enough epochs ({len(epochs)}) for {n_folds} folds. "
            f"Domain duration: {cropped_domain.end[0] - cropped_domain.start[0]:.1f}s"
        )

    folds = generate_folds(
        epochs,
        n_folds=n_folds,
        val_ratio=0.2,
        seed=seed,
    )
    logging.info(f"Generated {n_folds} folds")

    folds_dict = {f"fold_{i}": fold for i, fold in enumerate(folds)}
    splits = Data(**folds_dict, domain=epochs)

    subject_assignments = generate_string_kfold_assignment(
        string_id=subject_id, n_folds=n_folds, val_ratio=0.2, seed=seed
    )
    session_assignments = generate_string_kfold_assignment(
        string_id=f"{subject_id}_{session_id}",
        n_folds=n_folds,
        val_ratio=0.2,
        seed=seed,
    )

    for fold_idx, assignment in enumerate(subject_assignments):
        setattr(splits, f"intersubject_fold_{fold_idx}_assignment", assignment)
    for fold_idx, assignment in enumerate(session_assignments):
        setattr(splits, f"intersession_fold_{fold_idx}_assignment", assignment)

    return splits
