"""
LangChain tools for the Dataset Wizard.

Uses a decorator-based approach to reduce boilerplate while wrapping existing utilities.
"""

import json
import logging
import os
import re
from functools import wraps
from typing import Optional, List, Dict, Any

from langchain_core.tools import tool

from brainsets.utils.open_neuro import (
    fetch_metadata,
    fetch_readme,
    fetch_all_filenames,
    fetch_participants,
    download_file_from_s3,
    get_s3_file_size,
)
from brainsets.ds_wizard.context import (
    get_default_dataset_info,
    task_taxonomy,
    modalities,
    eeg_bids_specification,
)
from brainsets.utils.eeg_montages import (
    get_all_montage_matches,
    get_all_electrode_names_to_montage_mapping,
)
from brainsets.ds_wizard.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE, EEG_DATA_EXTENSIONS

logger = logging.getLogger(__name__)


def with_error_handling(func):
    """Decorator that adds error handling to tool functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            return f"Error: {str(e)}"
    return wrapper


# =============================================================================
# Dataset Information Tools
# =============================================================================

@tool
@with_error_handling
def fetch_dataset_metadata(dataset_id: str) -> str:
    """Fetch metadata for an OpenNeuro dataset including name, modalities, authors, etc."""
    metadata = fetch_metadata(dataset_id)
    return json.dumps(metadata, indent=2, default=str)


@tool
@with_error_handling
def fetch_dataset_readme(dataset_id: str) -> str:
    """Fetch the README content for an OpenNeuro dataset."""
    return fetch_readme(dataset_id)


@tool
@with_error_handling
def fetch_dataset_filenames(dataset_id: str, tag: Optional[str] = None) -> str:
    """Fetch all filenames in an OpenNeuro dataset."""
    filenames = fetch_all_filenames(dataset_id, tag)
    return json.dumps(filenames, indent=2)


@tool
@with_error_handling
def fetch_participants_list(dataset_id: str, tag: Optional[str] = None) -> str:
    """
    Fetch the complete list of participant IDs from an OpenNeuro dataset.
    Use this tool FIRST to ensure you don't miss any participants when creating recording info.
    """
    participants = fetch_participants(dataset_id, tag)
    return json.dumps({
        "dataset_id": dataset_id,
        "total_participants": len(participants),
        "participant_ids": participants,
    }, indent=2)


# =============================================================================
# Taxonomy and Reference Tools
# =============================================================================

@tool
@with_error_handling
def get_task_taxonomy() -> str:
    """Get the complete EEG task taxonomy with categories and subcategories."""
    taxonomy = task_taxonomy()
    return json.dumps(taxonomy, indent=2)


@tool
@with_error_handling
def get_modalities() -> str:
    """Get all available electrode/channel modalities (EEG, EOG, EMG, etc.)."""
    mods = modalities()
    return json.dumps(mods, indent=2)


@tool
@with_error_handling
def get_eeg_bids_specs() -> str:
    """Get the EEG BIDS specification reference."""
    return eeg_bids_specification()


@tool
@with_error_handling
def get_default_info(dataset_id: str) -> str:
    """Get default dataset information from internal database."""
    info = get_default_dataset_info(dataset_id)
    return json.dumps(info, indent=2, default=str)


# =============================================================================
# Montage and Channel Tools
# =============================================================================

@tool
@with_error_handling
def get_montage_positions(montage_name: str) -> str:
    """
    Get 3D electrode positions for all channels in a specified MNE montage.
    Returns channel names and their (x, y, z) coordinates in meters.
    """
    import mne

    try:
        montage = mne.channels.make_standard_montage(montage_name)
        positions_dict = montage.get_positions()
        ch_pos = positions_dict.get("ch_pos", {})

        positions = {
            ch: pos.tolist() if hasattr(pos, "tolist") else pos
            for ch, pos in ch_pos.items()
        }

        result = {
            "montage_name": montage_name,
            "total_channels": len(positions),
            "channel_names": list(positions.keys()),
            "positions": positions,
        }
        return json.dumps(result, indent=2)

    except ValueError as e:
        available_montages = mne.channels.get_builtin_montages()
        return json.dumps({
            "error": f"Invalid montage name: {montage_name}",
            "message": str(e),
            "available_montages": available_montages,
        }, indent=2)


@tool
@with_error_handling
def find_all_montage_matches(channel_names: List[str]) -> str:
    """
    Find ALL possible montage matches for a list of electrodes with detailed statistics.
    Returns match percentages, electrode positions, matched/unmatched channels for every available montage.
    """
    all_matches = get_all_montage_matches(channel_names)

    has_any_matches = any(
        match_info["match_to_input"] > 0 for match_info in all_matches.values()
    )

    if not has_any_matches:
        return json.dumps({
            "error": "No matching electrode names found in any montage",
            "input_channels": channel_names,
            "suggestion": "Try converting the channel names to standard names first.",
            "total_montages_analyzed": len(all_matches),
            "best_match_percentage": 0.0,
        }, indent=2)

    sorted_matches = dict(
        sorted(
            all_matches.items(),
            key=lambda x: x[1]["match_to_input"],
            reverse=True,
        )
    )

    summary = {
        "input_channels": channel_names,
        "total_montages_analyzed": len(sorted_matches),
        "best_match": {
            "montage": list(sorted_matches.keys())[0] if sorted_matches else None,
            "match_percentage": (
                list(sorted_matches.values())[0]["match_to_input"] * 100
                if sorted_matches
                else 0
            ),
        },
        "perfect_matches": [
            name for name, info in sorted_matches.items() if info["match_to_input"] == 1.0
        ],
        "good_matches_90_plus": [
            name for name, info in sorted_matches.items() if info["match_to_input"] >= 0.9
        ],
    }
    return json.dumps(summary, indent=2)


@tool
@with_error_handling
def get_electrode_to_montage_mapping() -> str:
    """
    Get a complete mapping of all electrode names to the montages they belong to.
    Useful for understanding which montages contain specific electrodes.
    """
    mapping = get_all_electrode_names_to_montage_mapping()

    summary = {
        "total_electrodes": len(mapping),
        "electrode_mapping": mapping,
        "montage_coverage": {
            montage: sum(1 for montages in mapping.values() if montage in montages)
            for montage in set(
                montage for montages in mapping.values() for montage in montages.split(",")
            )
        },
    }
    return json.dumps(summary, indent=2)


# =============================================================================
# Configuration File Tools (require base_dir)
# =============================================================================

def create_file_tools(base_dir: str) -> List:
    """Create file operation tools bound to a specific base directory."""

    @tool
    @with_error_handling
    def list_configuration_files() -> str:
        """List all downloaded configuration files."""
        if not os.path.exists(base_dir):
            return f"Error: Directory '{base_dir}' does not exist"
        if not os.path.isdir(base_dir):
            return f"Error: '{base_dir}' is not a directory"

        files = []
        for root, _, filenames in os.walk(base_dir):
            for filename in filenames:
                if filename.endswith(ALLOWED_EXTENSIONS):
                    abs_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(abs_path, base_dir)
                    files.append(rel_path)

        files.sort()
        return json.dumps({"files": files, "count": len(files)}, indent=2)

    @tool
    @with_error_handling
    def read_configuration_file(file_path: str) -> str:
        """
        Read the contents of a downloaded configuration file.
        Provide a relative path (e.g., 'dataset_description.json').
        """
        abs_path = os.path.join(base_dir, file_path)
        if not os.path.exists(abs_path):
            return f"Error: File '{file_path}' does not exist"
        if not os.path.isfile(abs_path):
            return f"Error: '{file_path}' is not a file"
        if not file_path.endswith(ALLOWED_EXTENSIONS):
            return f"Error: '{file_path}' is not a configuration file ({', '.join(ALLOWED_EXTENSIONS)})"

        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()

        if file_path.endswith(".json"):
            try:
                parsed = json.loads(content)
                return json.dumps(parsed, indent=2)
            except json.JSONDecodeError:
                pass

        return content

    @tool
    @with_error_handling
    def download_configuration_file(dataset_id: str, file_path: str) -> str:
        """
        Download a specific configuration file from an OpenNeuro dataset via AWS S3.
        Downloads small configuration files on-demand (≤5 MB).
        Supported file types: .json, .tsv, .txt, .md, .tsv.gz.
        """
        if not any(file_path.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
            return json.dumps({
                "error": f"File type not allowed. File must be one of: {', '.join(ALLOWED_EXTENSIONS)}",
                "file_path": file_path,
            }, indent=2)

        local_path = os.path.join(base_dir, file_path)

        if os.path.exists(local_path):
            logger.info(f"File already cached locally: {local_path}")
            with open(local_path, "r", encoding="utf-8") as f:
                content = f.read()

            preview = content[:500] if len(content) > 500 else content
            return json.dumps({
                "status": "cached",
                "file_path": file_path,
                "size_bytes": os.path.getsize(local_path),
                "content_preview": preview,
            }, indent=2)

        try:
            file_size = get_s3_file_size(dataset_id, file_path)

            if file_size > MAX_FILE_SIZE:
                return json.dumps({
                    "error": f"File too large ({file_size:,} bytes). Maximum allowed: {MAX_FILE_SIZE:,} bytes (5 MB)",
                    "file_path": file_path,
                    "file_size": file_size,
                }, indent=2)

            download_file_from_s3(dataset_id, file_path, base_dir)

            with open(local_path, "r", encoding="utf-8") as f:
                content = f.read()

            preview = content[:500] if len(content) > 500 else content
            return json.dumps({
                "status": "downloaded",
                "file_path": file_path,
                "size_bytes": file_size,
                "content_preview": preview,
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "error": str(e),
                "file_path": file_path,
                "dataset_id": dataset_id,
            }, indent=2)

    return [list_configuration_files, read_configuration_file, download_configuration_file]


# =============================================================================
# Recording Analysis Tool
# =============================================================================

def create_recording_analysis_tool(base_dir: str):
    """Create the analyze_all_recordings tool bound to a specific base directory."""

    def _download_and_read_file(dataset_id: str, file_path: str) -> Optional[str]:
        """Download a file from S3 and return its content."""
        local_path = os.path.join(base_dir, file_path)

        if os.path.exists(local_path):
            with open(local_path, "r", encoding="utf-8") as f:
                return f.read()

        try:
            download_file_from_s3(dataset_id, file_path, base_dir)
            with open(local_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to download {file_path}: {e}")
            return None

    @tool
    @with_error_handling
    def analyze_all_recordings(dataset_id: str, channel_maps: Dict[str, List[str]]) -> str:
        """
        Batch analyze ALL recordings in a dataset.
        Downloads and parses all *_eeg.json sidecar files and *_channels.tsv files.
        Extracts duration, sampling rate, channel names, channel types, and bad channels.
        Matches recordings to the appropriate channel maps.
        """
        filenames = fetch_all_filenames(dataset_id)

        eeg_json_files = [f for f in filenames if f.endswith("_eeg.json")]
        channels_tsv_files = [f for f in filenames if f.endswith("_channels.tsv")]

        channels_tsv_map = {}
        for f in channels_tsv_files:
            key = f.rsplit("_channels.tsv", 1)[0]
            channels_tsv_map[key] = f

        participants = {}
        participants_content = _download_and_read_file(dataset_id, "participants.tsv")
        if participants_content:
            participants = _parse_participants_tsv(participants_content)
        else:
            participant_ids = fetch_participants(dataset_id)
            participants = {pid: {"participant_id": pid} for pid in participant_ids}
            logger.info(f"participants.tsv not found, using API fallback: {len(participants)} participants")

        recordings = []
        errors = []

        if not eeg_json_files:
            logger.info("No *_eeg.json sidecar files found, extracting recordings from data filenames")
            recordings = _extract_recordings_from_filenames(filenames, participants, channel_maps)
            return json.dumps({
                "dataset_id": dataset_id,
                "total_recordings": len(recordings),
                "total_participants": len(participants),
                "recordings": recordings,
                "participants": participants,
                "channel_maps_provided": list(channel_maps.keys()),
                "extracted_from_filenames": True,
                "note": "Recording metadata unavailable - no sidecar JSON files found",
                "errors": None,
            }, indent=2, default=str)

        for eeg_json_path in eeg_json_files:
            try:
                entities = _parse_bids_filename(eeg_json_path)

                recording_id_parts = []
                if entities["subject_id"]:
                    recording_id_parts.append(f"sub-{entities['subject_id']}")
                if entities["session"]:
                    recording_id_parts.append(f"ses-{entities['session']}")
                if entities["task"]:
                    recording_id_parts.append(f"task-{entities['task']}")
                if entities["acquisition"]:
                    recording_id_parts.append(f"acq-{entities['acquisition']}")
                if entities["run"]:
                    recording_id_parts.append(f"run-{entities['run']}")

                recording_id = "_".join(recording_id_parts)

                eeg_json_content = _download_and_read_file(dataset_id, eeg_json_path)
                sidecar_data = {}
                if eeg_json_content:
                    try:
                        sidecar_data = json.loads(eeg_json_content)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON: {eeg_json_path}")

                duration_seconds = sidecar_data.get("RecordingDuration")
                sampling_frequency = sidecar_data.get("SamplingFrequency")
                manufacturer = sidecar_data.get("Manufacturer")

                base_key = eeg_json_path.rsplit("_eeg.json", 1)[0]
                channels_tsv_path = channels_tsv_map.get(base_key)

                channel_names = []
                channel_types = {}
                bad_channels = []
                num_channels = 0

                if channels_tsv_path:
                    channels_content = _download_and_read_file(dataset_id, channels_tsv_path)
                    if channels_content:
                        channel_info = _parse_channels_tsv(channels_content)
                        channel_names = channel_info["channel_names"]
                        channel_types = channel_info["channel_types"]
                        bad_channels = channel_info["bad_channels"]
                        num_channels = len(channel_names)

                matched_map_id, match_score = _match_to_channel_map(
                    channel_names, entities["acquisition"], channel_maps
                )

                subject_key = f"sub-{entities['subject_id']}" if entities["subject_id"] else None
                participant_info = participants.get(subject_key, {})

                recordings.append({
                    "recording_id": recording_id,
                    "subject_id": f"sub-{entities['subject_id']}" if entities["subject_id"] else None,
                    "session": entities["session"],
                    "task_id": entities["task"],
                    "acquisition": entities["acquisition"],
                    "run": entities["run"],
                    "duration_seconds": duration_seconds,
                    "sampling_frequency": sampling_frequency,
                    "manufacturer": manufacturer,
                    "channel_names": channel_names,
                    "channel_types": channel_types,
                    "num_channels": num_channels,
                    "bad_channels": bad_channels,
                    "matched_channel_map_id": matched_map_id,
                    "channel_map_match_score": match_score,
                    "participant_info": participant_info,
                    "source_files": {
                        "eeg_json": eeg_json_path,
                        "channels_tsv": channels_tsv_path,
                    },
                })

            except Exception as e:
                errors.append({"file": eeg_json_path, "error": str(e)})
                logger.error(f"Error processing {eeg_json_path}: {e}")

        return json.dumps({
            "dataset_id": dataset_id,
            "total_recordings": len(recordings),
            "total_participants": len(participants),
            "recordings": recordings,
            "participants": participants,
            "channel_maps_provided": list(channel_maps.keys()),
            "errors": errors if errors else None,
        }, indent=2, default=str)

    return analyze_all_recordings


# =============================================================================
# Helper Functions
# =============================================================================

def _parse_bids_filename(filename: str) -> Dict[str, Optional[str]]:
    """Extract BIDS entities from a filename."""
    basename = os.path.basename(filename)
    name_without_ext = (
        basename.rsplit("_eeg", 1)[0] if "_eeg" in basename else basename.rsplit(".", 1)[0]
    )

    entities = {
        "subject_id": None,
        "session": None,
        "task": None,
        "acquisition": None,
        "run": None,
    }

    patterns = {
        "subject_id": r"sub-([^_]+)",
        "session": r"ses-([^_]+)",
        "task": r"task-([^_]+)",
        "acquisition": r"acq-([^_]+)",
        "run": r"run-([^_]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, name_without_ext)
        if match:
            entities[key] = match.group(1)

    return entities


def _parse_channels_tsv(content: str) -> Dict[str, Any]:
    """Parse a channels.tsv file and extract channel information."""
    lines = content.strip().split("\n")
    if not lines:
        return {"channel_names": [], "channel_types": {}, "bad_channels": []}

    header = lines[0].split("\t")
    header_lower = [h.lower() for h in header]

    name_idx = header_lower.index("name") if "name" in header_lower else 0
    type_idx = header_lower.index("type") if "type" in header_lower else None
    status_idx = header_lower.index("status") if "status" in header_lower else None

    channel_names = []
    channel_types = {}
    bad_channels = []

    for line in lines[1:]:
        if not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) <= name_idx:
            continue

        ch_name = cols[name_idx].strip()
        if not ch_name:
            continue

        channel_names.append(ch_name)

        if type_idx is not None and len(cols) > type_idx:
            channel_types[ch_name] = cols[type_idx].strip()

        if status_idx is not None and len(cols) > status_idx:
            if cols[status_idx].strip().lower() == "bad":
                bad_channels.append(ch_name)

    return {
        "channel_names": channel_names,
        "channel_types": channel_types,
        "bad_channels": bad_channels,
    }


def _parse_participants_tsv(content: str) -> Dict[str, Dict[str, str]]:
    """Parse participants.tsv and return participant info keyed by participant_id."""
    lines = content.strip().split("\n")
    if len(lines) < 2:
        return {}

    header = lines[0].split("\t")
    participants = {}

    for line in lines[1:]:
        if not line.strip():
            continue
        cols = line.split("\t")
        if not cols:
            continue

        row_data = {}
        for i, col in enumerate(cols):
            if i < len(header):
                row_data[header[i]] = col.strip()

        participant_id = row_data.get("participant_id", "")
        if participant_id:
            participants[participant_id] = row_data

    return participants


def _match_to_channel_map(
    channel_names: List[str],
    acquisition: Optional[str],
    channel_maps: Dict[str, List[str]],
) -> tuple:
    """Match recording channels to the best channel map."""
    if not channel_names or not channel_maps:
        return None, 0.0

    best_match = None
    best_score = 0.0
    recording_channels = set(channel_names)

    for map_id, map_channels in channel_maps.items():
        map_channel_set = set(map_channels)
        overlap = len(map_channel_set & recording_channels)
        if overlap == 0:
            continue

        score = overlap / max(len(map_channel_set), len(recording_channels), 1)

        if acquisition and acquisition.lower() in map_id.lower():
            score += 0.3

        if score > best_score:
            best_score = score
            best_match = map_id

    return best_match, best_score


def _extract_recordings_from_filenames(
    filenames: List[str],
    participants: Dict[str, Dict[str, str]],
    channel_maps: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """Extract recording info from EEG data filenames when sidecar JSON files are unavailable."""
    eeg_data_files = [
        f
        for f in filenames
        if any(f.lower().endswith(ext) for ext in EEG_DATA_EXTENSIONS)
        and not f.startswith("derivatives/")
    ]

    seen_recordings = set()
    recordings = []

    for data_file in eeg_data_files:
        entities = _parse_bids_filename(data_file)

        if not entities["subject_id"]:
            continue

        recording_id_parts = []
        if entities["subject_id"]:
            recording_id_parts.append(f"sub-{entities['subject_id']}")
        if entities["session"]:
            recording_id_parts.append(f"ses-{entities['session']}")
        if entities["task"]:
            recording_id_parts.append(f"task-{entities['task']}")
        if entities["acquisition"]:
            recording_id_parts.append(f"acq-{entities['acquisition']}")
        if entities["run"]:
            recording_id_parts.append(f"run-{entities['run']}")

        recording_id = "_".join(recording_id_parts)

        if recording_id in seen_recordings:
            continue
        seen_recordings.add(recording_id)

        subject_key = f"sub-{entities['subject_id']}"
        participant_info = participants.get(subject_key, {})

        matched_map_id = list(channel_maps.keys())[0] if channel_maps else None

        recordings.append({
            "recording_id": recording_id,
            "subject_id": subject_key,
            "session": entities["session"],
            "task_id": entities["task"],
            "acquisition": entities["acquisition"],
            "run": entities["run"],
            "duration_seconds": None,
            "sampling_frequency": None,
            "manufacturer": None,
            "channel_names": [],
            "channel_types": {},
            "num_channels": None,
            "bad_channels": [],
            "matched_channel_map_id": matched_map_id,
            "channel_map_match_score": None,
            "participant_info": participant_info,
            "source_files": {"data_file": data_file},
            "extracted_from_filename": True,
        })

    return recordings


# =============================================================================
# Tool Factory Functions
# =============================================================================

def create_metadata_tools(base_dir: str) -> List:
    """Create metadata tools with the given base directory for file operations."""
    file_tools = create_file_tools(base_dir)
    return [
        fetch_dataset_metadata,
        fetch_dataset_readme,
        get_task_taxonomy,
        fetch_dataset_filenames,
        *file_tools,
    ]


def create_channel_tools(base_dir: str) -> List:
    """Create channel tools with the given base directory for file operations."""
    file_tools = create_file_tools(base_dir)
    return [
        fetch_dataset_readme,
        fetch_dataset_filenames,
        get_modalities,
        get_eeg_bids_specs,
        find_all_montage_matches,
        get_montage_positions,
        get_electrode_to_montage_mapping,
        *file_tools,
    ]


def create_recording_tools(base_dir: str) -> List:
    """Create recording tools with the given base directory for file operations."""
    file_tools = create_file_tools(base_dir)
    analyze_tool = create_recording_analysis_tool(base_dir)
    return [
        analyze_tool,
        *file_tools,
    ]
