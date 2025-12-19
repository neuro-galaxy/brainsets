"""
LangChain tools that wrap the existing brainsets_private utilities.
"""

import json
import logging
import os
from typing import Optional, List, Dict, Any, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from brainsets.utils.open_neuro import (
    fetch_metadata,
    fetch_readme,
    fetch_all_filenames,
    fetch_participants,
    validate_dataset_id,
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
    get_standard_montage,
    get_standard_ch_info,
    find_match_percentage_by_montage,
    get_all_montage_matches,
    get_all_electrode_names_to_montage_mapping,
)

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = (".json", ".tsv", ".txt", ".md", ".csv", ".elp")


def safe_tool_execution(
    tool_name: str, func, required_params: List[str] = None, **kwargs
):
    """
    Helper function to safely execute tool functions with error handling.

    Args:
        tool_name: Name of the tool for logging
        func: The function to execute
        required_params: List of required parameter names
        **kwargs: Parameters to pass to the function
    """
    try:
        # Check required parameters
        if required_params:
            missing_params = [
                param for param in required_params if not kwargs.get(param)
            ]
            if missing_params:
                return (
                    f"Error: Missing required parameter(s): {', '.join(missing_params)}"
                )

        return func(**kwargs)

    except TypeError as e:
        if "unexpected keyword argument" in str(e):
            logger.warning(
                f"{tool_name} called with unexpected parameters: {kwargs}. Attempting to proceed with valid parameters."
            )
            # Filter kwargs to only include expected parameters
            try:
                # Try to call with only the required/expected parameters
                import inspect

                sig = inspect.signature(func)
                valid_params = list(sig.parameters.keys())
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
                return func(**filtered_kwargs)
            except Exception as inner_e:
                logger.error(
                    f"Error in {tool_name} after filtering parameters: {inner_e}"
                )
                return f"Error: {str(inner_e)}"
        else:
            logger.error(f"TypeError in {tool_name}: {e}")
            return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"Error in {tool_name}: {e}")
        return f"Error: {str(e)}"


# Common input classes for dataset operations
class DatasetIdInput(BaseModel):
    """Common input class for tools that only require a dataset ID."""

    dataset_id: str = Field(description="Dataset ID (e.g., 'ds000247')")


class DatasetIdTagInput(BaseModel):
    """Common input class for tools that require a dataset ID and optional tag."""

    dataset_id: str = Field(description="Dataset ID (e.g., 'ds000247')")
    tag: Optional[str] = Field(
        default=None, description="Version tag (optional, uses latest if None)"
    )


# Common input classes for channel operations
class ChannelListInput(BaseModel):
    """Common input class for tools that require a list of channel names."""

    channel_names: List[str] = Field(description="List of channel names")


class MontageNameInput(BaseModel):
    """Common input class for tools that require a montage name."""

    montage_name: str = Field(
        description="Name of the montage (e.g., 'standard_1020', 'biosemi64')"
    )


# Common input classes for configuration file operations
class RelativeFilePathInput(BaseModel):
    """Input class for tools that require a relative file path."""

    file_path: str = Field(
        description="Relative path to file (e.g., 'dataset_description.json' or 'sub-01/eeg/sub-01_channels.tsv')"
    )


class DownloadFileInput(BaseModel):
    """Input class for downloading a specific file from OpenNeuro."""

    dataset_id: str = Field(description="Dataset ID (e.g., 'ds000247')")
    file_path: str = Field(
        description="Relative path to file within dataset (e.g., 'dataset_description.json' or 'sub-01/ses-01/eeg/sub-01_ses-01_channels.tsv')"
    )


class EmptyInput(BaseModel):
    """Input class for tools that take no parameters."""

    pass


class DatasetMetadataTool(BaseTool):
    args_schema: Type[BaseModel] = DatasetIdInput
    name: str = "fetch_dataset_metadata"
    description: str = (
        "Fetch metadata for an OpenNeuro dataset including name, modalities, authors, etc."
    )

    def _run(self, dataset_id: str) -> str:
        def _fetch_metadata(dataset_id: str):
            metadata = fetch_metadata(dataset_id)
            return json.dumps(metadata, indent=2, default=str)

        return safe_tool_execution(
            "DatasetMetadataTool",
            _fetch_metadata,
            ["dataset_id"],
            dataset_id=dataset_id,
        )


class DatasetReadmeTool(BaseTool):
    name: str = "fetch_dataset_readme"
    description: str = "Fetch the README content for an OpenNeuro dataset"
    args_schema: Type[BaseModel] = DatasetIdInput

    def _run(self, dataset_id: str) -> str:
        def _fetch_readme(dataset_id: str):
            return fetch_readme(dataset_id)

        return safe_tool_execution(
            "DatasetReadmeTool", _fetch_readme, ["dataset_id"], dataset_id=dataset_id
        )


class DatasetFilenamesTool(BaseTool):
    name: str = "fetch_dataset_filenames"
    description: str = "Fetch all filenames in an OpenNeuro dataset"
    args_schema: Type[BaseModel] = DatasetIdTagInput

    def _run(self, dataset_id: str, tag: Optional[str] = None) -> str:
        def _fetch_filenames(dataset_id: str, tag: Optional[str] = None):
            filenames = fetch_all_filenames(dataset_id, tag)
            return json.dumps(filenames, indent=2)

        return safe_tool_execution(
            "DatasetFilenamesTool",
            _fetch_filenames,
            ["dataset_id"],
            dataset_id=dataset_id,
            tag=tag,
        )


class FetchParticipantsTool(BaseTool):
    name: str = "fetch_participants"
    description: str = (
        "Fetch the complete list of participant IDs from an OpenNeuro dataset. "
        "This tool queries the dataset's GraphQL API to retrieve ALL participant IDs with 'sub-' prefix. "
        "Use this tool FIRST to ensure you don't miss any participants when creating recording info."
    )
    args_schema: Type[BaseModel] = DatasetIdTagInput

    def _run(self, dataset_id: str, tag: Optional[str] = None) -> str:
        def _fetch_participants(dataset_id: str, tag: Optional[str] = None):
            participants = fetch_participants(dataset_id, tag)
            return json.dumps(
                {
                    "dataset_id": dataset_id,
                    "total_participants": len(participants),
                    "participant_ids": participants,
                },
                indent=2,
            )

        return safe_tool_execution(
            "FetchParticipantsTool",
            _fetch_participants,
            ["dataset_id"],
            dataset_id=dataset_id,
            tag=tag,
        )


class TaskTaxonomyTool(BaseTool):
    name: str = "get_task_taxonomy"
    description: str = (
        "Get the complete EEG task taxonomy with categories and subcategories"
    )
    args_schema: Type[BaseModel] = EmptyInput

    def _run(self) -> str:
        def _get_taxonomy():
            taxonomy = task_taxonomy()
            return json.dumps(taxonomy, indent=2)

        return safe_tool_execution("TaskTaxonomyTool", _get_taxonomy)


class ModalitiesTool(BaseTool):
    name: str = "get_modalities"
    description: str = (
        "Get all available electrode/channel modalities (EEG, EOG, EMG, etc.)"
    )
    args_schema: Type[BaseModel] = EmptyInput

    def _run(self) -> str:
        def _get_modalities():
            mods = modalities()
            return json.dumps(mods, indent=2)

        return safe_tool_execution("ModalitiesTool", _get_modalities)


class EEGBidsSpecsTool(BaseTool):
    name: str = "get_eeg_bids_specs"
    description: str = "Get the EEG BIDS specification reference"
    args_schema: Type[BaseModel] = EmptyInput

    def _run(self) -> str:
        def _get_bids_specs():
            return eeg_bids_specification()

        return safe_tool_execution("EEGBidsSpecsTool", _get_bids_specs)


class DefaultDatasetInfoTool(BaseTool):
    name: str = "get_default_dataset_info"
    description: str = "Get default dataset information from internal database"
    args_schema: Type[BaseModel] = DatasetIdInput

    def _run(self, dataset_id: str) -> str:
        def _get_default_info(dataset_id: str):
            info = get_default_dataset_info(dataset_id)
            return json.dumps(info, indent=2, default=str)

        return safe_tool_execution(
            "DefaultDatasetInfoTool",
            _get_default_info,
            ["dataset_id"],
            dataset_id=dataset_id,
        )


class MontagePositionsTool(BaseTool):
    name: str = "get_montage_positions"
    description: str = (
        "Get 3D electrode positions for all channels in a specified MNE montage. "
        "Returns channel names and their (x, y, z) coordinates in meters."
    )
    args_schema: Type[BaseModel] = MontageNameInput

    def _run(self, montage_name: str) -> str:
        def _get_montage_positions(montage_name: str):
            try:
                import mne

                # Load the specified montage
                montage = mne.channels.make_standard_montage(montage_name)

                # Get positions
                positions_dict = montage.get_positions()
                ch_pos = positions_dict.get("ch_pos", {})

                # Convert numpy arrays to lists for JSON serialization
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
                # Invalid montage name
                available_montages = mne.channels.get_builtin_montages()
                return json.dumps(
                    {
                        "error": f"Invalid montage name: {montage_name}",
                        "message": str(e),
                        "available_montages": available_montages,
                    },
                    indent=2,
                )

        return safe_tool_execution(
            "MontagePositionsTool",
            _get_montage_positions,
            ["montage_name"],
            montage_name=montage_name,
        )


class AllMontageMatchesTool(BaseTool):
    name: str = "find_all_montage_matches"
    description: str = (
        "Find ALL possible montage matches for a list of electrodes with detailed statistics. "
        "Returns match percentages, electrode positions, matched/unmatched channels for every available montage. "
        "Useful for comprehensive montage analysis and comparison."
    )
    args_schema: Type[BaseModel] = ChannelListInput

    def _run(self, channel_names: List[str]) -> str:
        def _find_all_montage_matches(channel_names: List[str]):
            all_matches = get_all_montage_matches(channel_names)

            # Check if there are any matches at all
            has_any_matches = any(
                match_info["match_to_input"] > 0 for match_info in all_matches.values()
            )

            if not has_any_matches:
                error_message = {
                    "error": "No matching electrode names found in any montage",
                    "input_channels": channel_names,
                    "suggestion": "Try converting the channel names to standard names first using the convert_to_standard_names function, then run this tool again with the converted names.",
                    "total_montages_analyzed": len(all_matches),
                    "best_match_percentage": 0.0,
                }
                return json.dumps(error_message, indent=2)

            # Sort montages by match_to_input percentage (descending)
            sorted_matches = dict(
                sorted(
                    all_matches.items(),
                    key=lambda x: x[1]["match_to_input"],
                    reverse=True,
                )
            )

            # Create summary statistics
            summary = {
                "input_channels": channel_names,
                "total_montages_analyzed": len(sorted_matches),
                "best_match": {
                    "montage": (
                        list(sorted_matches.keys())[0] if sorted_matches else None
                    ),
                    "match_percentage": (
                        list(sorted_matches.values())[0]["match_to_input"] * 100
                        if sorted_matches
                        else 0
                    ),
                },
                "perfect_matches": [
                    name
                    for name, info in sorted_matches.items()
                    if info["match_to_input"] == 1.0
                ],
                "good_matches_90_plus": [
                    name
                    for name, info in sorted_matches.items()
                    if info["match_to_input"] >= 0.9
                ],
            }
            return json.dumps(summary, indent=2)

        return safe_tool_execution(
            "AllMontageMatchesTool",
            _find_all_montage_matches,
            ["channel_names"],
            channel_names=channel_names,
        )


class ListConfigurationFilesTool(BaseTool):
    name: str = "list_configuration_files"
    description: str = (
        "List all downloaded configuration files. "
        "Returns relative paths to all configuration files that have been downloaded."
    )
    args_schema: Type[BaseModel] = EmptyInput
    base_dir: str = ""

    def __init__(self, base_dir: str = "", **kwargs):
        super().__init__(**kwargs)
        self.base_dir = base_dir

    def _run(self) -> str:
        def _list_config_files():
            directory_path = self.base_dir
            if not os.path.exists(directory_path):
                return f"Error: Directory '{directory_path}' does not exist"
            if not os.path.isdir(directory_path):
                return f"Error: '{directory_path}' is not a directory"

            files = []
            for root, dirs, filenames in os.walk(directory_path):
                for filename in filenames:
                    if filename.endswith(ALLOWED_EXTENSIONS):
                        abs_path = os.path.join(root, filename)
                        rel_path = os.path.relpath(abs_path, directory_path)
                        files.append(rel_path)

            files.sort()
            return json.dumps({"files": files, "count": len(files)}, indent=2)

        return safe_tool_execution("ListConfigurationFilesTool", _list_config_files)


class ReadConfigurationFileTool(BaseTool):
    name: str = "read_configuration_file"
    description: str = (
        "Read the contents of a downloaded configuration file. "
        "Provide a relative path (e.g., 'dataset_description.json' or 'sub-01/eeg/sub-01_channels.tsv')."
    )
    args_schema: Type[BaseModel] = RelativeFilePathInput
    base_dir: str = ""

    def __init__(self, base_dir: str = "", **kwargs):
        super().__init__(**kwargs)
        self.base_dir = base_dir

    def _run(self, file_path: str) -> str:
        def _read_config_file(file_path: str):
            abs_path = os.path.join(self.base_dir, file_path)
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

        return safe_tool_execution(
            "ReadConfigurationFileTool",
            _read_config_file,
            ["file_path"],
            file_path=file_path,
        )


class ElectrodeToMontageMappingTool(BaseTool):
    name: str = "get_electrode_to_montage_mapping"
    description: str = (
        "Get a complete mapping of all electrode names to the montages they belong to. "
        "Returns a dictionary where each electrode name maps to a comma-separated list of montage names. "
        "Useful for understanding which montages contain specific electrodes, and for matching electrode names to standard names."
    )
    args_schema: Type[BaseModel] = EmptyInput

    def _run(self) -> str:
        def _get_electrode_mapping():
            mapping = get_all_electrode_names_to_montage_mapping()

            # Create a summary with statistics
            summary = {
                "total_electrodes": len(mapping),
                "electrode_mapping": mapping,
                "montage_coverage": {
                    montage: sum(
                        1 for montages in mapping.values() if montage in montages
                    )
                    for montage in set(
                        montage
                        for montages in mapping.values()
                        for montage in montages.split(",")
                    )
                },
            }

            return json.dumps(summary, indent=2)

        return safe_tool_execution(
            "ElectrodeToMontageMappingTool", _get_electrode_mapping
        )


class DownloadConfigurationFileTool(BaseTool):
    name: str = "download_configuration_file"
    description: str = (
        "Download a specific configuration file from an OpenNeuro dataset via AWS S3. "
        "This tool downloads small configuration files on-demand (≤5 MB). "
        "Supported file types: .json, .tsv, .txt, .md, .tsv.gz. "
        "The file will be cached locally to avoid redundant downloads. "
        "Provide relative file paths (e.g., 'dataset_description.json' or 'sub-01/eeg/sub-01_channels.tsv')."
    )
    args_schema: Type[BaseModel] = DownloadFileInput
    base_dir: str = ""

    def __init__(self, base_dir: str = "", **kwargs):
        super().__init__(**kwargs)
        self.base_dir = base_dir

    def _run(self, dataset_id: str, file_path: str) -> str:
        def _download_file(dataset_id: str, file_path: str):
            MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
            target_dir = self.base_dir

            if not any(file_path.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
                return json.dumps(
                    {
                        "error": f"File type not allowed. File must be one of: {', '.join(ALLOWED_EXTENSIONS)}",
                        "file_path": file_path,
                    },
                    indent=2,
                )

            local_path = os.path.join(target_dir, file_path)

            if os.path.exists(local_path):
                logger.info(f"File already cached locally: {local_path}")
                with open(local_path, "r", encoding="utf-8") as f:
                    content = f.read()

                preview = content[:500] if len(content) > 500 else content
                return json.dumps(
                    {
                        "status": "cached",
                        "file_path": file_path,
                        "size_bytes": os.path.getsize(local_path),
                        "content_preview": preview,
                    },
                    indent=2,
                )

            try:
                file_size = get_s3_file_size(dataset_id, file_path)

                if file_size > MAX_FILE_SIZE:
                    return json.dumps(
                        {
                            "error": f"File too large ({file_size:,} bytes). Maximum allowed: {MAX_FILE_SIZE:,} bytes (5 MB)",
                            "file_path": file_path,
                            "file_size": file_size,
                        },
                        indent=2,
                    )

                download_file_from_s3(dataset_id, file_path, target_dir)

                with open(local_path, "r", encoding="utf-8") as f:
                    content = f.read()

                preview = content[:500] if len(content) > 500 else content
                return json.dumps(
                    {
                        "status": "downloaded",
                        "file_path": file_path,
                        "size_bytes": file_size,
                        "content_preview": preview,
                    },
                    indent=2,
                )

            except Exception as e:
                return json.dumps(
                    {
                        "error": str(e),
                        "file_path": file_path,
                        "dataset_id": dataset_id,
                    },
                    indent=2,
                )

        return safe_tool_execution(
            "DownloadConfigurationFileTool",
            _download_file,
            ["dataset_id", "file_path"],
            dataset_id=dataset_id,
            file_path=file_path,
        )


class AnalyzeAllRecordingsInput(BaseModel):
    """Input class for batch recording analysis tool."""

    dataset_id: str = Field(description="Dataset ID (e.g., 'ds000247')")
    channel_maps: Dict[str, List[str]] = Field(
        description="Channel maps from ChannelAgent: {map_id: [channel_names]}"
    )


EEG_DATA_EXTENSIONS = (
    "_eeg.cdt",
    "_eeg.set",
    "_eeg.edf",
    "_eeg.bdf",
    "_eeg.fif",
    "_eeg.vhdr",
    "_eeg.cnt",
    "_eeg.mff",
)


def _parse_bids_filename(filename: str) -> Dict[str, Optional[str]]:
    """
    Extract BIDS entities from a filename.

    Args:
        filename: BIDS-compliant filename (e.g., 'sub-01_ses-02_task-rest_acq-eeg_run-01_eeg.json')

    Returns:
        Dictionary with extracted entities: subject_id, session, task, acquisition, run
    """
    import re

    basename = os.path.basename(filename)
    name_without_ext = (
        basename.rsplit("_eeg", 1)[0]
        if "_eeg" in basename
        else basename.rsplit(".", 1)[0]
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


def _extract_recordings_from_filenames(
    filenames: List[str],
    participants: Dict[str, Dict[str, str]],
    channel_maps: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """
    Extract recording info from EEG data filenames when sidecar JSON files are unavailable.

    Args:
        filenames: List of all filenames in the dataset
        participants: Participant info dictionary
        channel_maps: Channel maps from ChannelAgent

    Returns:
        List of recording info dictionaries
    """
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

        recordings.append(
            {
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
            }
        )

    return recordings


def _parse_channels_tsv(content: str) -> Dict[str, Any]:
    """
    Parse a channels.tsv file and extract channel information.

    Args:
        content: Raw TSV file content

    Returns:
        Dictionary with channel_names, channel_types, and bad_channels
    """
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


def _match_to_channel_map(
    channel_names: List[str],
    acquisition: Optional[str],
    channel_maps: Dict[str, List[str]],
) -> tuple[Optional[str], float]:
    """
    Match recording channels to the best channel map.

    Args:
        channel_names: List of channel names from the recording
        acquisition: Acquisition type from BIDS filename (e.g., 'headband', 'psg')
        channel_maps: Dictionary of {map_id: [channel_names]}

    Returns:
        Tuple of (best_match_map_id, match_score)
    """
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


def _parse_participants_tsv(content: str) -> Dict[str, Dict[str, str]]:
    """
    Parse participants.tsv and return participant info keyed by participant_id.

    Args:
        content: Raw TSV file content

    Returns:
        Dictionary of {participant_id: {field: value}}
    """
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


class AnalyzeAllRecordingsTool(BaseTool):
    """
    Batch analyze ALL recordings in a dataset.

    Downloads and parses all *_eeg.json sidecar files and *_channels.tsv files,
    extracts recording metadata, and matches recordings to channel maps.
    """

    name: str = "analyze_all_recordings"
    description: str = (
        "Batch analyze ALL recordings in a dataset. Downloads and parses all *_eeg.json "
        "sidecar files and *_channels.tsv files. Extracts duration, sampling rate, "
        "channel names, channel types, and bad channels. Matches recordings to the "
        "appropriate channel maps based on channel name overlap and acquisition type. "
        "Returns complete recording summaries ready for RecordingInfo generation."
    )
    args_schema: Type[BaseModel] = AnalyzeAllRecordingsInput
    base_dir: str = ""

    def __init__(self, base_dir: str = "", **kwargs):
        super().__init__(**kwargs)
        self.base_dir = base_dir

    def _download_and_read_file(self, dataset_id: str, file_path: str) -> Optional[str]:
        """Download a file from S3 and return its content."""
        local_path = os.path.join(self.base_dir, file_path)

        if os.path.exists(local_path):
            with open(local_path, "r", encoding="utf-8") as f:
                return f.read()

        try:
            download_file_from_s3(dataset_id, file_path, self.base_dir)
            with open(local_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to download {file_path}: {e}")
            return None

    def _run(self, dataset_id: str, channel_maps: Dict[str, List[str]]) -> str:
        def _analyze_all_recordings(
            dataset_id: str, channel_maps: Dict[str, List[str]]
        ) -> str:
            filenames = fetch_all_filenames(dataset_id)

            eeg_json_files = [f for f in filenames if f.endswith("_eeg.json")]
            channels_tsv_files = [f for f in filenames if f.endswith("_channels.tsv")]

            channels_tsv_map = {}
            for f in channels_tsv_files:
                key = f.rsplit("_channels.tsv", 1)[0]
                channels_tsv_map[key] = f

            participants = {}
            participants_content = self._download_and_read_file(
                dataset_id, "participants.tsv"
            )
            if participants_content:
                participants = _parse_participants_tsv(participants_content)
            else:
                participant_ids = fetch_participants(dataset_id)
                participants = {pid: {"participant_id": pid} for pid in participant_ids}
                logger.info(
                    f"participants.tsv not found, using API fallback: {len(participants)} participants"
                )

            recordings = []
            errors = []

            if not eeg_json_files:
                logger.info(
                    f"No *_eeg.json sidecar files found, extracting recordings from data filenames"
                )
                recordings = _extract_recordings_from_filenames(
                    filenames, participants, channel_maps
                )
                result = {
                    "dataset_id": dataset_id,
                    "total_recordings": len(recordings),
                    "total_participants": len(participants),
                    "recordings": recordings,
                    "participants": participants,
                    "channel_maps_provided": list(channel_maps.keys()),
                    "extracted_from_filenames": True,
                    "note": "Recording metadata (duration, sampling_frequency) unavailable - no sidecar JSON files found",
                    "errors": None,
                }
                return json.dumps(result, indent=2, default=str)

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

                    eeg_json_content = self._download_and_read_file(
                        dataset_id, eeg_json_path
                    )
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
                        channels_content = self._download_and_read_file(
                            dataset_id, channels_tsv_path
                        )
                        if channels_content:
                            channel_info = _parse_channels_tsv(channels_content)
                            channel_names = channel_info["channel_names"]
                            channel_types = channel_info["channel_types"]
                            bad_channels = channel_info["bad_channels"]
                            num_channels = len(channel_names)

                    matched_map_id, match_score = _match_to_channel_map(
                        channel_names, entities["acquisition"], channel_maps
                    )

                    subject_key = (
                        f"sub-{entities['subject_id']}"
                        if entities["subject_id"]
                        else None
                    )
                    participant_info = participants.get(subject_key, {})

                    recordings.append(
                        {
                            "recording_id": recording_id,
                            "subject_id": (
                                f"sub-{entities['subject_id']}"
                                if entities["subject_id"]
                                else None
                            ),
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
                        }
                    )

                except Exception as e:
                    errors.append({"file": eeg_json_path, "error": str(e)})
                    logger.error(f"Error processing {eeg_json_path}: {e}")

            result = {
                "dataset_id": dataset_id,
                "total_recordings": len(recordings),
                "total_participants": len(participants),
                "recordings": recordings,
                "participants": participants,
                "channel_maps_provided": list(channel_maps.keys()),
                "errors": errors if errors else None,
            }

            return json.dumps(result, indent=2, default=str)

        return safe_tool_execution(
            "AnalyzeAllRecordingsTool",
            _analyze_all_recordings,
            ["dataset_id", "channel_maps"],
            dataset_id=dataset_id,
            channel_maps=channel_maps,
        )


def create_metadata_tools(base_dir: str) -> List[BaseTool]:
    """Create metadata tools with the given base directory for file operations."""
    return [
        DatasetMetadataTool(),
        DatasetReadmeTool(),
        TaskTaxonomyTool(),
        DatasetFilenamesTool(),
        ListConfigurationFilesTool(base_dir=base_dir),
        ReadConfigurationFileTool(base_dir=base_dir),
        DownloadConfigurationFileTool(base_dir=base_dir),
    ]


def create_channel_tools(base_dir: str) -> List[BaseTool]:
    """Create channel tools with the given base directory for file operations."""
    return [
        DatasetReadmeTool(),
        DatasetFilenamesTool(),
        ModalitiesTool(),
        EEGBidsSpecsTool(),
        AllMontageMatchesTool(),
        MontagePositionsTool(),
        ElectrodeToMontageMappingTool(),
        ListConfigurationFilesTool(base_dir=base_dir),
        ReadConfigurationFileTool(base_dir=base_dir),
        DownloadConfigurationFileTool(base_dir=base_dir),
    ]


def create_recording_tools(base_dir: str) -> List[BaseTool]:
    """Create recording tools with the given base directory for file operations."""
    return [
        AnalyzeAllRecordingsTool(base_dir=base_dir),
        ListConfigurationFilesTool(base_dir=base_dir),
        ReadConfigurationFileTool(base_dir=base_dir),
        DownloadConfigurationFileTool(base_dir=base_dir),
    ]
