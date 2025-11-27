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
class DirectoryPathInput(BaseModel):
    """Input class for tools that require a directory path."""

    directory_path: str = Field(description="Path to directory")


class FilePathInput(BaseModel):
    """Input class for tools that require a file path."""

    file_path: str = Field(description="Path to file")


class DownloadFileInput(BaseModel):
    """Input class for downloading a specific file from OpenNeuro."""

    dataset_id: str = Field(description="Dataset ID (e.g., 'ds000247')")
    file_path: str = Field(
        description="Relative path to file within dataset (e.g., 'dataset_description.json' or 'sub-01/ses-01/eeg/sub-01_ses-01_channels.tsv')"
    )
    target_dir: str = Field(description="Directory where file should be downloaded")


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
    description: str = "List all downloaded configuration files in a directory"
    args_schema: Type[BaseModel] = DirectoryPathInput

    def _run(self, directory_path: str) -> str:
        def _list_config_files(directory_path: str):
            if not os.path.exists(directory_path):
                return f"Error: Directory '{directory_path}' does not exist"
            if not os.path.isdir(directory_path):
                return f"Error: '{directory_path}' is not a directory"

            files = []
            for root, dirs, filenames in os.walk(directory_path):
                for filename in filenames:
                    if filename.endswith((".json", ".tsv")):
                        abs_path = os.path.join(root, filename)
                        files.append(abs_path)

            files.sort()  # Sort for consistent ordering
            return json.dumps({"files": files, "count": len(files)}, indent=2)

        return safe_tool_execution(
            "ListConfigurationFilesTool",
            _list_config_files,
            ["directory_path"],
            directory_path=directory_path,
        )


class ReadConfigurationFileTool(BaseTool):
    name: str = "read_configuration_file"
    description: str = "Read the contents of a specific configuration file"
    args_schema: Type[BaseModel] = FilePathInput

    def _run(self, file_path: str) -> str:
        def _read_config_file(file_path: str):
            if not os.path.exists(file_path):
                return f"Error: File '{file_path}' does not exist"
            if not os.path.isfile(file_path):
                return f"Error: '{file_path}' is not a file"
            if not file_path.endswith((".json", ".tsv")):
                return (
                    f"Error: '{file_path}' is not a configuration file (.json or .tsv)"
                )

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Try to parse JSON files for better formatting
            if file_path.endswith(".json"):
                try:
                    parsed = json.loads(content)
                    return json.dumps(parsed, indent=2)
                except json.JSONDecodeError:
                    pass  # Return raw content if JSON parsing fails

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
        "Use this to access dataset metadata, channel files, participant info, etc."
    )
    args_schema: Type[BaseModel] = DownloadFileInput

    def _run(self, dataset_id: str, file_path: str, target_dir: str) -> str:
        def _download_file(dataset_id: str, file_path: str, target_dir: str):
            ALLOWED_EXTENSIONS = (".json", ".tsv", ".txt", ".md", ".tsv.gz")
            MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

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
                        "local_path": local_path,
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

                local_path = download_file_from_s3(dataset_id, file_path, target_dir)

                with open(local_path, "r", encoding="utf-8") as f:
                    content = f.read()

                preview = content[:500] if len(content) > 500 else content
                return json.dumps(
                    {
                        "status": "downloaded",
                        "local_path": local_path,
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
            ["dataset_id", "file_path", "target_dir"],
            dataset_id=dataset_id,
            file_path=file_path,
            target_dir=target_dir,
        )


# Categorize tools by agent responsibility
METADATA_TOOLS = [
    DatasetMetadataTool(),
    DatasetReadmeTool(),
    TaskTaxonomyTool(),
    DownloadConfigurationFileTool(),
]

CHANNEL_TOOLS = [
    DatasetReadmeTool(),
    ModalitiesTool(),
    EEGBidsSpecsTool(),
    AllMontageMatchesTool(),
    MontagePositionsTool(),
    ElectrodeToMontageMappingTool(),
    ListConfigurationFilesTool(),
    ReadConfigurationFileTool(),
    DownloadConfigurationFileTool(),
]

RECORDING_TOOLS = [
    FetchParticipantsTool(),
    DatasetReadmeTool(),
    DatasetFilenamesTool(),
    EEGBidsSpecsTool(),
    ListConfigurationFilesTool(),
    ReadConfigurationFileTool(),
    DownloadConfigurationFileTool(),
]
