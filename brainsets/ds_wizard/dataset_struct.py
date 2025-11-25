from typing import Optional, Dict, List
from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    """Structured output for dataset metadata"""

    name: str = Field(description="Dataset name in OpenNeuro")
    brainset_name: str = Field(
        description="Brainset name for the dataset, of the format <last_name_of_author>_<recognizable_word_from_title>_<dataset_id>_<year>. Example: 'smith_visual_ds001234_2023'"
    )
    version: str = Field(description="Dataset version, e.g. '1.0.0' or '2.1.3'")
    dataset_id: str = Field(description="Dataset identifier, e.g. 'ds001234'")
    dataset_summary: str = Field(
        description="Concise summary ≤150 words describing the purpose, methods, and key findings of the dataset"
    )
    task_description: str = Field(
        description="Concise task description ≤150 words explaining what participants were asked to do during the experiment"
    )
    task_category: str = Field(
        description="Best matching category from taxonomy or null. Example: 'Cognitive' or 'Motor'"
    )
    task_subcategory: str = Field(
        description="Best matching subcategory from taxonomy or null. Example: 'Working Memory' or 'Reaching'"
    )
    authors: Optional[list[str]] = Field(
        description="Authors from README or metadata as a list of strings, e.g. ['John Smith', 'Jane Doe', 'Robert Johnson']"
    )
    date: Optional[str] = Field(
        description="Date of the dataset when the dataset was created, in the format DD-MM-YYYY, e.g. '15-03-2023'"
    )


class Channel(BaseModel):
    """Structured output for channel"""

    new_name: str = Field(
        description="New name for the channel. This should be a MNE standard name for the channel. If you cannot match the channel to a standard name or the name is already a standard name, write an empty string. Examples: 'Fp1', 'Cz', 'O2'"
    )
    modality: str = Field(
        description="Channel type must be from the provided list of electrode types. Examples: 'EEG', 'EOG', 'EMG', 'ECG', 'MISC'"
    )
    unit: Optional[str] = Field(description="Channel unit. Examples: 'µV', 'mV', 'V'")
    x: Optional[float] = Field(
        description="Channel x coordinate in 3D space, e.g. -0.0456"
    )
    y: Optional[float] = Field(
        description="Channel y coordinate in 3D space, e.g. 0.0721"
    )
    z: Optional[float] = Field(
        description="Channel z coordinate in 3D space, e.g. 0.0893"
    )
    confidence: Optional[float] = Field(
        description="Confidence score between 0 and 1 indicating how confident you are in the mapping. Examples: 0.95 (high confidence), 0.6 (moderate confidence)"
    )


class ChannelMap(BaseModel):
    """Structured output for channel map"""

    map_name: str = Field(
        description="Map name, e.g. 'standard_64_channel' or 'biosemi_128'"
    )
    channels: dict[str, Channel] = Field(
        description="Dictionary of channels, the key is the channel name as it is reported in the dataset (e.g. 'A1', 'EEG001'), the value is the channel information"
    )
    device_name: Optional[str] = Field(
        description="Device name. Examples: 'BioSemi ActiveTwo', 'Brain Products actiCHamp', 'EGI NetAmps 300'"
    )
    device_manufacturer: Optional[str] = Field(
        description="Device manufacturer. Examples: 'BioSemi', 'Brain Products', 'EGI', 'ANT Neuro'"
    )


class RecordingInfo(BaseModel):
    """Structured output for recording info"""

    recording_id: str = Field(
        description="Recording identifier. Example: 'sub-01_ses-01_task-rest_run-01'"
    )
    subject_id: str = Field(description="Subject identifier. Example: 'sub-01'")
    task_id: str = Field(
        description="Task identifier. Examples: 'rest', 'nback', 'motor', 'visual'"
    )
    channel_map_id: str = Field(
        description="Channel map identifier. Examples: 'standard_64_channel', 'biosemi_128'"
    )
    duration_seconds: float = Field(
        description="Recording duration in seconds. Example: 600.5 (for a 10-minute recording)"
    )
    num_channels: int = Field(
        description="Number of channels. Example: 64, 128, or 256"
    )
    participant_info: Optional[dict[str, str]] = Field(
        description="Participant information, the key is the participant identifier, the value is the participant information. The participant information will depend on what is provided by the dataset. Example: {'age': '25', 'sex': 'F', 'handedness': 'right'}"
    )
    channels_to_remove: Optional[list[str]] = Field(
        description="Channels to remove. This is a list of channel names as they appear in the dataset, NOT new_name. Example: ['A32', 'B12', 'Status']"
    )


# Output wrapper for ChannelAgent
class ChannelMapsOutput(BaseModel):
    """Output schema for ChannelAgent"""

    channel_maps: Dict[str, ChannelMap] = Field(
        description="Dictionary of channel maps, where keys are map names and values are ChannelMap objects"
    )


# Output wrapper for RecordingAgent
class RecordingInfoOutput(BaseModel):
    """Output schema for RecordingAgent"""

    recording_info: List[RecordingInfo] = Field(
        description="List of recording information objects"
    )


class Dataset(BaseModel):
    """Structured output for dataset"""

    metadata: DatasetMetadata = Field(description="Dataset metadata")
    channel_maps: ChannelMapsOutput = Field(description="Channel maps")
    recording_info: RecordingInfoOutput = Field(description="Recording info")
