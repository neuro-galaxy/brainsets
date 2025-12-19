"""
Configuration constants for the Dataset Wizard.
"""

from typing import Literal

LLMProvider = Literal["ollama", "cerebras", "vertexai", "google"]

PROVIDER_MODELS = {
    "google": ["gemini-2.5-flash-lite", "gemini-2.5-flash"],
    "vertexai": ["Qwen3-Next-80B-Thinking"],
    "cerebras": ["gst-oss:200b"],
    "ollama": ["gpt-oss:20b"],
}

ALLOWED_EXTENSIONS = (".json", ".tsv", ".txt", ".md", ".csv", ".elp")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

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

DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_RETRIES = 3
DEFAULT_NUM_EXAMPLES = 2
DEFAULT_RECORDING_BATCH_SIZE = 100
