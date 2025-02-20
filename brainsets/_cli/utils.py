from __future__ import annotations
from functools import cached_property
import os
from pathlib import Path
import yaml
from typing import Dict, List
import click
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
import brainsets_pipelines
from dataclasses import dataclass, field


PIPELINES_PATH = Path(brainsets_pipelines.__path__[0])
EXISTING_FILEPATH_CLICK_TYPE = click.Path(exists=True, file_okay=True, dir_okay=False)
EXISTING_DIRPATH_CLICK_TYPE = click.Path(exists=True, file_okay=False, dir_okay=True)


def expand_path(path: str) -> Path:
    """Convert string path to absolute Path, expanding environment variables and user."""
    return Path(os.path.abspath(os.path.expandvars(os.path.expanduser(path))))


class CliConfig:

    def __init__(self, config_path: str | Path | None, error: bool = True):
        if config_path is None:
            config_path = self.find_config_file()
            if config_path is None and error == True:
                raise click.ClickException(
                    "No configuration file found. "
                    "Please run 'brainsets config init' to create one."
                )

        config_path = expand_path(config_path)
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        self._validate_config_dict(config_dict)

        self.config_path = expand_path(config_path)
        self.raw_dir = expand_path(config_dict["raw_dir"])
        self.processed_dir = expand_path(config_dict["processed_dir"])
        self.local_datasets = {
            k: Path(v) for k, v in config_dict.get("local_datasets", {}).items()
        }

    def save(self) -> Path:
        config_dict = {
            "raw_dir": str(self.raw_dir),
            "processed_dir": str(self.processed_dir),
        }
        if len(self.local_datasets) > 0:
            config_dict["local_datasets"] = {
                k: str(v) for k, v in self.local_datasets.items()
            }

        click.echo(f"Saving configuration file at {self.config_path}")
        with open(self.config_path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False)

    @staticmethod
    def _validate_config_dict(config_dict):
        """Validate that the configuration dictionary has required fields."""
        if "raw_dir" not in config_dict:
            raise click.ClickException("Configuration missing required 'raw_dir' field")
        if "processed_dir" not in config_dict:
            raise click.ClickException(
                "Configuration missing required 'processed_dir' field"
            )

    @cached_property
    def available_datasets(self) -> List[CliDatasetInfo]:
        """Get list of all available datasets from default and local pipelines."""
        default_pipelines = [
            CliDatasetInfo(d.name, expand_path(d), is_local=False)
            for d in PIPELINES_PATH.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        ]
        local_pipelines = [
            CliDatasetInfo(k, v, is_local=True) for k, v in self.local_datasets.items()
        ]
        return default_pipelines + local_pipelines

    def get_dataset_info(self, name: str, error: bool = True) -> CliDatasetInfo:
        """Get dataset info for a single dataset"""
        datasets = self.available_datasets
        for dataset in datasets:
            if dataset.name == name:
                return dataset

        if error:
            raise click.ClickException(
                f"Could not find dataset '{name}' in configuration file at "
                f"{self.config_path}"
            )

        return None

    @staticmethod
    def find_config_file():
        """Search for brainsets configuration file in standard locations.

        Searches for .brainsets.yaml in the following locations (in order):
        1. Current directory and all parent directories
        2. ~/.config/brainsets.yaml
        3. ~/.brainsets.yaml

        Returns:
            Path | None: Path to the first configuration file found, or None if no config file exists
        """
        # Try to find config file in curr directory and all parents
        current = Path.cwd()
        while current != current.parent:
            config_file = current / ".brainsets.yaml"
            if config_file.exists():
                return config_file
            current = current.parent

        # Try to find config in ~/.config/brainsets.yaml
        config_file = Path.home() / ".config/brainsets.yaml"
        if config_file.exists():
            return config_file

        # Try to find config in home dir
        config_file = Path.home() / ".brainsets.yaml"
        if config_file.exists():
            return config_file

        return None

    def __repr__(self):
        ans = []
        ans.append(f"Config file path: {self.config_path}")
        ans.append(f"Raw data directory: {self.raw_dir}")
        ans.append(f"Processed data directory: {self.processed_dir}")
        if len(self.local_datasets) > 0:
            ans.append(f"Local datasets:")
            for name, pipeline_path in self.local_datasets.items():
                ans.append(f"- {name} @ {pipeline_path}")
        return "\n".join(ans)


@dataclass
class CliDatasetInfo:
    name: str
    pipeline_path: Path
    is_local: bool = False

    def __repr__(self):
        if self.is_local:
            return f"{self.name}  (Local: {self.pipeline_path})"
        else:
            return self.name


class AutoSuggestFromList(AutoSuggest):
    """
    Give suggestions based on the lines in the history.
    """

    def __init__(self, suggestion_list: List[str]):
        self.suggestion_list = suggestion_list

    def get_suggestion(self, buffer, document) -> Suggestion | None:
        # Consider only the last line for the suggestion.
        text = document.text.rsplit("\n", 1)[-1]

        # Only create a suggestion when this is not an empty line.
        if text.strip():
            # Find first matching line in history.
            for string in reversed(self.suggestion_list):
                for line in reversed(string.splitlines()):
                    if line.startswith(text):
                        return Suggestion(line[len(text) :])

        return None
