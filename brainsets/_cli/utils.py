import os
from pathlib import Path
import yaml
from typing import List, Optional, Tuple
import click
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
import brainsets_pipelines
from dataclasses import dataclass


PIPELINES_PATH = Path(brainsets_pipelines.__path__[0])
CONFIG_PATH_CLICK_TYPE = click.Path(exists=True, file_okay=True, dir_okay=False)


@dataclass
class DatasetInfo:
    name: str | Path
    pipeline_path: str | Path
    is_local: bool = False


def get_datasets(config) -> List[DatasetInfo]:
    default_pipelines = [
        DatasetInfo(d.name, d, is_local=False)
        for d in PIPELINES_PATH.iterdir()
        if d.is_dir()
    ]

    local_pipelines = [
        DatasetInfo(k, v, is_local=True)
        for k, v in config.get("local_datasets", {}).items()
    ]

    return default_pipelines + local_pipelines


def get_dataset_names(config) -> List[str]:
    return [d.name for d in get_datasets(config)]


def get_dataset_info(name, config):
    datasets = get_datasets(config)
    for dataset in datasets:
        if dataset.name == name:
            return dataset
    return None


def find_config_file() -> Path | None:
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


def load_config(config_file: Optional[Path | str]) -> Tuple[dict, Path]:
    if config_file is None:
        config_file = find_config_file()
        if not config_file:
            raise click.ClickException(
                "No configuration file found. "
                "Please run 'brainsets config init' to create one."
            )

    if isinstance(config_file, str):
        config_file = Path(os.path.expanduser(config_file))

    try:
        with open(config_file, "r") as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise click.ClickException(f"Invalid YAML in config file: {e}")
    except FileNotFoundError:
        raise click.ClickException(f"Config file not found: {config_file}")
    except PermissionError:
        raise click.ClickException(
            f"Permission denied when reading config file: {config_file}"
        )

    validate_config(config)
    return config, config_file


def save_config(config: dict, filepath: Path):
    click.echo(f"Saving configuration file at {filepath}")
    with open(filepath, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)


def validate_config(config: dict):
    """Validate that the configuration dictionary has required fields."""
    if "raw_dir" not in config:
        raise click.ClickException("Configuration missing required 'raw_dir' field")
    if "processed_dir" not in config:
        raise click.ClickException(
            "Configuration missing required 'processed_dir' field"
        )


def expand_path(path: str) -> Path:
    """Convert string path to absolute Path, expanding environment variables and user."""
    return Path(os.path.abspath(os.path.expandvars(os.path.expanduser(path))))


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
