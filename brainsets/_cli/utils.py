import os
from pathlib import Path
import yaml
from typing import List, Optional, Tuple
import click
import brainsets_pipelines


PIPELINES_PATH = Path(brainsets_pipelines.__path__[0])


def get_datasets() -> List[str]:
    return [d.name for d in PIPELINES_PATH.iterdir() if d.is_dir()]


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
