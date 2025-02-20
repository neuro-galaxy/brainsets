import os
from pathlib import Path
import yaml
from typing import List, Optional, Tuple
import click


def get_pipelines_path() -> Path:
    import brainsets_pipelines

    return Path(brainsets_pipelines.__path__[0])


def get_datasets() -> List[str]:
    return [d.name for d in get_pipelines_path().iterdir() if d.is_dir()]


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

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        return config, config_file


def save_config(config: dict, filepath: Path):
    click.echo(f"Saving configuration file at {filepath}")
    with open(filepath, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)
