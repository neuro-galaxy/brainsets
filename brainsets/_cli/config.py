import os
from pathlib import Path
import yaml
from typing import Optional
import click
from prompt_toolkit import prompt

from .utils import load_config, save_config


@click.group(
    help="Create or manage configuration files",
)
def config():
    pass


@config.command()
@click.option(
    "--raw",
    help="Path for saving raw datasets",
    type=click.Path(),
)
@click.option(
    "--processed",
    help="Path for saving processed datasets",
    type=click.Path(),
)
@click.option(
    "--config-path",
    help="Path for saving the created configuration file",
    default="$HOME/.config/brainsets.yaml",
    show_default=True,
    type=click.Path(),
)
def init(raw: Optional[str], processed: Optional[str], config_path: Optional[str]):
    """Initialize brainsets configuration file.

    Creates a configuration file with paths for storing raw and processed datasets.
    If paths are not provided as command line arguments, prompts for input with defaults.
    The configuration file will be created at the specified path (default: ~/.config/brainsets.yaml).

    Raw and processed directories will be created if they don't exist.
    """
    # Get missing args from user prompts
    if raw is None or processed is None:
        default = "brainsets_data/raw"
        raw = prompt(f"Raw dataset directory [{default}]: ") or default

        default = "brainsets_data/processed"
        processed = prompt(f"Processed dataset directory [{default}]: ") or default

        default = "$HOME/.config/brainsets.yaml"
        config_path = prompt(f"Configuration file path [{default}]: ") or default

    # Convert all paths to absolute Paths
    raw_path = _expand_path(raw)
    processed_path = _expand_path(processed)
    config_path = _expand_path(config_path)

    # Create directories
    raw_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    config = {"raw_dir": str(raw_path), "processed_dir": str(processed_path)}
    save_config(config=config, filepath=config_path)


@config.command()
@click.option(
    "--config-path",
    "-c",
    help="Configuration file path",
    type=str,
)
def show(config_path: Optional[str]):
    config, config_file = load_config(config_path)
    click.echo(f"Config file found at: {config_file}")
    click.echo()
    click.echo(yaml.dump(config, default_flow_style=False, sort_keys=False))


def _expand_path(path: str) -> Path:
    """Convert string path to absolute Path, expanding environment variables and user."""
    return Path(os.path.abspath(os.path.expandvars(os.path.expanduser(path))))
