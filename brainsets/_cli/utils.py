import os
from typing import Union
import click
from pathlib import Path
import brainsets_pipelines

from brainsets.config import CONFIG_FILE, load_config as _load_config

PIPELINES_PATH = Path(brainsets_pipelines.__path__[0])


def expand_path(path: Union[str, Path]) -> Path:
    """
    Convert string path to absolute Path, expanding environment variables and user.
    """
    return Path(os.path.abspath(os.path.expandvars(os.path.expanduser(path))))


def load_config(path: Path = CONFIG_FILE):
    """Load config or raise a :class:`click.ClickException` on failure."""
    config = _load_config(path)
    if config is None:
        raise click.ClickException(
            f"Config not found or invalid at {path}. "
            "Please run `brainsets config set`."
        )
    return config


def get_available_brainsets():
    ret = [d.name for d in PIPELINES_PATH.iterdir() if d.is_dir()]
    ret = [name for name in ret if not name.startswith((".", "_"))]
    return ret


def debug_echo(msg: str, enable: bool):
    if enable:
        click.echo(f"DEBUG: {msg}")
