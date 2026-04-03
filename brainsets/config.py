from pathlib import Path
from typing import Optional

import yaml

CONFIG_FILE = Path.home() / ".brainsets.yaml"


def load_config(path: Path = CONFIG_FILE) -> Optional[dict]:
    """Load and validate the brainsets config file.

    Returns the config dict, or ``None`` if the file is missing or invalid.
    """
    if not path.exists():
        return None

    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None

    if not isinstance(config, dict):
        return None
    if "raw_dir" not in config or "processed_dir" not in config:
        return None

    return config


def save_config(config: dict, path: Path = CONFIG_FILE) -> Path:
    """Save the config dict to the brainsets config file."""
    with open(path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    return path


def get_processed_dir(path: Path = CONFIG_FILE) -> str:
    """Return ``processed_dir`` from config, or raise if unavailable."""
    config = load_config(path)
    if config is None:
        raise FileNotFoundError(
            f"Config not found at {path}. "
            "Please run `brainsets config set` or pass `root` explicitly."
        )
    return config["processed_dir"]
