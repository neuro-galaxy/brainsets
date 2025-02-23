import yaml
from pathlib import Path
import brainsets_pipelines

CONFIG_FILE = Path.home() / ".brainsets.yaml"
PIPELINES_PATH = Path(brainsets_pipelines.__path__[0])
DATASETS = [d.name for d in PIPELINES_PATH.iterdir() if d.is_dir()]


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return yaml.safe_load(f)
    return {"raw_dir": None, "processed_dir": None}


def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)
