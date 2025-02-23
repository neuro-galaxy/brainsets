import json
from pathlib import Path
import brainsets_pipelines

CONFIG_FILE = Path.home() / ".brainsets_config.json"
PIPELINES_PATH = Path(brainsets_pipelines.__path__[0])
DATASETS = [d.name for d in PIPELINES_PATH.iterdir() if d.is_dir()]


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {"raw_dir": None, "processed_dir": None}


def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
