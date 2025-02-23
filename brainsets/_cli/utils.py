from typing import List, Optional
import yaml
from pathlib import Path
import brainsets_pipelines
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion

CONFIG_FILE = Path.home() / ".brainsets.yaml"
PIPELINES_PATH = Path(brainsets_pipelines.__path__[0])


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return yaml.safe_load(f)
    return {"raw_dir": None, "processed_dir": None}


def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)


def get_available_datasets():
    return [d.name for d in PIPELINES_PATH.iterdir() if d.is_dir()]


class AutoSuggestFromList(AutoSuggest):
    """Provides auto-completion suggestions from a predefined list of strings.

    Args:
        suggestion_list (List[str]): A list of strings to use as the source of
            auto-completion suggestions.
    """

    def __init__(self, suggestion_list: List[str]):
        self.suggestion_list = suggestion_list

    def get_suggestion(self, buffer, document) -> Optional[Suggestion]:
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
