import os
from typing import List, Optional, Union
import click
import yaml
from pathlib import Path
import brainsets_pipelines
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion

CONFIG_FILE = Path.home() / ".brainsets.yaml"
PIPELINES_PATH = Path(brainsets_pipelines.__path__[0])


def expand_path(path: Union[str, Path]) -> Path:
    """
    Convert string path to absolute Path, expanding environment variables and user.
    """
    return Path(os.path.abspath(os.path.expandvars(os.path.expanduser(path))))


def load_config(path: Path = CONFIG_FILE):
    if path.exists():
        with open(path, "r") as f:
            return yaml.safe_load(f)
    else:
        raise FileNotFoundError(
            f"Config not found at {path}.\n" f"Please run `brainsets config`"
        )


def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    return CONFIG_FILE


def get_available_datasets():
    return [d.name for d in PIPELINES_PATH.iterdir() if d.is_dir()]


def debug_echo(msg: str, enable: bool):
    if enable:
        click.echo(f"DEBUG: {msg}")


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
