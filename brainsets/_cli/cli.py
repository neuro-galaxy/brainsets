import click
from pathlib import Path

from .cli_config import config
from .cli_prepare import prepare
from .cli_list import cli_list

from .utils import load_config, CONFIG_FILE


@click.group()
def cli():
    """
    Brainsets CLI\n
    A command line interface for downloading and preparing brainsets.

    \b
    Documentation: https://brainsets.readthedocs.io/en/latest
    Project page: https://github.com/neuro-galaxy/brainsets
    """
    pass


cli.add_command(prepare)
cli.add_command(cli_list, name="list")
cli.add_command(config)


if __name__ == "__main__":
    cli()
