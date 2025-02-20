import click

from .config import config
from .prepare import prepare
from .list import list_datasets


@click.group()
@click.version_option()
def cli():
    """
    Command line interface for downloading and preparing brainsets.

    \b
    Documentation: https://brainsets.readthedocs.io/en/latest/
    Project page: https://github.com/neuro-galaxy/brainsets
    """
    pass


cli.add_command(config)
cli.add_command(prepare)
cli.add_command(list_datasets, name="list")


if __name__ == "__main__":
    cli()
