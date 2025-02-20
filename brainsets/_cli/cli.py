import click

from .cli_config import config
from .cli_prepare import prepare
from .cli_list import list_datasets
from .cli_add import add
from .cli_completion import completion


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
cli.add_command(add)
cli.add_command(completion)


if __name__ == "__main__":
    cli()
