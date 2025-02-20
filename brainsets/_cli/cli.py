import click
import brainsets_pipelines

from .config import config
from .prepare import prepare


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


@cli.command()
def list():
    """List available datasets."""
    click.echo("Available datasets:")
    for dataset in DATASETS:
        click.echo(f"- {dataset}")


if __name__ == "__main__":
    cli()
