import click

from .utils import DATASETS


@click.command()
def cli_list():
    """List available datasets."""
    click.echo("Available datasets:")
    for dataset in DATASETS:
        click.echo(f"- {dataset}")
