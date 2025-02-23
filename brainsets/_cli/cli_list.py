import click

from .utils import get_available_datasets


@click.command()
def cli_list():
    """List available datasets."""
    click.echo("Available datasets:")
    for dataset in get_available_datasets():
        click.echo(f"- {dataset}")
