import click

from .utils import get_datasets


@click.command()
def list():
    """List available datasets."""
    click.echo("Available datasets:")
    for dataset in get_datasets():
        click.echo(f"- {dataset}")
