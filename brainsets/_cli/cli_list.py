import click

from .utils import get_available_brainsets


@click.command()
def cli_list():
    """List available datasets."""
    click.echo("Available datasets:")
    for dataset in get_available_brainsets():
        click.echo(f"- {dataset}")
