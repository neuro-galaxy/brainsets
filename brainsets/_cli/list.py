import click

from .utils import get_datasets


@click.command()
def list_datasets():
    """List available datasets."""
    click.echo("Available datasets:")
    echo_dataset_list()


def echo_dataset_list():
    for dataset in get_datasets():
        click.echo(f"- {dataset}")
