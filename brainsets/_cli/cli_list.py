import click

from .utils import CliConfig, EXISTING_FILEPATH_CLICK_TYPE


@click.command()
@click.option("--config-path", type=EXISTING_FILEPATH_CLICK_TYPE)
def cli_list(config_path):
    """List available datasets."""
    config = CliConfig(config_path)
    click.echo("Available datasets:")
    for dataset in config.avaiable_datasets:
        click.echo(f"- {dataset}")
