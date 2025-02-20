import click

from .utils import CliConfig, EXISTING_FILEPATH_CLICK_TYPE


@click.command()
@click.pass_context
def cli_list(ctx: click.Context):
    """List available datasets."""
    config = ctx.obj["CONFIG"]
    click.echo("Available datasets:")
    for dataset in config.available_datasets:
        click.echo(f"- {dataset}")
