import click

from .utils import get_datasets, load_config, CONFIG_PATH_CLICK_TYPE


@click.command()
@click.option("--config-path", type=CONFIG_PATH_CLICK_TYPE)
def list_datasets(config_path):
    """List available datasets."""
    click.echo("Available datasets:")
    config, _ = load_config(config_path)
    echo_dataset_list(config)


def echo_dataset_list(config):
    for dataset in get_datasets(config):
        to_echo = f"- {dataset.name}"
        if dataset.is_local:
            to_echo += f"\t (Local: {dataset.pipeline_path})"
        click.echo(to_echo)
