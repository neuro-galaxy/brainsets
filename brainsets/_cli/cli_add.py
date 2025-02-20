from pathlib import Path
import click

from .utils import CliConfig, expand_path, CONFIG_PATH_CLICK_TYPE


@click.command()
@click.argument("name", type=str)
@click.argument(
    "pipeline-path", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.option("--config-path", type=CONFIG_PATH_CLICK_TYPE)
@click.option("-u", "--update", default=False, is_flag=True)
def add(name, pipeline_path, config_path, update):
    """Add a local dataset to the brainsets configuration.

    This command registers a local dataset directory with brainsets. The directory
    must contain a Snakefile and will be referenced by the provided name.
    """
    pipeline_path = expand_path(pipeline_path)
    _validate_local_pipeline(pipeline_path)

    config = CliConfig.load(config_path)

    if name in config.local_datasets:
        if not update:
            raise click.ClickException(
                f"Dataset '{name}' already registered for {config.local_datasets[name]}."
                " Use --update to overwrite."
            )
        else:
            click.echo(
                f"Updating dataset '{name}' path from {config.local_datasets[name]} "
                f"to {pipeline_path}"
            )

    config.local_datasets[name] = str(pipeline_path)
    config.save()


def _validate_local_pipeline(path: Path):
    if not path.exists():
        raise click.ClickException(f"Dataset path does not exist: {path}")

    if not (path / "Snakefile").exists():
        raise click.ClickException(f"Dataset path must contain a Snakefile: {path}")
