from pathlib import Path
import click

from .utils import (
    CliConfig,
    expand_path,
    EXISTING_DIRPATH_CLICK_TYPE,
)


@click.command()
@click.argument("pipeline-path", type=EXISTING_DIRPATH_CLICK_TYPE)
@click.argument("name", type=str, required=False)
@click.option("-f", "--force", default=False, is_flag=True)
@click.pass_context
def add(ctx: click.Context, pipeline_path: str, name: str | None, force: bool):
    """Add a local dataset to the brainsets configuration.

    This command registers a local dataset directory with brainsets. The directory
    must contain a Snakefile and will be referenced by the provided name.
    """
    pipeline_path = expand_path(pipeline_path)
    _validate_local_pipeline(pipeline_path)

    config: CliConfig = ctx.obj["CONFIG"]

    if name is None:
        name = pipeline_path.name

    existing_dataset_info = config.get_dataset_info(name, error=False)
    if existing_dataset_info is not None:
        existing_pipeline_path = existing_dataset_info.pipeline_path
        if not force:
            raise click.ClickException(
                f"Dataset '{name}' already registered.\n"
                f"Existing pipeline path: {existing_pipeline_path}.\n"
                "Use --force/-f to overwrite."
            )
        else:
            click.echo(
                f"Updating pipeline for ({name}).\n"
                f"Previous pipeline path: {existing_pipeline_path}\n"
                f"New pipeline path: {pipeline_path}",
            )

    config.local_datasets[name] = str(pipeline_path)
    config.save()


def _validate_local_pipeline(path: Path):
    if not path.exists():
        raise click.ClickException(f"Dataset path does not exist: {path}")

    if not (path / "Snakefile").exists():
        raise click.ClickException(f"Dataset path must contain a Snakefile: {path}")
