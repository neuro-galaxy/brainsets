import subprocess
from prompt_toolkit import prompt
import click

from .utils import (
    CliConfig,
    expand_path,
    AutoSuggestFromList,
    expand_path,
)


@click.command()
@click.argument("dataset", type=str, required=False)
@click.option("-c", "--cores", default=4, help="Number of cores to use")
@click.pass_context
def prepare(ctx: click.Context, dataset: str | None, cores: int):
    """Download and process a dataset."""
    config: CliConfig = ctx.obj["CONFIG"]

    if dataset is None:
        click.echo(f"Available datasets: ")
        available_datasets = config.available_datasets
        for dataset in available_datasets:
            click.echo(f"- {dataset}")
        click.echo()

        dataset_names = [d.name for d in available_datasets]
        dataset = prompt(
            "Dataset: ",
            auto_suggest=AutoSuggestFromList(dataset_names),
        )

    dataset_info = config.get_dataset_info(dataset)

    pipeline_path = expand_path(dataset_info.pipeline_path)
    snakefile_filepath = pipeline_path / "Snakefile"
    reqs_filepath = pipeline_path / "requirements.txt"

    if not pipeline_path.exists():
        raise click.ClickException(
            f"Dataset pipeline path does not exist: {pipeline_path}"
        )
    if not snakefile_filepath.exists():
        raise click.ClickException(
            f"Dataset pipeline must contain a Snakefile: {snakefile_filepath}"
        )

    # Construct base Snakemake command with configuration
    command = [
        "snakemake",
        "-s",
        str(snakefile_filepath),
        "--config",
        f"RAW_DIR={config.raw_dir}",
        f"PROCESSED_DIR={config.processed_dir}",
        f"-c{cores}",
    ]

    # If dataset has additional requirements, prefix command with uv package manager
    if reqs_filepath.exists():
        uv_prefix_command = [
            "uv",
            "run",
            "--with-requirements",
            str(reqs_filepath),
            "--active",  # Prefer building temp environment on top of current venv
        ]
        command = uv_prefix_command + command
        click.echo(
            "Building temporary virtual environment using"
            f" requirements from {reqs_filepath}"
        )

    # Run snakemake workflow for dataset download with live output
    try:
        click.echo(f"Preparing {dataset}...")
        process = subprocess.run(
            command,
            check=True,
            capture_output=False,
            text=True,
        )

        if process.returncode == 0:
            click.echo(f"Successfully downloaded {dataset}")
        else:
            click.echo("Error downloading dataset")
    except subprocess.CalledProcessError as e:
        click.ClickException(f"Error: Command failed with return code {e.returncode}")
    except Exception as e:
        click.ClickException(f"Error: {str(e)}")
