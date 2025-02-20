import subprocess
from typing import Optional
from prompt_toolkit import prompt
import click

from .utils import (
    load_config,
    PIPELINES_PATH,
    expand_path,
    get_datasets,
    AutoSuggestFromList,
)
from .list import echo_dataset_list


@click.command()
@click.argument("dataset", type=str, required=False)
@click.option("-c", "--cores", default=4, help="Number of cores to use")
@click.option("--config-path", type=click.Path())
def prepare(dataset: Optional[str], cores: int, config_path: Optional[str]):
    """Download and process a dataset."""
    config, _ = load_config(config_path)

    if dataset is None:
        click.echo(f"Available datasets: ")
        echo_dataset_list()
        click.echo()
        dataset = prompt("Dataset: ", auto_suggest=AutoSuggestFromList(get_datasets()))

    click.echo(f"Preparing {dataset}...")

    snakefile_filepath = PIPELINES_PATH / dataset / "Snakefile"
    reqs_filepath = PIPELINES_PATH / dataset / "requirements.txt"

    # Construct base Snakemake command with configuration
    command = [
        "snakemake",
        "-s",
        str(snakefile_filepath),
        "--config",
        f"RAW_DIR={expand_path(config['raw_dir'])}",
        f"PROCESSED_DIR={expand_path(config['processed_dir'])}",
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
