import subprocess
from typing import Optional
from prompt_toolkit import prompt
import click

from .utils import (
    load_config,
    expand_path,
    get_dataset_names,
    get_dataset_info,
    AutoSuggestFromList,
    CONFIG_PATH_CLICK_TYPE,
    expand_path,
)
from .list import echo_dataset_list


@click.command()
@click.argument("dataset", type=str, required=False)
@click.option("-c", "--cores", default=4, help="Number of cores to use")
@click.option("--config-path", type=CONFIG_PATH_CLICK_TYPE)
def prepare(dataset: Optional[str], cores: int, config_path: Optional[str]):
    """Download and process a dataset."""
    config, config_path = load_config(config_path)

    if dataset is None:
        click.echo(f"Available datasets: ")
        echo_dataset_list(config)
        click.echo()
        dataset = prompt(
            "Dataset: ",
            auto_suggest=AutoSuggestFromList(get_dataset_names(config)),
        )

    click.echo(f"Preparing {dataset}...")

    dataset_info = get_dataset_info(dataset, config)

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
