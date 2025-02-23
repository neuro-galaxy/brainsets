from typing import Optional
import click
import subprocess
from prompt_toolkit import prompt

from .utils import (
    PIPELINES_PATH,
    load_config,
    AutoSuggestFromList,
    get_available_datasets,
    expand_path,
)


@click.command()
@click.argument("dataset", type=str, required=False)
@click.option("-c", "--cores", default=4, help="Number of cores to use")
@click.option("-v", "--verbose", is_flag=True, default=False)
@click.option("--use-active-env", is_flag=True, default=False)
@click.option(
    "--raw-dir",
    type=click.Path(file_okay=False),
    help="Path for storing raw data.",
)
@click.option(
    "--processed-dir",
    type=click.Path(file_okay=False),
    help="Path for storing processed brainset.",
)
def prepare(
    dataset: Optional[str],
    cores: int,
    verbose: bool,
    use_active_env: bool,
    raw_dir: Optional[str],
    processed_dir: Optional[str],
):
    """Download and process a specific dataset."""

    # Get raw and processed dirs
    if raw_dir is None or processed_dir is None:
        config = load_config()
        raw_dir = expand_path(raw_dir or config["raw_dir"])
        processed_dir = expand_path(processed_dir or config["processed_dir"])
    else:
        raw_dir = expand_path(raw_dir)
        processed_dir = expand_path(processed_dir)

    # Prompt user if dataset is not provided
    if dataset is None:
        click.echo(f"Available datasets: ")
        available_datasets = get_available_datasets()
        for dataset in available_datasets:
            click.echo(f"- {dataset}")
        click.echo()

        dataset = prompt(
            "Enter dataset name: ",
            auto_suggest=AutoSuggestFromList(available_datasets),
        )

    click.echo(f"Preparing {dataset}...")
    click.echo(f"Raw data directory: {raw_dir}")
    click.echo(f"Processed data directory: {processed_dir}")

    snakefile_filepath = PIPELINES_PATH / "Snakefile"
    reqs_filepath = PIPELINES_PATH / dataset / "requirements.txt"

    # Construct base Snakemake command with configuration
    command = [
        "snakemake",
        "-s",
        str(snakefile_filepath),
        "--config",
        f"raw_dir={raw_dir}",
        f"processed_dir={processed_dir}",
        f"-c{cores}",
        f"{dataset}",
        "--verbose" if verbose else "--quiet",
    ]

    if use_active_env:
        click.echo(
            "WARNING: Working in active environment due to --use-active-env.\n"
            "         This mode is only intended for brainset development purposes."
        )
        if reqs_filepath.exists():
            click.echo(
                f"WARNING: {reqs_filepath} found.\n"
                f"         These will not be installed automatically when --use-active-env flag is used.\n"
                f"         Make sure to install necessary requirements manually."
            )
    else:
        if reqs_filepath.exists():
            # If dataset has additional requirements, prefix command with uv package manager
            if not use_active_env:
                uv_prefix_command = [
                    "uv",
                    "run",
                    "--with-requirements",
                    str(reqs_filepath),
                    "--isolated",
                    "--no-project",
                ]
                if verbose:
                    uv_prefix_command.append("--verbose")

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
        click.echo(f"Error: Command failed with return code {e.returncode}")
    except Exception as e:
        click.echo(f"Error: {str(e)}")
