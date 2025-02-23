import click
import subprocess

from .utils import PIPELINES_PATH, DATASETS, load_config


@click.command()
@click.argument("dataset", type=click.Choice(DATASETS, case_sensitive=False))
@click.option("-c", "--cores", default=4, help="Number of cores to use")
@click.option("-v", "--verbose", is_flag=True, default=False)
@click.option("--use-active-env", is_flag=True, default=False)
def prepare(dataset: str, cores: int, verbose: bool, use_active_env: bool):
    """Download and process a specific dataset."""
    click.echo(f"Preparing {dataset}...")

    # Get config to check if directories are set
    config = load_config()
    if not config["raw_dir"] or not config["processed_dir"]:
        click.echo(
            "Error: Please set raw and processed directories first using 'brainsets config'"
        )
        return

    snakefile_filepath = PIPELINES_PATH / "Snakefile"
    reqs_filepath = PIPELINES_PATH / dataset / "requirements.txt"

    # Construct base Snakemake command with configuration
    command = [
        "snakemake",
        "-s",
        str(snakefile_filepath),
        "--config",
        f"raw_dir={config['raw_dir']}",
        f"processed_dir={config['processed_dir']}",
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
