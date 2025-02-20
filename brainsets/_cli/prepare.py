import subprocess
import click
from .utils import get_datasets


@click.command()
@click.argument("dataset", type=click.Choice(get_datasets(), case_sensitive=False))
@click.option("-c", "--cores", default=4, help="Number of cores to use")
@click.option(
    "--config-path",
)
def prepare(dataset, cores):
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
        click.echo(f"Error: Command failed with return code {e.returncode}")
    except Exception as e:
        click.echo(f"Error: {str(e)}")
