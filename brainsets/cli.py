import click
import json
from pathlib import Path
import subprocess
import brainsets_pipelines


CONFIG_FILE = Path.home() / ".brainsets_config.json"
PIPELINES_PATH = Path(brainsets_pipelines.__path__[0])
DATASETS = [d.name for d in PIPELINES_PATH.iterdir() if d.is_dir()]


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {"raw_dir": None, "processed_dir": None}


def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


@click.group()
def cli():
    """Brainsets CLI tool."""
    pass


@cli.command()
@click.argument("dataset", type=click.Choice(DATASETS, case_sensitive=False))
@click.option("-c", "--cores", default=4, help="Number of cores to use")
@click.option("-v", "--verbose", is_flag=True, default=False)
def prepare(dataset, cores, verbose):
    """Download and process a specific dataset."""
    click.echo(f"Preparing {dataset}...")

    # Get config to check if directories are set
    config = load_config()
    if not config["raw_dir"] or not config["processed_dir"]:
        click.echo(
            "Error: Please set raw and processed directories first using 'brainsets config'"
        )
        return

    pipelines_dirpath = PIPELINES_PATH
    snakefile_filepath = pipelines_dirpath / "Snakefile"
    reqs_filepath = pipelines_dirpath / dataset / "requirements.txt"

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
    command = " ".join(command)

    # Run snakemake workflow in a temporary environemnt
    tmpdir = Path(config["processed_dir"]) / dataset / "tmp"
    return_code = run_in_temp_venv(command, reqs_filepath, tmpdir, verbose)
    if return_code == 0:
        click.echo(f"Successfully downloaded {dataset}")


@cli.command()
def list():
    """List available datasets."""
    click.echo("Available datasets:")
    for dataset in DATASETS:
        click.echo(f"- {dataset}")


@cli.command()
@click.option(
    "--raw",
    prompt="Enter raw data directory",
    type=click.Path(file_okay=False, dir_okay=True),
    required=False,
)
@click.option(
    "--processed",
    prompt="Enter processed data directory",
    type=click.Path(file_okay=False, dir_okay=True),
    required=False,
)
def config(raw, processed):
    """Set raw and processed data directories."""
    # Create directories if they don't exist
    import os

    # If no arguments provided, prompt for input
    if raw is None or processed is None:
        if raw is None:
            raw = click.prompt(
                "Enter raw data directory",
                type=click.Path(file_okay=False, dir_okay=True),
            )
        if processed is None:
            processed = click.prompt(
                "Enter processed data directory",
                type=click.Path(file_okay=False, dir_okay=True),
            )

    raw = Path(os.path.expanduser(raw))
    processed = Path(os.path.expanduser(processed))

    # Create directories
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    # Save config
    raw = str(raw)
    processed = str(processed)

    config = load_config()
    config["raw_dir"] = raw
    config["processed_dir"] = processed
    save_config(config)
    click.echo("Configuration updated successfully.")
    click.echo(f"Raw data directory: {raw}")
    click.echo(f"Processed data directory: {processed}")


import subprocess
import tempfile
import sys
import shutil
import os
import atexit


def run_in_temp_venv(
    command: str,
    requirements_file: Path,
    tmpdir: Path,
    verbose: bool = False,
):
    """Runs a command inside a temporary virtual environment."""
    uv_cmd = "uv" if verbose else "uv -q"

    # Create venv dir in tmpdir
    venv_dir = tmpdir / "venv"
    if os.path.exists(venv_dir):
        shutil.rmtree(venv_dir)
    venv_dir.mkdir(parents=True, exist_ok=False)

    # Register deletion of venv_dir at script exit
    def cleanup():
        shutil.rmtree(venv_dir, ignore_errors=True)
        click.echo(f"Deleted {venv_dir}")

    atexit.register(cleanup)

    with tempfile.TemporaryDirectory(dir=venv_dir) as tmpdir:
        try:
            click.echo(f"Creating a temporary isolated environment @ {tmpdir}")

            # Get base environment
            process = subprocess.run(
                ["uv", "pip", "freeze"],
                capture_output=True,
                check=True,
            )
            base_requirements = process.stdout.decode().split("\n")

            # Find brainsets package
            brainsets_package = [x for x in base_requirements if "brainsets" in x]
            if len(brainsets_package) > 1:
                raise RuntimeError(
                    f"Found {len(brainsets_package)} candidates for brainsets. "
                    "Don't know how to handle this situation. "
                    "This is a bug."
                )
            if len(brainsets_package) == 0:  # This should never happen in practice
                raise RuntimeError(
                    "Weird situation. Could not find a brainsets package installed. "
                    "Do you see brainsets in `uv pip freeze`?"
                )
            brainsets_package: str = brainsets_package[0]

            # Handle case where package is like
            # brainsets @ git+https://...@brachname
            if " " in brainsets_package and not brainsets_package.startswith("-e "):
                parts = brainsets_package.split(" ")
                if parts[0] == "brainsets" and parts[1] == "@":
                    brainsets_package = parts[2]
                else:
                    raise ValueError(
                        f"Unknown package format {brainsets_package} in `uv pip freeze`"
                    )

            click.echo(f"Brainsets installation detected: {brainsets_package}")

            # Create temp venv
            subprocess.run([sys.executable, "-m", "venv", tmpdir], check=True)

            # Install brainsets
            subprocess.run(
                (
                    f". {tmpdir}/bin/activate && "
                    f"{uv_cmd} pip install {brainsets_package}"
                ),
                shell=True,
                check=True,
                capture_output=False,
            )

            # Install extra requirements
            click.echo(f"Installing requirements from: {requirements_file}")
            subprocess.run(
                (
                    f". {tmpdir}/bin/activate && "
                    f"{uv_cmd} pip install -r {requirements_file}"
                ),
                shell=True,
                check=True,
                text=True,
                capture_output=False,
            )

            # Run command
            process = subprocess.run(
                f". {tmpdir}/bin/activate && {command}",
                shell=True,
                check=True,
                text=True,
                capture_output=False,
            )
            return process.returncode

        except subprocess.CalledProcessError as e:
            import traceback

            click.echo(f"Error: Command failed with return code {e.returncode}")
            click.echo(traceback.format_exc())

        except Exception as e:
            import traceback

            click.echo(f"Error: {str(e)}")
            click.echo(traceback.format_exc())


if __name__ == "__main__":
    cli()
