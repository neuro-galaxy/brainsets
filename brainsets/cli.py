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
import traceback
import atexit


def run_in_temp_venv(
    command: str,
    requirements_file: Path,
    tmpdir: Path,
    verbose: bool = False,
):
    """Runs a command inside a temporary virtual environment.

    Creates an isolated virtual environment, installs the current brainsets package
    and additional requirements, then executes the specified command within this
    environment. The virtual environment is automatically cleaned up after execution.

    Args:
        command: Shell command to execute in the virtual environment.
        requirements_file: Path to a requirements.txt file containing additional
            dependencies to install.
        tmpdir: Path where the temporary virtual environment will be created.
        verbose: If True, prints detailed progress information.

    Returns:
        int: Return code from the executed command (0 for success, non-zero for failure).
            Returns None if an exception occurs during execution.
    """
    UV_CMD = "uv" if verbose else "uv -q"

    # Create venv dir in tmpdir
    venv_dir = tmpdir / "venv"
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
        if verbose:
            click.echo(f"Found existing {venv_dir}. Deleting.")
    venv_dir.mkdir(parents=True, exist_ok=False)
    if verbose:
        click.echo(f"Created {venv_dir}")

    # Register deletion of venv_dir at script exit
    def cleanup():
        if venv_dir.exists():
            shutil.rmtree(venv_dir)
            if verbose:
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
            brainsets_packages = [x for x in base_requirements if "brainsets" in x]
            if len(brainsets_packages) > 1:
                raise RuntimeError(
                    f"Found {len(brainsets_packages)} candidates for brainsets: "
                    f"{brainsets_packages}\n"
                    "This might be a bug. Please report this issue at "
                    "https://github.com/neuro-galaxy/brainsets/issues "
                    "with this error message."
                )
            if len(brainsets_packages) == 0:  # This should never happen in practice
                raise RuntimeError(
                    "Could not find a brainsets package installed.\n"
                    "This might be a bug. Please report this issue at "
                    "https://github.com/neuro-galaxy/brainsets/issues "
                    "with this error message and the output of `uv pip freeze`."
                )
            brainsets_package = brainsets_packages[0]

            if brainsets_package.startswith("brainsets=="):
                pass
            elif brainsets_package.startswith("-e "):
                pass
            elif brainsets_package.startswith("brainsets @ "):
                # Handle case where package is like
                # brainsets @ git+https://...@brachname
                brainsets_package = brainsets_package.removeprefix("brainsets @ ")
            else:
                raise ValueError(
                    f"Unknown package format {brainsets_package} in `uv pip freeze`\n"
                    "This is a bug. Please report this issue at "
                    "https://github.com/neuro-galaxy/brainsets/issues "
                    "with this error message."
                )
            click.echo(f"Brainsets installation detected: {brainsets_package}")

            # Create temp venv
            subprocess.run([sys.executable, "-m", "venv", tmpdir], check=True)

            # Install brainsets
            subprocess.run(
                (
                    f". {tmpdir}/bin/activate && "
                    f"{UV_CMD} pip install {brainsets_package}"
                ),
                shell=True,
                check=True,
                capture_output=False,
            )

            # Install  requirements
            if requirements_file.exists():
                click.echo(f"Installing requirements from: {requirements_file}")
                subprocess.run(
                    (
                        f". {tmpdir}/bin/activate && "
                        f"{UV_CMD} pip install -r {requirements_file}"
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
            click.echo(f"Error: Command failed with return code {e.returncode}")
            click.echo(traceback.format_exc())

        except Exception as e:
            click.echo(f"Error: {str(e)}")
            click.echo(traceback.format_exc())


if __name__ == "__main__":
    cli()
