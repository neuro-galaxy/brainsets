import sys
import shutil
import subprocess
import traceback
from contextlib import contextmanager
import click
import json
from pathlib import Path
import brainsets_pipelines
import re


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

    try:
        # Run snakemake workflow in a temporary environemnt
        tmpdir = Path(config["processed_dir"]) / dataset / "tmp"

        return_code = run_command_in_temp_venv(command, reqs_filepath, tmpdir, verbose)

        if return_code == 0:
            click.echo(f"Successfully downloaded {dataset}")
        else:
            click.echo(f"Error: Command failed with return code {return_code}")

        if tmpdir.exists():
            shutil.rmtree(tmpdir)

    except subprocess.CalledProcessError as e:
        click.echo(f"Error: Command failed with return code {e.returncode}")
        if verbose:
            click.echo(traceback.format_exc())

    except Exception as e:
        click.echo(f"Error: {str(e)}")
        if verbose:
            click.echo(traceback.format_exc())


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


@contextmanager
def temporary_venv(basedir: Path, verbose: bool = False):
    """Context manager for creating and cleaning up a temporary virtual environment.

    Args:
        basedir: Path where the temporary virtual environment will be created.
            It will be created at `basedir`/venv.
        verbose: If True, prints detailed progress information.

    Yields:
        Path: Path to the virtual environment directory
    """
    venv_dir = basedir / "venv"

    # Clean up existing tmpdir if needed
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
        if verbose:
            click.echo(f"Found existing {basedir}. Deleted.")
    venv_dir.mkdir(parents=True, exist_ok=False)

    try:
        # Create fresh venv
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
        yield venv_dir

    finally:
        # Clean up
        if venv_dir.exists():
            shutil.rmtree(venv_dir)
            if verbose:
                click.echo(f"Cleaned up {basedir}")


def run_command_in_temp_venv(
    command: str,
    reqs_filepath: Path,
    tmpdir: Path,
    verbose: bool = False,
) -> int:
    """Runs a command inside a temporary virtual environment.

    Creates an isolated virtual environment, installs the current brainsets package
    and additional requirements, then executes the specified command within this
    environment. The virtual environment is automatically cleaned up after execution.

    Args:
        command: Shell command to execute in the virtual environment.
        reqs_filepath: Path to a requirements.txt file containing additional
            dependencies to install.
        tmpdir: Path where the temporary virtual environment will be created.
        verbose: If True, prints detailed progress information.

    Returns:
        int: Return code from the executed command (0 for success, non-zero for failure).
            Returns None if an exception occurs during execution.
    """
    UV_CMD = "uv" if verbose else "uv -q"

    brainsets_package = _get_installed_brainsets_spec()
    click.echo(f"Brainsets installation detected: {brainsets_package}")

    with temporary_venv(tmpdir, verbose) as venv_dir:
        click.echo(f"Created virtual environment at {venv_dir}")
        VENV_ACTIVATE = venv_dir / "bin" / "activate"

        # Install brainsets
        click.echo(f"Installing brainsets '{brainsets_package}'")
        install_cmd = f". {VENV_ACTIVATE} && {UV_CMD} pip install {brainsets_package}"
        if verbose:
            click.echo(f"Running: {install_cmd}")
        subprocess.run(install_cmd, shell=True, check=True, capture_output=False)

        # Install requirements
        if reqs_filepath.exists():
            click.echo(f"Installing requirements from: {reqs_filepath}")
            install_cmd = (
                f". {VENV_ACTIVATE} && {UV_CMD} pip install -r {reqs_filepath}"
            )
            if verbose:
                click.echo(f"Running: {install_cmd}")
            subprocess.run(install_cmd, shell=True, check=True, capture_output=False)

        # Run requested command
        command = f". {VENV_ACTIVATE} && {command}"
        process = subprocess.run(command, shell=True, check=True, capture_output=False)
        return process.returncode


def _get_installed_brainsets_spec():
    """Get the currently installed brainsets package specification.

    Uses `uv pip freeze` to find the installed brainsets package and validates its format.
    The package can be in one of these formats:
    - brainsets==x.y.z (PyPI release)
    - -e /path/to/brainsets (editable install)
    - brainsets @ git+https://... (git install)

    Returns:
        str: Package specification that can be used with pip install.

    Raises:
        RuntimeError: If multiple brainsets packages are found, no package is found,
            or if the package format is not recognized.
    """
    PKG = "brainsets"

    # Get currently installed packages
    process = subprocess.run(["uv", "pip", "freeze"], capture_output=True, check=True)
    all_pkg_specs = process.stdout.decode().split("\n")

    # Find brainsets package
    """Regex used:
    brainsets==1.0.0                          ✓ PyPI release
    brainsets @ git+https://github.com/...    ✓ Git installation
    -e /path/to/brainsets                     ✓ Editable install
    -e /path/to/brainsets/                    ✓ Editable install with trailing slash
    brainsets-extra==1.0.0                    ✗ Different package
    mybrainsets==1.0.0                        ✗ Different package
    """
    pkg_pattern = re.compile(rf"^(?:{PKG}(?:==| @)|-e .*/{PKG}(?=/|$))")
    candidate_specs = [x for x in all_pkg_specs if pkg_pattern.match(x)]
    if len(candidate_specs) > 1:
        raise RuntimeError(
            f"Found {len(candidate_specs)} candidates for {PKG}: "
            f"{candidate_specs}.\n"
            "This normally should not occur. If detection of multiple packages with the "
            "name brainsets is not a issue from your end, then this is bug in brainsets CLI. "
            "Please report this issue at "
            "https://github.com/neuro-galaxy/brainsets/issues "
            "with this error message."
        )
    if len(candidate_specs) == 0:  # This should never happen in practice
        raise RuntimeError(
            f"Could not find a {PKG} installation.\n"
            "We use `uv pip freeze` to detect the current installation spec of brainsets, "
            "but could not find any. There are two possibilities: \n"
            "1. You have installed brainsets from a local clone, and the name of the "
            "clone directory is not 'brainsets'. In this case, please rename that directory "
            "to brainsets.\n"
            "2. This is a bug in Brainsets CLI. In this case, please report this issue "
            "https://github.com/neuro-galaxy/brainsets/issues "
            "with the output of `uv pip freeze`."
        )
    spec = candidate_specs[0]

    # Validate package format
    if spec.startswith(f"{PKG}=="):
        pass
    elif spec.startswith("-e "):
        pass
    elif spec.startswith(f"{PKG} @ "):
        # Handle case where package is like
        # brainsets @ git+https://...@brachname
        spec = spec.removeprefix(f"{PKG} @ ")
    else:
        raise RuntimeError(
            f"Unknown package format {spec} in `uv pip freeze`\n"
            "This is a bug. Please report this issue at "
            "https://github.com/neuro-galaxy/brainsets/issues "
            "with this error message."
        )

    return spec


if __name__ == "__main__":
    cli()
