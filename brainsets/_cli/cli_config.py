import click
from pathlib import Path

from .utils import load_config, save_config


@click.command()
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
