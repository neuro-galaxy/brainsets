import click
from prompt_toolkit import prompt

from .utils import CliConfig, expand_path


@click.group(
    help="Create or manage configuration files",
)
def config():
    pass


@config.command()
@click.option(
    "--raw",
    help="Path for saving raw datasets",
    type=click.Path(),
)
@click.option(
    "--processed",
    help="Path for saving processed datasets",
    type=click.Path(),
)
@click.option(
    "--config-path",
    help="Path for saving the created configuration file",
    default="$HOME/.config/brainsets.yaml",
    show_default=True,
    type=click.Path(),
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    default=False,
    help="Allow overwriting existing config file",
)
def init(raw: str | None, processed: str | None, config_path: str | None, force: bool):
    """Initialize brainsets configuration file.

    Creates a configuration file with paths for storing raw and processed datasets.
    If paths are not provided as command line arguments, prompts for input with defaults.
    The configuration file will be created at the specified path (default: ~/.config/brainsets.yaml).

    Raw and processed directories will be created if they don't exist.
    """
    # Get missing args from user prompts
    if raw is None or processed is None:
        default = "brainsets_data/raw"
        raw = prompt(f"Raw dataset directory [{default}]: ") or default

        default = "brainsets_data/processed"
        processed = prompt(f"Processed dataset directory [{default}]: ") or default

        default = "$HOME/.config/brainsets.yaml"
        config_path = prompt(f"Configuration file path [{default}]: ") or default

    # Convert all paths to absolute Paths
    raw_dir = expand_path(raw)
    processed_dir = expand_path(processed)
    config_path = expand_path(config_path)

    # Check config file does not exist
    if config_path.exists() and not force:
        raise click.ClickException(
            f"Configuration file already exists at {config_path}. "
            "Use --force to overwrite."
        )

    config = CliConfig(
        raw_dir=raw_dir, processed_dir=processed_dir, config_path=config_path
    )
    config.save()

    # Create directories
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)


@config.command()
@click.option(
    "--config-path",
    "-c",
    help="Configuration file path",
    type=str,
)
def show(config_path: str | None):
    """Display the current configuration settings.

    Shows the contents of the brainsets configuration file, including paths for raw and
    processed datasets. If --config-path is not specified, looks for the configuration
    file in the default location ($HOME/.config/brainsets.yaml).
    """
    config = CliConfig.load(config_path)
    click.echo(config)
