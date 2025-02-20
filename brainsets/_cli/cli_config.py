import click
from prompt_toolkit import prompt

from .utils import CliConfig, expand_path


@click.group(help="Initialize or view configuration files")
@click.pass_context
def config(ctx: click.Context):
    if ctx.invoked_subcommand != "init":
        ctx.obj["CONFIG"] = CliConfig(ctx.obj["CONFIG_PATH"])
        ctx.obj.pop("CONFIG_PATH")


@config.command()
@click.option(
    "--raw",
    help="Path for saving raw datasets",
    type=click.Path(file_okay=False, dir_okay=True),
)
@click.option(
    "--processed",
    help="Path for saving processed datasets",
    type=click.Path(file_okay=False, dir_okay=True),
)
@click.option(
    "--config-path",
    help="Path for saving the created configuration file (default: $HOME/.config/brainsets.yaml)",
    type=click.Path(file_okay=True, dir_okay=False),
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

    if config_path.name != ".brainsets.yaml":
        raise click.ClickException("Config file must be named '.brainsets.yaml'")

    # Check config file does not exist
    if config_path.exists() and not force:
        click.confirm("Overwrite existing config?", abort=True)

    config = CliConfig.__new__(CliConfig)
    config.raw_dir = raw_dir
    config.processed_dir = processed_dir
    config.local_datasets = {}
    config.config_path = config_path
    config.save()

    # Create directories
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)


@config.command()
@click.pass_context
def show(ctx: click.Context):
    """Display the current configuration settings.

    Shows the contents of the brainsets configuration file, including paths for raw and
    processed datasets. If --config-path is not specified, looks for the configuration
    file in the default location ($HOME/.config/brainsets.yaml).
    """
    config = ctx.obj["CONFIG"]
    click.echo(config)
