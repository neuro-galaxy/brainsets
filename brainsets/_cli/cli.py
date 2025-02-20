import click

from .utils import EXISTING_FILEPATH_CLICK_TYPE, CliConfig, expand_path
from .cli_config import config
from .cli_prepare import prepare
from .cli_list import cli_list
from .cli_add import add
from .cli_completion import completion


@click.group()
@click.version_option()
@click.option("--config-path", type=EXISTING_FILEPATH_CLICK_TYPE)
@click.pass_context
def cli(ctx: click.Context, config_path: str | None):
    """
    Command line interface for downloading and preparing brainsets.

    \b
    Documentation: https://brainsets.readthedocs.io/en/latest/
    Project page: https://github.com/neuro-galaxy/brainsets
    """
    ctx.ensure_object(dict)

    if ctx.invoked_subcommand != "config":
        ctx.obj["CONFIG"] = CliConfig(config_path)
    else:
        ctx.obj["CONFIG_PATH"] = config_path


cli.add_command(config)
cli.add_command(prepare)
cli.add_command(cli_list, name="list")
cli.add_command(add)
cli.add_command(completion)


if __name__ == "__main__":
    cli()
