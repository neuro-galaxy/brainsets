import os
import click


@click.command()
def completion():
    """Setup shell completion for brainsets"""

    shell = _get_shell_type()
    if shell == "bash":
        os.system("_BRAINSETS_COMPLETE=bash_source brainsets")
        eval_cmd = 'eval "$(brainsets completion)"'
    elif shell == "zsh":
        os.system("_BRAINSETS_COMPLETE=zsh_source brainsets")
        eval_cmd = 'eval "$(brainsets completion)"'
    elif shell == "fish":
        os.system("_BRAINSETS_COMPLETE=fish_source brainsets")
        eval_cmd = "_FOO_BAR_COMPLETE=fish_source foo-bar | source"
    else:
        click.ClickException(f"Unsupported shell: {shell}")

    click.echo("# INSTRUCTIONS:")
    click.echo("# Add the above to your ~/.bashrc or or ~/.bash_completions")
    click.echo("# Or run this command again like: ")
    click.echo(f"# {eval_cmd}")


def _get_shell_type():
    shell = os.environ.get("SHELL", "")
    if "bash" in shell:
        return "bash"
    elif "zsh" in shell:
        return "zsh"
    elif "fish" in shell:
        return "fish"
    else:
        return shell
