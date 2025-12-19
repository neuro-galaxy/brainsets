"""
Logging and diagnostics utilities for the Dataset Wizard.
"""

import json
import logging
from typing import Any, Dict, Optional, Tuple

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.tree import Tree
from rich import box

logger = logging.getLogger(__name__)


def setup_file_logging(log_dir: str, dataset_id: str) -> Tuple[Console, RichHandler]:
    """
    Set up file logging for agent operations.

    Args:
        log_dir: Directory to store log files
        dataset_id: Dataset ID for log file naming

    Returns:
        Tuple of (Console, RichHandler) for logging
    """
    import os

    log_file = os.path.join(log_dir, f"{dataset_id}_agent_log.txt")
    log_file_handle = open(log_file, "w", encoding="utf-8")

    console = Console(file=log_file_handle, width=120, record=True)

    file_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_level=True,
        show_path=False,
    )
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    console.print(
        Panel.fit(
            f"[bold cyan]Dataset Processing Log[/bold cyan]\n"
            f"[yellow]Dataset ID:[/yellow] {dataset_id}\n"
            f"[yellow]Log File:[/yellow] {log_file}",
            border_style="cyan",
            box=box.DOUBLE,
        )
    )

    return console, file_handler


def cleanup_logging(console: Optional[Console], handler: Optional[RichHandler]) -> None:
    """Clean up logging resources."""
    if handler:
        logger.removeHandler(handler)
        handler.close()

    if console and hasattr(console.file, "close"):
        console.file.close()


def log_agent_start(
    console: Optional[Console], agent_name: str, description: str, color: str = "blue"
) -> None:
    """Log agent start panel."""
    if not console:
        return

    console.print(
        Panel(
            f"[bold {color}]{agent_name}[/bold {color}]\n" f"[dim]{description}[/dim]",
            border_style=color,
            box=box.HEAVY,
        )
    )


def log_agent_complete(
    console: Optional[Console], agent_name: str, success: bool
) -> None:
    """Log agent completion panel."""
    if not console:
        return

    status = "[green]✓ COMPLETED[/green]" if success else "[red]✗ ERROR[/red]"
    color = "green" if success else "red"

    console.print(
        Panel(
            f"[bold]{status} - {agent_name}[/bold]",
            border_style=color,
            box=box.HEAVY,
        )
    )


def log_processing_complete(console: Optional[Console]) -> None:
    """Log processing completion panel."""
    if not console:
        return

    console.print(
        Panel(
            "[bold green]✓ Processing Completed Successfully[/bold green]",
            border_style="green",
            box=box.DOUBLE,
        )
    )


def log_error(console: Optional[Console], error: str, traceback_str: str = "") -> None:
    """Log error panel."""
    if not console:
        return

    content = f"[bold red]✗ SUPERVISOR AGENT FAILED[/bold red]\n\n[yellow]Error:[/yellow] {error}"
    if traceback_str:
        content += f"\n\n[dim]{traceback_str}[/dim]"

    console.print(
        Panel(
            content,
            border_style="red",
            box=box.HEAVY,
            title="ERROR",
        )
    )


def log_agent_diagnostics(
    console: Optional[Console],
    raw_result: Dict[str, Any],
    dataset_id: str,
    agent_name: str,
    token_callback: Optional[Any] = None,
) -> None:
    """Log diagnostic information about agent execution."""
    if not raw_result or not console:
        return

    console.print(
        Panel(
            f"[bold magenta]{agent_name}[/bold magenta] - Diagnostics for [cyan]{dataset_id}[/cyan]",
            border_style="magenta",
            box=box.ROUNDED,
        )
    )

    if "messages" in raw_result:
        _log_messages(console, raw_result["messages"])

    if "intermediate_steps" in raw_result:
        _log_intermediate_steps(console, raw_result["intermediate_steps"])

    if token_callback:
        _log_token_usage(console, token_callback)

    if "return_values" in raw_result:
        console.print(
            f"[dim]Return value keys: {list(raw_result['return_values'].keys())}[/dim]"
        )

    console.print("")


def _log_messages(console: Console, messages: list) -> None:
    """Log message tree."""
    message_tree = Tree(f"[bold]Messages Exchanged: {len(messages)}[/bold]")

    for i, msg in enumerate(messages):
        msg_type = getattr(msg, "type", "unknown")
        msg_node = message_tree.add(f"[yellow]Message {i+1}[/yellow] ({msg_type})")

        if hasattr(msg, "content"):
            content = msg.content

            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type", "unknown")

                        if block_type == "thinking" and "thinking" in block:
                            thinking_panel = Panel(
                                block["thinking"],
                                title="[bold blue]🧠 Thinking Output[/bold blue]",
                                border_style="blue",
                                box=box.ROUNDED,
                            )
                            msg_node.add(thinking_panel)

                        elif block_type == "text" and "text" in block:
                            text_content = block["text"][:500]
                            msg_node.add(f"[green]📝 Text:[/green] {text_content}...")

                        elif block_type == "tool_use":
                            console.print(block)
                            tool_name = block.get("name", "unknown")
                            tool_input = block.get("input", {})

                            tool_node = msg_node.add(
                                f"[cyan]🔧 Tool Call:[/cyan] {tool_name}"
                            )

                            if tool_input:
                                syntax = Syntax(
                                    json.dumps(tool_input, indent=2),
                                    "json",
                                    theme="monokai",
                                    line_numbers=False,
                                )
                                tool_node.add(syntax)

            elif isinstance(content, str):
                msg_node.add(f"[dim]{content[:300]}...[/dim]")

        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tools_node = msg_node.add(f"[cyan]Tool Calls: {len(msg.tool_calls)}[/cyan]")
            for j, tool_call in enumerate(msg.tool_calls):
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name", "unknown")
                    tool_args = tool_call.get("args", {})
                else:
                    tool_name = getattr(tool_call, "name", "unknown")
                    tool_args = getattr(tool_call, "args", {})
                tool_call_node = tools_node.add(f"{j+1}. {tool_name}")

                if tool_args:
                    syntax = Syntax(
                        json.dumps(tool_args, indent=2),
                        "json",
                        theme="monokai",
                        line_numbers=False,
                    )
                    tool_call_node.add(syntax)

    console.print(message_tree)


def _log_intermediate_steps(console: Console, steps: list) -> None:
    """Log intermediate execution steps."""
    if not steps:
        console.print("[yellow]⚠ No tool calls were made[/yellow]")
        return

    steps_table = Table(
        title=f"[bold]Tool Execution Steps: {len(steps)}[/bold]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    steps_table.add_column("Step", style="yellow", width=6)
    steps_table.add_column("Tool", style="cyan", width=25)
    steps_table.add_column("Input", style="green", width=40)
    steps_table.add_column("Result Preview", style="magenta", width=40)

    for i, (action, result) in enumerate(steps):
        tool_name = getattr(action, "tool", "unknown")
        tool_input = getattr(action, "tool_input", {})
        input_str = json.dumps(tool_input, indent=2)[:100] if tool_input else "N/A"
        result_str = (
            str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
        )

        steps_table.add_row(str(i + 1), tool_name, input_str, result_str)

    console.print(steps_table)


def _log_token_usage(console: Console, token_callback: Any) -> None:
    """Log token usage statistics."""
    usage_summary = token_callback.get_usage_summary()

    usage_table = Table(
        title="[bold]Token Usage Statistics[/bold]",
        box=box.SIMPLE,
        show_header=True,
        header_style="bold green",
    )
    usage_table.add_column("Metric", style="cyan")
    usage_table.add_column("Value", style="yellow", justify="right")

    usage_table.add_row("Input Tokens", f"{usage_summary['input_tokens']:,}")
    usage_table.add_row("Output Tokens", f"{usage_summary['output_tokens']:,}")
    usage_table.add_row("Total Tokens", f"{usage_summary['total_tokens']:,}")
    usage_table.add_row("LLM Calls", str(usage_summary["llm_calls"]))

    console.print(usage_table)


def log_dataset_summary(console: Optional[Console], dataset: Any) -> None:
    """Log dataset creation summary."""
    if not console:
        return

    summary_table = Table(
        title="[bold green]Dataset Creation Summary[/bold green]",
        box=box.DOUBLE,
        show_header=True,
        header_style="bold cyan",
    )
    summary_table.add_column("Component", style="yellow", width=25)
    summary_table.add_column("Count", style="green", justify="right", width=15)
    summary_table.add_column("Status", style="cyan", width=20)

    summary_table.add_row(
        "Channel Maps",
        str(len(dataset.channel_maps.channel_maps)),
        "✓ Created",
    )
    summary_table.add_row(
        "Recordings",
        str(len(dataset.recording_info.recording_info)),
        "✓ Analyzed",
    )
    summary_table.add_row(
        "Metadata Fields",
        str(len(dataset.metadata.model_dump())),
        "✓ Extracted",
    )

    console.print(summary_table)
