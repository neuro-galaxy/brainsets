"""
Orchestrator for the Dataset Wizard workflow.

Coordinates the execution of specialized agents to populate Dataset structures.
"""

import logging
import os
import shutil
import tempfile
import traceback
from typing import Any, Dict, Optional

from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich import box

from brainsets.ds_wizard.config import LLMProvider, DEFAULT_TEMPERATURE
from brainsets.ds_wizard.agents import MetadataAgent, ChannelAgent, RecordingAgent
from brainsets.ds_wizard.dataset_struct import Dataset
from brainsets.ds_wizard.diagnostics import (
    setup_file_logging,
    cleanup_logging,
    log_agent_start,
    log_agent_complete,
    log_processing_complete,
    log_error,
    log_dataset_summary,
)

logger = logging.getLogger(__name__)

AGENT_REGISTRY = {
    "metadata": {
        "class": MetadataAgent,
        "name": "🔍 STARTING METADATA AGENT",
        "description": "Extracting dataset metadata, descriptions, and task categorization",
        "color": "blue",
    },
    "channel": {
        "class": ChannelAgent,
        "name": "🎛️  STARTING CHANNEL AGENT",
        "description": "Analyzing channel configurations and creating channel maps",
        "color": "yellow",
    },
    "recording": {
        "class": RecordingAgent,
        "name": "📊 STARTING RECORDING AGENT",
        "description": "Analyzing individual recordings and sessions",
        "color": "magenta",
    },
}


class Orchestrator:
    """Coordinates agents to process datasets and populate Dataset structures."""

    def __init__(
        self,
        model_name: str = "llama-3.3-70b",
        temperature: float = DEFAULT_TEMPERATURE,
        provider: LLMProvider = "cerebras",
        verbose: bool = False,
        download_path: str = None,
        request_delay: float = 0.0,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.provider = provider
        self.verbose = verbose
        self.download_path = download_path
        self.request_delay = request_delay
        self._console: Optional[Console] = None
        self._log_handler = None

    def _create_agent(self, agent_type: str, base_dir: str):
        """Create an agent instance with current configuration."""
        agent_class = AGENT_REGISTRY[agent_type]["class"]
        return agent_class(
            model_name=self.model_name,
            temperature=self.temperature,
            provider=self.provider,
            verbose=self.verbose,
            log_console=self._console,
            base_dir=base_dir,
            request_delay=self.request_delay,
        )

    async def _run_agent(
        self,
        agent_type: str,
        dataset_id: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a single agent and log its execution."""
        config = AGENT_REGISTRY[agent_type]

        log_agent_start(
            self._console, config["name"], config["description"], config["color"]
        )

        agent = self._create_agent(agent_type, context.get("config_files_dir", ""))
        result = await agent.process(dataset_id, context)

        success = "error" not in result
        log_agent_complete(self._console, agent_type.upper() + " AGENT", success)

        return result

    def _setup_workspace(
        self, dataset_id: str, context: Dict[str, Any] = None
    ) -> tuple:
        """Set up temporary directory and logging for processing."""
        if self.download_path:
            work_dir = os.path.join(self.download_path, f"dataset_{dataset_id}")
            os.makedirs(work_dir, exist_ok=True)
        else:
            work_dir = tempfile.mkdtemp(prefix=f"dataset_{dataset_id}_")

        logger.info(f"Created workspace directory: {work_dir}")

        self._console, self._log_handler = setup_file_logging(work_dir, dataset_id)

        enhanced_context = context.copy() if context else {}
        enhanced_context["config_files_dir"] = work_dir

        return work_dir, enhanced_context

    def _cleanup_workspace(self, work_dir: str) -> None:
        """Clean up workspace directory and resources."""
        log_processing_complete(self._console)
        cleanup_logging(self._console, self._log_handler)

        self._console = None
        self._log_handler = None

        if os.path.exists(work_dir) and not self.download_path:
            logger.info(f"Cleaning up workspace: {work_dir}")
            shutil.rmtree(work_dir)

    def _assemble_dataset(self, results: Dict[str, Any]) -> Optional[Dataset]:
        """Assemble and validate the final Dataset object from agent results."""
        try:
            metadata = results.get("structured_metadata")
            channel_maps = results.get("channel_maps")
            recording_info = results.get("recording_info")

            if not all([metadata, channel_maps, recording_info]):
                missing = [
                    k
                    for k, v in [
                        ("metadata", metadata),
                        ("channel_maps", channel_maps),
                        ("recording_info", recording_info),
                    ]
                    if not v
                ]
                logger.error(f"Missing required components: {missing}")
                return None

            dataset = Dataset(
                metadata=metadata,
                channel_maps=channel_maps,
                recording_info=recording_info,
            )

            log_dataset_summary(self._console, dataset)

            logger.info(
                f"Successfully created Dataset with {len(dataset.channel_maps.channel_maps)} "
                f"channel maps and {len(dataset.recording_info.recording_info)} recordings"
            )
            return dataset

        except (ValidationError, KeyError) as e:
            logger.error(f"Failed to create Dataset object: {e}")
            if self._console:
                self._console.print(
                    Panel(
                        f"[bold red]✗ Dataset Creation Failed[/bold red]\n{str(e)}",
                        border_style="red",
                        box=box.HEAVY,
                    )
                )
            return None

    async def run(
        self, dataset_id: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run the complete dataset processing workflow.

        Executes agents in sequence: Metadata → Channel → Recording,
        then assembles the final Dataset object.
        """
        logger.info(f"Starting dataset processing for {dataset_id}")

        work_dir, ctx = self._setup_workspace(dataset_id, context)
        results = {"config_files_dir": work_dir}

        try:
            metadata_result = await self._run_agent("metadata", dataset_id, ctx)
            if isinstance(metadata_result, Exception):
                results["metadata_error"] = str(metadata_result)
            else:
                results.update(metadata_result)
                ctx.update(metadata_result)

            channel_result = await self._run_agent("channel", dataset_id, ctx)
            results.update(channel_result)

            if "error" not in channel_result:
                recording_ctx = {**ctx, **results}
                recording_result = await self._run_agent(
                    "recording", dataset_id, recording_ctx
                )
                results.update(recording_result)
            else:
                logger.error("Cannot proceed with recording agent without channel maps")

            if self._console:
                self._console.print(
                    Panel(
                        "[bold cyan]🔨 Creating Final Dataset Object[/bold cyan]",
                        border_style="cyan",
                        box=box.ROUNDED,
                    )
                )

            dataset = self._assemble_dataset(results)
            return {"dataset": dataset, "results": results}

        except Exception as e:
            logger.error(f"Orchestrator failed: {e}")
            traceback.print_exc()
            log_error(self._console, str(e), traceback.format_exc())
            return {"error": str(e), "partial_results": results}

        finally:
            self._cleanup_workspace(work_dir)


# Backwards compatibility alias
SupervisorAgent = Orchestrator
