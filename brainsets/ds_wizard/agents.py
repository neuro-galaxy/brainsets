"""
Agent implementations for the Dataset Wizard using LangChain.
"""

from abc import ABC, abstractmethod
import asyncio
import json
import logging
import traceback
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from rich.console import Console

from brainsets.ds_wizard.config import (
    LLMProvider,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_NUM_EXAMPLES,
    DEFAULT_RECORDING_BATCH_SIZE,
)
from brainsets.ds_wizard.llm import (
    create_llm,
    TokenUsageCallbackHandler,
    extract_output_text,
)
from brainsets.ds_wizard.diagnostics import log_agent_diagnostics
from brainsets.ds_wizard.dataset_struct import (
    DatasetMetadata,
    ChannelMapsOutput,
    RecordingInfoOutput,
)
from brainsets.ds_wizard.tools import (
    create_metadata_tools,
    create_channel_tools,
    create_recording_tools,
)
from brainsets.ds_wizard.examples import (
    METADATA_EXAMPLES,
    CHANNEL_EXAMPLES,
    RECORDING_EXAMPLES,
)
from brainsets.ds_wizard.prompts import (
    METADATA_SYSTEM_PROMPT,
    METADATA_USER_PROMPT,
    CHANNEL_SYSTEM_PROMPT,
    CHANNEL_USER_PROMPT,
    RECORDING_SYSTEM_PROMPT,
    RECORDING_USER_PROMPT,
    RECORDING_BATCH_SYSTEM_PROMPT,
    RECORDING_BATCH_USER_PROMPT,
)

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all agents in the Dataset Wizard."""

    name: str = "BaseAgent"

    def __init__(
        self,
        provider: LLMProvider = "cerebras",
        model_name: str = "llama-3.3-70b",
        temperature: float = DEFAULT_TEMPERATURE,
        verbose: bool = False,
        use_few_shot: bool = True,
        num_examples: int = DEFAULT_NUM_EXAMPLES,
        max_retries: int = DEFAULT_MAX_RETRIES,
        log_console: Optional[Console] = None,
        base_dir: Optional[str] = None,
        request_delay: float = 0.0,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.provider = provider
        self.verbose = verbose
        self.use_few_shot = use_few_shot
        self.num_examples = num_examples
        self.max_retries = max_retries
        self.log_console = log_console
        self.base_dir = base_dir or ""
        self.request_delay = request_delay
        rate_limit = 1.0 / request_delay if request_delay > 0 else None
        self.llm = create_llm(provider, model_name, temperature, rate_limit=rate_limit)
        self.output_parser = self._create_output_parser()

    def _create_output_parser(self) -> Optional[PydanticOutputParser]:
        """Get the Pydantic output parser for this agent's output schema."""
        schema = self.get_output_schema()
        if schema:
            return PydanticOutputParser(pydantic_object=schema)
        return None

    def _get_simplified_schema(self) -> str:
        """Get a simplified JSON schema for prompt formatting."""
        schema = self.get_output_schema()
        if schema:
            schema_dict = schema.model_json_schema()
            schema_json = json.dumps(schema_dict, indent=2)
            return schema_json.replace("{", "[").replace("}", "]")
        return "{}"

    def _build_system_prompt_with_examples(self) -> str:
        """Build system prompt with few-shot examples."""
        base_prompt = self.get_system_prompt().format(
            format_instructions=self._get_simplified_schema()
        )

        if not self.use_few_shot:
            return base_prompt

        examples = self.get_few_shot_examples()
        if not examples:
            return base_prompt

        examples = examples[: self.num_examples]
        examples_text = "\n\nEXAMPLES:\n" + "\n\n".join(
            f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in examples
        )

        return base_prompt + examples_text

    def create_agent(self):
        """Create the agent using LangChain 1.x API with built-in retry."""
        tools = self.get_tools()
        system_prompt = self._build_system_prompt_with_examples()
        agent = create_agent(self.llm, tools, system_prompt=system_prompt)

        return agent.with_retry(
            stop_after_attempt=self.max_retries,
            wait_exponential_jitter=True,
        )

    def _extract_output_from_result(self, raw_result: Dict[str, Any]) -> Optional[str]:
        """Extract and validate output text from agent result."""
        if not raw_result or "messages" not in raw_result:
            return None

        messages = raw_result["messages"]
        if not messages:
            return None

        last_message = messages[-1]

        if hasattr(last_message, "content"):
            output_text = extract_output_text(last_message.content)
        elif isinstance(last_message, dict):
            output_text = extract_output_text(last_message.get("content", ""))
        else:
            output_text = extract_output_text(str(last_message))

        return output_text if output_text and output_text.strip() else None

    async def _invoke_with_validation(
        self, agent, prompt: str, dataset_id: str
    ) -> Dict[str, Any]:
        """Invoke agent and validate response, retrying on empty outputs."""
        last_error = None

        for attempt in range(self.max_retries):
            token_callback = TokenUsageCallbackHandler()

            try:
                logger.info(
                    f"{self.name} attempt {attempt + 1}/{self.max_retries} for {dataset_id}"
                )

                raw_result = await agent.ainvoke(
                    {"messages": [("user", prompt)]},
                    config={"callbacks": [token_callback]},
                )

                log_agent_diagnostics(
                    self.log_console, raw_result, dataset_id, self.name, token_callback
                )

                output_text = self._extract_output_from_result(raw_result)

                if output_text:
                    logger.info(
                        f"{self.name} succeeded on attempt {attempt + 1} for {dataset_id}"
                    )
                    return {
                        "output": output_text,
                        "messages": raw_result.get("messages", []),
                    }

                last_error = "Agent returned empty or invalid output"
                logger.warning(
                    f"{self.name} returned empty output on attempt {attempt + 1}, retrying..."
                )

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** (attempt + 1))

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"{self.name} attempt {attempt + 1} failed with exception: {e}"
                )

        logger.error(
            f"{self.name} failed after {self.max_retries} attempts for {dataset_id}. Last error: {last_error}"
        )
        return {
            "error": f"Agent failed after {self.max_retries} retries",
            "last_error": last_error,
        }

    async def process(
        self, dataset_id: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process the dataset and return results."""
        agent = self.create_agent()
        prompt = self.build_prompt(dataset_id, context or {})

        try:
            result = await self._invoke_with_validation(agent, prompt, dataset_id)

            if "error" in result:
                return result

            output = extract_output_text(result["output"])
            logger.info(
                f"{self.name} raw output (first 500 chars): {str(output)[:500]}"
            )

            structured_result = self.output_parser.parse(output)
            return self.format_result(output, structured_result)

        except Exception as e:
            logger.error(f"Error in {self.name} for {dataset_id}: {e}")
            traceback.print_exc()
            return {"error": str(e)}

    @abstractmethod
    def get_output_schema(self) -> Optional[BaseModel]:
        """Return the Pydantic model for structured output."""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt template for this agent."""
        pass

    @abstractmethod
    def get_tools(self) -> List:
        """Return the tools available to this agent."""
        pass

    @abstractmethod
    def build_prompt(self, dataset_id: str, context: Dict[str, Any]) -> str:
        """Build the user prompt for this agent."""
        pass

    @abstractmethod
    def format_result(self, raw: str, structured: Any) -> Dict[str, Any]:
        """Format the result dict with appropriate keys."""
        pass

    def get_few_shot_examples(self) -> List[Dict[str, str]]:
        """Return few-shot examples for in-context learning. Override in subclasses."""
        return []


class MetadataAgent(BaseAgent):
    """Agent responsible for extracting dataset metadata."""

    name = "MetadataAgent"

    def get_output_schema(self) -> Optional[BaseModel]:
        return DatasetMetadata

    def get_system_prompt(self) -> str:
        return METADATA_SYSTEM_PROMPT

    def get_tools(self) -> List:
        return create_metadata_tools(self.base_dir)

    def get_few_shot_examples(self) -> List[Dict[str, str]]:
        return METADATA_EXAMPLES

    def build_prompt(self, dataset_id: str, context: Dict[str, Any]) -> str:
        return METADATA_USER_PROMPT.format(dataset_id=dataset_id)

    def format_result(self, raw: str, structured: Any) -> Dict[str, Any]:
        return {"raw_metadata": raw, "structured_metadata": structured}


class ChannelAgent(BaseAgent):
    """Agent responsible for analyzing channel configurations and creating channel maps."""

    name = "ChannelAgent"

    def get_output_schema(self) -> Optional[BaseModel]:
        return ChannelMapsOutput

    def get_system_prompt(self) -> str:
        return CHANNEL_SYSTEM_PROMPT

    def get_tools(self) -> List:
        return create_channel_tools(self.base_dir)

    def get_few_shot_examples(self) -> List[Dict[str, str]]:
        return CHANNEL_EXAMPLES

    def build_prompt(self, dataset_id: str, context: Dict[str, Any]) -> str:
        return CHANNEL_USER_PROMPT.format(dataset_id=dataset_id)

    def format_result(self, raw: str, structured: Any) -> Dict[str, Any]:
        return {"channel_maps_raw": raw, "channel_maps": structured}


class RecordingAgent(BaseAgent):
    """Agent that processes recordings in batches to avoid context overflow."""

    name = "RecordingAgent"

    def __init__(self, *args, batch_size: int = DEFAULT_RECORDING_BATCH_SIZE, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def get_output_schema(self) -> Optional[BaseModel]:
        return RecordingInfoOutput

    def get_system_prompt(self) -> str:
        return RECORDING_BATCH_SYSTEM_PROMPT

    def get_tools(self) -> List:
        return []

    def get_few_shot_examples(self) -> List[Dict[str, str]]:
        return []

    def build_prompt(self, dataset_id: str, context: Dict[str, Any]) -> str:
        recordings = context.get("batch_recordings", [])
        channel_map_ids = context.get("channel_map_ids", [])

        compact_recordings = []
        for r in recordings:
            compact_recordings.append(
                {
                    "recording_id": r.get("recording_id"),
                    "subject_id": r.get("subject_id"),
                    "task_id": r.get("task_id"),
                    "matched_channel_map_id": r.get("matched_channel_map_id"),
                    "duration_seconds": r.get("duration_seconds"),
                    "num_channels": r.get("num_channels"),
                    "bad_channels": r.get("bad_channels", []),
                    "participant_info": r.get("participant_info", {}),
                }
            )

        return RECORDING_BATCH_USER_PROMPT.format(
            num_recordings=len(compact_recordings),
            channel_map_ids=channel_map_ids,
            recordings_json=json.dumps(compact_recordings, indent=2),
        )

    def format_result(self, raw: str, structured: Any) -> Dict[str, Any]:
        return {"raw_recording_info": raw, "recording_info": structured}

    def _extract_channel_maps_for_tool(
        self, channel_maps_info: Any
    ) -> Dict[str, List[str]]:
        """Extract channel map IDs and their channel names."""
        result = {}

        if hasattr(channel_maps_info, "channel_maps"):
            maps = channel_maps_info.channel_maps
        elif (
            isinstance(channel_maps_info, dict) and "channel_maps" in channel_maps_info
        ):
            maps = channel_maps_info["channel_maps"]
        elif isinstance(channel_maps_info, dict):
            maps = channel_maps_info
        else:
            return result

        for map_id, channel_map in maps.items():
            if hasattr(channel_map, "channels"):
                result[map_id] = list(channel_map.channels.keys())
            elif isinstance(channel_map, dict) and "channels" in channel_map:
                result[map_id] = list(channel_map["channels"].keys())

        return result

    def _run_analysis_tool(
        self, dataset_id: str, channel_maps: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Run analyze_all_recordings programmatically (not via LLM)."""
        from brainsets.ds_wizard.tools import create_recording_analysis_tool

        analyze_tool = create_recording_analysis_tool(self.base_dir)
        result_json = analyze_tool.invoke(
            {"dataset_id": dataset_id, "channel_maps": channel_maps}
        )

        return json.loads(result_json)

    async def _process_batch(
        self,
        dataset_id: str,
        batch: List[Dict],
        channel_map_ids: List[str],
        batch_num: int,
    ) -> List[Any]:
        """Process a single batch of recordings through the LLM."""
        batch_context = {
            "batch_recordings": batch,
            "channel_map_ids": channel_map_ids,
        }

        logger.info(
            f"{self.name} processing batch {batch_num} with {len(batch)} recordings"
        )

        agent = self.create_agent()
        prompt = self.build_prompt(dataset_id, batch_context)
        result = await self._invoke_with_validation(
            agent, prompt, f"{dataset_id}_batch{batch_num}"
        )

        if "error" in result:
            logger.error(f"Batch {batch_num} failed: {result.get('error')}")
            return []

        output = extract_output_text(result["output"])
        structured = self.output_parser.parse(output)

        return (
            structured.recording_info if hasattr(structured, "recording_info") else []
        )

    async def process(
        self, dataset_id: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process recordings in batches to avoid context overflow."""
        if not context or "channel_maps" not in context:
            return {"error": "RecordingAgent requires channel_maps from ChannelAgent"}

        channel_maps_info = context.get("channel_maps", {})
        channel_maps_for_tool = self._extract_channel_maps_for_tool(channel_maps_info)
        channel_map_ids = list(channel_maps_for_tool.keys())

        logger.info(f"{self.name} running analysis tool for {dataset_id}")
        try:
            analysis_result = self._run_analysis_tool(dataset_id, channel_maps_for_tool)
        except Exception as e:
            logger.error(f"Analysis tool failed: {e}")
            return {"error": f"Analysis tool failed: {e}"}

        recordings = analysis_result.get("recordings", [])
        total_recordings = len(recordings)
        logger.info(
            f"{self.name} found {total_recordings} recordings, batch_size={self.batch_size}"
        )

        if total_recordings == 0:
            return {"error": "No recordings found in dataset"}

        all_recording_info = []
        batches = [
            recordings[i : i + self.batch_size]
            for i in range(0, total_recordings, self.batch_size)
        ]

        for batch_num, batch in enumerate(batches, 1):
            logger.info(f"{self.name} processing batch {batch_num}/{len(batches)}")
            batch_results = await self._process_batch(
                dataset_id, batch, channel_map_ids, batch_num
            )
            all_recording_info.extend(batch_results)

            if self.request_delay > 0 and batch_num < len(batches):
                await asyncio.sleep(self.request_delay)

        logger.info(
            f"{self.name} completed: {len(all_recording_info)}/{total_recordings} recordings processed"
        )

        combined_output = RecordingInfoOutput(recording_info=all_recording_info)
        return {
            "raw_recording_info": json.dumps(
                {
                    "total_batches": len(batches),
                    "total_recordings": len(all_recording_info),
                }
            ),
            "recording_info": combined_output,
        }
