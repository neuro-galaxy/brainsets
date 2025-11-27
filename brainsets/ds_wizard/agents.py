"""
Agent implementations for the Dataset Wizard using LangChain.
"""

from abc import ABC, abstractmethod
import asyncio
import json
import logging
import os
import shutil
import tempfile
import traceback
from typing import Any, Dict, List, Literal, Optional

from langchain.agents import create_agent
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_cerebras import ChatCerebras
from langchain_ollama import ChatOllama
from langchain_google_vertexai import ChatVertexAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.tree import Tree
from rich import box

from brainsets.ds_wizard.dataset_struct import (
    Dataset,
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


logger = logging.getLogger(__name__)


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """Callback handler to track token usage from LLM calls."""

    def __init__(self):
        super().__init__()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.llm_calls = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when LLM finishes running."""
        # Extract token usage from the response
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            self.total_input_tokens += usage.get("prompt_tokens", 0)
            self.total_output_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)
            self.llm_calls += 1
        # For providers that use usage_metadata instead (like Google)
        elif hasattr(response, "generations") and response.generations:
            for generation_list in response.generations:
                for generation in generation_list:
                    if hasattr(generation, "message") and hasattr(
                        generation.message, "usage_metadata"
                    ):
                        usage_metadata = generation.message.usage_metadata
                        self.total_input_tokens += usage_metadata.get("input_tokens", 0)
                        self.total_output_tokens += usage_metadata.get(
                            "output_tokens", 0
                        )
                        self.total_tokens += usage_metadata.get("total_tokens", 0)
                        self.llm_calls += 1

    def reset(self):
        """Reset the token counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.llm_calls = 0

    def get_usage_summary(self) -> Dict[str, int]:
        """Get a summary of token usage."""
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
        }


# Supported LLM providers
LLMProvider = Literal["ollama", "cerebras", "vertexai", "google"]

# Model mappings for different providers
PROVIDER_MODELS = {
    "google": ["gemini-2.5-flash-lite", "gemini-2.5-flash"],
    "vertexai": ["Qwen3-Next-80B-Thinking"],
    "cerebras": ["gst-oss:200b"],
    "ollama": ["gpt-oss:20b"],
}


class BaseAgent(ABC):
    """Base class for all agents in the Dataset Wizard."""

    def __init__(
        self,
        provider: LLMProvider = "cerebras",
        model_name: str = "llama-3.3-70b",
        temperature: float = 0.1,
        verbose: bool = False,
        use_few_shot: bool = True,
        num_examples: int = 2,
        max_retries: int = 3,
        log_console: Optional[Console] = None,
        base_dir: Optional[str] = None,
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
        self._validate_model_provider_combination()
        self.llm = self._create_llm()
        self.output_parser = self.get_output_parser()

    def _validate_model_provider_combination(self):
        """Validate that the model is supported by the provider."""
        if self.provider in PROVIDER_MODELS:
            supported_models = PROVIDER_MODELS[self.provider]
            if self.model_name not in supported_models:
                logger.warning(
                    f"Model '{self.model_name}' may not be officially supported by provider '{self.provider}'. "
                    f"Supported models: {supported_models}"
                )

    def _create_llm(self):
        """Create the language model instance based on provider."""
        if self.provider == "cerebras":
            return ChatCerebras(
                model=self.model_name,
                temperature=self.temperature,
            )
        elif self.provider == "ollama":
            return ChatOllama(
                model=self.model_name,
                temperature=self.temperature,
                base_url="http://localhost:11434",
            )
        elif self.provider == "vertexai":
            return ChatVertexAI(model=self.model_name, temperature=self.temperature)
        elif self.provider == "google":
            # Add model-specific configurations for Gemini
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                max_output_tokens=65536,
                include_thoughts=True,
                thinking_budget=10000,
            )
        else:
            raise ValueError(
                f"Unsupported provider: {self.provider}. Supported providers: {list(PROVIDER_MODELS.keys())}"
            )

    def _extract_output_text(self, raw_output: Any) -> str:
        """
        Extract the actual text output from potentially complex output structure.

        LangChain 1.x with thinking/structured outputs returns:
        - A simple string (basic format)
        - A list of content blocks: [{'type': 'thinking', 'thinking': '...'}, {'type': 'text', 'text': '...'}]

        Args:
            raw_output: The raw output from the agent

        Returns:
            The extracted text string, or empty string if not found
        """
        # Handle None or empty
        if not raw_output:
            return ""

        # If it's already a string, return it
        if isinstance(raw_output, str):
            return raw_output

        # If it's a list, handle content blocks or string items
        if isinstance(raw_output, list):
            text_parts = []

            for item in raw_output:
                # Handle content block format: {'type': 'text', 'text': '...'}
                if isinstance(item, dict):
                    if item.get("type") == "text" and "text" in item:
                        text_parts.append(item["text"])
                    # Skip 'thinking' blocks - they're internal reasoning
                    elif item.get("type") == "thinking":
                        continue
                # Handle plain string items
                elif isinstance(item, str):
                    text_parts.append(item)

            # Join all text parts
            if text_parts:
                return "\n".join(text_parts)

            # If no text found, return empty
            return ""

        # For any other type, convert to string
        return str(raw_output)

    def _log_agent_diagnostics(
        self,
        raw_result: Dict[str, Any],
        dataset_id: str,
        agent_name: str,
        token_callback: Optional[TokenUsageCallbackHandler] = None,
    ) -> None:
        """Log diagnostic information about agent execution."""
        if not raw_result or not self.log_console:
            return

        self.log_console.print(
            Panel(
                f"[bold magenta]{agent_name}[/bold magenta] - Diagnostics for [cyan]{dataset_id}[/cyan]",
                border_style="magenta",
                box=box.ROUNDED,
            )
        )

        if "messages" in raw_result:
            messages = raw_result["messages"]

            message_tree = Tree(f"[bold]Messages Exchanged: {len(messages)}[/bold]")

            for i, msg in enumerate(messages):
                msg_type = getattr(msg, "type", "unknown")
                msg_node = message_tree.add(
                    f"[yellow]Message {i+1}[/yellow] ({msg_type})"
                )

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
                                    msg_node.add(
                                        f"[green]📝 Text:[/green] {text_content}..."
                                    )

                                elif block_type == "tool_use":
                                    self.log_console.print(block)
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
                    tools_node = msg_node.add(
                        f"[cyan]Tool Calls: {len(msg.tool_calls)}[/cyan]"
                    )
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

            self.log_console.print(message_tree)

        if "intermediate_steps" in raw_result:
            steps = raw_result["intermediate_steps"]

            if steps:
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
                    input_str = (
                        json.dumps(tool_input, indent=2)[:100] if tool_input else "N/A"
                    )
                    result_str = (
                        str(result)[:100] + "..."
                        if len(str(result)) > 100
                        else str(result)
                    )

                    steps_table.add_row(str(i + 1), tool_name, input_str, result_str)

                self.log_console.print(steps_table)
            else:
                self.log_console.print("[yellow]⚠ No tool calls were made[/yellow]")

        if token_callback:
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

            self.log_console.print(usage_table)

        if "return_values" in raw_result:
            self.log_console.print(
                f"[dim]Return value keys: {list(raw_result['return_values'].keys())}[/dim]"
            )

        self.log_console.print("")

    async def _invoke_agent_with_retry(
        self, agent, prompt: str, agent_name: str, dataset_id: str
    ) -> Dict[str, Any]:
        """
        Invoke agent with retry logic for empty outputs.

        Args:
            agent: The LangGraph agent to invoke
            prompt: The prompt to send to the agent
            agent_name: Name of the agent for logging
            dataset_id: Dataset ID being processed

        Returns:
            Dict containing the agent's output
        """
        last_error = None

        for attempt in range(self.max_retries):
            # Create a new token callback handler for each attempt
            token_callback = TokenUsageCallbackHandler()

            try:
                logger.info(
                    f"{agent_name} attempt {attempt + 1}/{self.max_retries} for {dataset_id}"
                )

                # LangGraph agents use a different invocation pattern
                raw_result = await agent.ainvoke(
                    {"messages": [("user", prompt)]},
                    config={"callbacks": [token_callback]},
                )

                # Log diagnostics about agent execution
                self._log_agent_diagnostics(
                    raw_result, dataset_id, agent_name, token_callback
                )

                # Validate raw_result structure
                if not raw_result:
                    last_error = "Agent returned None"
                    logger.warning(f"{agent_name} returned None, retrying...")
                    continue

                # LangGraph returns messages instead of output
                if "messages" not in raw_result:
                    last_error = f"Agent result missing 'messages' key. Keys: {raw_result.keys()}"
                    logger.warning(f"{agent_name} missing messages key, retrying...")
                    continue

                # Extract the last message content
                messages = raw_result["messages"]
                if not messages:
                    last_error = "Agent returned empty messages list"
                    logger.warning(f"{agent_name} returned empty messages, retrying...")
                    continue

                # Get the last AI message content
                last_message = messages[-1]

                # Extract content from the message
                if hasattr(last_message, "content"):
                    output_text = self._extract_output_text(last_message.content)
                elif isinstance(last_message, dict):
                    output_text = self._extract_output_text(
                        last_message.get("content", "")
                    )
                else:
                    output_text = self._extract_output_text(str(last_message))

                if not output_text or not output_text.strip():
                    last_error = "Agent returned empty output"
                    logger.warning(
                        f"{agent_name} returned empty output on attempt {attempt + 1}, retrying..."
                    )
                    continue

                # Success! Return the result with output key for backward compatibility
                logger.info(
                    f"{agent_name} succeeded on attempt {attempt + 1} for {dataset_id}"
                )
                return {
                    "output": output_text,
                    "messages": messages,
                    "intermediate_steps": [],  # LangGraph doesn't expose intermediate steps the same way
                }

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"{agent_name} attempt {attempt + 1} failed with exception: {e}"
                )
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying {agent_name}...")
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                continue

        # All retries exhausted
        logger.error(
            f"{agent_name} failed after {self.max_retries} attempts for {dataset_id}. Last error: {last_error}"
        )
        return {
            "error": f"Agent failed after {self.max_retries} retries",
            "last_error": last_error,
        }

    @abstractmethod
    def get_output_schema(self) -> Optional[BaseModel]:
        """Return the Pydantic model for structured output, or None for unstructured."""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass

    @abstractmethod
    def get_tools(self) -> List:
        """Return the tools available to this agent."""
        pass

    def get_few_shot_examples(self) -> List[Dict[str, str]]:
        """Return few-shot examples for in-context learning. Override in subclasses."""
        return []

    def _create_few_shot_prompt(self) -> Optional[FewShotChatMessagePromptTemplate]:
        """Create a few-shot prompt template from examples."""
        examples = self.get_few_shot_examples()
        if not examples or not self.use_few_shot:
            return None

        # Limit number of examples
        examples = examples[: self.num_examples]

        # Create example prompt template
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )

        # Create few-shot template
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )

        return few_shot_prompt

    def create_agent(self):
        """Create the agent using new LangChain 1.x API."""
        tools = self.get_tools()

        output_parser = self.get_output_parser()

        # Include format instructions in system prompt if we have a parser
        system_prompt = self.get_system_prompt()
        if output_parser:
            system_prompt = system_prompt.format(
                format_instructions=output_parser.get_format_instructions()
            )

        # Add few-shot examples to system prompt if available
        if self.use_few_shot:
            examples = self.get_few_shot_examples()
            if examples:
                examples = examples[: self.num_examples]
                examples_text = "\n\nEXAMPLES:\n"
                for i, example in enumerate(examples, 1):
                    examples_text += f"\nExample {i}:\nInput: {example['input']}\nOutput: {example['output']}\n"
                system_prompt += examples_text

        # Create agent using new LangChain 1.x create_agent API
        agent = create_agent(self.llm, tools, system_prompt=system_prompt)

        return agent

    def get_output_parser(self) -> Optional[PydanticOutputParser]:
        """Get the Pydantic output parser for this agent's output schema."""
        schema = self.get_output_schema()
        if schema:
            return PydanticOutputParser(pydantic_object=schema)
        return None

    def get_simplified_schema(self) -> str:
        """Get a simplified JSON schema for prompt formatting"""
        schema = self.get_output_schema()
        if schema:
            import json

            schema_dict = schema.model_json_schema()
            schema_json = json.dumps(schema_dict, indent=2)
            return schema_json.replace("{", "[").replace("}", "]")
        return "{}"

    @abstractmethod
    async def process(
        self, dataset_id: str, context: Dict[str, Any] = None
    ) -> BaseModel:
        """Process the dataset and return results."""
        pass


class MetadataAgent(BaseAgent):
    """Agent responsible for extracting dataset metadata, descriptions, and task categorization."""

    def get_few_shot_examples(self) -> List[Dict[str, str]]:
        """Return metadata extraction examples for in-context learning."""
        return METADATA_EXAMPLES

    def get_system_prompt(self) -> str:
        return f"""You are a metadata extraction specialist for neuroscience datasets.

Your task is to analyze datasets and extract structured metadata information.

OUTPUT FORMAT:
Return ONLY a valid JSON object matching this schema (no other text, explanation, or markdown):
{self.get_simplified_schema()}"""

    def get_tools(self) -> List:
        return create_metadata_tools(self.base_dir)

    def get_output_schema(self) -> Optional[BaseModel]:
        return DatasetMetadata

    async def process(
        self, dataset_id: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Extract metadata information for the dataset."""
        agent = self.create_agent()

        prompt = f"""
Analyze dataset {dataset_id} and extract metadata information.

REQUIRED STEPS:
1. fetch_dataset_metadata - Get dataset information (name, modalities, authors, etc.)
2. fetch_dataset_readme - Get README content for summaries and descriptions
3. get_task_taxonomy - Get valid task categories for classification

OPTIONAL TOOLS (use if you need additional context):
If the above tools don't provide sufficient information, you can:
- fetch_dataset_filenames - List all files in the dataset
- download_configuration_file - Download specific files (e.g., dataset_description.json, participants.tsv)
- read_configuration_file - Read contents of downloaded files
- list_configuration_files - List all downloaded files

GUIDELINES:
- Summaries: ≤150 words, derived from README
- Task matching: Use most specific taxonomy match or null if no match
- Brainset name format: <last_author>_<recognizable_word>_<dataset_id>_<year>
- Authors: Extract from README or metadata fields
- Call each tool at most once with the same parameters
- Use relative file paths (e.g., 'dataset_description.json' or 'sub-01/eeg/sub-01_channels.tsv')

Return a complete, valid JSON object. If information is missing, use your best judgment."""

        try:
            # Use retry logic
            raw_result = await self._invoke_agent_with_retry(
                agent, prompt, "MetadataAgent", dataset_id
            )

            # Check if we got an error from retry logic
            if "error" in raw_result:
                return raw_result

            # Extract the actual text output (handles thinking output format)
            raw_output = raw_result["output"]
            output = self._extract_output_text(raw_output)

            # Log the output for debugging
            logger.info(
                f"MetadataAgent raw output (first 500 chars): {str(output)[:500]}"
            )

            structured_result = self.output_parser.parse(output)
            return {
                "raw_metadata": output,
                "structured_metadata": structured_result,
            }
        except Exception as e:
            logger.error(f"Error in MetadataAgent for {dataset_id}: {e}")
            traceback.print_exc()
            return {"error": str(e)}


class ChannelAgent(BaseAgent):
    """Agent responsible for analyzing channel configurations and creating channel maps."""

    def get_few_shot_examples(self) -> List[Dict[str, str]]:
        """Return channel mapping examples for in-context learning."""
        return CHANNEL_EXAMPLES

    def get_system_prompt(self) -> str:
        return f"""You are an EEG channel mapping specialist.

Your task is to create channel maps for neuroscience datasets. Each map should:
- Map channel names to standard names (if not already standard)
- Classify electrode types (modalities) based on available information
- Extract 3D coordinates for EEG electrodes

OUTPUT FORMAT:
Return ONLY a valid JSON object matching this schema (no other text, explanation, or markdown):
{self.get_simplified_schema()}"""

    def get_tools(self) -> List:
        return create_channel_tools(self.base_dir)

    def get_output_schema(self) -> Optional[BaseModel]:
        return ChannelMapsOutput

    async def process(
        self, dataset_id: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Analyze channel configurations and create channel maps."""
        agent = self.create_agent()

        prompt = f"""
Create channel maps for dataset {dataset_id}.

FILE WORKFLOW:
1. fetch_dataset_readme - Get dataset overview and device information
2. fetch_dataset_filenames - List all files to find *_channels.tsv, *_electrodes.tsv files
3. download_configuration_file - Download channel/electrode configuration files
4. read_configuration_file - Read the downloaded files to extract channel information
5. list_configuration_files - List all downloaded files

Use relative file paths (e.g., 'channels.tsv' or 'sub-01/eeg/sub-01_channels.tsv').

CHANNEL MAPPING PROCESS:
1. Get all unique channels per device/session from configuration files
2. Use get_modalities to get valid electrode types for classification
3. For EEG channels with non-standard names, use get_electrode_to_montage_mapping to map them to standard names (e.g., 'EEG_Fp1' -> 'Fp1')
4. If no standard match found, use modality EEG-OTHER
5. Use find_all_montage_matches to get electrode coordinates
6. Use get_eeg_bids_specs for BIDS format reference if needed
7. Create one ChannelMap per device, including device name and manufacturer

NOTES:
- Datasets may use multiple devices with different channel configurations
- Sessions with the same device may have variations due to manual data collection

Return a complete, valid JSON object. If information is missing, use your best judgment."""

        try:
            # Use retry logic
            raw_result = await self._invoke_agent_with_retry(
                agent, prompt, "ChannelAgent", dataset_id
            )

            # Check if we got an error from retry logic
            if "error" in raw_result:
                return raw_result

            # Extract the actual text output (handles thinking output format)
            raw_output = raw_result["output"]
            output = self._extract_output_text(raw_output)

            # Log the output for debugging
            logger.info(
                f"ChannelAgent raw output (first 500 chars): {str(output)[:500]}"
            )

            structured_result = self.output_parser.parse(output)
            return {
                "channel_maps_raw": output,
                "channel_maps": structured_result,
            }
        except Exception as e:
            logger.error(f"Error in ChannelAgent for {dataset_id}: {e}")
            traceback.print_exc()
            return {"error": str(e)}


class RecordingAgent(BaseAgent):
    """Agent responsible for analyzing individual recordings and sessions."""

    def get_few_shot_examples(self) -> List[Dict[str, str]]:
        """Return recording info examples for in-context learning."""
        return RECORDING_EXAMPLES

    def get_system_prompt(self) -> str:
        return f"""You are a recording analysis specialist.

Your task is to analyze individual recordings and sessions in neuroscience datasets, extracting metadata for each recording and mapping them to the appropriate channel configurations.

OUTPUT FORMAT:
Return ONLY a valid JSON object matching this schema (no other text, explanation, or markdown):
{self.get_simplified_schema()}"""

    def get_tools(self) -> List:
        return create_recording_tools(self.base_dir)

    def get_output_schema(self) -> Optional[BaseModel]:
        return RecordingInfoOutput

    def _extract_channel_maps_for_tool(
        self, channel_maps_info: Any
    ) -> Dict[str, List[str]]:
        """
        Extract channel map IDs and their channel names for the analyze_all_recordings tool.

        Args:
            channel_maps_info: Channel maps from ChannelAgent (ChannelMapsOutput or dict)

        Returns:
            Dictionary of {map_id: [channel_names]}
        """
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

    async def process(
        self, dataset_id: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Analyze recordings and create recording info."""
        if not context or "channel_maps" not in context:
            return {"error": "RecordingAgent requires channel_maps from ChannelAgent"}

        agent = self.create_agent()

        channel_maps_info = context.get("channel_maps", {})
        channel_maps_for_tool = self._extract_channel_maps_for_tool(channel_maps_info)
        channel_maps_json = json.dumps(channel_maps_for_tool, indent=2)

        prompt = f"""
Analyze recordings for dataset {dataset_id}.

STEP 1: Call analyze_all_recordings with:
- dataset_id: "{dataset_id}"
- channel_maps: {channel_maps_json}

This tool will batch-download all *_eeg.json and *_channels.tsv files and extract:
- Recording IDs, subject IDs, task IDs, sessions, acquisitions
- Duration and sampling frequency from sidecar files
- Channel names, types, and bad channels from channels.tsv
- Matched channel_map_id based on channel overlap
- Participant info from participants.tsv

STEP 2: Review the tool output and generate the final RecordingInfoOutput JSON.

For each recording in the tool output, create a RecordingInfo entry with:
- recording_id: Use the extracted recording_id
- subject_id: Use the extracted subject_id
- task_id: Use the extracted task_id
- channel_map_id: Use matched_channel_map_id (verify it makes sense)
- duration_seconds: Use the extracted duration_seconds
- num_channels: Use the extracted num_channels
- participant_info: Use the extracted participant_info (exclude participant_id key)
- channels_to_remove: Use bad_channels list

VALIDATION:
- Verify channel_map matches make sense 
- If a match seems wrong, use acquisition type or channel count to determine correct map
- Ensure all recordings have valid channel_map_id from: {list(channel_maps_for_tool.keys())}

Return a complete, valid JSON object with the recording_info list."""

        try:
            # Use retry logic
            raw_result = await self._invoke_agent_with_retry(
                agent, prompt, "RecordingAgent", dataset_id
            )

            # Check if we got an error from retry logic
            if "error" in raw_result:
                return raw_result

            # Extract the actual text output (handles thinking output format)
            raw_output = raw_result["output"]
            output = self._extract_output_text(raw_output)

            # Log the output for debugging
            logger.info(
                f"RecordingAgent raw output (first 500 chars): {str(output)[:500]}"
            )

            filtered_result = self.output_parser.parse(output)
            return {
                "raw_recording_info": output,
                "recording_info": filtered_result,
            }
        except Exception as e:
            logger.error(f"Error in RecordingAgent for {dataset_id}: {e}")
            traceback.print_exc()
            return {"error": str(e)}


class SupervisorAgent(BaseAgent):
    """Supervisor agent that orchestrates other agents and manages workflow."""

    def __init__(
        self,
        model_name: str = "llama-3.3-70b",
        temperature: float = 0.1,
        provider: LLMProvider = "cerebras",
        verbose: bool = False,
        download_path: str = None,
    ):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            provider=provider,
            verbose=verbose,
            log_console=None,
        )
        self.download_path = download_path
        self.log_file_handler = None

    def get_system_prompt(self) -> str:
        return """You are the supervisor for the Dataset Wizard. Your responsibilities:

            1. Orchestrate the workflow between specialized agents
            2. Handle errors and conflicts
            3. Ensure the final Dataset object is complete and valid

            You coordinate Metadata, Channel, and Recording agents to populate a complete Dataset structure."""

    def get_tools(self) -> List:
        return []  # Supervisor doesn't use external tools directly

    def get_output_schema(self) -> Optional[BaseModel]:
        return Dataset  # Supervisor returns the final Dataset object

    def create_agent(self):
        """Supervisor doesn't need tools, just coordination logic."""
        return None

    def _setup_processing_context(
        self, dataset_id: str, context: Dict[str, Any] = None
    ) -> tuple[str, Dict[str, Any]]:
        """Set up temporary directory and enhanced context for processing."""
        if self.download_path:
            temp_dir = os.path.join(self.download_path, f"dataset_{dataset_id}")
            os.makedirs(temp_dir, exist_ok=True)
        else:
            temp_dir = tempfile.mkdtemp(prefix=f"dataset_{dataset_id}_")
        logger.info(f"Created temporary directory for configuration files: {temp_dir}")

        self._setup_file_logging(temp_dir, dataset_id)

        enhanced_context = context.copy() if context else {}
        enhanced_context["config_files_dir"] = temp_dir

        return temp_dir, enhanced_context

    def _setup_file_logging(self, log_dir: str, dataset_id: str) -> None:
        """Set up file handler for detailed logging of agent operations."""
        log_file = os.path.join(log_dir, f"{dataset_id}_agent_log.txt")

        log_file_handle = open(log_file, "w", encoding="utf-8")
        self.log_console = Console(file=log_file_handle, width=120, record=True)

        self.log_file_handler = RichHandler(
            console=self.log_console,
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_level=True,
            show_path=False,
        )
        self.log_file_handler.setLevel(logging.DEBUG)

        logger.addHandler(self.log_file_handler)

        self.log_console.print(
            Panel.fit(
                f"[bold cyan]Dataset Processing Log[/bold cyan]\n"
                f"[yellow]Dataset ID:[/yellow] {dataset_id}\n"
                f"[yellow]Log File:[/yellow] {log_file}",
                border_style="cyan",
                box=box.DOUBLE,
            )
        )

    def _cleanup_resources(self, temp_dir: str) -> None:
        """Clean up temporary directory and resources."""
        if self.log_console:
            self.log_console.print(
                Panel(
                    "[bold green]✓ Processing Completed Successfully[/bold green]",
                    border_style="green",
                    box=box.DOUBLE,
                )
            )

        if self.log_file_handler:
            logger.removeHandler(self.log_file_handler)
            self.log_file_handler.close()
            self.log_file_handler = None

        if self.log_console and hasattr(self.log_console.file, "close"):
            self.log_console.file.close()
            self.log_console = None

        if os.path.exists(temp_dir) and not self.download_path:
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

    async def _run_metadata_agent(
        self, dataset_id: str, enhanced_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run metadata agent and return raw results."""
        if self.log_console:
            self.log_console.print(
                Panel(
                    "[bold blue]🔍 STARTING METADATA AGENT[/bold blue]\n"
                    "[dim]Extracting dataset metadata, descriptions, and task categorization[/dim]",
                    border_style="blue",
                    box=box.HEAVY,
                )
            )

        metadata_agent = MetadataAgent(
            model_name=self.model_name,
            temperature=self.temperature,
            provider=self.provider,
            verbose=self.verbose,
            log_console=self.log_console,
            base_dir=enhanced_context.get("config_files_dir", ""),
        )
        result = await metadata_agent.process(dataset_id, enhanced_context)

        if self.log_console:
            status = (
                "[green]✓ COMPLETED[/green]"
                if "error" not in result
                else "[red]✗ ERROR[/red]"
            )
            self.log_console.print(
                Panel(
                    f"[bold]{status} - METADATA AGENT[/bold]",
                    border_style="green" if "error" not in result else "red",
                    box=box.HEAVY,
                )
            )

        return result

    async def _run_channel_agent(
        self, dataset_id: str, enhanced_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run channel agent and return raw results."""
        if self.log_console:
            self.log_console.print(
                Panel(
                    "[bold yellow]🎛️  STARTING CHANNEL AGENT[/bold yellow]\n"
                    "[dim]Analyzing channel configurations and creating channel maps[/dim]",
                    border_style="yellow",
                    box=box.HEAVY,
                )
            )

        channel_agent = ChannelAgent(
            model_name=self.model_name,
            temperature=self.temperature,
            provider=self.provider,
            verbose=self.verbose,
            log_console=self.log_console,
            base_dir=enhanced_context.get("config_files_dir", ""),
        )
        result = await channel_agent.process(dataset_id, enhanced_context)

        if self.log_console:
            status = (
                "[green]✓ COMPLETED[/green]"
                if "error" not in result
                else "[red]✗ ERROR[/red]"
            )
            self.log_console.print(
                Panel(
                    f"[bold]{status} - CHANNEL AGENT[/bold]",
                    border_style="green" if "error" not in result else "red",
                    box=box.HEAVY,
                )
            )

        return result

    async def _run_recording_agent(
        self, dataset_id: str, recording_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run recording agent and return raw results."""
        if self.log_console:
            self.log_console.print(
                Panel(
                    "[bold magenta]📊 STARTING RECORDING AGENT[/bold magenta]\n"
                    "[dim]Analyzing individual recordings and sessions[/dim]",
                    border_style="magenta",
                    box=box.HEAVY,
                )
            )

        recording_agent = RecordingAgent(
            model_name=self.model_name,
            temperature=self.temperature,
            provider=self.provider,
            verbose=self.verbose,
            log_console=self.log_console,
            base_dir=recording_context.get("config_files_dir", ""),
        )
        result = await recording_agent.process(dataset_id, recording_context)

        if self.log_console:
            status = (
                "[green]✓ COMPLETED[/green]"
                if "error" not in result
                else "[red]✗ ERROR[/red]"
            )
            self.log_console.print(
                Panel(
                    f"[bold]{status} - RECORDING AGENT[/bold]",
                    border_style="green" if "error" not in result else "red",
                    box=box.HEAVY,
                )
            )

        return result

    def _create_dataset(self, results: Dict[str, Any]) -> Optional[Dataset]:
        """Create and validate the final Dataset object."""
        try:
            metadata = results.get("structured_metadata")
            channel_maps = results.get("channel_maps")
            recording_info = results.get("recording_info")

            if not metadata:
                logger.error("No metadata found in results")
                return None

            if not channel_maps:
                logger.error("No channel maps found in results")
                return None

            if not recording_info:
                logger.error("No recording info found in results")
                return None

            dataset = Dataset(
                metadata=metadata,
                channel_maps=channel_maps,
                recording_info=recording_info,
            )

            if self.log_console:
                summary_table = Table(
                    title="[bold green]Dataset Creation Summary[/bold green]",
                    box=box.DOUBLE,
                    show_header=True,
                    header_style="bold cyan",
                )
                summary_table.add_column("Component", style="yellow", width=25)
                summary_table.add_column(
                    "Count", style="green", justify="right", width=15
                )
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

                self.log_console.print(summary_table)

            logger.info(
                f"Successfully created Dataset with {len(dataset.channel_maps.channel_maps)} channel maps and {len(dataset.recording_info.recording_info)} recordings"
            )
            return dataset

        except (ValidationError, KeyError) as e:
            logger.error(f"Failed to create Dataset object: {e}")
            if self.log_console:
                self.log_console.print(
                    Panel(
                        f"[bold red]✗ Dataset Creation Failed[/bold red]\n{str(e)}",
                        border_style="red",
                        box=box.HEAVY,
                    )
                )
            return None

    async def process(
        self, dataset_id: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Orchestrate the complete dataset processing workflow."""
        logger.info(f"Starting dataset processing for {dataset_id}")

        # Setup processing context and temp directory
        temp_dir, enhanced_context = self._setup_processing_context(dataset_id, context)
        results = {"config_files_dir": temp_dir}

        try:
            logger.info(f"Starting metadata extraction for {dataset_id}")

            # Step 1: Run metadata agent (files will be downloaded on-demand)
            metadata_result = await self._run_metadata_agent(
                dataset_id, enhanced_context
            )

            # Step 2: Collect results from metadata agent
            if isinstance(metadata_result, Exception):
                logger.error(f"Metadata processing failed: {metadata_result}")
                results["metadata_error"] = str(metadata_result)
            else:
                results.update(metadata_result)

            # Add the metadata to the context to be used by the channel and recording agents
            enhanced_context.update(metadata_result)

            channel_result = await self._run_channel_agent(dataset_id, enhanced_context)
            results.update(channel_result)

            if "error" not in channel_result:
                recording_context = {**enhanced_context, **results}
                recording_result = await self._run_recording_agent(
                    dataset_id, recording_context
                )
                results.update(recording_result)
            else:
                logger.error("Cannot proceed with recording agent without channel maps")

            # Step 5: Create final dataset object
            if self.log_console:
                self.log_console.print(
                    Panel(
                        "[bold cyan]🔨 Creating Final Dataset Object[/bold cyan]",
                        border_style="cyan",
                        box=box.ROUNDED,
                    )
                )

            dataset = self._create_dataset(results)
            return {
                "dataset": dataset,
                "results": results,
            }

        except Exception as e:
            logger.error(f"Supervisor Agent failed: {e}")
            traceback.print_exc()

            if self.log_console:
                self.log_console.print(
                    Panel(
                        f"[bold red]✗ SUPERVISOR AGENT FAILED[/bold red]\n\n"
                        f"[yellow]Error:[/yellow] {str(e)}\n\n"
                        f"[dim]{traceback.format_exc()}[/dim]",
                        border_style="red",
                        box=box.HEAVY,
                        title="ERROR",
                    )
                )

            return {"error": str(e), "partial_results": results}
        finally:
            self._cleanup_resources(temp_dir)
