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
    MessagesPlaceholder,
    FewShotChatMessagePromptTemplate,
)
from pydantic import BaseModel, Field, ValidationError

from brainsets.utils.open_neuro import download_configuration_files
from brainsets.ds_wizard.dataset_struct import (
    Dataset,
    DatasetMetadata,
    ChannelMapsOutput,
    RecordingInfoOutput,
)
from brainsets.ds_wizard.tools import (
    METADATA_TOOLS,
    CHANNEL_TOOLS,
    RECORDING_TOOLS,
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
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.provider = provider
        self.verbose = verbose
        self.use_few_shot = use_few_shot
        self.num_examples = num_examples
        self.max_retries = max_retries
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
                thinking_budget=5000,
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
        if not raw_result:
            return

        # Log intermediate steps if available
        if "intermediate_steps" in raw_result:
            steps = raw_result["intermediate_steps"]
            logger.info(
                f"{agent_name} executed {len(steps)} tool calls for {dataset_id}"
            )

            if steps:
                for i, (action, result) in enumerate(steps):
                    if hasattr(action, "tool"):
                        tool_name = action.tool
                        logger.info(f"  Step {i+1}: Called tool '{tool_name}'")
                        logger.info(f"  Result preview: {str(result)[:100]}...")
            else:
                logger.warning(f"{agent_name} made NO tool calls before finishing")

        # Log token usage from callback handler
        if token_callback:
            usage_summary = token_callback.get_usage_summary()
            logger.info(
                f"{agent_name} token usage for {dataset_id}: "
                f"input={usage_summary['input_tokens']}, "
                f"output={usage_summary['output_tokens']}, "
                f"total={usage_summary['total_tokens']}, "
                f"llm_calls={usage_summary['llm_calls']}"
            )

        # Log other metadata
        if "return_values" in raw_result:
            logger.info(
                f"{agent_name} return_values keys: {raw_result['return_values'].keys()}"
            )

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
        return f"""You are a metadata extraction specialist for neuroscience datasets. Use your tools to extract information!

    YOUR TASKS:
    1. Use fetch_dataset_metadata to get dataset information
    2. Use fetch_dataset_readme to get README content  
    3. Use get_task_taxonomy to get valid task categories
    4. Extract and categorize all required metadata fields

    GUIDELINES:
    - Summaries: ≤150 words, derived from README
    - Task matching: Use most specific taxonomy match or null
    - Brainset name format: <last_author>_<recognizable_word>_<dataset_id>_<year>
    - Authors: Extract from README or metadata fields
    - Categories: Match to taxonomy using get_task_taxonomy tool
    - Tools: Call each tool at most once with the same parameters.

    START by fetching metadata, README, and taxonomy, then extract all fields systematically.
    
    CRITICAL OUTPUT REQUIREMENTS:
    - You MUST always return a complete, valid JSON object
    - NEVER return an empty string, incomplete output, or stop mid-generation
    - If information is missing, use your best judgment to provide reasonable values
    - Complete the entire JSON structure before finishing

    Return ONLY a structured json object with the following schema, no other text or explanation or markdown syntax:
    {self.get_simplified_schema()}
    """

    def get_tools(self) -> List:
        return METADATA_TOOLS

    def get_output_schema(self) -> Optional[BaseModel]:
        return DatasetMetadata

    async def process(
        self, dataset_id: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Extract metadata information for the dataset."""
        agent = self.create_agent()

        prompt = f"""
        Analyze dataset {dataset_id} and extract metadata information.
        
        MANDATORY STEPS:
        1. Use fetch_dataset_metadata to get dataset information
        2. Use fetch_dataset_readme to get README content
        3. Use get_task_taxonomy to get available categories
        4. Extract all required DatasetMetadata fields from these sources
        
        CRITICAL: You MUST return a valid JSON object. Do NOT return an empty string or incomplete output.
        If you're unsure about any field, provide your best estimate based on available information.

        DO NOT ask for examples, figure out what to do using your tools!"""

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
        return f"""
        You are an EEG channel mapping specialist. 

        Each dataset is usually recorded with a single EEG device, but some datasets might use multiple device. 
        Given that data is manually collected, sessions recorded using the same devices could have different channel configurations, names and types from human error.
        Your goal is to create one map for each of the device used in the dataset.
        A channel map should map channel names as found in the dataset and map them to some standard names if that is already not the case.
        Additionally, For each electrode in a channel map, you'll need to classify its type (modality) based on all the information available.
        Additionally, you'll need to extract the coordinates of the electrodes for EEG electrodes.

        PROCESS:
        1. Use the DatasetReadmeTool to get README content. This is the primary source of information about the dataset that should be used to extract information.
        2. Use the ModalitiesTool to get valid electrode types.
        3. Use the ListConfigurationFilesTool to find channel configuration files (*_channels.tsv, *_electrodes.tsv)
        4. Use the ReadConfigurationFileTool to examine file structure and content
        5. Use the GetEEGBidsSpecsTool to understand BIDS format specifications
        6. Get all the unique channels that exist per device session from the configuration files.
        7. For each of those channels, estimate the modality type based on the information available.
        8. For all the channels that you identified as EEG, if some of the channel names seem to not be standard names, try to map them to a standard name 
        as could be found in an MNE montage. Use the ElectrodeToMontageMappingTool to get the mapping. Examples: 'EEG_Fp1 -> Fp1', 'Fp2-Fz -> Fp2'...
        If you cannot find a match, use the modality EEG-OTHER.
        9. For all channels that you identified as EEG, extract the coordinates based on the information available.
        Use the AllMontageMatchesTool to get the mapping of electrode names to montages.
        10. Create a ChannelMap object for each device, extracting the device name and manufacturer.

        CRITICAL OUTPUT REQUIREMENTS:
        - You MUST always return a complete, valid JSON object
        - NEVER return an empty string, incomplete output, or stop mid-generation
        - If information is missing, use your best judgment to provide reasonable values
        - Complete the entire JSON structure before finishing

        DO NOT ask for examples - discover the structure using your tools!"""

    def get_tools(self) -> List:
        return CHANNEL_TOOLS

    def get_output_schema(self) -> Optional[BaseModel]:
        return ChannelMapsOutput

    async def process(
        self, dataset_id: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Analyze channel configurations and create channel maps."""
        agent = self.create_agent()

        config_dir = context.get("config_files_dir", "") if context else ""
        config_info = (
            f"Configuration files have been downloaded to: {config_dir}"
            if config_dir
            else ""
        )

        # Available metadata: {context.get("raw_metadata", {}).replace("{", "[").replace("}", "]")}
        prompt = f"""
        Create channel maps for dataset {dataset_id}.
        
        {config_info}
        
        CRITICAL: You MUST return a valid JSON object. Do NOT return an empty string or incomplete output.
        If you're unsure about any field, provide your best estimate based on available information.

        Return ONLY a structured json object with the following schema, no other text or explanation or markdown syntax:
        {self.get_simplified_schema()}
        """

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
        return f"""You are a recording analysis specialist. Use your tools to discover recording information!

    YOUR TASKS:
    1. Use fetch_participants to get the COMPLETE list of all participant IDs in the dataset
    2. Use fetch_dataset_filenames to identify ALL recording files for each participant
    3. Use list_configuration_files to find participant/session metadata files (participants.tsv, sessions.tsv, etc.)
    4. Use read_configuration_file to examine participant metadata and session information
    5. Map each recording to the appropriate channel_map_id from ChannelAgent results
    6. Extract recording metadata and participant information for EVERY participant

    CRITICAL REQUIREMENTS:
    - You MUST use fetch_participants FIRST to get the complete list of ALL participants
    - You MUST create recording_info entries for EVERY participant returned by fetch_participants
    - Cross-reference the participant list with recording files to ensure nothing is missed
    - If a participant has no recording files, investigate why and include all available info

    CRITICAL DATA EXTRACTION REQUIREMENTS:
    - duration_seconds: MUST be extracted from the SPECIFIC recording's *_eeg.json sidecar file
      * Look for fields like "RecordingDuration" or calculate from samples/sampling_rate
      * DO NOT assume all recordings have the same duration
      * DO NOT copy duration from other recordings
      * Each recording may have different duration - read each one individually
    - num_channels: MUST be counted from the SPECIFIC recording's *_channels.tsv file
      * Count the actual number of rows in the recording-specific channels.tsv file
      * DO NOT assume all recordings have the same number of channels
      * DO NOT copy channel count from other recordings
      * Each recording may have different channels - read each one individually

    PROCESS:
    1. START by calling fetch_participants to get the authoritative list of ALL participant IDs
    2. CALL fetch_dataset_filenames to get all recording files in the dataset
    3. LIST configuration files to find all metadata files (participants.tsv, sessions.tsv, *_channels.tsv, *_eeg.json)
    4. READ participants.tsv to understand participant information structure
    5. For EACH recording found in the dataset:
       a. Identify the recording (BIDS naming: sub-XX_[ses-YY_]task-ZZ_eeg.*)
       b. Find and READ the recording's *_eeg.json sidecar file for metadata (duration, sampling rate)
       c. Find and READ the recording's *_channels.tsv file to COUNT the actual number of channels
       d. Extract task, session, and subject information from the filename
       e. Match recording to appropriate channel_map_id based on device/setup from channel file
       f. Get participant info from participants.tsv for that subject
    6. VERIFY that you have created entries for ALL recordings and participants

    IMPORTANT: DO NOT make assumptions about duration or channel counts. Each recording is unique and must be examined individually!

    You must reference channel_maps from context to assign channel_map_id correctly!
    
    CRITICAL OUTPUT REQUIREMENTS:
    - You MUST always return a complete, valid JSON object
    - NEVER return an empty string, incomplete output, or stop mid-generation
    - If information is missing, use your best judgment to provide reasonable values
    - Complete the entire JSON structure before finishing"""

    def get_tools(self) -> List:
        return RECORDING_TOOLS

    def get_output_schema(self) -> Optional[BaseModel]:
        return RecordingInfoOutput

    async def process(
        self, dataset_id: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Analyze recordings and create recording info."""
        # Check if we have channel maps
        if not context or "channel_maps" not in context:
            return {"error": "RecordingAgent requires channel_maps from ChannelAgent"}

        agent = self.create_agent()

        config_dir = context.get("config_files_dir", "")
        config_info = (
            f"Configuration files have been downloaded to: {config_dir}"
            if config_dir
            else ""
        )

        # Format channel maps for display in prompt
        channel_maps_info = context.get("channel_maps", {})
        if isinstance(channel_maps_info, dict):
            channel_maps_display = json.dumps(channel_maps_info, indent=2, default=str)
        else:
            channel_maps_display = str(channel_maps_info)

        prompt = f"""
        Analyze recordings for dataset {dataset_id}.
        
        Available Channel Maps: {channel_maps_display}
        {config_info}
        
        MANDATORY STEPS:
        1. FIRST: Use fetch_participants tool to get the complete list of ALL participant IDs for {dataset_id}
        2. Use fetch_dataset_filenames to get all recording files
        3. Use list_configuration_files and read_configuration_file to get participant metadata
        4. Create a recording_info entry for EVERY participant from step 1
        5. Verify the total count of recording_info matches the number of participants
        
        You MUST return a valid JSON object. Do NOT return an empty string or incomplete output.
        If you're unsure about any field, provide your best estimate based on available information.
        
        Return ONLY a structured json object with the following schema, no other text or explanation or markdown syntax:
        {self.get_simplified_schema()}
        """

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
        )
        self.download_path = download_path

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

        enhanced_context = context.copy() if context else {}
        enhanced_context["config_files_dir"] = temp_dir

        return temp_dir, enhanced_context

    def _cleanup_resources(self, temp_dir: str) -> None:
        """Clean up temporary directory and resources."""
        if os.path.exists(temp_dir) and not self.download_path:
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

    async def _run_metadata_agent(
        self, dataset_id: str, enhanced_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run metadata agent and return raw results."""
        metadata_agent = MetadataAgent(
            model_name=self.model_name,
            temperature=self.temperature,
            provider=self.provider,
            verbose=self.verbose,
        )
        return await metadata_agent.process(dataset_id, enhanced_context)

    async def _run_channel_agent(
        self, dataset_id: str, enhanced_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run channel agent and return raw results."""
        channel_agent = ChannelAgent(
            model_name=self.model_name,
            temperature=self.temperature,
            provider=self.provider,
            verbose=self.verbose,
        )
        return await channel_agent.process(dataset_id, enhanced_context)

    async def _run_recording_agent(
        self, dataset_id: str, recording_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run recording agent and return raw results."""
        recording_agent = RecordingAgent(
            model_name=self.model_name,
            temperature=self.temperature,
            provider=self.provider,
            verbose=self.verbose,
        )
        return await recording_agent.process(dataset_id, recording_context)

    def _create_dataset(self, results: Dict[str, Any]) -> Optional[Dataset]:
        """Create and validate the final Dataset object."""
        try:
            # Get the structured objects directly from agent results
            metadata = results.get("structured_metadata")
            channel_maps = results.get("channel_maps")
            recording_info = results.get("recording_info")

            # Validate we have the required data
            if not metadata:
                logger.error("No metadata found in results")
                return None

            if not channel_maps:
                logger.error("No channel maps found in results")
                return None

            if not recording_info:
                logger.error("No recording info found in results")
                return None

            # Create Dataset object
            dataset = Dataset(
                metadata=metadata,
                channel_maps=channel_maps,
                recording_info=recording_info,
            )

            logger.info(
                f"Successfully created Dataset with {len(dataset.channel_maps.channel_maps)} channel maps and {len(dataset.recording_info.recording_info)} recordings"
            )
            return dataset

        except (ValidationError, KeyError) as e:
            logger.error(f"Failed to create Dataset object: {e}")
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
            logger.info(f"Starting download and metadata extraction for {dataset_id}")

            # Step 1: Download configuration files and run metadata agent in parallel
            # Check if files are already downloaded to avoid redundant downloads
            if os.listdir(temp_dir):
                logger.info(
                    f"Configuration files already exist in {temp_dir}, skipping download"
                )
                download_task = None
            else:
                download_task = asyncio.get_event_loop().run_in_executor(
                    None, download_configuration_files, dataset_id, temp_dir
                )

            metadata_task = self._run_metadata_agent(dataset_id, enhanced_context)

            # Wait for both to complete (or just metadata if download was skipped)
            if download_task:
                _, metadata_result = await asyncio.gather(
                    download_task, metadata_task, return_exceptions=True
                )
            else:
                metadata_result = await metadata_task

            logger.info(f"Configuration files downloaded to {temp_dir}")

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
            dataset = self._create_dataset(results)
            return {
                "dataset": dataset,
                "results": results,
            }

        except Exception as e:
            logger.error(f"Supervisor Agent failed: {e}")
            traceback.print_exc()
            return {"error": str(e), "partial_results": results}
        finally:
            self._cleanup_resources(temp_dir)
