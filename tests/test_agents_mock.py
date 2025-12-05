"""
Unit tests for agents using mock LLMs for fast, deterministic testing.
"""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import FakeChatModel

from brainsets.ds_wizard.agents import MetadataAgent, ChannelAgent
from brainsets.ds_wizard.dataset_struct import DatasetMetadata


class MockLLMForMetadata(FakeChatModel):
    """Mock LLM that returns valid metadata JSON."""

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        return """
{
    "name": "Test Dataset",
    "brainset_name": "smith_test_ds001234_2023",
    "version": "1.0.0",
    "dataset_id": "ds001234",
    "dataset_summary": "This is a test dataset for unit testing purposes.",
    "task_description": "Participants performed a simple test task.",
    "task_category": "Resting & Baseline States",
    "task_subcategory": "Eyes open / closed rest",
    "authors": ["Smith, John", "Doe, Jane"],
    "date": "01-01-2023"
}
"""


def test_metadata_agent_configuration():
    """Test that MetadataAgent configuration methods work correctly."""
    # Test basic configuration without initializing LLM
    from brainsets.ds_wizard.agents import PROVIDER_MODELS

    # Verify provider models are defined
    assert "google" in PROVIDER_MODELS
    assert "cerebras" in PROVIDER_MODELS
    assert "ollama" in PROVIDER_MODELS

    # Test output schema
    from brainsets.ds_wizard.dataset_struct import DatasetMetadata

    assert DatasetMetadata is not None


def test_extract_output_text_content_blocks():
    """Test _extract_output_text handles new LangChain 1.x content block format."""
    # Create a minimal agent just to test the extraction method
    # Use ollama which doesn't require authentication at init time
    from unittest.mock import Mock

    agent = Mock()
    agent._extract_output_text = MetadataAgent._extract_output_text.__get__(agent, Mock)

    # Test with content blocks (new format)
    content_blocks = [
        {"type": "thinking", "thinking": "Internal reasoning here..."},
        {"type": "text", "text": "This is the actual output"},
        {"type": "text", "text": "More output text"},
    ]
    result = agent._extract_output_text(content_blocks)
    assert result == "This is the actual output\nMore output text"

    # Test with simple string
    result = agent._extract_output_text("Simple string output")
    assert result == "Simple string output"

    # Test with list of strings (legacy format)
    result = agent._extract_output_text(["text1", "text2"])
    assert result == "text1\ntext2"

    # Test with empty/None
    assert agent._extract_output_text(None) == ""
    assert agent._extract_output_text([]) == ""


def test_agent_tools_are_valid():
    """Test that all agent tools are properly configured."""
    from brainsets.ds_wizard.tools import METADATA_TOOLS, CHANNEL_TOOLS, RECORDING_TOOLS

    # Verify tools are lists
    assert isinstance(METADATA_TOOLS, list)
    assert isinstance(CHANNEL_TOOLS, list)
    assert isinstance(RECORDING_TOOLS, list)

    # Verify all tools have names
    for tool in METADATA_TOOLS + CHANNEL_TOOLS + RECORDING_TOOLS:
        assert hasattr(tool, "name")
        assert hasattr(tool, "description")
        assert tool.name is not None
        assert tool.description is not None
