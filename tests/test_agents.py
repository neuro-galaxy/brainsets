import os
import pytest
from dotenv import load_dotenv, find_dotenv
from brainsets.ds_wizard.agents import (
    MetadataAgent,
    ChannelAgent,
)
from brainsets.ds_wizard.dataset_struct import DatasetMetadata


@pytest.fixture(autouse=True)
def load_env_and_check_api_key():
    """
    Pytest fixture that runs before every test in this file to:
    1. Load the .env file if it exists
    2. Skip all tests if CEREBRAS_API_KEY is not set
    """
    # Attempt to find and load the .env file
    dotenv_path = find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path)
        print(f"Loaded environment variables from: {dotenv_path}")
    else:
        print("No .env file found")

    # Check if CEREBRAS_API_KEY is set
    cerebras_api_key = os.getenv("CEREBRAS_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not cerebras_api_key and not google_api_key:
        pytest.skip(
            "CEREBRAS_API_KEY or GOOGLE_API_KEY is not set. Skipping test_agents.py tests."
        )

    print(f"CEREBRAS_API_KEY is set (length: {len(cerebras_api_key)})")
    return cerebras_api_key, google_api_key


@pytest.fixture
def model_information():
    # return {"provider": "cerebras", "model_name": "llama-3.3-70b", "temperature": 0.1}
    # return {
    #     "provider": "vertexai",
    #     "model_name": "Qwen3-Next-80B-Thinking",
    #     "temperature": 0.1,
    # }
    return {
        "provider": "google",
        "model_name": "gemini-2.5-flash-lite",
        "temperature": 0.1,
    }


@pytest.mark.asyncio
async def test_metadata_agent(model_information):
    """Test the MetadataAgent."""
    metadata_agent = MetadataAgent(
        provider=model_information["provider"],
        model_name=model_information["model_name"],
        temperature=model_information["temperature"],
    )
    assert metadata_agent is not None
    assert metadata_agent.provider == model_information["provider"]
    assert metadata_agent.model_name == model_information["model_name"]
    assert metadata_agent.temperature == model_information["temperature"]

    result = await metadata_agent.process("ds006695")
    assert result is not None
    assert "error" not in result.keys()
    assert "raw_metadata" in result.keys()
    assert "structured_metadata" in result.keys()
    assert type(result["structured_metadata"]) == DatasetMetadata


@pytest.mark.asyncio
async def test_channel_agent(model_information):
    """Test the ChannelAgent."""
    channel_agent = ChannelAgent(
        provider=model_information["provider"],
        model_name=model_information["model_name"],
        temperature=model_information["temperature"],
    )
    assert channel_agent is not None
    assert channel_agent.provider == model_information["provider"]
    assert channel_agent.model_name == model_information["model_name"]
    assert channel_agent.temperature == model_information["temperature"]
