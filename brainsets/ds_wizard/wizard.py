"""
Main Dataset Wizard interface for populating Dataset structures.
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from pathlib import Path

from brainsets.ds_wizard.agents import (
    PROVIDER_MODELS,
    SupervisorAgent,
    LLMProvider,
)
from brainsets.ds_wizard.dataset_struct import Dataset

logger = logging.getLogger(__name__)


class DatasetWizard:
    """
    Main interface for the Dataset Wizard.

    Uses agentic workflow to automatically populate Dataset structures
    from OpenNeuro dataset IDs.
    """

    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.1,
        provider: LLMProvider = "cerebras",
        api_key: Optional[str] = None,
        verbose: bool = False,
        download_path: Optional[str] = None,
    ):
        """
        Initialize the Dataset Wizard.

        Args:
            model_name: LLM model to use (default: gpt-4)
            temperature: Temperature for LLM (default: 0.1 for consistency)
            provider: LLM provider to use ("openai", "google", or "ollama")
            api_key: API key for the provider (uses environment variable if None)
            verbose: Whether to enable verbose output for debugging (default: False)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.provider = provider
        self.download_path = download_path
        # Set up API key based on provider
        if api_key:
            if provider == "openai":
                os.environ["OPENAI_API_KEY"] = api_key
            elif provider == "google":
                os.environ["GOOGLE_API_KEY"] = api_key
            elif provider == "ollama":
                # Ollama runs locally, no API key needed
                pass

        # Also load the .env file
        load_dotenv()

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Initialize supervisor agent
        self.supervisor = SupervisorAgent(
            model_name=model_name,
            temperature=temperature,
            provider=provider,
            verbose=verbose,
            download_path=download_path,
        )

    async def populate_dataset(
        self,
        dataset_id: str,
        context: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Populate a Dataset structure from an OpenNeuro dataset ID.

        Args:
            dataset_id: OpenNeuro dataset identifier (e.g., 'ds000247')
            context: Additional context to provide to agents
            save_path: Optional path to save the results

        Returns:
            Dictionary containing the populated Dataset and processing metadata
        """
        logger.info(f"Starting dataset population for {dataset_id}")

        try:
            # Run the supervisor agent workflow
            results = await self.supervisor.process(dataset_id, context or {})

            # Save results if path provided
            if save_path and results:
                self._save_results(results, save_path)

            return results

        except Exception as e:
            logger.error(f"Failed to populate dataset {dataset_id}: {e}")
            return {"error": str(e)}

    def populate_dataset_sync(
        self,
        dataset_id: str,
        context: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for populate_dataset.

        Args:
            dataset_id: OpenNeuro dataset identifier
            context: Additional context for agents
            save_path: Optional path to save results

        Returns:
            Dictionary containing populated Dataset and metadata
        """
        return asyncio.run(self.populate_dataset(dataset_id, context, save_path))

    def _save_results(self, results: Dict[str, Any], save_path: str) -> None:
        """Save results to file."""
        try:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Convert Dataset object to dict for JSON serialization
            serializable_results = {}

            if "dataset" in results and results["dataset"]:
                dataset = results["dataset"]
                if isinstance(dataset, Dataset):
                    serializable_results["dataset"] = dataset.model_dump()
                else:
                    serializable_results["dataset"] = dataset

            with open(path, "w") as f:
                json.dump(serializable_results, f, indent=2, default=str)

            logger.info(f"Results saved to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save results to {save_path}: {e}")

    def load_results(self, load_path: str) -> Dict[str, Any]:
        """Load previously saved results."""
        try:
            with open(load_path, "r") as f:
                data = json.load(f)

            # Reconstruct Dataset object if present
            if "dataset" in data and data["dataset"]:
                data["dataset"] = Dataset(**data["dataset"])

            return data

        except Exception as e:
            logger.error(f"Failed to load results from {load_path}: {e}")
            return {"error": str(e)}


# Convenience functions for common usage patterns
async def populate_dataset(
    dataset_id: str,
    model_name: str = "gpt-4",
    provider: LLMProvider = "openai",
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to populate a dataset with default settings.

    Args:
        dataset_id: OpenNeuro dataset identifier
        model_name: LLM model to use
        provider: LLM provider ("openai", "google", or "ollama")
        save_path: Optional path to save results

    Returns:
        Dictionary containing populated Dataset and metadata
    """
    wizard = DatasetWizard(model_name=model_name, provider=provider)
    return await wizard.populate_dataset(dataset_id, save_path=save_path)


def populate_dataset_sync(
    dataset_id: str,
    model_name: str = "gpt-4",
    provider: LLMProvider = "openai",
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Synchronous convenience function to populate a dataset.

    Args:
        dataset_id: OpenNeuro dataset identifier
        model_name: LLM model to use
        provider: LLM provider ("openai", "google", or "ollama")
        save_path: Optional path to save results

    Returns:
        Dictionary containing populated Dataset and metadata
    """
    wizard = DatasetWizard(model_name=model_name, provider=provider)
    return wizard.populate_dataset_sync(dataset_id, save_path=save_path)


if __name__ == "__main__":
    import sys

    import argparse

    parser = argparse.ArgumentParser(
        description="Populate dataset structures from OpenNeuro dataset IDs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python wizard.py ds005555 --provider=google --model=gemini-2.0-flash-lite
  python wizard.py ds005555 --output=result.json --provider=openai --model=gpt-4
  python wizard.py ds005555 -o custom_output.json -p ollama -m gpt-oss:20b""",
    )

    parser.add_argument(
        "dataset_id", help="OpenNeuro dataset identifier (e.g., ds000247)"
    )
    parser.add_argument(
        "-o", "--output", help="Output file path (default: <dataset_id>_result.json)"
    )
    parser.add_argument(
        "-p",
        "--provider",
        choices=list(PROVIDER_MODELS.keys()),
        default="cerebras",
        help="LLM provider to use (default: google)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="llama-3.3-70b",
        help="Model name to use (default: gemini-1.5-flash)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output for debugging (shows all tool calls)",
    )

    parser.add_argument(
        "-d",
        "--download_path",
        type=str,
        help="Path to download configuration files to",
    )

    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for LLM (default: 0.1)",
    )

    args = parser.parse_args()

    dataset_id = args.dataset_id

    # Parse arguments
    output_file = args.output
    provider = args.provider
    model = args.model
    verbose = args.verbose
    download_path = args.download_path
    temperature = args.temperature

    if output_file is None:
        output_file = f"{dataset_id}_result.json"

    print(f"Populating dataset {dataset_id} using {provider}:{model}...")
    if verbose:
        print("Verbose mode enabled - you'll see all tool calls and outputs")

    wizard = DatasetWizard(
        model_name=model,
        provider=provider,
        verbose=verbose,
        download_path=download_path,
        temperature=temperature,
    )
    results = wizard.populate_dataset_sync(dataset_id, save_path=output_file)
