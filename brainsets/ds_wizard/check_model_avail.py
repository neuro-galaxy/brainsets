#!/usr/bin/env python3
"""
Script to test which LLM models are available and working across different providers.
Helps diagnose issues like models getting stuck or being unavailable.
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Tuple, Any
from dotenv import load_dotenv

# Import the LLM classes from your existing setup
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Test models for each provider
TEST_MODELS = {
    "openai": [],
    "google": [],
    "ollama": ["gpt-oss:20b"],
}

# Simple test prompt
TEST_PROMPT = "Hello! Please respond with exactly: 'Model test successful'"


class ModelTester:
    """Test model availability and basic functionality."""

    def __init__(self, timeout_seconds: int = 30):
        self.timeout = timeout_seconds
        self.results = {
            "working_models": [],
            "failed_models": [],
            "timeout_models": [],
            "unauthorized_models": [],
            "not_found_models": [],
        }

    async def test_openai_model(self, model_name: str) -> Tuple[str, bool, str, float]:
        """Test an OpenAI model."""
        start_time = time.time()

        try:
            if not os.getenv("OPENAI_API_KEY"):
                return model_name, False, "No OPENAI_API_KEY found", 0.0

            llm = ChatOpenAI(model=model_name, temperature=0.1, timeout=self.timeout)

            # Test with timeout
            response = await asyncio.wait_for(
                llm.ainvoke(TEST_PROMPT), timeout=self.timeout
            )

            duration = time.time() - start_time
            return model_name, True, response.content.strip(), duration

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            return model_name, False, "Timeout", duration
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e).lower()
            if "unauthorized" in error_msg or "401" in error_msg:
                return model_name, False, f"Unauthorized: {str(e)}", duration
            elif "not found" in error_msg or "404" in error_msg:
                return model_name, False, f"Model not found: {str(e)}", duration
            else:
                return model_name, False, f"Error: {str(e)}", duration

    async def test_google_model(self, model_name: str) -> Tuple[str, bool, str, float]:
        """Test a Google/Gemini model."""
        start_time = time.time()

        try:
            if not os.getenv("GOOGLE_API_KEY"):
                return model_name, False, "No GOOGLE_API_KEY found", 0.0

            llm = ChatGoogleGenerativeAI(
                model=model_name, temperature=0.1, convert_system_message_to_human=True
            )

            # Test with timeout
            response = await asyncio.wait_for(
                llm.ainvoke(TEST_PROMPT), timeout=self.timeout
            )

            duration = time.time() - start_time
            return model_name, True, response.content.strip(), duration

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            return model_name, False, "Timeout", duration
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e).lower()
            if "unauthorized" in error_msg or "401" in error_msg:
                return model_name, False, f"Unauthorized: {str(e)}", duration
            elif (
                "not found" in error_msg or "404" in error_msg or "invalid" in error_msg
            ):
                return model_name, False, f"Model not found: {str(e)}", duration
            else:
                return model_name, False, f"Error: {str(e)}", duration

    async def test_ollama_model(self, model_name: str) -> Tuple[str, bool, str, float]:
        """Test an Ollama model."""
        start_time = time.time()

        try:
            # Ollama runs locally, no API key needed but check if service is accessible
            llm = ChatOllama(
                model=model_name, temperature=0.1, base_url="http://localhost:11434"
            )

            # Test with timeout
            response = await asyncio.wait_for(
                llm.ainvoke(TEST_PROMPT), timeout=self.timeout
            )

            duration = time.time() - start_time
            return model_name, True, response.content.strip(), duration

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            return model_name, False, "Timeout", duration
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e).lower()
            if "connection" in error_msg or "refused" in error_msg:
                return (
                    model_name,
                    False,
                    "Ollama server not running (http://localhost:11434)",
                    duration,
                )
            elif "not found" in error_msg or "404" in error_msg:
                return model_name, False, f"Model not found: {str(e)}", duration
            else:
                return model_name, False, f"Error: {str(e)}", duration

    async def test_provider_models(
        self, provider: str, models: List[str]
    ) -> List[Dict[str, Any]]:
        """Test all models for a specific provider."""
        logger.info(f"Testing {len(models)} {provider} models...")

        # Choose the appropriate test function
        if provider == "openai":
            test_func = self.test_openai_model
        elif provider == "google":
            test_func = self.test_google_model
        elif provider == "ollama":
            test_func = self.test_ollama_model
        else:
            logger.error(f"Unknown provider: {provider}")
            return []

        # Test models with limited concurrency to avoid rate limits
        semaphore = asyncio.Semaphore(2)  # Max 2 concurrent tests per provider

        async def test_with_semaphore(model):
            async with semaphore:
                return await test_func(model)

        # Run tests
        tasks = [test_with_semaphore(model) for model in models]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        provider_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Exception during test: {result}")
                continue

            model_name, success, response, duration = result
            result_dict = {
                "provider": provider,
                "model": model_name,
                "success": success,
                "response": response,
                "duration_seconds": round(duration, 2),
            }

            provider_results.append(result_dict)

            # Categorize results
            if success:
                self.results["working_models"].append(result_dict)
                logger.info(f"✓ {provider}:{model_name} - {duration:.2f}s")
            else:
                if "Timeout" in response:
                    self.results["timeout_models"].append(result_dict)
                    logger.warning(f"⏰ {provider}:{model_name} - TIMEOUT")
                elif "Unauthorized" in response:
                    self.results["unauthorized_models"].append(result_dict)
                    logger.warning(f"🔒 {provider}:{model_name} - UNAUTHORIZED")
                elif "not found" in response.lower():
                    self.results["not_found_models"].append(result_dict)
                    logger.warning(f"❓ {provider}:{model_name} - NOT FOUND")
                else:
                    self.results["failed_models"].append(result_dict)
                    logger.error(f"❌ {provider}:{model_name} - {response}")

        return provider_results

    async def test_all_models(self) -> Dict[str, Any]:
        """Test all models across all providers."""
        logger.info("Starting comprehensive model testing...")
        start_time = time.time()

        all_results = []

        # Test each provider
        for provider, models in TEST_MODELS.items():
            if models:  # Skip empty model lists
                provider_results = await self.test_provider_models(provider, models)
                all_results.extend(provider_results)

                # Brief pause between providers to avoid rate limits
                await asyncio.sleep(1)

        total_duration = time.time() - start_time

        # Create summary
        summary = {
            "test_summary": {
                "total_models_tested": len(all_results),
                "working_models": len(self.results["working_models"]),
                "failed_models": len(self.results["failed_models"]),
                "timeout_models": len(self.results["timeout_models"]),
                "unauthorized_models": len(self.results["unauthorized_models"]),
                "not_found_models": len(self.results["not_found_models"]),
                "total_test_duration_seconds": round(total_duration, 2),
            },
            "results_by_category": self.results,
            "all_test_results": all_results,
        }

        return summary

    def print_summary(self, results: Dict[str, Any]):
        """Print a nice summary of the test results."""
        print("\n" + "=" * 60)
        print("MODEL AVAILABILITY TEST RESULTS")
        print("=" * 60)

        summary = results["test_summary"]
        print(f"Total models tested: {summary['total_models_tested']}")
        print(f"Total test duration: {summary['total_test_duration_seconds']}s")
        print()

        # Working models
        working = results["results_by_category"]["working_models"]
        if working:
            print("✓ WORKING MODELS:")
            for model in working:
                print(
                    f"  {model['provider']}:{model['model']} ({model['duration_seconds']}s)"
                )
            print()

        # Failed models
        failed = results["results_by_category"]["failed_models"]
        if failed:
            print("❌ FAILED MODELS:")
            for model in failed:
                print(f"  {model['provider']}:{model['model']} - {model['response']}")
            print()

        # Timeout models
        timeout = results["results_by_category"]["timeout_models"]
        if timeout:
            print("⏰ TIMEOUT MODELS:")
            for model in timeout:
                print(
                    f"  {model['provider']}:{model['model']} - took >{model['duration_seconds']}s"
                )
            print()

        # Unauthorized models
        unauthorized = results["results_by_category"]["unauthorized_models"]
        if unauthorized:
            print("🔒 UNAUTHORIZED MODELS (need valid API key):")
            for model in unauthorized:
                print(f"  {model['provider']}:{model['model']}")
            print()

        # Not found models
        not_found = results["results_by_category"]["not_found_models"]
        if not_found:
            print("❓ NOT FOUND MODELS:")
            for model in not_found:
                print(f"  {model['provider']}:{model['model']}")
            print()

        # Recommendations
        print("RECOMMENDATIONS:")
        if working:
            fastest = min(working, key=lambda x: x["duration_seconds"])
            print(
                f"• Fastest working model: {fastest['provider']}:{fastest['model']} ({fastest['duration_seconds']}s)"
            )

        if timeout:
            print(f"• Avoid timeout models for production use")

        if "gemini-2.0-flash-lite" in [m["model"] for m in working]:
            print(f"• Your default model (gemini-2.0-flash-lite) is working!")
        elif "gemini-2.0-flash-light" in [m["model"] for m in working]:
            print(
                f"• The typo version (gemini-2.0-flash-light) is working - fix your code!"
            )
        else:
            print(f"• Your default model may have issues - consider switching")

        print("=" * 60)


async def main():
    global TEST_MODELS
    """Main function to run model tests."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test LLM model availability and functionality"
    )
    parser.add_argument(
        "--timeout", type=int, default=30, help="Timeout per model test in seconds"
    )
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument(
        "--provider",
        choices=["openai", "google", "ollama"],
        help="Test only specific provider",
    )
    parser.add_argument("--model", type=str, help="Test only specific model name")

    args = parser.parse_args()

    # Filter models based on arguments
    test_models = TEST_MODELS.copy()
    if args.provider:
        test_models = {args.provider: test_models[args.provider]}

    if args.model:
        # Find which provider has this model
        found_provider = None
        for provider, models in test_models.items():
            if args.model in models:
                found_provider = provider
                break

        if found_provider:
            test_models = {found_provider: [args.model]}
        else:
            # Add the model to all providers to test
            for provider in test_models:
                test_models[provider] = [args.model]

    # Create tester and run tests
    tester = ModelTester(timeout_seconds=args.timeout)

    # Temporarily override TEST_MODELS for this run
    original_test_models = TEST_MODELS
    TEST_MODELS = test_models

    try:
        results = await tester.test_all_models()

        # Print summary
        tester.print_summary(results)

        # Save results if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nDetailed results saved to: {args.output}")

    finally:
        # Restore original TEST_MODELS
        TEST_MODELS = original_test_models


if __name__ == "__main__":
    asyncio.run(main())
