"""Async client for interacting with Ollama."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List

import httpx
import structlog


class OllamaClient:
    """Client for interacting with Ollama LLM service with configurable models and fallback strategies."""

    def __init__(self, base_url: str = "http://ollama:11434") -> None:
        self.base_url = base_url
        self.logger = structlog.get_logger("ollama_client")
        self.current_model = None
        self.available_models: List[str] = []
        self.fallback_chains: Dict[str, List[str]] = {}
        self.retry_config = {
            "max_retries": 3,
            "base_delay": 1.0,
            "max_delay": 30.0,
            "backoff_factor": 2.0,
        }

    async def initialize(self) -> None:
        """Initialize Ollama client and discover available models."""
        try:
            await self._discover_available_models()
            await self._setup_fallback_chains()
            self.logger.info("Ollama client initialized", available_models=len(self.available_models))
        except Exception as exc:
            self.logger.error("Failed to initialize Ollama client", error=str(exc))

    async def _discover_available_models(self) -> None:
        """Discover which models are available in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    self.available_models = [model["name"] for model in data.get("models", [])]
                    self.logger.info("Discovered Ollama models", models=self.available_models)
                else:
                    self.logger.warning("Failed to discover models", status_code=response.status_code)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Error discovering Ollama models", error=str(exc))

    async def _setup_fallback_chains(self) -> None:
        """Setup intelligent fallback chains for different model families."""
        self.fallback_chains = {
            "llama3.2:8b": ["llama3.2:3b", "llama3.2:1b"],
            "llama3.2:3b": ["llama3.2:1b"],
            "qwen3:8b": ["qwen3:4b", "qwen3:1.7b", "qwen3:0.6b"],
            "qwen3:4b": ["qwen3:1.7b", "qwen3:0.6b"],
            "qwen3:1.7b": ["qwen3:0.6b"],
            "phi4:3.8b": ["gemma3n:e4b", "gemma3n:e2b"],
            "mistral:7b": ["llama3.2:3b", "llama3.2:1b"],
            "codellama:7b": ["phi4:3.8b", "qwen3:4b"],
        }

    async def ensure_model_available(self, model_id: str) -> bool:
        """Ensure model is available, pull if necessary."""
        if model_id in self.available_models:
            return True

        try:
            self.logger.info("Pulling model from Ollama", model_id=model_id)
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(f"{self.base_url}/api/pull", json={"name": model_id})
                if response.status_code == 200:
                    self.available_models.append(model_id)
                    self.logger.info("Successfully pulled model", model_id=model_id)
                    return True
                self.logger.error(
                    "Failed to pull model",
                    model_id=model_id,
                    status_code=response.status_code,
                )
                return False
        except Exception as exc:
            self.logger.error("Error pulling model", model_id=model_id, error=str(exc))
            return False

    async def generate_with_fallback(self, prompt: str, model_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Generate text with automatic fallback to smaller models on failure."""
        models_to_try = [model_id] + self.fallback_chains.get(model_id, [])

        for attempt_model in models_to_try:
            try:
                result = await self._generate_with_retry(prompt, attempt_model, **kwargs)
                if result["success"]:
                    if attempt_model != model_id:
                        self.logger.info(
                            "Used fallback model",
                            requested=model_id,
                            used=attempt_model,
                        )
                    return result
            except Exception as exc:
                self.logger.warning("Model generation failed", model=attempt_model, error=str(exc))
                continue

        return {
            "success": False,
            "error": f"All models failed for prompt generation: {models_to_try}",
            "response": "",
            "model_used": None,
        }

    async def _generate_with_retry(self, prompt: str, model_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Generate text with retry mechanism."""
        if not await self.ensure_model_available(model_id):
            raise RuntimeError(f"Model {model_id} not available")

        for attempt in range(self.retry_config["max_retries"]):
            try:
                start_time = time.time()

                async with httpx.AsyncClient(timeout=120.0) as client:
                    payload = {
                        "model": model_id,
                        "prompt": prompt,
                        "stream": False,
                        **kwargs,
                    }

                    response = await client.post(f"{self.base_url}/api/generate", json=payload)

                    if response.status_code == 200:
                        data = response.json()
                        response_time = time.time() - start_time

                        return {
                            "success": True,
                            "response": data.get("response", ""),
                            "model_used": model_id,
                            "response_time": response_time,
                            "tokens_evaluated": data.get("eval_count", 0),
                            "tokens_per_second": data.get("eval_count", 0) / response_time if response_time > 0 else 0,
                        }
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

            except Exception as exc:
                if attempt < self.retry_config["max_retries"] - 1:
                    delay = min(
                        self.retry_config["base_delay"]
                        * (self.retry_config["backoff_factor"] ** attempt),
                        self.retry_config["max_delay"],
                    )
                    self.logger.warning("Retrying after error", attempt=attempt + 1, delay=delay, error=str(exc))
                    await asyncio.sleep(delay)
                else:
                    raise

    async def summarize_content(self, content: str, model_id: str = "llama3.2:3b") -> Dict[str, Any]:
        """Generate a summary of the content."""
        prompt = f"""Please provide a concise summary of the following content. Focus on the main points and key information:

Content:
{content[:4000]}  # Limit content length

Summary:"""

        return await self.generate_with_fallback(prompt, model_id)

    async def extract_keywords(self, content: str, model_id: str = "llama3.2:3b") -> Dict[str, Any]:
        """Extract keywords and key phrases from content."""
        prompt = f"""Extract the most important keywords and key phrases from the following content. Provide them as a comma-separated list:

Content:
{content[:4000]}

Keywords:"""

        return await self.generate_with_fallback(prompt, model_id)

    async def assess_content_quality(self, content: str, model_id: str = "llama3.2:3b") -> Dict[str, Any]:
        """Assess the quality and characteristics of content."""
        prompt = """Analyze the following content and provide a quality assessment. Consider factors like:
- Clarity and coherence
- Information density
- Relevance and usefulness
- Writing quality

Provide a brief assessment and a quality score from 1-10:

Content:
{content}

Assessment:""".format(
            content=content[:4000]
        )

        return await self.generate_with_fallback(prompt, model_id)


__all__ = ["OllamaClient"]
