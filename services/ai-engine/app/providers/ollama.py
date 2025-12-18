"""Ollama provider implementation."""

import os
from typing import Optional, AsyncIterator

import httpx

from ..core.llm_client import BaseLLMProvider, BaseEmbeddingProvider, LLMConfig, EmbeddingConfig


class OllamaLLMProvider(BaseLLMProvider):
    """Ollama LLM provider for local models."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url or os.getenv("OLLAMA_HOST", "http://ollama:11434")
    
    async def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate text from prompt."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                    },
                },
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
    
    async def generate_stream(
        self, prompt: str, system: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Stream generated text."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                    },
                },
                timeout=120.0,
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Ollama embedding provider for local models."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.base_url = config.base_url or os.getenv("OLLAMA_HOST", "http://ollama:11434")
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        embeddings = []
        async with httpx.AsyncClient() as client:
            for text in texts:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.config.model,
                        "prompt": text,
                    },
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
                embeddings.append(data.get("embedding", []))
        return embeddings
    
    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for single text."""
        result = await self.embed([text])
        return result[0] if result else []
