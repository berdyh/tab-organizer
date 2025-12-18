"""DeepSeek provider implementation."""

import os
from typing import Optional, AsyncIterator

import httpx

from ..core.llm_client import BaseLLMProvider, BaseEmbeddingProvider, LLMConfig, EmbeddingConfig


class DeepSeekLLMProvider(BaseLLMProvider):
    """DeepSeek LLM provider."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url or "https://api.deepseek.com"
        self.api_key = config.api_key or os.getenv("DEEPSEEK_API_KEY")
    
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    async def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate text from prompt."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    
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
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "stream": True,
                },
                timeout=120.0,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        import json
                        data = json.loads(line[6:])
                        content = data["choices"][0].get("delta", {}).get("content", "")
                        if content:
                            yield content


class DeepSeekEmbeddingProvider(BaseEmbeddingProvider):
    """DeepSeek embedding provider."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.base_url = config.base_url or "https://api.deepseek.com"
        self.api_key = config.api_key or os.getenv("DEEPSEEK_API_KEY")
    
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers=self._headers(),
                json={
                    "model": self.config.model,
                    "input": texts,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]
    
    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for single text."""
        result = await self.embed([text])
        return result[0] if result else []
