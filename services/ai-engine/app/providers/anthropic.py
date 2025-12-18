"""Anthropic Claude provider implementation."""

import os
from typing import Optional, AsyncIterator

import httpx

from ..core.llm_client import BaseLLMProvider, LLMConfig


class AnthropicLLMProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url or "https://api.anthropic.com"
        self.api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
    
    def _headers(self) -> dict:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
    
    async def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate text from prompt."""
        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        if system:
            payload["system"] = system
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/v1/messages",
                headers=self._headers(),
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract text from content blocks
            content = data.get("content", [])
            text_parts = [
                block.get("text", "") 
                for block in content 
                if block.get("type") == "text"
            ]
            return "".join(text_parts)
    
    async def generate_stream(
        self, prompt: str, system: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Stream generated text."""
        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        
        if system:
            payload["system"] = system
        
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/messages",
                headers=self._headers(),
                json=payload,
                timeout=120.0,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        import json
                        try:
                            data = json.loads(line[6:])
                            if data.get("type") == "content_block_delta":
                                delta = data.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    yield delta.get("text", "")
                        except json.JSONDecodeError:
                            continue
