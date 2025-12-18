"""Google Gemini provider implementation."""

import os
from typing import Optional, AsyncIterator

import httpx

from ..core.llm_client import BaseLLMProvider, BaseEmbeddingProvider, LLMConfig, EmbeddingConfig


class GeminiLLMProvider(BaseLLMProvider):
    """Google Gemini LLM provider."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url or "https://generativelanguage.googleapis.com/v1beta"
        self.api_key = config.api_key or os.getenv("GOOGLE_API_KEY")
    
    async def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate text from prompt."""
        contents = []
        
        if system:
            contents.append({
                "role": "user",
                "parts": [{"text": f"System instruction: {system}"}]
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "Understood. I will follow these instructions."}]
            })
        
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/models/{self.config.model}:generateContent",
                params={"key": self.api_key},
                json={
                    "contents": contents,
                    "generationConfig": {
                        "temperature": self.config.temperature,
                        "maxOutputTokens": self.config.max_tokens,
                    },
                },
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
            
            candidates = data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                return "".join(part.get("text", "") for part in parts)
            return ""
    
    async def generate_stream(
        self, prompt: str, system: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Stream generated text."""
        contents = []
        
        if system:
            contents.append({
                "role": "user",
                "parts": [{"text": f"System instruction: {system}"}]
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "Understood. I will follow these instructions."}]
            })
        
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })
        
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/models/{self.config.model}:streamGenerateContent",
                params={"key": self.api_key, "alt": "sse"},
                json={
                    "contents": contents,
                    "generationConfig": {
                        "temperature": self.config.temperature,
                        "maxOutputTokens": self.config.max_tokens,
                    },
                },
                timeout=120.0,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        import json
                        try:
                            data = json.loads(line[6:])
                            candidates = data.get("candidates", [])
                            if candidates:
                                content = candidates[0].get("content", {})
                                parts = content.get("parts", [])
                                for part in parts:
                                    text = part.get("text", "")
                                    if text:
                                        yield text
                        except json.JSONDecodeError:
                            continue


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Google Gemini embedding provider."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.base_url = config.base_url or "https://generativelanguage.googleapis.com/v1beta"
        self.api_key = config.api_key or os.getenv("GOOGLE_API_KEY")
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        embeddings = []
        
        async with httpx.AsyncClient() as client:
            for text in texts:
                response = await client.post(
                    f"{self.base_url}/models/{self.config.model}:embedContent",
                    params={"key": self.api_key},
                    json={
                        "model": f"models/{self.config.model}",
                        "content": {
                            "parts": [{"text": text}]
                        },
                    },
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
                embedding = data.get("embedding", {}).get("values", [])
                embeddings.append(embedding)
        
        return embeddings
    
    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for single text."""
        result = await self.embed([text])
        return result[0] if result else []
