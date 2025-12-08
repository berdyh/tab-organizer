import os
import asyncio
from typing import Dict, Any, List, Optional
import httpx
import structlog
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

logger = structlog.get_logger("llm_client")

class BaseLLMClient:
    async def generate(self, prompt: str, context: str = "", model: str = None) -> str:
        raise NotImplementedError

    async def get_embedding(self, text: str, model: str = None) -> List[float]:
        raise NotImplementedError

class OllamaClient(BaseLLMClient):
    def __init__(self, base_url: str):
        self.client = httpx.AsyncClient(base_url=base_url, timeout=60.0)
        self.default_model = os.getenv("DEFAULT_LLM_MODEL", "phi4:3.8b")
        self.default_embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", "mxbai-embed-large")

    async def generate(self, prompt: str, context: str = "", model: str = None) -> str:
        model = model or self.default_model
        full_prompt = f"""Context: {context}\n\nUser question: {prompt}"""
        try:
            response = await self.client.post(
                "/api/generate",
                json={"model": model, "prompt": full_prompt, "stream": False}
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.error("Ollama generation failed", error=str(e))
            raise

    async def get_embedding(self, text: str, model: str = None) -> List[float]:
        model = model or self.default_embedding_model
        try:
            response = await self.client.post(
                "/api/embeddings",
                json={"model": model, "prompt": text}
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            logger.error("Ollama embedding failed", error=str(e))
            raise

class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, base_url: str = None):
        if not AsyncOpenAI:
            raise ImportError("openai package not installed")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.default_model = "gpt-3.5-turbo"
        self.default_embedding_model = "text-embedding-3-small"

    async def generate(self, prompt: str, context: str = "", model: str = None) -> str:
        model = model or self.default_model
        messages = [
            {"role": "system", "content": f"Context: {context}"},
            {"role": "user", "content": prompt}
        ]
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("OpenAI generation failed", error=str(e))
            raise

    async def get_embedding(self, text: str, model: str = None) -> List[float]:
        model = model or self.default_embedding_model
        try:
            response = await self.client.embeddings.create(
                input=text,
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error("OpenAI embedding failed", error=str(e))
            raise

class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str):
        if not genai:
            raise ImportError("google-generativeai package not installed")
        genai.configure(api_key=api_key)
        self.default_model = "gemini-pro"
        self.default_embedding_model = "models/embedding-001"

    async def generate(self, prompt: str, context: str = "", model: str = None) -> str:
        model_name = model or self.default_model
        try:
            # Run blocking call in executor
            loop = asyncio.get_running_loop()
            model = genai.GenerativeModel(model_name)
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}"
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(full_prompt)
            )
            return response.text
        except Exception as e:
            logger.error("Gemini generation failed", error=str(e))
            raise

    async def get_embedding(self, text: str, model: str = None) -> List[float]:
        model = model or self.default_embedding_model
        try:
            # Run blocking call in executor
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: genai.embed_content(
                    model=model,
                    content=text,
                    task_type="retrieval_document"
                )
            )
            return result['embedding']
        except Exception as e:
            logger.error("Gemini embedding failed", error=str(e))
            raise

class LLMFactory:
    @staticmethod
    def get_client(provider: str, api_key: str = None, base_url: str = None) -> BaseLLMClient:
        provider = provider.lower()
        if provider == "openai":
            return OpenAIClient(api_key=api_key)
        elif provider == "deepseek":
            # DeepSeek V3.2 (beta) endpoint
            base_url = base_url or "https://api.deepseek.com/v1"
            return OpenAIClient(api_key=api_key, base_url=base_url)
        elif provider == "gemini":
            return GeminiClient(api_key=api_key)
        else:
            return OllamaClient(base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"))
