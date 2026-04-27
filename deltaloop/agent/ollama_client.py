import time
from pathlib import Path

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from deltaloop.config import settings


class OllamaClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.ollama_base_url, timeout=120.0
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
    ) -> str:
        start = time.monotonic()
        logger.debug(
            f"ollama.complete model={model} temperature={temperature} "
            f"n_messages={len(messages)}"
        )
        try:
            response = await self._client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
            )
            response.raise_for_status()
            content: str = response.json()["message"]["content"]
            logger.debug(
                f"ollama.complete done latency={time.monotonic() - start:.2f}s "
                f"response_len={len(content)}"
            )
            return content
        except Exception as exc:
            logger.warning(f"ollama.complete failed ({type(exc).__name__}: {exc}), will retry")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def complete_multimodal(
        self,
        model: str,
        messages: list[dict],
        images: list[str],  # base64-encoded strings
    ) -> str:
        start = time.monotonic()
        logger.debug(f"ollama.complete_multimodal model={model} n_images={len(images)}")
        # Attach images to the last user message
        messages_with_images = [*messages[:-1], {**messages[-1], "images": images}]
        try:
            response = await self._client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages_with_images,
                    "stream": False,
                },
            )
            response.raise_for_status()
            content: str = response.json()["message"]["content"]
            logger.debug(
                f"ollama.complete_multimodal done latency={time.monotonic() - start:.2f}s"
            )
            return content
        except Exception as exc:
            logger.warning(
                f"ollama.complete_multimodal failed ({type(exc).__name__}: {exc}), will retry"
            )
            raise

    async def aclose(self) -> None:
        await self._client.aclose()


# Module-level singleton — nodes and tools import this
_instance: OllamaClient | None = None


def get_ollama_client() -> OllamaClient:
    global _instance
    if _instance is None:
        _instance = OllamaClient()
    return _instance
