from __future__ import annotations

import logging
import os
import time
from typing import Iterable, List, Optional

import requests
from openai import OpenAI

try:
    from openai import APIError  # type: ignore
except ImportError:  # Backwards compat for older SDKs
    class APIError(Exception):
        pass

from .config import TranslationConfig
from .types import TranscriptSegment

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_PREFIX = (
    "将以下英文口语内容翻译成符合中文口语习惯的表达，用于视频配音。"
    "保持原意和语气，不要添加额外说明或数字编号。\n\n"
)


class BaseTranslator:
    def translate_segments(self, segments: Iterable[TranscriptSegment]) -> List[TranscriptSegment]:
        raise NotImplementedError


class OpenAITranslator(BaseTranslator):
    """Translate transcript segments using an OpenAI-compatible Responses API."""

    def __init__(self, config: TranslationConfig, client: Optional[OpenAI] = None):
        self.config = config
        if client is not None and config.api_base:
            # override base via provided client duplicating base_url not supported; use new instance
            logger.warning("Ignoring provided OpenAI client because custom api_base was supplied.")
            client = None
        kwargs = {}
        if config.api_base:
            kwargs["base_url"] = config.api_base
        if config.api_key_env:
            api_key = os.getenv(config.api_key_env)
            if api_key:
                kwargs["api_key"] = api_key
        self.client = client or OpenAI(**kwargs)

    def translate_segments(self, segments: Iterable[TranscriptSegment]) -> List[TranscriptSegment]:
        translated: List[TranscriptSegment] = []
        for segment in segments:
            segment.translation = self._translate_text(segment.text)
            translated.append(segment)
        return translated

    def _translate_text(self, text: str) -> str:
        prompt = DEFAULT_PROMPT_PREFIX + text.strip()
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self.client.responses.create(
                    model=self.config.model,
                    input=[
                        {"role": "system", "content": "You are a professional bilingual translation assistant."},
                        {"role": "user", "content": [{"type": "text", "text": prompt}]},
                    ],
                    temperature=self.config.temperature,
                )
                content = response.output[0].content[0].text.strip()
                logger.debug("Translation result: %s -> %s", text, content)
                return content
            except APIError as exc:
                logger.warning("Translation attempt %s failed: %s", attempt, exc)
                if attempt >= self.config.max_retries:
                    raise
                time.sleep(self.config.retry_delay)
        raise RuntimeError("Unreachable translation retry loop")


class DeepSeekTranslator(BaseTranslator):
    """Translate transcript segments via the DeepSeek REST API."""

    def __init__(self, config: TranslationConfig):
        self.config = config
        self.api_key = os.getenv(config.api_key_env or "DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                f"DeepSeek API key not found. Please set environment variable '{config.api_key_env or 'DEEPSEEK_API_KEY'}'."
            )
        self.base_url = (config.api_base or "https://api.deepseek.com").rstrip("/")

    def translate_segments(self, segments: Iterable[TranscriptSegment]) -> List[TranscriptSegment]:
        translated: List[TranscriptSegment] = []
        for segment in segments:
            segment.translation = self._translate_text(segment.text)
            translated.append(segment)
        return translated

    def _translate_text(self, text: str) -> str:
        prompt = DEFAULT_PROMPT_PREFIX + text.strip()
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.config.model or "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a professional bilingual translation assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
        }
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                if response.status_code >= 400:
                    logger.warning("DeepSeek translation failed (HTTP %s): %s", response.status_code, response.text)
                    raise RuntimeError(response.text)
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                logger.debug("DeepSeek translation result: %s -> %s", text, content)
                return content
            except Exception as exc:  # broad to include network errors
                logger.warning("DeepSeek translation attempt %s failed: %s", attempt, exc)
                if attempt >= self.config.max_retries:
                    raise
                time.sleep(self.config.retry_delay)
        raise RuntimeError("Unreachable translation retry loop")


def build_translator(config: TranslationConfig, client: Optional[OpenAI] = None) -> BaseTranslator:
    provider = (config.provider or "openai").lower()
    if provider == "openai":
        return OpenAITranslator(config=config, client=client)
    if provider == "deepseek":
        return DeepSeekTranslator(config=config)
    raise ValueError(f"Unsupported translation provider: {config.provider}")
