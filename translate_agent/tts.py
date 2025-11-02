from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Iterable, Optional

from openai import OpenAI

try:
    from openai import APIError  # type: ignore
except ImportError:  # Backwards compat for older SDKs
    class APIError(Exception):
        pass

from .config import TTSConfig
from .types import TranscriptSegment

logger = logging.getLogger(__name__)


class BaseTTS:
    def synthesize_segments(self, segments: Iterable[TranscriptSegment], output_dir: Path) -> None:
        raise NotImplementedError


class OpenAITTS(BaseTTS):
    """Use OpenAI's TTS models to generate Chinese speech audio segments."""

    def __init__(self, config: TTSConfig, client: Optional[OpenAI] = None, max_retries: int = 3, retry_delay: float = 3.0):
        self.config = config
        if client is not None and config.api_base:
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
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def synthesize_segments(self, segments: Iterable[TranscriptSegment], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for idx, segment in enumerate(segments, start=1):
            if not segment.translation:
                raise ValueError("Segment must include translation text before TTS synthesis.")
            filename = output_dir / f"segment_{idx:04d}.{self.config.format}"
            self._synthesize_to_file(segment.translation, filename)
            segment.tts_path = filename
            if idx == 1 or idx % 10 == 0:
                logger.info("TTS progress (OpenAI): %s segments synthesized", idx)

    def _synthesize_to_file(self, text: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, self.max_retries + 1):
            try:
                with self.client.audio.speech.with_streaming_response.create(
                    model=self.config.model,
                    voice=self.config.voice,
                    input=text,
                    format=self.config.format,
                    voice_settings={"speaking_rate": self.config.speaking_rate},
                ) as response:
                    response.stream_to_file(output_path)
                logger.debug("Generated TTS segment at %s", output_path)
                return
            except APIError as exc:
                logger.warning("TTS attempt %s failed: %s", attempt, exc)
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_delay)


class EdgeTTS(BaseTTS):
    """Use Microsoft Edge neural voices without requiring Azure credentials."""

    def __init__(self, config: TTSConfig):
        try:
            import edge_tts  # type: ignore
        except ImportError as exc:
            raise RuntimeError("edge-tts package is required for Edge TTS provider.") from exc
        self.edge_tts = edge_tts
        self.config = config

    def synthesize_segments(self, segments: Iterable[TranscriptSegment], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for idx, segment in enumerate(segments, start=1):
            if not segment.translation:
                raise ValueError("Segment must include translation text before TTS synthesis.")
            filename = output_dir / f"segment_{idx:04d}.{self.config.format}"
            self._run_async(self._synthesize_to_file(segment.translation, filename))
            segment.tts_path = filename
            if idx == 1 or idx % 10 == 0:
                logger.info("TTS progress (Edge): %s segments synthesized", idx)

    async def _synthesize_to_file(self, text: str, output_path: Path) -> None:
        communicate = self.edge_tts.Communicate(
            text=text,
            voice=self.config.edge_voice,
            rate=self.config.edge_rate,
            volume=self.config.edge_volume,
            output_format=self.config.edge_output_format,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as outfile:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    outfile.write(chunk["data"])
        logger.debug("Generated Edge TTS segment at %s", output_path)

    def _run_async(self, coroutine) -> None:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coroutine)
        finally:
            loop.close()


def build_tts(config: TTSConfig, client: Optional[OpenAI] = None) -> BaseTTS:
    provider = (config.provider or "openai").lower()
    if provider == "openai":
        return OpenAITTS(config=config, client=client)
    if provider == "edge":
        return EdgeTTS(config=config)
    raise ValueError(f"Unsupported TTS provider: {config.provider}")
