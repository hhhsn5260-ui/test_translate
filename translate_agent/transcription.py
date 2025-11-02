from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

import whisper

from .config import TranscriptionConfig
from .types import TranscriptSegment

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Wrapper around OpenAI Whisper for generating timestamped transcripts."""

    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        logger.info("Loading Whisper model '%s'...", self.config.model_size)
        self._model = whisper.load_model(self.config.model_size, device=self.config.device)
        return self._model

    def transcribe(self, media_path: Path) -> List[TranscriptSegment]:
        model = self._load_model()
        logger.info("Transcribing audio from %s", media_path)
        result = model.transcribe(
            str(media_path),
            language=self.config.language,
            task="transcribe",
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        segments: Iterable[dict] = result.get("segments", [])
        return [
            TranscriptSegment(start=float(seg["start"]), end=float(seg["end"]), text=seg["text"].strip())
            for seg in segments
        ]
