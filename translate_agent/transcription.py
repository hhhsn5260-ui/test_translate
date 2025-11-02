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
        logger.info("Loading Whisper model '%s' on device '%s'...", self.config.model_size, self.config.device or "default")
        try:
            self._model = whisper.load_model(self.config.model_size, device=self.config.device)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error("Failed to load Whisper model: %s", e)
            raise
        return self._model

    def transcribe(self, media_path: Path) -> List[TranscriptSegment]:
        model = self._load_model()
        logger.info("Transcribing audio from %s", media_path)
        try:
            result = model.transcribe(
                str(media_path),
                language=self.config.language,
                task="transcribe",
                condition_on_previous_text=False,
                word_timestamps=False,
            )
            logger.info("Transcription completed, processing segments")
            segments: Iterable[dict] = result.get("segments", [])
            collected = [
                TranscriptSegment(start=float(seg["start"]), end=float(seg["end"]), text=seg["text"].strip())
                for seg in segments
            ]
            logger.info("Transcription complete: %s segments", len(collected))
            if logger.isEnabledFor(logging.DEBUG):
                for idx, seg in enumerate(collected, start=1):
                    logger.debug("Segment %03d: %.2f-%.2f %s", idx, seg.start, seg.end, seg.text)
            return collected
        except Exception as e:
            logger.error("Error during transcription: %s", e)
            raise