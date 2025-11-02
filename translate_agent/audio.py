from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable, Optional

from pydub import AudioSegment

from .types import TranscriptSegment

logger = logging.getLogger(__name__)


def build_dub_track(
    segments: Iterable[TranscriptSegment],
    duration_seconds: float,
    output_path: Path,
    sample_rate_hint: Optional[int] = None,
) -> Path:
    """Combine per-segment TTS audio into a single track aligned with the transcript."""

    segments = list(segments)
    if not segments:
        raise ValueError("No segments provided for dub track assembly.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_rate = sample_rate_hint
    primer_audio: Optional[AudioSegment] = None
    for segment in segments:
        if segment.tts_path and segment.tts_path.exists():
            primer_audio = AudioSegment.from_file(segment.tts_path)
            sample_rate = sample_rate or primer_audio.frame_rate
            break

    if sample_rate is None:
        sample_rate = 24000
        primer_audio = AudioSegment.silent(duration=500, frame_rate=sample_rate)

    total_duration_ms = int(math.ceil(duration_seconds * 1000.0)) + 500
    base_track = AudioSegment.silent(duration=total_duration_ms, frame_rate=sample_rate)

    for segment in segments:
        if not segment.tts_path or not segment.tts_path.exists():
            logger.warning("Skipping segment without TTS audio: %s", segment)
            continue
        clip = AudioSegment.from_file(segment.tts_path)
        if clip.frame_rate != sample_rate:
            clip = clip.set_frame_rate(sample_rate)
        position_ms = max(0, int(segment.start * 1000))
        base_track = base_track.overlay(clip, position=position_ms)

    finalized = base_track[: int(duration_seconds * 1000)]
    audio_format = output_path.suffix.lstrip(".") or "wav"
    finalized.export(output_path, format=audio_format)
    logger.info("Created dubbed audio track at %s", output_path)
    return output_path
