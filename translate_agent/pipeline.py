from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .audio import build_dub_track
from .config import PipelineConfig
from .subtitles import write_bilingual_srt
from .transcription import WhisperTranscriber
from .translation import build_translator
from .tts import build_tts
from .types import PipelineArtifacts, TranscriptSegment
from .video import get_video_duration, mux_video_with_audio

logger = logging.getLogger(__name__)


class VideoTranslationAgent:
    """High-level orchestrator that drives the translation, dubbing, and muxing pipeline."""

    def __init__(self, config: Optional[PipelineConfig] = None, client: Optional[OpenAI] = None):
        self.config = config or PipelineConfig()
        self.client = client or (OpenAI() if self._requires_openai_client() else None)
        self.transcriber = WhisperTranscriber(self.config.transcription)
        self.translator = build_translator(self.config.translation, client=self.client)
        self.tts = build_tts(self.config.tts, client=self.client)

    def run(self, video_path: Path, run_name: Optional[str] = None) -> PipelineArtifacts:
        video_path = video_path.resolve()
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        base_output_dir = self.config.output_root.resolve()
        run_name = run_name or self._default_run_name(video_path)
        run_dir = base_output_dir / run_name

        if run_dir.exists() and not self.config.overwrite:
            raise FileExistsError(f"{run_dir} already exists and overwrite=False")

        logger.info("Starting translation run '%s' for %s", run_name, video_path)
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Step 1/4: Transcribing source audio...")
        segments = self._create_transcript(video_path)

        logger.info("Step 2/4: Translating transcript segments...")
        self._translate_segments(segments)

        logger.info("Step 3/4: Generating TTS audio segments...")
        self._synthesize_audio(segments, run_dir / "tts_segments")

        duration = get_video_duration(video_path)
        logger.info("Step 4/4: Building dubbed track and muxing final video...")
        dubbed_audio_path = build_dub_track(
            segments=segments,
            duration_seconds=duration,
            output_path=run_dir / "audio" / f"{run_name}_zh.wav",
            sample_rate_hint=self.config.sample_rate_hint,
        )
        video_output_path = mux_video_with_audio(
            video_path=video_path,
            dubbed_audio_path=dubbed_audio_path,
            output_path=run_dir / "video" / f"{run_name}_zh.mp4",
            mix_original_audio=self.config.mix_original_audio,
            original_volume=self.config.original_audio_mix_level,
        )
        subtitles_path = write_bilingual_srt(
            segments=segments,
            output_path=run_dir / "subtitles" / f"{run_name}_bilingual.srt",
        )
        transcript_path = self._write_transcript_json(segments, run_dir / "transcript" / f"{run_name}.json")

        artifacts = PipelineArtifacts(
            video_path=video_output_path,
            dubbed_audio_path=dubbed_audio_path,
            subtitles_path=subtitles_path,
            transcript_json=transcript_path,
        )

        logger.info("Translation run completed. Artifacts: %s", artifacts)
        return artifacts

    def _create_transcript(self, video_path: Path) -> list[TranscriptSegment]:
        return self.transcriber.transcribe(video_path)

    def _translate_segments(self, segments: list[TranscriptSegment]) -> None:
        logger.info("Translating %s transcript segments...", len(segments))
        self.translator.translate_segments(segments)

    def _synthesize_audio(self, segments: list[TranscriptSegment], output_dir: Path) -> None:
        logger.info("Generating Chinese TTS audio for %s segments...", len(segments))
        self.tts.synthesize_segments(segments, output_dir=output_dir)

    def _write_transcript_json(self, segments: list[TranscriptSegment], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "start": segment.start,
                "end": segment.end,
                "text_en": segment.text,
                "text_zh": segment.translation,
                "tts_path": str(segment.tts_path) if segment.tts_path else None,
            }
            for segment in segments
        ]
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path

    def _default_run_name(self, video_path: Path) -> str:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{video_path.stem}_{timestamp}"

    def _requires_openai_client(self) -> bool:
        translation_provider = (self.config.translation.provider or "openai").lower()
        tts_provider = (self.config.tts.provider or "openai").lower()
        return "openai" in {translation_provider, tts_provider}
