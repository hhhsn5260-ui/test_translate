from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TranscriptSegment:
    """Single transcript segment with timing data."""

    start: float
    end: float
    text: str
    translation: Optional[str] = None
    tts_path: Optional[Path] = None


@dataclass
class PipelineArtifacts:
    """Paths to the generated artifacts for a translation run."""

    video_path: Path
    dubbed_audio_path: Path
    subtitles_path: Path
    transcript_json: Path
