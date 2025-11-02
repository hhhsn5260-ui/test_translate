from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable

import srt

from .types import TranscriptSegment


def write_bilingual_srt(segments: Iterable[TranscriptSegment], output_path: Path, zh_first: bool = True) -> Path:
    subtitles = []
    for idx, segment in enumerate(segments, start=1):
        if not segment.translation:
            raise ValueError("All segments must be translated before creating subtitles.")
        content_lines = [segment.translation, segment.text] if zh_first else [segment.text, segment.translation]
        subtitle = srt.Subtitle(
            index=idx,
            start=dt.timedelta(seconds=segment.start),
            end=dt.timedelta(seconds=segment.end),
            content="\n".join(content_lines),
        )
        subtitles.append(subtitle)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(srt.compose(subtitles), encoding="utf-8")
    return output_path
