from __future__ import annotations

import logging
from pathlib import Path

from moviepy.editor import AudioFileClip, CompositeAudioClip, VideoFileClip

logger = logging.getLogger(__name__)


def get_video_duration(video_path: Path) -> float:
    with VideoFileClip(str(video_path)) as clip:
        return float(clip.duration)


def mux_video_with_audio(
    video_path: Path,
    dubbed_audio_path: Path,
    output_path: Path,
    mix_original_audio: bool = True,
    original_volume: float = 0.25,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with VideoFileClip(str(video_path)) as clip, AudioFileClip(str(dubbed_audio_path)) as dubbed_audio:
        if mix_original_audio and clip.audio is not None:
            original_audio = clip.audio.volumex(original_volume)
            final_audio = CompositeAudioClip([original_audio, dubbed_audio])
        else:
            final_audio = dubbed_audio

        clip_with_audio = clip.set_audio(final_audio)
        logger.info("Writing final video to %s", output_path)
        clip_with_audio.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
            threads=4,
        )
    return output_path
