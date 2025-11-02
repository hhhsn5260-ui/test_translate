from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TranscriptionConfig:
    """Configuration for Whisper transcription."""

    model_size: str = "tiny"  # 更改默认模型为 tiny，提高CPU上的运行速度
    language: str = "en"
    device: Optional[str] = None  # 默认不指定设备，让系统自动选择


@dataclass
class TranslationConfig:
    """Configuration for text translation."""

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_retries: int = 3
    retry_delay: float = 3.0
    api_base: Optional[str] = None
    api_key_env: Optional[str] = "OPENAI_API_KEY"


@dataclass
class TTSConfig:
    """Configuration for text-to-speech synthesis."""

    provider: str = "openai"
    model: str = "gpt-4o-mini-tts"
    voice: str = "alloy"
    format: str = "wav"
    speaking_rate: float = 1.0
    api_base: Optional[str] = None
    api_key_env: Optional[str] = "OPENAI_API_KEY"
    edge_voice: str = "zh-CN-XiaoxiaoNeural"
    edge_rate: str = "+0%"
    edge_volume: str = "+0%"
    edge_output_format: str = "audio-24khz-48kbitrate-mono-mp3"


@dataclass
class PipelineConfig:
    """Top level configuration for the translation agent."""

    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    output_root: Path = Path("artifacts")
    overwrite: bool = True
    mix_original_audio: bool = True
    original_audio_mix_level: float = 0.25
    sample_rate_hint: Optional[int] = None