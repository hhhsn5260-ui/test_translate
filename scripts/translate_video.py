import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from translate_agent import PipelineConfig, VideoTranslationAgent
from translate_agent.config import TTSConfig, TranscriptionConfig, TranslationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate an English video into Chinese audio with bilingual subtitles.")
    parser.add_argument("video", type=Path, help="Path to the source video file.")
    parser.add_argument("--run-name", type=str, help="Optional name for this translation run.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Directory to store generated artifacts.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite previous runs with the same name.")
    parser.add_argument("--whisper-model", type=str, default="base", help="Whisper model size (tiny/base/small/medium/large).")
    parser.add_argument("--translation-provider", type=str, choices=["openai", "deepseek"], default="openai", help="Translation backend to use.")
    parser.add_argument("--translation-model", type=str, default="gpt-4o-mini", help="Model name for translation provider.")
    parser.add_argument("--translation-api-base", type=str, help="Custom base URL for translation API (optional).")
    parser.add_argument("--translation-api-key-env", type=str, help="Environment variable containing translation API key.")
    parser.add_argument("--tts-provider", type=str, choices=["openai", "edge"], default="openai", help="TTS backend to use.")
    parser.add_argument("--tts-model", type=str, default="gpt-4o-mini-tts", help="Model name for OpenAI TTS provider.")
    parser.add_argument("--tts-voice", type=str, default="alloy", help="Voice name for the TTS model.")
    parser.add_argument("--tts-format", type=str, default="wav", help="Audio format for synthesized speech (wav/mp3).")
    parser.add_argument("--speaking-rate", type=float, default=1.0, help="Relative speaking rate for the TTS voice.")
    parser.add_argument("--tts-api-base", type=str, help="Custom base URL for TTS API (optional).")
    parser.add_argument("--tts-api-key-env", type=str, help="Environment variable containing TTS API key.")
    parser.add_argument("--edge-voice", type=str, default="zh-CN-XiaoxiaoNeural", help="Voice ID for Edge TTS provider.")
    parser.add_argument("--edge-rate", type=str, default="+0%", help="Speech rate adjustment for Edge TTS (e.g., +10%).")
    parser.add_argument("--edge-volume", type=str, default="+0%", help="Volume adjustment for Edge TTS (e.g., +0%).")
    parser.add_argument(
        "--edge-output-format",
        type=str,
        default="audio-24khz-48kbitrate-mono-mp3",
        help="Edge TTS output format (controls bitrate and sample rate).",
    )
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for the translation model.")
    parser.add_argument("--no-mix", action="store_true", help="Do not keep the original English audio underneath the dub.")
    parser.add_argument("--mix-level", type=float, default=0.25, help="Volume multiplier for the original English audio.")
    parser.add_argument("--sample-rate", type=int, help="Optional sample rate hint for the dubbed audio track.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    transcription = TranscriptionConfig(model_size=args.whisper_model)

    translation_model = args.translation_model
    if args.translation_provider == "deepseek" and translation_model == "gpt-4o-mini":
        translation_model = "deepseek-chat"
    translation_api_key_env = args.translation_api_key_env
    if not translation_api_key_env:
        translation_api_key_env = "OPENAI_API_KEY" if args.translation_provider == "openai" else "DEEPSEEK_API_KEY"

    translation = TranslationConfig(
        provider=args.translation_provider,
        model=translation_model,
        temperature=args.temperature,
        api_base=args.translation_api_base,
        api_key_env=translation_api_key_env,
    )

    tts_format = args.tts_format
    if args.tts_provider == "edge" and tts_format.lower() == "wav":
        tts_format = "mp3"
    tts_api_key_env = args.tts_api_key_env or ("OPENAI_API_KEY" if args.tts_provider == "openai" else None)

    tts = TTSConfig(
        provider=args.tts_provider,
        model=args.tts_model,
        voice=args.tts_voice,
        format=tts_format,
        speaking_rate=args.speaking_rate,
        api_base=args.tts_api_base,
        api_key_env=tts_api_key_env,
        edge_voice=args.edge_voice,
        edge_rate=args.edge_rate,
        edge_volume=args.edge_volume,
        edge_output_format=args.edge_output_format,
    )

    return PipelineConfig(
        transcription=transcription,
        translation=translation,
        tts=tts,
        output_root=args.output_dir,
        overwrite=args.overwrite,
        mix_original_audio=not args.no_mix,
        original_audio_mix_level=args.mix_level,
        sample_rate_hint=args.sample_rate,
    )


def main() -> None:
    args = parse_args()
    load_dotenv()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    config = build_config(args)
    agent = VideoTranslationAgent(config=config)

    artifacts = agent.run(video_path=args.video, run_name=args.run_name)

    logging.info("Video with Chinese dub: %s", artifacts.video_path)
    logging.info("Dubbed audio track: %s", artifacts.dubbed_audio_path)
    logging.info("Bilingual subtitles: %s", artifacts.subtitles_path)
    logging.info("Transcript metadata: %s", artifacts.transcript_json)


if __name__ == "__main__":
    main()
