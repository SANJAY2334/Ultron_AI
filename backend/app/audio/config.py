"""Audio Engine Configuration Manager (Phase 4E.3).

Provides unified configuration resolution for VAD, Wake Word, STT, and TTS subsystems
derived from application Settings and environment variables.
"""

from pydantic import BaseModel, Field

from app.audio.stt import STTConfig
from app.audio.tts import TTSConfig
from app.audio.vad import VADConfig
from app.audio.wake_word import WakeWordConfig
from app.core.config import Settings, get_settings


class AudioEngineConfig(BaseModel):
    """Unified configuration container for all production audio engine adapters."""

    vad_provider: str = Field(default="local", description="Active VAD engine provider name")
    wake_word_provider: str = Field(
        default="local", description="Active Wake Word engine provider name"
    )
    stt_provider: str = Field(default="whisper_api", description="Active STT provider name")
    tts_provider: str = Field(default="openai_tts", description="Active TTS provider name")
    language: str = Field(default="en-IN", description="Default ISO language code")
    sample_rate: int = Field(default=16000, description="Default audio sample rate in Hz")
    channels: int = Field(default=1, description="Default audio channels count")
    wake_word: str = Field(default="ultron", description="Primary wake-word trigger phrase")
    wake_word_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Wake-word threshold"
    )


def get_audio_engine_config(settings: Settings | None = None) -> AudioEngineConfig:
    """Builds AudioEngineConfig from system Settings instance."""
    cfg = settings or get_settings()
    return AudioEngineConfig(
        vad_provider=cfg.AUDIO_VAD_PROVIDER,
        wake_word_provider=cfg.AUDIO_WAKE_WORD_PROVIDER,
        stt_provider=cfg.AUDIO_STT_PROVIDER,
        tts_provider=cfg.AUDIO_TTS_PROVIDER,
        language=cfg.AUDIO_LANGUAGE,
        sample_rate=cfg.AUDIO_SAMPLE_RATE,
        channels=cfg.AUDIO_CHANNELS,
        wake_word=cfg.AUDIO_WAKE_WORD,
        wake_word_threshold=cfg.AUDIO_WAKE_WORD_THRESHOLD,
    )


def create_vad_config(settings: Settings | None = None) -> VADConfig:
    """Constructs VADConfig from system settings."""
    _cfg = settings or get_settings()
    return VADConfig(
        speech_threshold=_cfg.AUDIO_WAKE_WORD_THRESHOLD,
    )


def create_wake_word_config(settings: Settings | None = None) -> WakeWordConfig:
    """Constructs WakeWordConfig with target wake-word trigger phrase."""
    cfg = settings or get_settings()
    return WakeWordConfig(
        provider=cfg.AUDIO_WAKE_WORD_PROVIDER,
        wake_words=[cfg.AUDIO_WAKE_WORD.lower(), f"hey {cfg.AUDIO_WAKE_WORD.lower()}"],
        confidence_threshold=cfg.AUDIO_WAKE_WORD_THRESHOLD,
    )


def create_stt_config(settings: Settings | None = None) -> STTConfig:
    """Constructs STTConfig for transcription provider."""
    cfg = settings or get_settings()
    return STTConfig(
        provider_name=cfg.AUDIO_STT_PROVIDER,
        language=cfg.AUDIO_LANGUAGE[:2],
    )


def create_tts_config(settings: Settings | None = None) -> TTSConfig:
    """Constructs TTSConfig for speech synthesis provider."""
    cfg = settings or get_settings()
    return TTSConfig(
        provider_name=cfg.AUDIO_TTS_PROVIDER,
        language=cfg.AUDIO_LANGUAGE[:2],
        sample_rate=cfg.AUDIO_SAMPLE_RATE,
        channels=cfg.AUDIO_CHANNELS,
    )
