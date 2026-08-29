"""Audio Engine Composition Factory (Phase 4E.4).

Provides centralized composition and instantiation of concrete production audio engine adapters
(VAD, Wake Word, STT, TTS, Capture, Playback) driven cleanly by application configuration settings.
Strictly eliminates ad-hoc conditional provider checks across application domain layers.
"""

import logging

from app.audio.adapters.microphone import MicrophoneCaptureAdapter
from app.audio.adapters.mock_playback import MockPlaybackAdapter
from app.audio.adapters.real_stt import RealSTTProvider
from app.audio.adapters.real_tts import RealTTSProvider
from app.audio.adapters.sounddevice_capture import SoundDeviceCaptureAdapter
from app.audio.adapters.sounddevice_playback import SoundDevicePlaybackAdapter
from app.audio.adapters.stt import STTAdapter
from app.audio.adapters.tts import TTSAdapter
from app.audio.adapters.vad import VADAdapter
from app.audio.adapters.wake_word import WakeWordAdapter
from app.audio.base import (
    IAudioCapture,
    IAudioPlayback,
    ISpeechToTextProvider,
    ITextToSpeechProvider,
    IVoiceActivityDetector,
    IWakeWordDetector,
)
from app.audio.config import (
    create_stt_config,
    create_tts_config,
    create_vad_config,
    create_wake_word_config,
)
from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class AudioEngineFactory:
    """Centralized Composition Factory for ULTRON Audio Engine Adapters."""

    @staticmethod
    def create_vad_detector(settings: Settings | None = None) -> IVoiceActivityDetector:
        """Constructs IVoiceActivityDetector based on AUDIO_VAD_PROVIDER setting."""
        cfg = settings or get_settings()
        provider_name = cfg.AUDIO_VAD_PROVIDER.lower()
        vad_config = create_vad_config(cfg)

        logger.info(f"AudioEngineFactory: Constructing VAD provider '{provider_name}'.")
        # Both local energy VAD and real VAD utilize VADAdapter
        return VADAdapter(config=vad_config)

    @staticmethod
    def create_wake_word_detector(settings: Settings | None = None) -> IWakeWordDetector:
        """Constructs IWakeWordDetector based on AUDIO_WAKE_WORD_PROVIDER setting."""
        cfg = settings or get_settings()
        provider_name = cfg.AUDIO_WAKE_WORD_PROVIDER.lower()
        ww_config = create_wake_word_config(cfg)

        logger.info(f"AudioEngineFactory: Constructing Wake-Word provider '{provider_name}'.")
        return WakeWordAdapter(config=ww_config, mock_trigger_word=cfg.AUDIO_WAKE_WORD.lower())

    @staticmethod
    def create_stt_provider(settings: Settings | None = None) -> ISpeechToTextProvider:
        """Constructs ISpeechToTextProvider based on AUDIO_STT_PROVIDER setting."""
        cfg = settings or get_settings()
        provider_name = cfg.AUDIO_STT_PROVIDER.lower()
        stt_config = create_stt_config(cfg)

        logger.info(f"AudioEngineFactory: Constructing STT provider '{provider_name}'.")
        if provider_name in ("real", "whisper_api", "openai_whisper"):
            return RealSTTProvider(config=stt_config, settings=cfg)
        return STTAdapter(config=stt_config)

    @staticmethod
    def create_tts_provider(settings: Settings | None = None) -> ITextToSpeechProvider:
        """Constructs ITextToSpeechProvider based on AUDIO_TTS_PROVIDER setting."""
        cfg = settings or get_settings()
        provider_name = cfg.AUDIO_TTS_PROVIDER.lower()
        tts_config = create_tts_config(cfg)

        logger.info(f"AudioEngineFactory: Constructing TTS provider '{provider_name}'.")
        if provider_name in ("real", "openai_tts", "elevenlabs"):
            return RealTTSProvider(config=tts_config, settings=cfg)
        return TTSAdapter(config=tts_config)

    @staticmethod
    def create_audio_capture(settings: Settings | None = None) -> IAudioCapture:
        """Constructs IAudioCapture based on system environment."""
        cfg = settings or get_settings()
        logger.info("AudioEngineFactory: Constructing SoundDevice capture adapter.")
        if cfg.ULTRON_ENV == "testing":
            return MicrophoneCaptureAdapter(mock_mode=True)
        return SoundDeviceCaptureAdapter()

    @staticmethod
    def create_audio_playback(settings: Settings | None = None) -> IAudioPlayback:
        """Constructs IAudioPlayback based on system environment."""
        cfg = settings or get_settings()
        logger.info("AudioEngineFactory: Constructing SoundDevice playback adapter.")
        if cfg.ULTRON_ENV == "testing":
            return MockPlaybackAdapter()
        return SoundDevicePlaybackAdapter()
