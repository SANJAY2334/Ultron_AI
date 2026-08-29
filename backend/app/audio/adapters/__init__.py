"""Audio Hardware and Engine Adapters for ULTRON.

Provides concrete adapters for microphone capture, SoundDevice capture/playback, VAD, STT, TTS,
Wake Word Detection, Voice Planner Gateway, Mock Playback, Real STT Provider, and Real TTS Provider.
"""

from app.audio.adapters.microphone import MicrophoneCaptureAdapter
from app.audio.adapters.mock_playback import MockPlaybackAdapter
from app.audio.adapters.planner_gateway import VoicePlannerGateway
from app.audio.adapters.real_stt import RealSTTProvider
from app.audio.adapters.real_tts import RealTTSProvider
from app.audio.adapters.sounddevice_capture import SoundDeviceCaptureAdapter
from app.audio.adapters.sounddevice_playback import SoundDevicePlaybackAdapter
from app.audio.adapters.stt import STTAdapter
from app.audio.adapters.tts import TTSAdapter
from app.audio.adapters.vad import VADAdapter
from app.audio.adapters.wake_word import WakeWordAdapter

__all__ = [
    "MicrophoneCaptureAdapter",
    "SoundDeviceCaptureAdapter",
    "SoundDevicePlaybackAdapter",
    "VADAdapter",
    "STTAdapter",
    "TTSAdapter",
    "WakeWordAdapter",
    "VoicePlannerGateway",
    "MockPlaybackAdapter",
    "RealSTTProvider",
    "RealTTSProvider",
]
