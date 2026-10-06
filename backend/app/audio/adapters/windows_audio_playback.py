"""Windows Speaker Audio Playback Device Adapter (Phase 4H.5).

Concrete implementation of IAudioPlayback interfacing with physical Windows speakers
and headphones via SoundDevice/PortAudio and the Hardware Abstraction Layer (HAL).

Features:
- Seamless resolution with HAL DeviceManager (explicit index, HAL ID 'audio_out_X', or system default).
- Non-blocking asynchronous audio playback offloaded from the asyncio event loop.
- Bounded-buffer queue with DROP_OLDEST backpressure policy to avoid unbounded memory accumulation.
- Real-time barge-in interruption / speech cancellation: cleans up PortAudio buffers and stops playback instantly.
- Strict Zero-Trust and privacy invariants: audio is ephemeral and never persisted to disk or DB.
"""

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Any

from app.audio.base import IAudioPlayback
from app.audio.models import (
    VALID_PLAYBACK_STATE_TRANSITIONS,
    AudioChunk,
    AudioEncoding,
    AudioFormat,
    PlaybackState,
)
from app.audio.playback import (
    PlaybackConfig,
    PlaybackDeviceNotFoundError,
    PlaybackStateError,
    PlaybackTelemetry,
    PlaybackValidationError,
)
from app.hardware.device_manager import DeviceManager
from app.hardware.models import DeviceType

logger = logging.getLogger(__name__)

# Conditional import for sounddevice and numpy
try:
    import numpy as np  # type: ignore[import-untyped,import-not-found]
    import sounddevice as sd  # type: ignore[import-untyped,import-not-found]

    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    sd = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    SOUNDDEVICE_AVAILABLE = False


class WindowsAudioPlaybackDevice(IAudioPlayback):
    """Hardware Audio Playback Adapter for Windows output speakers and headphones."""

    def __init__(
        self,
        device_id: int | str | None = None,
        config: PlaybackConfig | None = None,
        device_manager: DeviceManager | None = None,
        simulated_mode: bool = False,
    ) -> None:
        """Initializes WindowsAudioPlaybackDevice.

        Args:
            device_id: Optional speaker index (e.g. 3), HAL ID ('audio_out_3'), or friendly name.
            config: Optional PlaybackConfig configuration instance.
            device_manager: Optional HAL DeviceManager for hardware discovery.
            simulated_mode: If True, executes in simulated mode without opening physical speakers.
        """
        self.raw_device_id = device_id
        self.config = config or PlaybackConfig()
        self.device_manager = device_manager or DeviceManager()
        self.simulated_mode = simulated_mode or not SOUNDDEVICE_AVAILABLE

        self._resolved_device_index: int | None = None
        self._resolved_device_name: str | None = None

        self._state = PlaybackState.IDLE
        self._queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=self.config.max_buffered_chunks)
        self._last_sequence_number: int | None = None
        self._playback_task: asyncio.Task[Any] | None = None

        # Telemetry metrics
        self._chunks_received = 0
        self._chunks_played = 0
        self._chunks_dropped = 0
        self._underruns = 0
        self._overruns = 0
        self._total_duration_ms = 0.0
        self._session_id: str | None = None
        self._correlation_id: str | None = None

        logger.info(
            f"WindowsAudioPlaybackDevice initialized: requested_device={self.raw_device_id}, "
            f"simulated={self.simulated_mode}."
        )

    def get_state(self) -> PlaybackState:
        """Returns active PlaybackState."""
        return self._state

    def is_speaking(self) -> bool:
        """Returns True if the device is actively outputting speech."""
        return self._state == PlaybackState.PLAYING

    def _transition(self, target_state: PlaybackState) -> PlaybackState:
        """Validates and updates playback state machine."""
        allowed = VALID_PLAYBACK_STATE_TRANSITIONS.get(self._state, set())
        if target_state not in allowed:
            raise PlaybackStateError(
                f"Illegal PlaybackState transition: Cannot transition from {self._state} to {target_state}."
            )
        self._state = target_state
        return self._state

    async def resolve_device(self) -> tuple[int | None, str]:
        """Resolves target physical speaker device using HAL DeviceManager."""
        if self._resolved_device_index is not None and self._resolved_device_name is not None:
            return self._resolved_device_index, self._resolved_device_name

        if self.simulated_mode or not SOUNDDEVICE_AVAILABLE:
            self._resolved_device_index = None
            self._resolved_device_name = "Simulated Speaker"
            return None, self._resolved_device_name

        # If integer device_id was passed
        if isinstance(self.raw_device_id, int):
            try:
                info = sd.query_devices(self.raw_device_id)
                self._resolved_device_index = self.raw_device_id
                self._resolved_device_name = str(info.get("name", f"Speaker {self.raw_device_id}"))
                return self._resolved_device_index, self._resolved_device_name
            except Exception as exc:
                raise PlaybackDeviceNotFoundError(f"Output device index {self.raw_device_id} not found: {exc}") from exc

        # If string device_id was passed
        if isinstance(self.raw_device_id, str):
            # Check for HAL format "audio_out_X"
            if self.raw_device_id.startswith("audio_out_"):
                try:
                    idx = int(self.raw_device_id.replace("audio_out_", ""))
                    info = sd.query_devices(idx)
                    self._resolved_device_index = idx
                    self._resolved_device_name = str(info.get("name", f"Speaker {idx}"))
                    return self._resolved_device_index, self._resolved_device_name
                except Exception as exc:
                    raise PlaybackDeviceNotFoundError(f"HAL audio device {self.raw_device_id} not found: {exc}") from exc

            # Substring match across available devices
            devs = sd.query_devices()
            for idx, d in enumerate(devs):
                if d.get("max_output_channels", 0) > 0 and self.raw_device_id.lower() in str(d.get("name", "")).lower():
                    self._resolved_device_index = idx
                    self._resolved_device_name = str(d.get("name"))
                    return self._resolved_device_index, self._resolved_device_name

            raise PlaybackDeviceNotFoundError(f"Speaker device matching '{self.raw_device_id}' not found.")

        # Fallback to system default output
        default_dev = await self.device_manager.get_default_device(DeviceType.AUDIO_OUTPUT)
        if default_dev:
            try:
                idx = int(default_dev.device_id.replace("audio_out_", ""))
                self._resolved_device_index = idx
                self._resolved_device_name = default_dev.name
                return self._resolved_device_index, self._resolved_device_name
            except Exception:
                pass

        # Query sounddevice default output
        try:
            default_out_idx = sd.default.device[1]
            if default_out_idx is not None and default_out_idx >= 0:
                info = sd.query_devices(default_out_idx)
                self._resolved_device_index = int(default_out_idx)
                self._resolved_device_name = str(info.get("name", "Default Speaker"))
                return self._resolved_device_index, self._resolved_device_name
        except Exception:
            pass

        self._resolved_device_index = None
        self._resolved_device_name = "Default System Audio Output"
        return None, self._resolved_device_name

    def _validate_chunk(self, chunk: AudioChunk) -> None:
        """Validates AudioChunk payload and sequence continuity."""
        if not chunk.payload:
            raise PlaybackValidationError("Cannot play empty AudioChunk payload.")

        if self._last_sequence_number is not None:
            if chunk.sequence_number == self._last_sequence_number:
                raise PlaybackValidationError(f"Duplicate sequence number seq={chunk.sequence_number}.")
            if chunk.sequence_number < self._last_sequence_number:
                raise PlaybackValidationError(
                    f"Audio sequence regression: seq={chunk.sequence_number} < last_seq={self._last_sequence_number}."
                )

        self._last_sequence_number = chunk.sequence_number

    async def play_chunk(self, chunk: AudioChunk) -> None:
        """Enqueues and plays an AudioChunk frame with DROP_OLDEST backpressure."""
        self._validate_chunk(chunk)
        self._chunks_received += 1
        self._session_id = chunk.session_id
        self._correlation_id = chunk.correlation_id

        # Enforce bounded queue with DROP_OLDEST
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._chunks_dropped += 1
                self._overruns += 1
            except asyncio.QueueEmpty:
                pass

        self._queue.put_nowait(chunk)

        if self._state in (PlaybackState.IDLE, PlaybackState.STOPPED):
            self._transition(PlaybackState.PLAYING)

        if self._playback_task is None or self._playback_task.done():
            self._playback_task = asyncio.create_task(self._playback_loop())

    async def play_bytes(self, pcm_bytes: bytes, sample_rate: int = 16000) -> None:
        """Helper to play contiguous PCM bytes by wrapping into an AudioChunk."""
        if not pcm_bytes:
            raise PlaybackValidationError("Cannot play empty PCM bytes.")

        chunk = AudioChunk(
            chunk_id=f"chk_play_{self._chunks_received}",
            sequence_number=self._chunks_received,
            timestamp=datetime.now(UTC),
            duration_ms=(len(pcm_bytes) // 2 / sample_rate) * 1000.0,
            audio_format=AudioFormat(
                sample_rate=sample_rate,
                channels=1,
                sample_width=2,
                encoding=AudioEncoding.PCM_S16LE,
            ),
            payload=pcm_bytes,
        )
        await self.play_chunk(chunk)

    async def pause(self) -> None:
        """Pauses active playback stream."""
        if self._state == PlaybackState.PLAYING:
            self._transition(PlaybackState.PAUSED)
            logger.info("WindowsAudioPlaybackDevice paused.")

    async def resume(self) -> None:
        """Resumes paused playback stream."""
        if self._state == PlaybackState.PAUSED:
            self._transition(PlaybackState.PLAYING)
            logger.info("WindowsAudioPlaybackDevice resumed.")

    async def stop(self) -> None:
        """Immediately halts playback, interrupts physical audio, and flushes queues."""
        if self._state in (PlaybackState.STOPPED, PlaybackState.STOPPING, PlaybackState.IDLE):
            return

        try:
            self._transition(PlaybackState.STOPPING)
        except PlaybackStateError:
            self._state = PlaybackState.STOPPING

        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
            try:
                await self._playback_task
            except asyncio.CancelledError:
                pass
            self._playback_task = None

        if not self.simulated_mode and SOUNDDEVICE_AVAILABLE:
            try:
                sd.stop()
            except Exception as exc:
                logger.warning(f"Error stopping PortAudio playback: {exc}")

        # Flush queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._last_sequence_number = None
        self._transition(PlaybackState.STOPPED)
        self._transition(PlaybackState.IDLE)
        logger.info("WindowsAudioPlaybackDevice stopped and reset cleanly.")

    async def _playback_loop(self) -> None:
        """Background worker loop driving physical audio playback."""
        dev_idx, _ = await self.resolve_device()

        try:
            while not self._queue.empty():
                if self._state == PlaybackState.PAUSED:
                    await asyncio.sleep(0.05)
                    continue

                if self._state != PlaybackState.PLAYING:
                    break

                chunk = await self._queue.get()
                start_time = time.perf_counter()

                if not self.simulated_mode and SOUNDDEVICE_AVAILABLE and np is not None:
                    try:
                        int16_arr = np.frombuffer(chunk.payload, dtype=np.int16)
                        sd.play(int16_arr, samplerate=chunk.audio_format.sample_rate, device=dev_idx, blocking=False)
                    except Exception as exc:
                        self._underruns += 1
                        logger.error(f"PortAudio speaker output error: {exc}")

                # Sleep chunk duration or simulated duration
                sleep_sec = max(0.01, min(chunk.duration_ms / 1000.0, 1.0))
                await asyncio.sleep(sleep_sec)

                self._chunks_played += 1
                self._total_duration_ms += chunk.duration_ms
                _elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if self._state == PlaybackState.PLAYING and self._queue.empty():
                self._transition(PlaybackState.IDLE)

        except asyncio.CancelledError:
            logger.debug("Playback worker loop cancelled.")
            raise
        except Exception as exc:
            logger.error(f"Unexpected error in playback worker loop: {exc}")
            self._state = PlaybackState.ERROR

    async def recover(self) -> None:
        """Recovers device from ERROR state and re-initializes hardware handles."""
        logger.info("WindowsAudioPlaybackDevice: Recovering from error...")
        await self.stop()
        self._state = PlaybackState.IDLE
        self._resolved_device_index = None
        self._resolved_device_name = None
        await self.resolve_device()

    async def health(self) -> dict[str, Any]:
        """Probes health and readiness status of physical speaker hardware."""
        dev_idx, dev_name = await self.resolve_device()
        return {
            "subsystem_playback": True,
            "device_index": dev_idx,
            "device_name": dev_name,
            "healthy": self._state != PlaybackState.ERROR,
            "state": self._state.value,
            "is_speaking": self.is_speaking(),
            "simulated_mode": self.simulated_mode,
            "queue_depth": self._queue.qsize(),
            "chunks_played": self._chunks_played,
            "chunks_dropped": self._chunks_dropped,
            "underruns": self._underruns,
        }

    def get_telemetry(self) -> PlaybackTelemetry:
        """Returns privacy-safe PlaybackTelemetry metadata."""
        return PlaybackTelemetry(
            state=self._state,
            chunks_received=self._chunks_received,
            chunks_played=self._chunks_played,
            chunks_dropped=self._chunks_dropped,
            total_duration_ms=self._total_duration_ms,
            latency_ms=1.0,
            timestamp=datetime.now(UTC),
            session_id=self._session_id,
            correlation_id=self._correlation_id,
        )
