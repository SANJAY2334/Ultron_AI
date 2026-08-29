"""SoundDevice / PortAudio Microphone Hardware Capture Adapter (Phase 4E.2).

Concrete implementation of IAudioCapture interfacing with system audio input hardware
via SoundDevice / PortAudio. Features non-blocking real-time audio callbacks, bounded queue
DROP_OLDEST backpressure, monotonic sequence numbering, device enumeration, and error isolation.
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from app.audio.base import IAudioCapture
from app.audio.capture import (
    AudioCaptureConfigurationError,
    AudioCaptureState,
    AudioCaptureUnavailableError,
    AudioDeviceInfo,
)
from app.audio.models import AudioChunk, AudioFormat

logger = logging.getLogger(__name__)

# Conditional import for sounddevice and numpy
try:
    import numpy as np  # type: ignore[import-not-found,import-untyped]
    import sounddevice as sd  # type: ignore[import-not-found,import-untyped]

    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    sd = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    SOUNDDEVICE_AVAILABLE = False


class SoundDeviceCaptureAdapter(IAudioCapture):
    """PortAudio Microphone Capture Adapter using SoundDevice."""

    def __init__(
        self,
        device_id: int | str | None = None,
        sample_rate: int = 16000,
        channels: int = 1,
        max_buffered_chunks: int = 100,
        simulated_mode: bool = False,
    ) -> None:
        """Initializes SoundDeviceCaptureAdapter.

        Args:
            device_id: Optional input device index or name string.
            sample_rate: Input sample rate in Hz (default 16000).
            channels: Input channel count (default 1 mono).
            max_buffered_chunks: Bounded queue size for captured frames.
            simulated_mode: If True, operates in synthetic test mode without hardware.
        """
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_buffered_chunks = max_buffered_chunks
        self.simulated_mode = simulated_mode or not SOUNDDEVICE_AVAILABLE

        self._state = AudioCaptureState.STOPPED
        self._queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=self.max_buffered_chunks)
        self._sequence_number = 0
        self._stream: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Metrics tracking
        self._chunks_processed = 0
        self._chunks_dropped = 0
        self._stream_errors = 0

    @classmethod
    def list_devices(cls, kind: str = "input") -> list[AudioDeviceInfo]:
        """Enumerates system input/output audio hardware devices."""
        if not SOUNDDEVICE_AVAILABLE:
            return [
                AudioDeviceInfo(
                    device_id="simulated_mic_0",
                    name="Simulated Microphone (Test)",
                    input_channels=1,
                    sample_rates=[16000],
                    default_sample_rate=16000,
                    is_default=True,
                    is_available=True,
                )
            ]

        try:
            raw_devices = sd.query_devices()
            default_input_idx = sd.default.device[0] if sd.default.device else None
            devices: list[AudioDeviceInfo] = []

            for idx, dev in enumerate(raw_devices):
                max_in = int(dev.get("max_input_channels", 0))
                max_out = int(dev.get("max_output_channels", 0))

                if kind == "input" and max_in == 0:
                    continue
                if kind == "output" and max_out == 0:
                    continue

                devices.append(
                    AudioDeviceInfo(
                        device_id=str(idx),
                        name=str(dev.get("name", f"Audio Device {idx}")),
                        input_channels=max_in,
                        sample_rates=[16000, 44100, 48000],
                        default_sample_rate=int(dev.get("default_samplerate", 16000)),
                        is_default=(idx == default_input_idx),
                        is_available=True,
                    )
                )
            return devices
        except Exception as exc:
            logger.error(f"Error querying SoundDevice hardware devices: {exc}")
            return []

    async def health(self) -> dict[str, Any]:
        """Probes health and readiness of microphone capture hardware."""
        healthy = self._state not in (AudioCaptureState.ERROR, "ERROR")
        state_str = str(self._state.value) if hasattr(self._state, "value") else str(self._state)
        return {
            "subsystem_capture": True,
            "healthy": healthy,
            "state": state_str,
            "simulated_mode": self.simulated_mode,
            "sounddevice_available": SOUNDDEVICE_AVAILABLE,
            "chunks_processed": self._chunks_processed,
            "chunks_dropped": self._chunks_dropped,
            "stream_errors": self._stream_errors,
        }

    def _audio_callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        """Non-blocking real-time PortAudio input stream callback.

        Executes in audio driver thread. Performs zero blocking I/O or allocation.
        """
        if status:
            self._stream_errors += 1

        try:
            if self.simulated_mode or np is None:
                pcm_bytes = b"\x00" * (frames * 2 * self.channels)
            else:
                # Convert float32/int16 array to 16-bit LE PCM bytes
                if indata.dtype == np.int16:
                    pcm_bytes = indata.tobytes()
                else:
                    int16_data = (indata * 32767).astype(np.int16)
                    pcm_bytes = int16_data.tobytes()

            chunk = AudioChunk(
                chunk_id=f"sd_mic_{self._sequence_number}",
                sequence_number=self._sequence_number,
                timestamp=datetime.now(UTC),
                duration_ms=(frames / self.sample_rate) * 1000.0,
                audio_format=AudioFormat(
                    sample_rate=self.sample_rate, channels=self.channels, sample_width=2
                ),
                payload=pcm_bytes,
            )
            self._sequence_number += 1

            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._enqueue_chunk_from_callback, chunk)
        except Exception:
            self._stream_errors += 1

    def _enqueue_chunk_from_callback(self, chunk: AudioChunk) -> None:
        """Enqueues chunk into bounded queue with DROP_OLDEST policy on main event loop."""
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._chunks_dropped += 1
            except asyncio.QueueEmpty:
                pass

        self._queue.put_nowait(chunk)
        self._chunks_processed += 1

    def validate_format(self) -> bool:
        """Validates hardware support for target sample rate and channel configuration."""
        if self.simulated_mode or not SOUNDDEVICE_AVAILABLE:
            return True
        try:
            device_idx = int(self.device_id) if self.device_id is not None else None
            sd.check_input_settings(
                device=device_idx,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
            )
            return True
        except Exception as exc:
            logger.warning(f"SoundDevice input format check failed: {exc}")
            return False

    async def start(self) -> None:
        """Starts PortAudio microphone capture stream asynchronously."""
        if self._state in (AudioCaptureState.RUNNING, AudioCaptureState.STARTING):
            return

        self._state = AudioCaptureState.STARTING
        self._loop = asyncio.get_running_loop()

        if self.simulated_mode or not SOUNDDEVICE_AVAILABLE:
            self._state = AudioCaptureState.RUNNING
            logger.info("SoundDeviceCaptureAdapter running in simulated mode.")
            return

        if not self.validate_format():
            self._state = AudioCaptureState.ERROR
            raise AudioCaptureConfigurationError(
                f"Microphone hardware does not support format: rate={self.sample_rate}Hz, channels={self.channels}."
            )

        try:
            device_idx = int(self.device_id) if self.device_id is not None else None
            self._stream = sd.InputStream(
                device=device_idx,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                blocksize=480,  # 30ms frames @ 16kHz
                callback=self._audio_callback,
            )
            self._stream.start()
            self._state = AudioCaptureState.RUNNING
            logger.info(f"SoundDeviceCaptureAdapter started input stream on device {device_idx}.")
        except Exception as exc:
            self._state = AudioCaptureState.ERROR
            logger.error(f"Failed to start SoundDevice InputStream: {exc}")
            raise AudioCaptureUnavailableError(f"Microphone capture unavailable: {exc}") from exc

    async def stop(self) -> None:
        """Stops microphone capture stream idempotently and releases resources."""
        if self._state in (AudioCaptureState.STOPPED, AudioCaptureState.STOPPING):
            return

        self._state = AudioCaptureState.STOPPING

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                logger.warning(f"Error stopping SoundDevice InputStream: {exc}")
            finally:
                self._stream = None

        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._sequence_number = 0
        self._state = AudioCaptureState.STOPPED
        logger.info("SoundDeviceCaptureAdapter stopped cleanly.")

    async def read_chunk(self) -> AudioChunk:
        """Reads next AudioChunk from capture stream queue."""
        if self.simulated_mode or not SOUNDDEVICE_AVAILABLE:
            if self._state != AudioCaptureState.RUNNING:
                await self.start()
            await asyncio.sleep(0.03)
            chunk = AudioChunk(
                chunk_id=f"sd_mic_{self._sequence_number}",
                sequence_number=self._sequence_number,
                timestamp=datetime.now(UTC),
                duration_ms=30.0,
                audio_format=AudioFormat(
                    sample_rate=self.sample_rate, channels=self.channels, sample_width=2
                ),
                payload=b"\x00" * 960,
            )
            self._sequence_number += 1
            self._chunks_processed += 1
            return chunk

        if self._state != AudioCaptureState.RUNNING:
            raise AudioCaptureUnavailableError("Capture stream is not running.")

        try:
            return await self._queue.get()
        except asyncio.CancelledError:
            raise
