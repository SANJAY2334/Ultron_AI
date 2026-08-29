"""Energy-Based Voice Activity Detection (VAD) Adapter (Phase 4C.3).

Concrete implementation of IVoiceActivityDetector providing normalized RMS energy detection,
configurable thresholds, hysteresis debouncing, minimum speech/silence duration accumulation,
timeout protection, and privacy-preserving telemetry without raw audio payload leakage.
"""

import asyncio
import logging
import math
import struct
import time
from datetime import UTC, datetime
from typing import Any

from app.audio.base import IVoiceActivityDetector
from app.audio.models import AudioChunk, VoiceActivityEvent, VoiceActivityState
from app.audio.vad import (
    VADBackendError,
    VADConfig,
    VADError,
    VADProcessingError,
    VADTelemetry,
    VADTimeoutError,
)

logger = logging.getLogger(__name__)


class VADAdapter(IVoiceActivityDetector):
    """Adapter encapsulating Voice Activity Detection processing."""

    def __init__(self, config: VADConfig | None = None, processing_delay_sec: float = 0.0) -> None:
        """Initializes VADAdapter.

        Args:
            config: Optional VADConfig instance.
            processing_delay_sec: Optional artificial delay in seconds for timeout testing.
        """
        self.config = config or VADConfig()
        self._processing_delay_sec = processing_delay_sec
        self._current_state = VoiceActivityState.SILENCE

        self._speech_accumulator_ms = 0.0
        self._silence_accumulator_ms = 0.0

        self._processed_chunks = 0
        self._total_speech_events = 0
        self._last_telemetry: VADTelemetry | None = None

    @property
    def current_state(self) -> VoiceActivityState:
        """Returns current active VAD state."""
        return self._current_state

    def _compute_normalized_energy(self, chunk: AudioChunk) -> float:
        """Calculates normalized RMS energy and confidence deterministically from PCM payload bytes.

        Args:
            chunk: Input AudioChunk payload.

        Returns:
            float: Speech confidence score in range [0.0, 1.0].
        """
        payload = chunk.payload
        if not payload:
            raise VADProcessingError("Cannot compute energy of empty audio chunk payload.")

        sample_width = chunk.audio_format.sample_width

        # Strictly enforce sample alignment
        if len(payload) % sample_width != 0:
            raise VADProcessingError(
                f"Audio payload size ({len(payload)} bytes) is misaligned with sample_width ({sample_width} bytes)."
            )

        num_samples = len(payload) // sample_width
        if num_samples == 0:
            return 0.0

        try:
            total_sq = 0.0
            if sample_width == 2:
                # 16-bit signed PCM LE
                fmt = f"<{num_samples}h"
                samples = struct.unpack(fmt, payload)
                for s in samples:
                    total_sq += float(s) * float(s)
            elif sample_width == 4:
                # 32-bit float or signed PCM LE
                fmt = f"<{num_samples}f"
                samples = struct.unpack(fmt, payload)
                for s in samples:
                    total_sq += float(s) * float(s)
            else:
                # Fallback byte energy calculation
                for b in payload:
                    s_val = float(b - 128)
                    total_sq += s_val * s_val

            rms = math.sqrt(total_sq / float(num_samples))

            # Normalize RMS against standard 16-bit reference amplitude (12000.0)
            normalized = min(1.0, rms / 12000.0)
            return round(normalized, 4)
        except VADProcessingError:
            raise
        except Exception as exc:
            raise VADProcessingError(f"Failed to calculate audio payload energy: {exc}") from exc

    async def process(self, chunk: AudioChunk) -> VoiceActivityEvent:
        """Processes an AudioChunk frame and returns a VoiceActivityEvent with timeout and debouncing.

        Args:
            chunk: Input AudioChunk object.

        Returns:
            VoiceActivityEvent: State transition event object.

        Raises:
            VADTimeoutError: If processing exceeds processing_timeout_ms.
            VADProcessingError: If payload processing fails.
        """
        timeout_sec = self.config.processing_timeout_ms / 1000.0

        try:
            return await asyncio.wait_for(self._process_internal(chunk), timeout=timeout_sec)
        except TimeoutError as exc:
            self._record_telemetry(
                chunk, self._current_state, 0.0, timeout_sec * 1000.0, error_code="VAD_TIMEOUT"
            )
            raise VADTimeoutError(
                f"VAD chunk processing timed out after {self.config.processing_timeout_ms}ms."
            ) from exc
        except asyncio.CancelledError:
            logger.info("VAD processing cancelled cleanly.")
            raise
        except VADError:
            raise
        except Exception as exc:
            raise VADBackendError(f"Unexpected VAD backend failure: {exc}") from exc

    async def _process_internal(self, chunk: AudioChunk) -> VoiceActivityEvent:
        """Internal deterministic VAD state machine evaluation logic."""
        start_time = time.perf_counter()

        if self._processing_delay_sec > 0:
            await asyncio.sleep(self._processing_delay_sec)

        confidence = self._compute_normalized_energy(chunk)
        duration = chunk.duration_ms
        is_speech = confidence >= self.config.speech_threshold

        next_state = self._current_state

        if is_speech:
            self._silence_accumulator_ms = 0.0
            self._speech_accumulator_ms += duration

            if self._current_state == VoiceActivityState.SILENCE:
                if self._speech_accumulator_ms >= self.config.minimum_speech_duration_ms:
                    next_state = VoiceActivityState.SPEECH_START
                    self._total_speech_events += 1
            elif self._current_state in (
                VoiceActivityState.SPEECH_START,
                VoiceActivityState.SPEAKING,
                VoiceActivityState.UNKNOWN,
            ):
                next_state = VoiceActivityState.SPEAKING
            elif self._current_state == VoiceActivityState.SPEECH_END:
                next_state = VoiceActivityState.SPEECH_START

        else:
            self._speech_accumulator_ms = 0.0
            self._silence_accumulator_ms += duration

            if self._current_state in (
                VoiceActivityState.SPEECH_START,
                VoiceActivityState.SPEAKING,
            ):
                if self._silence_accumulator_ms >= self.config.minimum_silence_duration_ms:
                    next_state = VoiceActivityState.SPEECH_END
                else:
                    # Hysteresis: Maintain SPEAKING state during brief inter-word silence pauses
                    next_state = VoiceActivityState.SPEAKING
            elif self._current_state == VoiceActivityState.SPEECH_END:
                next_state = VoiceActivityState.SILENCE
            elif self._current_state in (VoiceActivityState.SILENCE, VoiceActivityState.UNKNOWN):
                next_state = VoiceActivityState.SILENCE

        self._current_state = next_state
        self._processed_chunks += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        self._record_telemetry(chunk, next_state, confidence, elapsed_ms)

        return VoiceActivityEvent(
            event_id=f"vad_evt_{self._processed_chunks}",
            state=next_state,
            timestamp=datetime.now(UTC),
            confidence=confidence,
            chunk_id=chunk.chunk_id,
            session_id=chunk.session_id,
            correlation_id=chunk.correlation_id,
        )

    def _record_telemetry(
        self,
        chunk: AudioChunk,
        state: VoiceActivityState,
        confidence: float,
        latency_ms: float,
        error_code: str | None = None,
    ) -> None:
        """Records privacy-preserving telemetry without raw audio bytes."""
        self._last_telemetry = VADTelemetry(
            timestamp=datetime.now(UTC),
            chunk_id=chunk.chunk_id,
            sequence_number=chunk.sequence_number,
            duration_ms=chunk.duration_ms,
            detected_state=state,
            confidence=confidence,
            processing_latency_ms=latency_ms,
            error_code=error_code,
        )

    async def reset(self) -> None:
        """Resets internal state machine accumulators and VAD state back to SILENCE."""
        self._current_state = VoiceActivityState.SILENCE
        self._speech_accumulator_ms = 0.0
        self._silence_accumulator_ms = 0.0
        logger.info("VADAdapter state reset to SILENCE.")

    async def health(self) -> dict[str, Any]:
        """Probes health and readiness status of VAD detector engine."""
        return {
            "subsystem": "vad_detector",
            "status": "RUNNING",
            "current_state": self._current_state.value,
            "processed_chunks": self._processed_chunks,
            "total_speech_events": self._total_speech_events,
            "speech_threshold": self.config.speech_threshold,
            "last_telemetry": self._last_telemetry.model_dump() if self._last_telemetry else None,
        }
