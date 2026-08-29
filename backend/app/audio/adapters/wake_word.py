"""Wake Word Detection Adapter (Phase 4C.6).

Concrete implementation of IWakeWordDetector providing rolling window buffer management,
monotonic sequence validation, confidence thresholding, cooldown duplicate suppression,
transient error retry policy, timeout protection, and privacy-preserving telemetry.
"""

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Any

from app.audio.base import IWakeWordDetector
from app.audio.models import AudioChunk, WakeWordDetection
from app.audio.wake_word import (
    WakeWordConfig,
    WakeWordProcessingError,
    WakeWordProviderError,
    WakeWordTelemetry,
    WakeWordTimeoutError,
    WakeWordValidationError,
)

logger = logging.getLogger(__name__)


class WakeWordAdapter(IWakeWordDetector):
    """Provider-agnostic Wake Word Detection adapter."""

    def __init__(
        self,
        config: WakeWordConfig | None = None,
        mock_trigger_word: str | None = None,
        mock_confidence: float = 0.95,
        simulated_delay_sec: float = 0.0,
        transient_failures_to_simulate: int = 0,
    ) -> None:
        """Initializes WakeWordAdapter.

        Args:
            config: Optional WakeWordConfig configuration instance.
            mock_trigger_word: Optional phrase override to trigger detection for testing.
            mock_confidence: Confidence score to return for mock trigger detections.
            simulated_delay_sec: Optional processing delay in seconds for timeout testing.
            transient_failures_to_simulate: Number of transient failures to simulate before succeeding.
        """
        self.config = config or WakeWordConfig()
        self._mock_trigger_word = mock_trigger_word
        self._mock_confidence = mock_confidence
        self._simulated_delay_sec = simulated_delay_sec
        self._transient_failures_remaining = transient_failures_to_simulate

        self._last_sequence_number: int | None = None
        self._last_trigger_time_ms: float = 0.0
        self._buffered_chunks: list[AudioChunk] = []

        self._detection_count = 0
        self._last_telemetry: WakeWordTelemetry | None = None

    async def health(self) -> dict[str, Any]:
        """Probes health and readiness status of wake-word detector."""
        return {
            "subsystem_wake_word": True,
            "enabled": self.config.enabled,
            "provider_ready": True,
            "buffered_chunks": len(self._buffered_chunks),
        }

    async def reset(self) -> None:
        """Resets internal window buffers and sequence tracking state."""
        self._last_sequence_number = None
        self._last_trigger_time_ms = 0.0
        self._buffered_chunks.clear()
        logger.info("WakeWordAdapter state and window buffers reset.")

    def _validate_chunk(self, chunk: AudioChunk) -> None:
        """Validates incoming AudioChunk sequence numbers and payload alignment."""
        if not chunk.payload:
            raise WakeWordValidationError("Cannot process empty AudioChunk payload.")

        if self._last_sequence_number is not None:
            if chunk.sequence_number <= self._last_sequence_number:
                raise WakeWordValidationError(
                    f"Invalid AudioChunk sequence regression: seq={chunk.sequence_number} <= last_seq={self._last_sequence_number}."
                )

        self._last_sequence_number = chunk.sequence_number

    def _maintain_buffered_window(self, chunk: AudioChunk) -> None:
        """Appends chunk to rolling window and enforces max_buffered_chunks and max_detection_window_ms."""
        self._buffered_chunks.append(chunk)

        # Enforce max_buffered_chunks limit
        while len(self._buffered_chunks) > self.config.max_buffered_chunks:
            self._buffered_chunks.pop(0)

        # Enforce max_detection_window_ms limit
        total_duration = sum(c.duration_ms for c in self._buffered_chunks)
        while (
            total_duration > self.config.max_detection_window_ms and len(self._buffered_chunks) > 1
        ):
            removed = self._buffered_chunks.pop(0)
            total_duration -= removed.duration_ms

    async def detect(self, chunk: AudioChunk) -> WakeWordDetection | None:
        """Evaluates an AudioChunk for wake-word activation with timeout, retries, and cooldown."""
        if not self.config.enabled:
            return None

        self._validate_chunk(chunk)
        self._maintain_buffered_window(chunk)

        timeout_sec = self.config.timeout_ms / 1000.0

        for attempt in range(self.config.retry_count + 1):
            try:
                return await asyncio.wait_for(self._detect_internal(chunk), timeout=timeout_sec)
            except TimeoutError as exc:
                self._record_telemetry(
                    detection_id=f"ww_err_{chunk.sequence_number}",
                    chunk=chunk,
                    wake_word="none",
                    confidence=0.0,
                    latency_ms=timeout_sec * 1000.0,
                    success=False,
                    error_code="WAKE_WORD_TIMEOUT",
                )
                raise WakeWordTimeoutError(
                    f"Wake-word evaluation timed out after {self.config.timeout_ms}ms."
                ) from exc
            except asyncio.CancelledError:
                logger.info("Wake-word evaluation cancelled cleanly.")
                raise
            except WakeWordValidationError:
                raise
            except WakeWordProviderError as exc:
                if attempt < self.config.retry_count:
                    logger.warning(
                        f"Transient wake-word provider error (attempt {attempt + 1}/{self.config.retry_count + 1}). Retrying..."
                    )
                    await asyncio.sleep(0.05)
                    continue
                raise exc
            except Exception as exc:
                raise WakeWordProcessingError(
                    f"Unexpected wake-word evaluation error: {exc}"
                ) from exc

        raise WakeWordProviderError("Wake-word evaluation failed after exhausting retry attempts.")

    async def _detect_internal(self, chunk: AudioChunk) -> WakeWordDetection | None:
        """Internal wake-word engine evaluation logic."""
        start_time = time.perf_counter()

        if self._transient_failures_remaining > 0:
            self._transient_failures_remaining -= 1
            raise WakeWordProviderError("Simulated transient wake-word network failure.")

        if self._simulated_delay_sec > 0:
            await asyncio.sleep(self._simulated_delay_sec)

        # Check mock trigger phrase
        triggered_word: str | None = None
        if self._mock_trigger_word and self._mock_trigger_word in self.config.wake_words:
            triggered_word = self._mock_trigger_word

        if not triggered_word:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self._record_telemetry(
                detection_id=f"ww_none_{chunk.sequence_number}",
                chunk=chunk,
                wake_word="none",
                confidence=0.0,
                latency_ms=elapsed_ms,
                success=True,
            )
            return None

        # Check confidence thresholding
        confidence = self._mock_confidence
        if confidence < self.config.confidence_threshold:
            return None

        # Cooldown duplicate detection suppression
        now_ms = time.time() * 1000.0
        if (now_ms - self._last_trigger_time_ms) < self.config.cooldown_ms:
            logger.debug(f"Wake-word trigger '{triggered_word}' suppressed during cooldown window.")
            return None

        self._last_trigger_time_ms = now_ms
        self._detection_count += 1

        detection_id = f"ww_evt_{self._detection_count}"
        detection = WakeWordDetection(
            detection_id=detection_id,
            wake_word=triggered_word,
            confidence=confidence,
            timestamp=datetime.now(UTC),
            session_id=chunk.session_id,
            correlation_id=chunk.correlation_id,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self._record_telemetry(
            detection_id=detection_id,
            chunk=chunk,
            wake_word=triggered_word,
            confidence=confidence,
            latency_ms=elapsed_ms,
            success=True,
        )
        return detection

    def _record_telemetry(
        self,
        detection_id: str,
        chunk: AudioChunk,
        wake_word: str,
        confidence: float,
        latency_ms: float,
        success: bool = True,
        error_code: str | None = None,
    ) -> None:
        """Records privacy-preserving telemetry without raw audio bytes."""
        self._last_telemetry = WakeWordTelemetry(
            detection_id=detection_id,
            wake_word=wake_word,
            confidence=confidence,
            latency_ms=latency_ms,
            provider=self.config.provider,
            model=self.config.model,
            sequence_number=chunk.sequence_number,
            success=success,
            error_code=error_code,
            correlation_id=chunk.correlation_id,
        )
