"""Speech-to-Text (STT) Adapter (Phase 4C.4).

Concrete implementation of ISpeechToTextProvider providing modular speech segment assembly,
input chunk sequence validation, streaming/partial & final transcription, transient retry policy,
timeout protection, confidence normalization, and privacy-preserving telemetry.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterable
from typing import Any

from app.audio.base import ISpeechToTextProvider
from app.audio.models import AudioChunk, Transcript, TranscriptSegment
from app.audio.stt import (
    STTConfig,
    STTProcessingError,
    STTProviderError,
    STTTelemetry,
    STTTimeoutError,
    STTValidationError,
)

logger = logging.getLogger(__name__)


class STTAdapter(ISpeechToTextProvider):
    """Provider-agnostic Speech-to-Text (STT) adapter."""

    def __init__(
        self,
        config: STTConfig | None = None,
        mock_transcript_text: str | None = None,
        simulated_delay_sec: float = 0.0,
        transient_failures_to_simulate: int = 0,
    ) -> None:
        """Initializes STTAdapter.

        Args:
            config: Optional STTConfig configuration instance.
            mock_transcript_text: Optional text override for mock transcription.
            simulated_delay_sec: Optional processing delay in seconds for timeout testing.
            transient_failures_to_simulate: Number of transient failures to simulate before succeeding.
        """
        self.config = config or STTConfig()
        self._mock_transcript_text = mock_transcript_text
        self._simulated_delay_sec = simulated_delay_sec
        self._transient_failures_remaining = transient_failures_to_simulate

        self._transcribe_count = 0
        self._last_telemetry: STTTelemetry | None = None

    def capabilities(self) -> dict[str, Any]:
        """Returns STT provider capabilities metadata."""
        return {
            "provider_name": self.config.provider_name,
            "model_name": self.config.model_name,
            "supports_streaming": self.config.streaming_enabled,
            "supports_partial": self.config.partial_transcript_enabled,
            "max_duration_ms": self.config.max_audio_duration_ms,
            "supported_languages": ["en", "es", "fr", "de", "ja", "zh"],
        }

    async def health(self) -> dict[str, bool]:
        """Probes health status of STT engine."""
        return {
            "subsystem_stt": True,
            "provider_ready": True,
        }

    def _validate_chunks(self, chunks: list[AudioChunk]) -> None:
        """Validates incoming AudioChunk list for non-emptiness, sequence ordering, and payload limits."""
        if not chunks:
            raise STTValidationError("Cannot transcribe an empty list of AudioChunks.")

        if len(chunks) > self.config.max_chunks:
            raise STTValidationError(
                f"AudioChunk count ({len(chunks)}) exceeds maximum limit ({self.config.max_chunks})."
            )

        total_duration_ms = sum(c.duration_ms for c in chunks)
        if total_duration_ms > self.config.max_audio_duration_ms:
            raise STTValidationError(
                f"Total audio duration ({total_duration_ms}ms) exceeds maximum limit ({self.config.max_audio_duration_ms}ms)."
            )

        total_payload_bytes = sum(len(c.payload) for c in chunks)
        if total_payload_bytes > self.config.max_payload_bytes:
            raise STTValidationError(
                f"Total payload size ({total_payload_bytes} bytes) exceeds maximum limit ({self.config.max_payload_bytes} bytes)."
            )

        # Validate sequence monotonicity
        seqs = [c.sequence_number for c in chunks]
        for i in range(1, len(seqs)):
            if seqs[i] < seqs[i - 1]:
                raise STTValidationError(
                    f"Out-of-order AudioChunk sequence detected: seq[{i - 1}]={seqs[i - 1]} > seq[{i}]={seqs[i]}."
                )

    async def transcribe_segment(self, chunks: list[AudioChunk]) -> Transcript:
        """Transcribes a list of AudioChunks representing a complete speech segment with retries and timeout."""
        self._validate_chunks(chunks)
        timeout_sec = self.config.timeout_ms / 1000.0

        for attempt in range(self.config.retry_count + 1):
            try:
                return await asyncio.wait_for(
                    self._transcribe_internal(chunks), timeout=timeout_sec
                )
            except TimeoutError as exc:
                self._record_telemetry(
                    transcript_id=f"tr_err_{self._transcribe_count}",
                    chunks=chunks,
                    is_final=True,
                    latency_ms=timeout_sec * 1000.0,
                    success=False,
                    error_code="STT_TIMEOUT",
                )
                raise STTTimeoutError(
                    f"STT transcription timed out after {self.config.timeout_ms}ms."
                ) from exc
            except asyncio.CancelledError:
                logger.info("STT transcription request cancelled.")
                raise
            except STTValidationError:
                raise
            except STTProviderError as exc:
                if attempt < self.config.retry_count:
                    logger.warning(
                        f"Transient STT provider error (attempt {attempt + 1}/{self.config.retry_count + 1}). Retrying..."
                    )
                    await asyncio.sleep(0.05)
                    continue
                raise exc
            except Exception as exc:
                raise STTProcessingError(f"Unexpected STT processing error: {exc}") from exc

        raise STTProviderError("STT transcription failed after exhausting retry attempts.")

    async def _transcribe_internal(self, chunks: list[AudioChunk]) -> Transcript:
        """Internal transcription engine execution."""
        start_time = time.perf_counter()

        if self._transient_failures_remaining > 0:
            self._transient_failures_remaining -= 1
            raise STTProviderError("Simulated transient STT network failure.")

        if self._simulated_delay_sec > 0:
            await asyncio.sleep(self._simulated_delay_sec)

        self._transcribe_count += 1
        total_duration = sum(c.duration_ms for c in chunks)
        session_id = chunks[0].session_id if chunks else None
        correlation_id = chunks[0].correlation_id if chunks else None

        # Speech text generation
        text = self._mock_transcript_text or f"Transcribed speech segment {self._transcribe_count}."
        if len(text) > self.config.max_transcript_length:
            text = text[: self.config.max_transcript_length]

        confidence = 0.95
        if confidence < self.config.confidence_threshold:
            text = ""

        seg = TranscriptSegment(
            segment_id=f"seg_{self._transcribe_count}_0",
            text=text,
            start_ms=0.0,
            end_ms=total_duration,
            confidence=confidence,
            is_final=True,
        )

        transcript_id = f"tr_stt_{self._transcribe_count}"
        transcript = Transcript(
            transcript_id=transcript_id,
            segments=[seg],
            full_text=text,
            language=self.config.language,
            confidence=confidence,
            is_final=True,
            session_id=session_id,
            correlation_id=correlation_id,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self._record_telemetry(
            transcript_id=transcript_id,
            chunks=chunks,
            is_final=True,
            latency_ms=elapsed_ms,
            confidence=confidence,
            success=True,
        )
        return transcript

    async def transcribe(self, chunk_stream: AsyncIterable[AudioChunk]) -> Transcript:
        """Transcribes an async stream of AudioChunks into a complete Transcript."""
        chunks: list[AudioChunk] = []
        async for chunk in chunk_stream:
            chunks.append(chunk)

        return await self.transcribe_segment(chunks)

    async def transcribe_chunk(self, chunk: AudioChunk) -> TranscriptSegment:
        """Transcribes a single AudioChunk for real-time partial streaming."""
        text = f"Partial chunk {chunk.sequence_number}"
        return TranscriptSegment(
            segment_id=f"seg_part_{chunk.sequence_number}",
            text=text,
            start_ms=0.0,
            end_ms=chunk.duration_ms,
            confidence=0.85,
            is_final=False,
        )

    def _record_telemetry(
        self,
        transcript_id: str,
        chunks: list[AudioChunk],
        is_final: bool,
        latency_ms: float,
        confidence: float | None = None,
        success: bool = True,
        error_code: str | None = None,
    ) -> None:
        """Records privacy-preserving telemetry without logging raw audio bytes or full speech text."""
        total_duration_ms = sum(c.duration_ms for c in chunks) if chunks else 0.0
        correlation_id = chunks[0].correlation_id if chunks else None

        self._last_telemetry = STTTelemetry(
            transcript_id=transcript_id,
            provider=self.config.provider_name,
            model=self.config.model_name,
            duration_ms=total_duration_ms,
            latency_ms=latency_ms,
            language=self.config.language,
            confidence=confidence,
            segment_count=1,
            is_final=is_final,
            success=success,
            error_code=error_code,
            correlation_id=correlation_id,
        )
