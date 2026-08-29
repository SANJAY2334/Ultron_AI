"""Text-to-Speech (TTS) Adapter (Phase 4C.5).

Concrete implementation of ITextToSpeechProvider providing modular speech synthesis,
text input validation, voice & speed/pitch parameter checking, streaming AudioChunk generation,
transient retry policy, timeout protection, and privacy-preserving telemetry.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterable
from datetime import UTC, datetime
from typing import Any

from app.audio.base import ITextToSpeechProvider
from app.audio.models import (
    AudioChunk,
    AudioEncoding,
    AudioFormat,
    AudioPrivacy,
    SpeechSynthesisRequest,
    SpeechSynthesisResult,
)
from app.audio.tts import (
    TTSConfig,
    TTSProcessingError,
    TTSProviderError,
    TTSTelemetry,
    TTSTimeoutError,
    TTSValidationError,
)

logger = logging.getLogger(__name__)


class TTSAdapter(ITextToSpeechProvider):
    """Provider-agnostic Text-to-Speech (TTS) synthesis adapter."""

    def __init__(
        self,
        config: TTSConfig | None = None,
        simulated_delay_sec: float = 0.0,
        transient_failures_to_simulate: int = 0,
    ) -> None:
        """Initializes TTSAdapter.

        Args:
            config: Optional TTSConfig configuration instance.
            simulated_delay_sec: Optional processing delay in seconds for timeout testing.
            transient_failures_to_simulate: Number of transient failures to simulate before succeeding.
        """
        self.config = config or TTSConfig()
        self._simulated_delay_sec = simulated_delay_sec
        self._transient_failures_remaining = transient_failures_to_simulate

        self._synthesis_count = 0
        self._last_telemetry: TTSTelemetry | None = None

    def capabilities(self) -> dict[str, Any]:
        """Returns TTS provider capabilities metadata."""
        return {
            "provider_name": self.config.provider_name,
            "model_name": self.config.model_name,
            "supports_streaming": self.config.streaming_enabled,
            "available_voices": ["en-US-Standard-A", "en-US-Standard-B", "en-GB-Standard-A"],
            "supported_sample_rates": [8000, 16000, 24000, 44100, 48000],
            "min_speed": 0.25,
            "max_speed": 4.0,
            "min_pitch": -20.0,
            "max_pitch": 20.0,
        }

    async def health(self) -> dict[str, bool]:
        """Probes health status of TTS engine."""
        return {
            "subsystem_tts": True,
            "provider_ready": True,
        }

    def _validate_request(self, request: SpeechSynthesisRequest) -> None:
        """Validates SpeechSynthesisRequest text, speed, pitch, voice, and length bounds."""
        text = request.text.strip() if request.text else ""
        if not text:
            raise TTSValidationError("Cannot synthesize empty or whitespace-only text.")

        if len(request.text) > self.config.max_text_length:
            raise TTSValidationError(
                f"Input text length ({len(request.text)}) exceeds maximum limit ({self.config.max_text_length})."
            )

        if not (0.25 <= request.speed <= 4.0):
            raise TTSValidationError(
                f"Speed multiplier ({request.speed}) must be in range [0.25, 4.0]."
            )

        if not (-20.0 <= request.pitch <= 20.0):
            raise TTSValidationError(
                f"Pitch shift ({request.pitch}) must be in range [-20.0, 20.0]."
            )

        supported_voices = self.capabilities()["available_voices"]
        if request.voice and request.voice not in supported_voices:
            raise TTSValidationError(f"Unsupported TTS voice '{request.voice}'.")

    async def synthesize(self, request: SpeechSynthesisRequest) -> SpeechSynthesisResult:
        """Synthesizes text into a complete SpeechSynthesisResult with retries and timeout handling."""
        self._validate_request(request)
        timeout_sec = self.config.timeout_ms / 1000.0

        for attempt in range(self.config.retry_count + 1):
            try:
                return await asyncio.wait_for(
                    self._synthesize_internal(request), timeout=timeout_sec
                )
            except TimeoutError as exc:
                self._record_telemetry(
                    synthesis_id=request.request_id,
                    request=request,
                    duration_ms=0.0,
                    latency_ms=timeout_sec * 1000.0,
                    chunk_count=0,
                    success=False,
                    error_code="TTS_TIMEOUT",
                )
                raise TTSTimeoutError(
                    f"TTS synthesis request timed out after {self.config.timeout_ms}ms."
                ) from exc
            except asyncio.CancelledError:
                logger.info("TTS synthesis request cancelled.")
                raise
            except TTSValidationError:
                raise
            except TTSProviderError as exc:
                if attempt < self.config.retry_count:
                    logger.warning(
                        f"Transient TTS provider error (attempt {attempt + 1}/{self.config.retry_count + 1}). Retrying..."
                    )
                    await asyncio.sleep(0.05)
                    continue
                raise exc
            except Exception as exc:
                raise TTSProcessingError(f"Unexpected TTS processing error: {exc}") from exc

        raise TTSProviderError("TTS synthesis failed after exhausting retry attempts.")

    async def _synthesize_internal(self, request: SpeechSynthesisRequest) -> SpeechSynthesisResult:
        """Internal speech rendering logic."""
        start_time = time.perf_counter()

        if self._transient_failures_remaining > 0:
            self._transient_failures_remaining -= 1
            raise TTSProviderError("Simulated transient TTS network failure.")

        if self._simulated_delay_sec > 0:
            await asyncio.sleep(self._simulated_delay_sec)

        self._synthesis_count += 1

        # Calculate synthetic audio duration and payload bytes
        char_count = len(request.text)
        base_duration = (char_count * 100.0) / max(0.25, request.speed)
        duration_ms = min(base_duration, self.config.max_audio_duration_ms)

        if base_duration > self.config.max_audio_duration_ms:
            raise TTSValidationError(
                f"Generated audio duration ({base_duration}ms) exceeds max limit ({self.config.max_audio_duration_ms}ms)."
            )

        sample_rate = request.sample_rate or self.config.sample_rate
        num_samples = int(sample_rate * (duration_ms / 1000.0))
        audio_payload = b"\x00\x00" * num_samples  # 16-bit PCM silent samples

        res_fmt = AudioFormat(
            sample_rate=sample_rate,
            channels=1,
            sample_width=2,
            encoding=AudioEncoding.PCM_S16LE,
        )

        result = SpeechSynthesisResult(
            request_id=request.request_id,
            audio_format=res_fmt,
            duration_ms=duration_ms,
            audio_payload=audio_payload,
            session_id=request.session_id,
            correlation_id=request.correlation_id,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self._record_telemetry(
            synthesis_id=request.request_id,
            request=request,
            duration_ms=duration_ms,
            latency_ms=elapsed_ms,
            chunk_count=1,
            success=True,
        )
        return result

    async def stream_chunks(self, request: SpeechSynthesisRequest) -> AsyncIterable[AudioChunk]:
        """Streams synthesized audio as a sequence of deterministic AudioChunk objects."""
        full_res = await self.synthesize(request)
        chunk_duration_ms = 30.0
        total_duration = full_res.duration_ms
        num_chunks = int(total_duration / chunk_duration_ms) or 1
        bytes_per_chunk = len(full_res.audio_payload) // num_chunks

        for seq in range(num_chunks):
            start_byte = seq * bytes_per_chunk
            end_byte = (
                start_byte + bytes_per_chunk
                if seq < num_chunks - 1
                else len(full_res.audio_payload)
            )
            chunk_bytes = full_res.audio_payload[start_byte:end_byte]

            yield AudioChunk(
                chunk_id=f"chk_tts_{request.request_id}_{seq}",
                sequence_number=seq,
                timestamp=datetime.now(UTC),
                duration_ms=chunk_duration_ms,
                audio_format=full_res.audio_format,
                payload=chunk_bytes,
                session_id=request.session_id,
                correlation_id=request.correlation_id,
                privacy_level=AudioPrivacy.EPHEMERAL,
            )

    async def stream(self, request: SpeechSynthesisRequest) -> AsyncIterable[bytes]:
        """Streams synthesized PCM audio chunks asynchronously as binary bytes."""
        async for chunk in self.stream_chunks(request):
            yield chunk.payload

    def _record_telemetry(
        self,
        synthesis_id: str,
        request: SpeechSynthesisRequest,
        duration_ms: float,
        latency_ms: float,
        chunk_count: int,
        success: bool = True,
        error_code: str | None = None,
    ) -> None:
        """Records privacy-preserving telemetry without raw audio bytes or full spoken text."""
        self._last_telemetry = TTSTelemetry(
            synthesis_id=synthesis_id,
            provider=self.config.provider_name,
            model=self.config.model_name,
            voice=request.voice or self.config.voice,
            duration_ms=duration_ms,
            latency_ms=latency_ms,
            audio_chunk_count=chunk_count,
            success=success,
            error_code=error_code,
            correlation_id=request.correlation_id,
        )
