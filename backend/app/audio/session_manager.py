"""Full-Duplex Audio Session Orchestrator with Voice Planner Gateway & Playback Integration (Phase 4E.1).

Concrete implementation of IAudioSessionManager orchestrating IAudioCapture, IVoiceActivityDetector,
IWakeWordDetector, ISpeechToTextProvider, ITextToSpeechProvider, IAudioPlayback, and IVoicePlannerGateway
into a full-duplex voice session pipeline with barge-in interruption, backpressure queues, and telemetry.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterable
from datetime import UTC, datetime
from typing import Any

from app.audio.adapters.microphone import MicrophoneCaptureAdapter
from app.audio.adapters.mock_playback import MockPlaybackAdapter
from app.audio.adapters.planner_gateway import VoicePlannerGateway
from app.audio.adapters.stt import STTAdapter
from app.audio.adapters.tts import TTSAdapter
from app.audio.adapters.vad import VADAdapter
from app.audio.adapters.wake_word import WakeWordAdapter
from app.audio.base import (
    IAudioCapture,
    IAudioPlayback,
    IAudioSessionManager,
    ISpeechToTextProvider,
    ITextToSpeechProvider,
    IVoiceActivityDetector,
    IWakeWordDetector,
)
from app.audio.models import (
    AudioChunk,
    AudioPrivacy,
    AudioSessionState,
    AudioSessionStateMachine,
    SpeechSynthesisRequest,
    SpeechSynthesisResult,
    Transcript,
    VoiceActivityState,
)
from app.audio.planner_gateway import IVoicePlannerGateway
from app.audio.session import (
    AudioSessionConfig,
    AudioSessionError,
    AudioSessionLimitError,
    AudioSessionTelemetry,
)

logger = logging.getLogger(__name__)


class AudioSessionManager(IAudioSessionManager):
    """Full-duplex Audio Session Manager orchestrating capture, VAD, wake-word, STT, TTS, playback, and Voice Planner Gateway."""

    def __init__(
        self,
        capture: IAudioCapture | None = None,
        vad: IVoiceActivityDetector | None = None,
        stt: ISpeechToTextProvider | None = None,
        tts: ITextToSpeechProvider | None = None,
        wake_word: IWakeWordDetector | None = None,
        playback: IAudioPlayback | None = None,
        planner_gateway: IVoicePlannerGateway | None = None,
        config: AudioSessionConfig | None = None,
    ) -> None:
        """Initializes AudioSessionManager with component adapters or default implementations."""
        self.config = config or AudioSessionConfig()
        self.capture = capture or MicrophoneCaptureAdapter(mock_mode=True)
        self.vad = vad or VADAdapter()
        self.stt = stt or STTAdapter()
        self.tts = tts or TTSAdapter()
        self.wake_word = wake_word or WakeWordAdapter()
        self.playback = playback or MockPlaybackAdapter()
        self.planner_gateway = planner_gateway or VoicePlannerGateway()

        self._active_session_id: str | None = None
        self._correlation_id: str = "corr_session_default"
        self._state_machine = AudioSessionStateMachine(initial_state=AudioSessionState.IDLE)
        self._active_sessions: set[str] = set()

        self._capture_queue: asyncio.Queue[AudioChunk] = asyncio.Queue(
            maxsize=self.config.max_queued_chunks
        )
        self._synthesis_queue: asyncio.Queue[AudioChunk] = asyncio.Queue(
            maxsize=self.config.max_pending_synthesis_chunks
        )

        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._session_start_time: float = 0.0

        # Metrics counters
        self._chunks_processed = 0
        self._chunks_dropped = 0
        self._speech_segments = 0
        self._transcripts_generated = 0
        self._tts_requests = 0
        self._errors = 0

    def current_state(self) -> AudioSessionState:
        """Returns current active AudioSessionState."""
        return self._state_machine.current_state

    async def transition(self, new_state: AudioSessionState) -> AudioSessionState:
        """Transitions state machine to target state if legal."""
        try:
            return self._state_machine.transition(new_state)
        except Exception as exc:
            self._errors += 1
            raise exc

    async def start_session(self, session_id: str) -> None:
        """Starts a new full-duplex voice interaction session."""
        if len(self._active_sessions) >= self.config.max_concurrent_sessions:
            raise AudioSessionLimitError(
                f"Concurrent sessions limit reached ({self.config.max_concurrent_sessions})."
            )

        if self._active_session_id and self._active_session_id != session_id:
            await self.cancel()

        self._active_session_id = session_id
        self._correlation_id = f"corr_{session_id}"
        self._active_sessions.add(session_id)
        self._session_start_time = time.perf_counter()

        # Reset metrics
        self._chunks_processed = 0
        self._chunks_dropped = 0
        self._speech_segments = 0
        self._transcripts_generated = 0
        self._tts_requests = 0
        self._errors = 0

        # Transition state to LISTENING
        await self.transition(AudioSessionState.LISTENING)
        await self.capture.start()

        # Spawn background processing loop
        task = asyncio.create_task(self._session_pipeline_loop())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        logger.info(f"AudioSessionManager started session '{session_id}'.")

    async def stop_session(self, session_id: str | None = None) -> None:
        """Stops active voice session cleanly and releases all resources."""
        target_id = session_id or self._active_session_id
        if target_id and target_id in self._active_sessions:
            self._active_sessions.remove(target_id)

        await self._cleanup_tasks_and_resources()

        if self._state_machine.current_state != AudioSessionState.IDLE:
            try:
                await self.transition(AudioSessionState.IDLE)
            except Exception:
                self._state_machine = AudioSessionStateMachine(initial_state=AudioSessionState.IDLE)

        self._active_session_id = None
        logger.info(f"AudioSessionManager stopped session '{target_id}'.")

    async def cancel(self) -> None:
        """Cancels active voice session idempotently and safely transitions to IDLE."""
        await self.cancel_session(self._active_session_id)

    async def cancel_session(self, session_id: str | None = None) -> None:
        """Cancels specified session idempotently and releases resources."""
        target_id = session_id or self._active_session_id
        if target_id and target_id in self._active_sessions:
            self._active_sessions.remove(target_id)

        await self._cleanup_tasks_and_resources()

        # Reset state machine to IDLE
        self._state_machine = AudioSessionStateMachine(initial_state=AudioSessionState.IDLE)
        self._active_session_id = None
        logger.info(f"AudioSessionManager cancelled session '{target_id}'.")

    async def interrupt_speech(self) -> None:
        """Executes barge-in interruption, cancelling ongoing TTS/playback and returning to LISTENING state."""
        curr_state = self.current_state()
        if curr_state in (AudioSessionState.SYNTHESIZING, AudioSessionState.PLAYING):
            logger.info("Barge-in speech interruption triggered: cancelling playback output.")
            await self.playback.stop()

            # Drain synthesis queue
            while not self._synthesis_queue.empty():
                try:
                    self._synthesis_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            # Safely transition state machine back to LISTENING
            try:
                if curr_state == AudioSessionState.SYNTHESIZING:
                    await self.transition(AudioSessionState.PLAYING)
                await self.transition(AudioSessionState.LISTENING)
            except Exception:
                self._state_machine = AudioSessionStateMachine(
                    initial_state=AudioSessionState.LISTENING
                )

    async def _cleanup_tasks_and_resources(self) -> None:
        """Cancels all background tasks, drains queues, and stops capture safely."""
        await self.capture.stop()
        await self.playback.stop()
        await self.vad.reset()
        await self.wake_word.reset()

        for task in list(self._background_tasks):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._background_tasks.clear()

        # Drain queues
        while not self._capture_queue.empty():
            try:
                self._capture_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        while not self._synthesis_queue.empty():
            try:
                self._synthesis_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _session_pipeline_loop(self) -> None:
        """Background loop reading capture chunks and routing through wake-word, VAD, STT, Planner, and TTS."""
        accumulated_speech_chunks: list[AudioChunk] = []

        try:
            while self._state_machine.current_state != AudioSessionState.IDLE:
                curr_state = self._state_machine.current_state

                if curr_state in (
                    AudioSessionState.LISTENING,
                    AudioSessionState.SPEECH_DETECTED,
                    AudioSessionState.SYNTHESIZING,
                    AudioSessionState.PLAYING,
                ):
                    chunk = await self.capture.read_chunk()
                    chunk.session_id = self._active_session_id
                    chunk.correlation_id = self._correlation_id

                    self._chunks_processed += 1

                    # Bounded Queue Backpressure Policy: DROP_OLDEST
                    if self._capture_queue.full():
                        try:
                            self._capture_queue.get_nowait()
                            self._chunks_dropped += 1
                        except asyncio.QueueEmpty:
                            pass
                    self._capture_queue.put_nowait(chunk)

                    # Barge-in check during playback/synthesis
                    vad_evt = await self.vad.process(chunk)
                    ww_detect = await self.wake_word.detect(chunk)

                    if curr_state in (AudioSessionState.SYNTHESIZING, AudioSessionState.PLAYING):
                        if ww_detect or vad_evt.state in (
                            VoiceActivityState.SPEECH_START,
                            VoiceActivityState.SPEAKING,
                        ):
                            await self.interrupt_speech()
                            continue

                    # 1. Wake word evaluation
                    if (
                        ww_detect
                        and self._state_machine.current_state == AudioSessionState.LISTENING
                    ):
                        await self.transition(AudioSessionState.SPEECH_DETECTED)

                    # 2. VAD evaluation
                    if vad_evt.state in (
                        VoiceActivityState.SPEECH_START,
                        VoiceActivityState.SPEAKING,
                    ):
                        accumulated_speech_chunks.append(chunk)

                    elif vad_evt.state == VoiceActivityState.SPEECH_END:
                        if accumulated_speech_chunks:
                            self._speech_segments += 1
                            speech_chunks_to_process = list(accumulated_speech_chunks)
                            accumulated_speech_chunks.clear()

                            # Transition to TRANSCRIBING -> THINKING -> Planner Gateway -> SYNTHESIZING -> PLAYING
                            await self.transition(AudioSessionState.TRANSCRIBING)
                            transcript = await self._execute_stt_transcription(
                                speech_chunks_to_process
                            )

                            # Submit to Voice Planner Gateway
                            planner_resp = await self.planner_gateway.submit_transcript(transcript)
                            if planner_resp and planner_resp.text_response:
                                await self.process_speech_text(planner_resp.text_response)

                else:
                    await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"AudioSessionManager pipeline loop error: {exc}")
            self._errors += 1
            self._state_machine = AudioSessionStateMachine(initial_state=AudioSessionState.ERROR)

    async def _execute_stt_transcription(self, chunks: list[AudioChunk]) -> Transcript:
        """Invokes STT engine and transitions to THINKING state."""

        async def _stream_generator() -> AsyncIterable[AudioChunk]:
            for c in chunks:
                yield c

        try:
            if isinstance(self.stt, STTAdapter):
                transcript = await self.stt.transcribe_segment(chunks)
            else:
                transcript = await self.stt.transcribe(_stream_generator())

            self._transcripts_generated += 1
            await self.transition(AudioSessionState.THINKING)
            return transcript
        except Exception as exc:
            self._errors += 1
            logger.error(f"STT transcription failed: {exc}")
            raise AudioSessionError(f"STT transcription error: {exc}") from exc

    async def process_speech_text(self, text: str) -> SpeechSynthesisResult:
        """Executes TTS synthesis for a text response and streams chunks to synthesis playback queue."""
        curr = self.current_state()
        if curr == AudioSessionState.LISTENING:
            await self.transition(AudioSessionState.SPEECH_DETECTED)
            await self.transition(AudioSessionState.TRANSCRIBING)
            await self.transition(AudioSessionState.THINKING)
        elif curr == AudioSessionState.SPEECH_DETECTED:
            await self.transition(AudioSessionState.TRANSCRIBING)
            await self.transition(AudioSessionState.THINKING)
        elif curr == AudioSessionState.TRANSCRIBING:
            await self.transition(AudioSessionState.THINKING)
        elif curr == AudioSessionState.IDLE:
            await self.transition(AudioSessionState.LISTENING)
            await self.transition(AudioSessionState.SPEECH_DETECTED)
            await self.transition(AudioSessionState.TRANSCRIBING)
            await self.transition(AudioSessionState.THINKING)

        await self.transition(AudioSessionState.SYNTHESIZING)

        req = SpeechSynthesisRequest(
            request_id=f"tts_req_{self._tts_requests + 1}",
            text=text,
            voice="en-US-Standard-A",
            session_id=self._active_session_id,
            correlation_id=self._correlation_id,
        )

        try:
            result = await self.tts.synthesize(req)
            self._tts_requests += 1

            # Feed chunks to playback queue and IAudioPlayback adapter
            if isinstance(self.tts, TTSAdapter):
                async for synth_chunk in self.tts.stream_chunks(req):
                    if self._synthesis_queue.full():
                        try:
                            self._synthesis_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    self._synthesis_queue.put_nowait(synth_chunk)
                    await self.playback.play_chunk(synth_chunk)
            else:
                seq = 0
                async for chunk_bytes in self.tts.stream(req):
                    chunk_obj = AudioChunk(
                        chunk_id=f"chk_tts_{req.request_id}_{seq}",
                        sequence_number=seq,
                        timestamp=datetime.now(UTC),
                        duration_ms=30.0,
                        audio_format=result.audio_format,
                        payload=chunk_bytes,
                        session_id=req.session_id,
                        correlation_id=req.correlation_id,
                        privacy_level=AudioPrivacy.EPHEMERAL,
                    )
                    seq += 1
                    if self._synthesis_queue.full():
                        try:
                            self._synthesis_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    self._synthesis_queue.put_nowait(chunk_obj)
                    await self.playback.play_chunk(chunk_obj)

            await self.transition(AudioSessionState.PLAYING)
            return result
        except Exception as exc:
            self._errors += 1
            raise AudioSessionError(f"TTS synthesis error: {exc}") from exc

    async def read_playback_chunk(self) -> AudioChunk:
        """Reads next synthesized AudioChunk from the playback boundary queue."""
        try:
            return await self._synthesis_queue.get()
        except asyncio.CancelledError:
            raise

    def get_telemetry(self) -> AudioSessionTelemetry:
        """Returns privacy-safe AudioSessionTelemetry metadata without raw audio or full text."""
        elapsed_ms = (
            (time.perf_counter() - self._session_start_time) * 1000.0
            if self._session_start_time > 0
            else 0.0
        )
        return AudioSessionTelemetry(
            session_id=self._active_session_id or "none",
            correlation_id=self._correlation_id,
            state=self.current_state(),
            duration_ms=elapsed_ms,
            chunks_processed=self._chunks_processed,
            chunks_dropped=self._chunks_dropped,
            speech_segments=self._speech_segments,
            transcripts_generated=self._transcripts_generated,
            tts_requests=self._tts_requests,
            errors=self._errors,
            latency_ms=25.0,
            timestamp=datetime.now(UTC),
        )

    async def health(self) -> dict[str, Any]:
        """Probes health status of session manager and child audio components."""
        cap_health = await self.capture.health()
        vad_health = await self.vad.health()
        stt_health = await self.stt.health()
        tts_health = await self.tts.health()
        ww_health = await self.wake_word.health()
        playback_health = await self.playback.health()
        gw_health = await self.planner_gateway.health()

        return {
            "subsystem_session_manager": True,
            "state": self.current_state().value,
            "active_session_id": self._active_session_id,
            "active_sessions_count": len(self._active_sessions),
            "capture": cap_health,
            "vad": vad_health,
            "stt": stt_health,
            "tts": tts_health,
            "wake_word": ww_health,
            "playback": playback_health,
            "planner_gateway": gw_health,
        }
