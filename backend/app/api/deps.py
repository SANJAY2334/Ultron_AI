"""FastAPI Dependency Injection Module.

Provides standard FastAPI dependency providers for Settings, Database Sessions,
Redis Clients, Security Contexts, Autonomous Planner instances, Memory Manager facade,
Audio Session Manager orchestrator, and Voice Planner Gateway boundary.
"""

from collections.abc import AsyncGenerator

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.planner.base import BasePlanner
from app.ai.planner.langgraph_planner import LangGraphPlanner
from app.audio.base import (
    IAudioCapture,
    IAudioPlayback,
    IAudioSessionManager,
    ISpeechToTextProvider,
    ITextToSpeechProvider,
    IVoiceActivityDetector,
    IWakeWordDetector,
)
from app.audio.planner_gateway import IVoicePlannerGateway
from app.core.config import Settings, get_settings
from app.core.container import ContainerKeyError, container
from app.core.redis import get_redis_client
from app.core.security import decode_jwt_token
from app.database.session import get_async_db_session
from app.memory.base import IMemoryManager
from app.vision.base import (
    IObjectDetector,
    ISceneAnalyzer,
    IVisionCapture,
    IVisionPlannerContextBuilder,
    IVisionProcessor,
    IVisionSessionManager,
    IVisionTracker,
)

security_bearer = HTTPBearer(auto_error=False)


def get_current_settings() -> Settings:
    """FastAPI Dependency Provider for Application Settings."""
    return get_settings()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI Dependency Provider yielding an Async Database Session."""
    async for session in get_async_db_session():
        yield session


def get_redis() -> Redis:
    """FastAPI Dependency Provider for the Async Redis Client."""
    return get_redis_client()


def get_planner() -> BasePlanner:
    """FastAPI Dependency Provider resolving the BasePlanner interface via DI Container."""
    try:
        return container.resolve_sync(BasePlanner)
    except ContainerKeyError:
        planner = LangGraphPlanner()
        container.register_singleton(BasePlanner, planner)
        return planner


def get_memory_manager() -> IMemoryManager:
    """FastAPI Dependency Provider resolving the IMemoryManager facade via DI Container."""
    try:
        return container.resolve_sync(IMemoryManager)
    except ContainerKeyError:
        from app.memory.manager import MemoryManager

        manager = MemoryManager()
        container.register_singleton(IMemoryManager, manager)
        return manager


def get_vad_detector() -> IVoiceActivityDetector:
    """FastAPI Dependency Provider resolving IVoiceActivityDetector via DI Container."""
    try:
        return container.resolve_sync(IVoiceActivityDetector)
    except ContainerKeyError:
        from app.audio.factory import AudioEngineFactory

        vad = AudioEngineFactory.create_vad_detector()
        container.register_singleton(IVoiceActivityDetector, vad)
        return vad


def get_wake_word_detector() -> IWakeWordDetector:
    """FastAPI Dependency Provider resolving IWakeWordDetector via DI Container."""
    try:
        return container.resolve_sync(IWakeWordDetector)
    except ContainerKeyError:
        from app.audio.factory import AudioEngineFactory

        ww = AudioEngineFactory.create_wake_word_detector()
        container.register_singleton(IWakeWordDetector, ww)
        return ww


def get_stt_provider() -> ISpeechToTextProvider:
    """FastAPI Dependency Provider resolving ISpeechToTextProvider via DI Container."""
    try:
        return container.resolve_sync(ISpeechToTextProvider)
    except ContainerKeyError:
        from app.audio.factory import AudioEngineFactory

        stt = AudioEngineFactory.create_stt_provider()
        container.register_singleton(ISpeechToTextProvider, stt)
        return stt


def get_tts_provider() -> ITextToSpeechProvider:
    """FastAPI Dependency Provider resolving ITextToSpeechProvider via DI Container."""
    try:
        return container.resolve_sync(ITextToSpeechProvider)
    except ContainerKeyError:
        from app.audio.factory import AudioEngineFactory

        tts = AudioEngineFactory.create_tts_provider()
        container.register_singleton(ITextToSpeechProvider, tts)
        return tts


def get_audio_session_manager() -> IAudioSessionManager:
    """FastAPI Dependency Provider resolving the IAudioSessionManager orchestrator via DI Container."""
    try:
        return container.resolve_sync(IAudioSessionManager)
    except ContainerKeyError:
        from app.audio.session_manager import AudioSessionManager

        capture = get_audio_capture()
        vad = get_vad_detector()
        stt = get_stt_provider()
        tts = get_tts_provider()
        wake_word = get_wake_word_detector()
        playback = get_audio_playback()
        gateway = get_voice_planner_gateway()

        session_mgr = AudioSessionManager(
            capture=capture,
            vad=vad,
            stt=stt,
            tts=tts,
            wake_word=wake_word,
            playback=playback,
            planner_gateway=gateway,
        )
        container.register_singleton(IAudioSessionManager, session_mgr)
        return session_mgr


def get_voice_planner_gateway() -> IVoicePlannerGateway:
    """FastAPI Dependency Provider resolving the IVoicePlannerGateway boundary via DI Container."""
    try:
        return container.resolve_sync(IVoicePlannerGateway)
    except ContainerKeyError:
        from app.audio.adapters.planner_gateway import VoicePlannerGateway

        gateway = VoicePlannerGateway()
        container.register_singleton(IVoicePlannerGateway, gateway)
        return gateway


def get_audio_capture() -> IAudioCapture:
    """FastAPI Dependency Provider resolving the IAudioCapture boundary via DI Container."""
    try:
        return container.resolve_sync(IAudioCapture)
    except ContainerKeyError:
        from app.audio.factory import AudioEngineFactory

        capture = AudioEngineFactory.create_audio_capture()
        container.register_singleton(IAudioCapture, capture)
        return capture


def get_audio_playback() -> IAudioPlayback:
    """FastAPI Dependency Provider resolving the IAudioPlayback boundary via DI Container."""
    try:
        return container.resolve_sync(IAudioPlayback)
    except ContainerKeyError:
        from app.audio.factory import AudioEngineFactory

        playback = AudioEngineFactory.create_audio_playback()
        container.register_singleton(IAudioPlayback, playback)
        return playback


def get_vision_capture() -> IVisionCapture:
    """FastAPI Dependency Provider resolving the IVisionCapture boundary via DI Container."""
    try:
        return container.resolve_sync(IVisionCapture)
    except ContainerKeyError:
        from app.vision.adapters.camera import CameraCaptureAdapter

        capture = CameraCaptureAdapter()
        container.register_singleton(IVisionCapture, capture)
        return capture


def get_vision_processor() -> IVisionProcessor:
    """FastAPI Dependency Provider resolving the IVisionProcessor boundary via DI Container."""
    try:
        return container.resolve_sync(IVisionProcessor)
    except ContainerKeyError:
        from app.vision.processor import VisionProcessor

        processor = VisionProcessor()
        container.register_singleton(IVisionProcessor, processor)
        return processor


def get_object_detector() -> IObjectDetector:
    """FastAPI Dependency Provider resolving the IObjectDetector boundary via DI Container."""
    try:
        return container.resolve_sync(IObjectDetector)
    except ContainerKeyError:
        from app.vision.adapters.object_detector import ObjectDetectorAdapter

        detector = ObjectDetectorAdapter()
        container.register_singleton(IObjectDetector, detector)
        return detector


def get_vision_tracker() -> IVisionTracker:
    """FastAPI Dependency Provider resolving the IVisionTracker boundary via DI Container."""
    try:
        return container.resolve_sync(IVisionTracker)
    except ContainerKeyError:
        from app.vision.adapters.tracker import VisionTrackerAdapter

        tracker = VisionTrackerAdapter()
        container.register_singleton(IVisionTracker, tracker)
        return tracker


def get_scene_analyzer() -> ISceneAnalyzer:
    """FastAPI Dependency Provider resolving the ISceneAnalyzer boundary via DI Container."""
    try:
        return container.resolve_sync(ISceneAnalyzer)
    except ContainerKeyError:
        from app.vision.adapters.scene_analyzer import SceneAnalyzerAdapter

        analyzer = SceneAnalyzerAdapter()
        container.register_singleton(ISceneAnalyzer, analyzer)
        return analyzer


def get_vision_planner_context_builder() -> IVisionPlannerContextBuilder:
    """FastAPI Dependency Provider resolving IVisionPlannerContextBuilder via DI Container."""
    try:
        return container.resolve_sync(IVisionPlannerContextBuilder)
    except ContainerKeyError:
        from app.vision.adapters.planner_context_builder import VisionPlannerContextBuilder

        builder = VisionPlannerContextBuilder()
        container.register_singleton(IVisionPlannerContextBuilder, builder)
        return builder


def get_vision_session_manager() -> IVisionSessionManager:
    """FastAPI Dependency Provider resolving the IVisionSessionManager via DI Container."""
    return container.resolve_sync(IVisionSessionManager)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security_bearer),
    settings: Settings = Depends(get_current_settings),
) -> str:
    """FastAPI Dependency Provider for JWT Authentication.

    Returns the authenticated user subject ID ('sub').
    Raises HTTP 401 Unauthorized for missing or invalid tokens.
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        secret_key = (
            settings.ULTRON_SECRET_KEY.get_secret_value()
            or "test_secret_key_for_jwt_auth_1234567890"
        )
        payload = decode_jwt_token(credentials.credentials, secret_key)
        sub: str = payload.get("sub", "")
        if not sub:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return sub
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None
