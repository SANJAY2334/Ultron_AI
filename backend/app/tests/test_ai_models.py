"""Unit Tests for Canonical AI Domain Models.

Validates provider-independent Message, GenerationRequest, GenerationResponse,
Usage, ProviderMetadata, ToolCall, and StreamChunk models.
"""

from app.ai.models import (
    Conversation,
    GenerationRequest,
    GenerationResponse,
    Message,
    ProviderMetadata,
    StreamChunk,
    ToolCall,
    ToolResultReference,
    Usage,
)


def test_message_instantiation_and_defaults() -> None:
    """Verify Message model fields and auto-generated defaults."""
    msg = Message(role="user", content="Hello, Ultron.")
    assert msg.id.startswith("msg_")
    assert msg.role == "user"
    assert msg.content == "Hello, Ultron."
    assert msg.timestamp is not None
    assert msg.tool_calls == []


def test_tool_call_and_result_reference() -> None:
    """Verify ToolCall and ToolResultReference schemas."""
    tc = ToolCall(function_name="get_weather", arguments={"location": "Tokyo"})
    assert tc.id.startswith("call_")
    assert tc.function_name == "get_weather"
    assert tc.arguments["location"] == "Tokyo"

    tr = ToolResultReference(
        tool_call_id=tc.id,
        tool_name="get_weather",
        status="success",
        output={"temp": "22C"},
        execution_time_ms=45.2,
    )
    assert tr.tool_call_id == tc.id
    assert tr.status == "success"
    assert tr.output["temp"] == "22C"


def test_generation_request_and_response() -> None:
    """Verify GenerationRequest and GenerationResponse models."""
    req = GenerationRequest(
        messages=[Message(role="user", content="What is your operating state?")],
        temperature=0.5,
    )
    assert len(req.messages) == 1
    assert req.temperature == 0.5
    assert req.stream is False

    res = GenerationResponse(
        message=Message(role="assistant", content="ULTRON Kernel is fully operational."),
        usage=Usage(
            prompt_tokens=15, completion_tokens=8, total_tokens=23, estimated_cost_usd=0.0001
        ),
        metadata=ProviderMetadata(
            provider_name="openai",
            model_name="gpt-4o",
            latency_ms=120.5,
        ),
    )
    assert res.id.startswith("gen_")
    assert res.message.content == "ULTRON Kernel is fully operational."
    assert res.usage.total_tokens == 23
    assert res.metadata.provider_name == "openai"


def test_stream_chunk() -> None:
    """Verify StreamChunk model."""
    chunk = StreamChunk(content_delta="Hello")
    assert chunk.id.startswith("chunk_")
    assert chunk.content_delta == "Hello"
    assert chunk.finish_reason is None


def test_conversation() -> None:
    """Verify Conversation model handling message history and metadata."""
    conv = Conversation(system_prompt="You are ULTRON.")
    assert conv.id.startswith("conv_")
    assert conv.system_prompt == "You are ULTRON."

    conv.messages.append(Message(role="user", content="Identify yourself."))
    assert len(conv.messages) == 1
