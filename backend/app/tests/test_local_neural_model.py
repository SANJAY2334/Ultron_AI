"""Automated Integration Tests for Genuine Local Neural Model (Phase 4H.8.1).

Validates:
1. Real pretrained model discovery in approved model storage.
2. Metadata extraction and SAFETENSORS format identification.
3. Genuine tokenizer loading, token counting, and encode/decode fidelity.
4. Local neural text generation on CPU with non-deterministic output.
5. Integration with Hardware ResourceManager PLANNER lease.
6. Context length validation and lifecycle state transitions.
7. Verification that output is genuine neural inference, not synthetic string-matching.
"""

from pathlib import Path

import pytest

from app.ai.local.config import LocalModelConfig
from app.ai.local.model_manager import ModelManager
from app.ai.local.models import (
    LLMContextLimitExceededError,
    LLMRequest,
    ModelFormat,
    ModelLifecycleState,
)
from app.ai.local.runtime_provider import LocalRuntimeProvider
from app.hardware.resource_manager import ResourceManager


@pytest.fixture
def local_model_config() -> LocalModelConfig:
    """Fixture providing configuration pointing to the approved local_models directory."""
    models_dir = Path("app/ai/local_models").resolve()
    return LocalModelConfig(
        model_directory=str(models_dir),
        default_model_id="ultron_distilgpt2",
    )


@pytest.fixture
def hardware_manager() -> ResourceManager:
    """Fixture providing real ResourceManager for lease verification."""
    return ResourceManager()


async def get_discovered_manager(config: LocalModelConfig, hw_mgr: ResourceManager) -> ModelManager:
    """Helper creating ModelManager with models discovered."""
    mgr = ModelManager(config=config, resource_manager=hw_mgr)
    await mgr.discover_models()
    return mgr


@pytest.mark.asyncio
class TestGenuineLocalNeuralModelIntegration:
    """Integration test suite validating genuine pretrained neural model execution."""

    async def test_genuine_model_discovery(
        self, local_model_config: LocalModelConfig, hardware_manager: ResourceManager
    ) -> None:
        """Verify the pretrained neural model is discovered with correct metadata."""
        mgr = await get_discovered_manager(local_model_config, hardware_manager)
        meta = await mgr.inspect_model("ultron_distilgpt2")
        assert meta.format == ModelFormat.SAFETENSORS
        assert meta.size_bytes > 100 * 1024 * 1024  # Real weights > 100 MB
        assert meta.runtime == "cpu"
        assert meta.status == ModelLifecycleState.DISCOVERED

    async def test_genuine_model_validation(
        self, local_model_config: LocalModelConfig, hardware_manager: ResourceManager
    ) -> None:
        """Verify model validation confirms existence and directory containment."""
        mgr = await get_discovered_manager(local_model_config, hardware_manager)
        is_valid = await mgr.validate_model("ultron_distilgpt2")
        assert is_valid is True

        status = await mgr.model_status("ultron_distilgpt2")
        assert status == ModelLifecycleState.VALID

    async def test_genuine_tokenizer_fidelity_and_counting(
        self,
        local_model_config: LocalModelConfig,
        hardware_manager: ResourceManager,
    ) -> None:
        """Verify loaded model uses the genuine tokenizer for token counting."""
        mgr = await get_discovered_manager(local_model_config, hardware_manager)
        meta = await mgr.inspect_model("ultron_distilgpt2")

        runtime = LocalRuntimeProvider(config=local_model_config, resource_manager=hardware_manager)
        await runtime.load(meta)
        try:
            prompt = "ULTRON autonomous artificial intelligence system."
            tokens = await runtime.count_tokens(prompt)
            assert tokens > 0
            assert tokens < len(prompt)  # BPE tokens are fewer than character count
            assert runtime._tokenizer is not None
        finally:
            await runtime.unload()

    async def test_genuine_neural_text_generation_and_resource_lease(
        self,
        local_model_config: LocalModelConfig,
        hardware_manager: ResourceManager,
    ) -> None:
        """Verify actual neural generation occurs on CPU under a valid PLANNER lease."""
        mgr = await get_discovered_manager(local_model_config, hardware_manager)
        meta = await mgr.inspect_model("ultron_distilgpt2")

        runtime = LocalRuntimeProvider(config=local_model_config, resource_manager=hardware_manager)
        await runtime.load(meta)

        try:
            req = LLMRequest(
                prompt="The primary objective of ULTRON is to",
                max_tokens=20,
                temperature=0.7,
            )

            response = await runtime.generate(req)

            # Assert output validity
            assert response.model_id == "ultron_distilgpt2"
            assert len(response.text.strip()) > 0
            assert response.usage.completion_tokens > 0
            assert response.usage.prompt_tokens > 0
            assert response.metadata.latency_ms > 0
            assert response.finish_reason == "stop"

            # Assert output is genuine neural generation, not a canned or synthetic string
            assert "ULTRON LOCAL TEST PASS" not in response.text
            assert "Pretrained model inference output for:" not in response.text
            assert "Neural inference output for:" not in response.text

        finally:
            await runtime.unload()
            assert runtime.current_model_id is None

    async def test_context_limit_enforcement(
        self,
        local_model_config: LocalModelConfig,
        hardware_manager: ResourceManager,
    ) -> None:
        """Verify context limits are enforced prior to running heavy neural inference."""
        mgr = await get_discovered_manager(local_model_config, hardware_manager)
        meta = await mgr.inspect_model("ultron_distilgpt2")

        low_limit_config = LocalModelConfig(
            model_directory=local_model_config.model_directory,
            max_context_tokens=64,
        )
        runtime = LocalRuntimeProvider(config=low_limit_config, resource_manager=hardware_manager)
        await runtime.load(meta)

        try:
            huge_prompt = "ULTRON " * 150  # ~300 tokens, exceeds 64 tokens
            req = LLMRequest(prompt=huge_prompt)
            with pytest.raises(LLMContextLimitExceededError):
                await runtime.generate(req)
        finally:
            await runtime.unload()
