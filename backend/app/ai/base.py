"""Abstract LLM Provider Base Interface.

Defines the contract for all AI model providers (OpenAI, Anthropic, Local LLM/vLLM)
in ULTRON. Enforces canonical domain schemas and telemetry collection across all drivers.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

from app.ai.models import GenerationRequest, GenerationResponse, Message, StreamChunk


class BaseLLMProvider(ABC):
    """Abstract Base Class for LLM Provider Drivers.

    All LLM drivers (OpenAI, Anthropic, Ollama/vLLM) must inherit from BaseLLMProvider
    and implement these canonical methods.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the identifier name of the provider driver (e.g. 'openai', 'anthropic')."""

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Returns the default model string name associated with this provider driver."""

    @abstractmethod
    async def complete(self, request: GenerationRequest) -> GenerationResponse:
        """Asynchronously generates a complete response for an LLM generation request.

        Args:
            request: Canonical GenerationRequest object.

        Returns:
            GenerationResponse: Canonical response containing generated message, usage, and telemetry.
        """

    @abstractmethod
    def stream(self, request: GenerationRequest) -> AsyncGenerator[StreamChunk, None]:
        """Asynchronously streams generated response chunks in real-time.

        Args:
            request: Canonical GenerationRequest object.

        Yields:
            StreamChunk: Canonical incremental content delta or tool call snippet.
        """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously generates vector embeddings for a list of text strings.

        Args:
            texts: List of text strings to embed.

        Returns:
            list[list[float]]: List of high-dimensional vector embeddings.
        """

    @abstractmethod
    async def count_tokens(self, input_data: str | list[Message]) -> int:
        """Calculates or estimates the token count for a text string or list of messages.

        Args:
            input_data: String text or list of Message objects.

        Returns:
            int: Estimated or exact token count.
        """

    @abstractmethod
    async def health(self) -> bool:
        """Probes the provider API endpoint to verify connectivity and operational status.

        Returns:
            bool: True if provider API is responsive, False otherwise.
        """

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        """Returns provider driver capability metadata.

        Returns:
            dict[str, Any]: Capability map (e.g., {'streaming': True, 'function_calling': True, 'vision': True}).
        """
