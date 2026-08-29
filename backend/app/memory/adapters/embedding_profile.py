"""Embedding Profile and Production Compatibility Contract.

Defines EmbeddingProfile and DistanceMetric enums for explicit vector model compatibility checks.
Documented Production Policy:
- Provider: 'openai'
- Primary Model: 'text-embedding-3-small'
- Canonical Dimension: 1536
- Version: 'v1'
- Distance Metric: COSINE

The system rejects any incompatible vector profile or dimension before storage or retrieval.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

from app.memory.models import VectorEmbedding

# Canonical Production Embedding Constants
PRODUCTION_EMBEDDING_PROVIDER = "openai"
PRODUCTION_EMBEDDING_MODEL = "text-embedding-3-small"
PRODUCTION_EMBEDDING_DIMENSION = 1536
PRODUCTION_EMBEDDING_VERSION = "v1"
PRODUCTION_DISTANCE_METRIC = "COSINE"


class DistanceMetric(StrEnum):
    """Supported vector distance metric types."""

    COSINE = "COSINE"
    EUCLIDEAN = "EUCLIDEAN"
    INNER_PRODUCT = "INNER_PRODUCT"


class EmbeddingProfileMismatchError(Exception):
    """Raised when an embedding or query is incompatible with the production vector store profile."""


class EmbeddingProfile(BaseModel):
    """Explicit embedding profile declaring model, dimension, version, and distance metric parameters."""

    provider: str = Field(
        default=PRODUCTION_EMBEDDING_PROVIDER, description="Embedding provider identifier"
    )
    model_name: str = Field(
        default=PRODUCTION_EMBEDDING_MODEL, description="Embedding model identifier name"
    )
    dimension: int = Field(
        default=PRODUCTION_EMBEDDING_DIMENSION,
        ge=1,
        description="Vector dimension size (enforced at 1536 in production)",
    )
    version: str = Field(
        default=PRODUCTION_EMBEDDING_VERSION, description="Embedding model version string"
    )
    distance_metric: DistanceMetric = Field(
        default=DistanceMetric.COSINE, description="Target vector similarity metric"
    )

    def is_compatible(self, other: "EmbeddingProfile") -> bool:
        """Evaluates whether another EmbeddingProfile is fully compatible."""
        return (
            self.provider == other.provider
            and self.model_name == other.model_name
            and self.dimension == other.dimension
            and self.version == other.version
            and self.distance_metric == other.distance_metric
        )

    def validate_vector(self, embedding: VectorEmbedding) -> None:
        """Validates that a VectorEmbedding conforms to this profile.

        Raises:
            EmbeddingProfileMismatchError: If model_name, dimension, or version mismatch.
        """
        if embedding.dimension != self.dimension:
            raise EmbeddingProfileMismatchError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {embedding.dimension}."
            )
        if len(embedding.vector) != self.dimension:
            raise EmbeddingProfileMismatchError(
                f"Vector length mismatch: expected {self.dimension}, got {len(embedding.vector)}."
            )
        if embedding.model_name != self.model_name:
            raise EmbeddingProfileMismatchError(
                f"Embedding model mismatch: expected '{self.model_name}', got '{embedding.model_name}'."
            )
        if embedding.version != self.version:
            raise EmbeddingProfileMismatchError(
                f"Embedding version mismatch: expected '{self.version}', got '{embedding.version}'."
            )
