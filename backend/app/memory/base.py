"""Framework-Agnostic Hybrid Memory Matrix Interfaces.

Defines abstract base classes (contracts) for Working Memory, Episodic Memory,
Semantic Memory, Knowledge Graph, Memory Retriever, Memory Writer, and Memory Manager facade.
Enforces MemoryAccessContext privacy filtering, decoupled tier operations, and deletion semantics.
"""

from abc import ABC, abstractmethod

from app.memory.knowledge_graph import IKnowledgeGraph
from app.memory.models import (
    MemoryAccessContext,
    MemoryDeleteResult,
    MemoryItem,
    MemoryQuery,
    MemorySearchResult,
    MemoryWriteResult,
)

__all__ = [
    "IWorkingMemory",
    "IEpisodicMemory",
    "ISemanticMemory",
    "IKnowledgeGraph",
    "IMemoryRetriever",
    "IMemoryWriter",
    "IMemoryManager",
]


class IWorkingMemory(ABC):
    """Abstract Interface for Transient Working Memory (Session context & active window).

    Responsibility: Manages short-term, session-bound context items and recent message history.
    """

    @abstractmethod
    async def add(self, item: MemoryItem) -> MemoryWriteResult:
        """Adds a new transient item to Working Memory."""

    @abstractmethod
    async def get(self, item_id: str) -> MemoryItem | None:
        """Retrieves a single working memory item by ID."""

    @abstractmethod
    async def clear_session(self, session_id: str) -> bool:
        """Clears all transient working memory items associated with a session."""

    @abstractmethod
    async def get_history(self, session_id: str, limit: int = 50) -> list[MemoryItem]:
        """Retrieves ordered conversation working memory history for a session."""


class IEpisodicMemory(ABC):
    """Abstract Interface for Episodic Memory (Temporal interaction histories & episodes).

    Responsibility: Stores and recalls time-ordered conversation episodes and planner execution traces.
    """

    @abstractmethod
    async def store_episode(self, item: MemoryItem) -> MemoryWriteResult:
        """Stores a temporal episode item in Episodic Memory."""

    @abstractmethod
    async def recall_episodes(
        self, query: MemoryQuery, access_context: MemoryAccessContext
    ) -> MemorySearchResult:
        """Recalls relevant temporal episode items matching search parameters and authorization."""


class ISemanticMemory(ABC):
    """Abstract Interface for Semantic Memory (Factual knowledge & vector embeddings).

    Responsibility: Manages long-term facts, concepts, and vector embeddings for similarity search.
    """

    @abstractmethod
    async def store_concept(
        self, item: MemoryItem, access_context: MemoryAccessContext | None = None
    ) -> MemoryWriteResult:
        """Stores a domain concept or factual item in Semantic Memory."""

    @abstractmethod
    async def search_concepts(
        self, query: MemoryQuery, access_context: MemoryAccessContext
    ) -> MemorySearchResult:
        """Performs vector or keyword search over long-term semantic knowledge."""

    @abstractmethod
    async def delete_concept(
        self,
        memory_id: str,
        access_context: MemoryAccessContext,
        soft_delete: bool = True,
    ) -> MemoryDeleteResult:
        """Deletes or tombstones a concept item by ID."""


class IMemoryRetriever(ABC):
    """Abstract Interface for Cross-Tier Memory Retrieval.

    Responsibility: Executes unified search queries across hybrid memory tiers, enforcing MemoryAccessContext privacy.
    """

    @abstractmethod
    async def retrieve(
        self, query: MemoryQuery, access_context: MemoryAccessContext
    ) -> MemorySearchResult:
        """Executes cross-tier retrieval enforcing privacy access rules and returning ranked results."""


class IMemoryWriter(ABC):
    """Abstract Interface for Memory Persistence Operations.

    Responsibility: Handles item creation, updates, and logical/permanent deletion across storage tiers.
    """

    @abstractmethod
    async def write(
        self, item: MemoryItem, access_context: MemoryAccessContext
    ) -> MemoryWriteResult:
        """Writes a memory item to storage enforcing authorization rules."""

    @abstractmethod
    async def delete(
        self,
        memory_id: str,
        access_context: MemoryAccessContext,
        soft_delete: bool = True,
    ) -> MemoryDeleteResult:
        """Deletes or tombstones a memory item by ID."""


class IMemoryManager(ABC):
    """Abstract Unified Facade Interface for the Hybrid Memory Matrix Subsystem.

    Responsibility: Serves as the single primary memory contract consumed by the Autonomous Planner,
    coordinating Working, Episodic, Semantic, Relational Graph, and Memory Consolidation lifecycles.
    """

    @abstractmethod
    async def remember(
        self, item: MemoryItem, access_context: MemoryAccessContext
    ) -> MemoryWriteResult:
        """Stores a memory item into the Hybrid Memory Matrix."""

    @abstractmethod
    async def recall(
        self, query: MemoryQuery, access_context: MemoryAccessContext
    ) -> MemorySearchResult:
        """Recalls relevant memory items matching query parameters and authorization context."""

    @abstractmethod
    async def forget(
        self,
        memory_id: str,
        access_context: MemoryAccessContext,
        soft_delete: bool = True,
    ) -> MemoryDeleteResult:
        """Removes or tombstones a memory item from storage tiers."""

    @abstractmethod
    async def consolidate(
        self, session_id: str, access_context: MemoryAccessContext
    ) -> list[MemoryWriteResult]:
        """Consolidates short-term session working memory into long-term episodic/semantic tiers."""
