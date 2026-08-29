"""Alembic migration script for ultron_memory_items table with pgvector extension and HNSW index.

Revision ID: 002_create_pgvector_memory_table
Revises: 001_initial_schema
Create Date: 2026-08-08 09:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op  # type: ignore[import-not-found]

revision: str = "002_create_pgvector_memory_table"
down_revision: str | None = "001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Enables pgvector extension, creates ultron_memory_items table, B-Tree indexes, and HNSW vector index."""
    # 1. Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 2. Create persistent memory table
    op.create_table(
        "ultron_memory_items",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("memory_type", sa.String(length=32), nullable=False),
        sa.Column("scope", sa.String(length=32), nullable=False),
        sa.Column("privacy", sa.String(length=32), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=True),
        sa.Column("session_id", sa.String(length=128), nullable=True),
        sa.Column("project_id", sa.String(length=128), nullable=True),
        sa.Column("importance", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("embedding_provider", sa.String(length=64), nullable=False),
        sa.Column("embedding_model", sa.String(length=64), nullable=False),
        sa.Column("embedding_dimension", sa.Integer(), nullable=False),
        sa.Column("embedding_version", sa.String(length=32), nullable=False),
        sa.Column("distance_metric", sa.String(length=32), nullable=False, server_default="COSINE"),
        sa.Column("embedding", sa.Text(), nullable=False),  # Stored as vector(1536) in pgvector
        sa.Column("extra_metadata", sa.JSON(), nullable=False, server_default="{}"),
        sa.PrimaryKeyConstraint("id"),
    )

    # 3. Create B-Tree indexes
    op.create_index("ix_ultron_memory_items_id", "ultron_memory_items", ["id"])
    op.create_index("ix_ultron_memory_items_memory_type", "ultron_memory_items", ["memory_type"])
    op.create_index("ix_ultron_memory_items_scope", "ultron_memory_items", ["scope"])
    op.create_index("ix_ultron_memory_items_privacy", "ultron_memory_items", ["privacy"])
    op.create_index("ix_ultron_memory_items_user_id", "ultron_memory_items", ["user_id"])
    op.create_index("ix_ultron_memory_items_session_id", "ultron_memory_items", ["session_id"])
    op.create_index("ix_ultron_memory_items_project_id", "ultron_memory_items", ["project_id"])
    op.create_index("ix_ultron_memory_items_is_active", "ultron_memory_items", ["is_active"])
    op.create_index("ix_ultron_memory_items_created_at", "ultron_memory_items", ["created_at"])
    op.create_index("ix_ultron_memory_items_expires_at", "ultron_memory_items", ["expires_at"])
    op.create_index("ix_memory_user_session", "ultron_memory_items", ["user_id", "session_id"])
    op.create_index("ix_memory_type_scope", "ultron_memory_items", ["memory_type", "scope"])

    # 4. Create HNSW Cosine Vector Index
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_memory_items_embedding_hnsw ON ultron_memory_items USING hnsw (embedding vector_cosine_ops);"
    )


def downgrade() -> None:
    """Drops HNSW index, table, and pgvector extension."""
    op.execute("DROP INDEX IF EXISTS ix_memory_items_embedding_hnsw;")
    op.drop_table("ultron_memory_items")
