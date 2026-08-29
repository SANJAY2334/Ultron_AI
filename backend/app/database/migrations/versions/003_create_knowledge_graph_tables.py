"""Alembic migration script for ultron_graph_nodes and ultron_graph_edges tables.

Revision ID: 003_create_knowledge_graph_tables
Revises: 002_create_pgvector_memory_table
Create Date: 2026-08-08 20:04:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op  # type: ignore[import-not-found]

revision: str = "003_create_knowledge_graph_tables"
down_revision: str | None = "002_create_pgvector_memory_table"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Creates ultron_graph_nodes and ultron_graph_edges persistent graph tables with indexes and FK constraints."""
    # 1. Create Nodes Table
    op.create_table(
        "ultron_graph_nodes",
        sa.Column("node_id", sa.String(length=64), nullable=False),
        sa.Column("entity_type", sa.String(length=32), nullable=False),
        sa.Column("canonical_name", sa.String(length=256), nullable=False),
        sa.Column("properties", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("privacy", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=True),
        sa.Column("project_id", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.PrimaryKeyConstraint("node_id"),
    )

    op.create_index("ix_ultron_graph_nodes_node_id", "ultron_graph_nodes", ["node_id"])
    op.create_index("ix_ultron_graph_nodes_entity_type", "ultron_graph_nodes", ["entity_type"])
    op.create_index(
        "ix_ultron_graph_nodes_canonical_name", "ultron_graph_nodes", ["canonical_name"]
    )
    op.create_index("ix_ultron_graph_nodes_user_id", "ultron_graph_nodes", ["user_id"])
    op.create_index("ix_ultron_graph_nodes_project_id", "ultron_graph_nodes", ["project_id"])
    op.create_index("ix_ultron_graph_nodes_is_active", "ultron_graph_nodes", ["is_active"])

    # 2. Create Edges Table with Foreign Key Constraints
    op.create_table(
        "ultron_graph_edges",
        sa.Column("edge_id", sa.String(length=64), nullable=False),
        sa.Column("source_node_id", sa.String(length=64), nullable=False),
        sa.Column("target_node_id", sa.String(length=64), nullable=False),
        sa.Column("relationship_type", sa.String(length=32), nullable=False),
        sa.Column("properties", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("source", sa.String(length=32), nullable=False, server_default="USER_INPUT"),
        sa.Column("privacy", sa.String(length=32), nullable=False, server_default="PRIVATE"),
        sa.Column("user_id", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.ForeignKeyConstraint(
            ["source_node_id"], ["ultron_graph_nodes.node_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["target_node_id"], ["ultron_graph_nodes.node_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("edge_id"),
    )

    op.create_index("ix_ultron_graph_edges_edge_id", "ultron_graph_edges", ["edge_id"])
    op.create_index(
        "ix_ultron_graph_edges_source_node_id", "ultron_graph_edges", ["source_node_id"]
    )
    op.create_index(
        "ix_ultron_graph_edges_target_node_id", "ultron_graph_edges", ["target_node_id"]
    )
    op.create_index(
        "ix_ultron_graph_edges_relationship_type",
        "ultron_graph_edges",
        ["relationship_type"],
    )
    op.create_index("ix_ultron_graph_edges_is_active", "ultron_graph_edges", ["is_active"])
    op.create_index(
        "ix_graph_edge_source_target",
        "ultron_graph_edges",
        ["source_node_id", "target_node_id", "relationship_type"],
    )


def downgrade() -> None:
    """Drops ultron_graph_edges and ultron_graph_nodes tables."""
    op.drop_table("ultron_graph_edges")
    op.drop_table("ultron_graph_nodes")
