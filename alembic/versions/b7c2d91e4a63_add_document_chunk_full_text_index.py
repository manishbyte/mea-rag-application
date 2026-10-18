
"""Add a GIN index for document chunk full-text search."""

from alembic import op

revision = "b7c2d91e4a63"
down_revision = "42421081707d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_document_chunks_content_fts
        ON document_chunks
        USING GIN (to_tsvector('english', content))
        """
    )


def downgrade() -> None:
    op.execute(
        "DROP INDEX IF EXISTS ix_document_chunks_content_fts"
    )
