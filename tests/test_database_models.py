"""Tests for the initial SQLAlchemy schema."""

from pgvector.sqlalchemy import Vector

from database.base import Base
from database.models import (
    Conversation,
    DocumentChunk,
    Message,
    SourceDocument,
    User,
)


def test_all_expected_tables_are_registered() -> None:
    expected = {
        "users",
        "conversations",
        "messages",
        "source_documents",
        "document_chunks",
    }

    assert expected.issubset(Base.metadata.tables.keys())


def test_document_embedding_dimension_is_384() -> None:
    column = DocumentChunk.__table__.c.embedding

    assert isinstance(column.type, Vector)
    assert column.type.dim == 384


def test_message_role_constraint_exists() -> None:
    constraints = {
        constraint.name for constraint in Message.__table__.constraints
    }

    assert "ck_messages_role" in constraints


def test_document_chunks_have_unique_position_per_document() -> None:
    constraint_names = {
        constraint.name
        for constraint in DocumentChunk.__table__.constraints
    }

    assert "uq_document_chunks_document_index" in constraint_names


def test_user_conversation_and_document_models_are_importable() -> None:
    assert User.__tablename__ == "users"
    assert Conversation.__tablename__ == "conversations"
    assert SourceDocument.__tablename__ == "source_documents"