"""Chat and conversation-history schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: UUID | None = None

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Question must not be empty.")

        return value


class ChatResponse(BaseModel):
    conversation_id: UUID
    user_message_id: UUID
    assistant_message_id: UUID
    question: str
    answer: str
    citations: list[dict] = Field(default_factory=list)
    abstained: bool
    validation_passed: bool


class MessageResponse(BaseModel):
    id: UUID
    role: str
    content: str
    citations: list[dict] | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ConversationSummary(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ConversationHistoryResponse(BaseModel):
    conversation: ConversationSummary
    messages: list[MessageResponse]