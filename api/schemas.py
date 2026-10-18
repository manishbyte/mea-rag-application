
"""Pydantic request and response schemas for the API."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        value = value.strip().lower()

        if (
            not value
            or "@" not in value
            or value.startswith("@")
            or value.endswith("@")
            or any(char.isspace() for char in value)
        ):
            raise ValueError("Enter a valid email address.")

        return value


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().lower()


class UserResponse(BaseModel):
    id: UUID
    email: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


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


class CitationResponse(BaseModel):
    """Citation fields are flexible because retrieval metadata may vary."""

    model_config = ConfigDict(extra="allow")

    source_id: str | None = None
    page_number: int | None = None


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


class HealthResponse(BaseModel):
    status: str
