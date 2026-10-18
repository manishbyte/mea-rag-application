
"""Chat and persistent conversation-history endpoints."""

import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status

from api.dependencies import CurrentUser, DatabaseSession
from api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationHistoryResponse,
    ConversationSummary,
    MessageResponse,
)
from database.models.conversation import Conversation
from database.repositories import (
    add_message,
    create_conversation,
    get_owned_conversation,
    list_conversation_messages,
    list_user_conversations,
)
from graph.workflow import run_workflow

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    user: CurrentUser,
    session: DatabaseSession,
) -> ChatResponse:
    """Ask a question and persist the user and assistant messages."""

    conversation: Conversation | None = None

    if payload.conversation_id is not None:
        conversation = await get_owned_conversation(
            session,
            conversation_id=payload.conversation_id,
            user_id=user.id,
        )

        if conversation is None:
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found.",
            )

    try:
        if conversation is None:
            title = payload.question[:200] or "New conversation"
            conversation = await create_conversation(
                session,
                user_id=user.id,
                title=title,
            )
            await session.commit()
        elif conversation.title == "New conversation":
            conversation.title = payload.question[:200]

        user_message = await add_message(
            session,
            conversation_id=conversation.id,
            role="user",
            content=payload.question,
        )
        await session.commit()

    except Exception:
        await session.rollback()
        logger.exception("Failed to persist the incoming chat message.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to save your message.",
        ) from None

    try:
        result = await run_workflow(payload.question)
    except Exception:
        logger.exception("RAG workflow failed.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "The question was saved, but the answer service "
                "could not complete the request. You can retry."
            ),
        ) from None

    answer = result.get("answer") or (
        "I could not produce a validated answer for this question."
    )
    citations = result.get("citations") or []
    abstained = bool(result.get("abstained", False))
    validation_passed = bool(result.get("validation_passed", False))

    try:
        assistant_message = await add_message(
            session,
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            citations=citations,
        )
        await session.commit()
    except Exception:
        await session.rollback()
        logger.exception("Failed to persist the assistant response.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "An answer was generated, but it could not be saved. "
                "Please retry the request."
            ),
        ) from None

    return ChatResponse(
        conversation_id=conversation.id,
        user_message_id=user_message.id,
        assistant_message_id=assistant_message.id,
        question=payload.question,
        answer=answer,
        citations=citations,
        abstained=abstained,
        validation_passed=validation_passed,
    )


@router.get("/conversations", response_model=list[ConversationSummary])
async def get_conversations(
    user: CurrentUser,
    session: DatabaseSession,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[ConversationSummary]:
    """List only the authenticated user's conversations."""

    conversations = await list_user_conversations(
        session,
        user_id=user.id,
        limit=limit,
        offset=offset,
    )
    return [
        ConversationSummary.model_validate(item)
        for item in conversations
    ]


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationHistoryResponse,
)
async def get_conversation_history(
    conversation_id: UUID,
    user: CurrentUser,
    session: DatabaseSession,
    limit: int = Query(default=200, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> ConversationHistoryResponse:
    """Retrieve messages only if the conversation belongs to the user."""

    conversation = await get_owned_conversation(
        session,
        conversation_id=conversation_id,
        user_id=user.id,
    )

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    messages = await list_conversation_messages(
        session,
        conversation_id=conversation.id,
        limit=limit,
        offset=offset,
    )

    return ConversationHistoryResponse(
        conversation=ConversationSummary.model_validate(conversation),
        messages=[
            MessageResponse.model_validate(message)
            for message in messages
        ],
    )
