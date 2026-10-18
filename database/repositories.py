
"""Database operations for users, conversations, and messages."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models.conversation import Conversation, Message
from database.models.user import User


async def get_user_by_email(
    session: AsyncSession,
    email: str,
) -> User | None:
    result = await session.execute(
        select(User).where(User.email == email.lower().strip())
    )
    return result.scalar_one_or_none()


async def get_user_by_id(
    session: AsyncSession,
    user_id: UUID,
) -> User | None:
    result = await session.execute(
        select(User).where(User.id == user_id)
    )
    return result.scalar_one_or_none()


async def create_user(
    session: AsyncSession,
    *,
    email: str,
    hashed_password: str,
) -> User:
    user = User(
        email=email.lower().strip(),
        hashed_password=hashed_password,
    )
    session.add(user)
    await session.flush()
    return user


async def create_conversation(
    session: AsyncSession,
    *,
    user_id: UUID,
    title: str = "New conversation",
) -> Conversation:
    conversation = Conversation(user_id=user_id, title=title[:200])
    session.add(conversation)
    await session.flush()
    return conversation


async def get_owned_conversation(
    session: AsyncSession,
    *,
    conversation_id: UUID,
    user_id: UUID,
) -> Conversation | None:
    result = await session.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def add_message(
    session: AsyncSession,
    *,
    conversation_id: UUID,
    role: str,
    content: str,
    citations: list[dict] | None = None,
) -> Message:
    if role not in {"user", "assistant"}:
        raise ValueError("Message role must be user or assistant.")

    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        citations=citations,
    )
    session.add(message)
    await session.flush()
    return message


async def list_user_conversations(
    session: AsyncSession,
    *,
    user_id: UUID,
    limit: int = 50,
    offset: int = 0,
) -> list[Conversation]:
    result = await session.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(
            Conversation.updated_at.desc(),
            Conversation.id.desc(),
        )
        .offset(offset)
        .limit(limit)
    )
    return list(result.scalars().all())


async def list_conversation_messages(
    session: AsyncSession,
    *,
    conversation_id: UUID,
    limit: int = 200,
    offset: int = 0,
) -> list[Message]:
    result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at, Message.id)
        .offset(offset)
        .limit(limit)
    )
    return list(result.scalars().all())
