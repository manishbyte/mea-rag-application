
"""Registration and login endpoints."""

from fastapi import APIRouter, HTTPException, status
from sqlalchemy.exc import IntegrityError

from api.dependencies import CurrentUser, DatabaseSession
from api.schemas import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from core.config import get_settings
from core.security import create_access_token, hash_password, verify_password
from database.repositories import create_user, get_user_by_email
from fastapi import APIRouter, Depends, Response, status

from api.dependencies import CurrentUser

router = APIRouter(prefix="/auth", tags=["Authentication"])



@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(
    payload: RegisterRequest,
    session: DatabaseSession,
) -> TokenResponse:
    existing = await get_user_by_email(session, payload.email)

    if existing is not None:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    try:
        user = await create_user(
            session,
            email=payload.email,
            hashed_password=hash_password(payload.password),
        )
        await session.commit()

    except IntegrityError:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        ) from None

    except Exception:
        await session.rollback()
        raise

    settings = get_settings()

    return TokenResponse(
        access_token=create_access_token(user.id),
        expires_in=settings.access_token_expire_minutes * 60,
        user=UserResponse.model_validate(user),
    )



@router.post("/login", response_model=TokenResponse)
async def login(
    payload: LoginRequest,
    session: DatabaseSession,
) -> TokenResponse:
    user = await get_user_by_email(session, payload.email)

    if user is None or not verify_password(
        payload.password,
        user.hashed_password,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    settings = get_settings()

    return TokenResponse(
        access_token=create_access_token(user.id),
        expires_in=settings.access_token_expire_minutes * 60,
        user=UserResponse.model_validate(user),
    )


@router.get("/me", response_model=UserResponse)
async def get_me(user: CurrentUser) -> UserResponse:
    return UserResponse.model_validate(user)

@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    current_user: CurrentUser,
) -> Response:
    """
    Stateless JWT logout.

    The client must delete its access token.
    The token remains valid until expiry unless server-side
    token revocation is implemented.
    """
    return Response(status_code=status.HTTP_204_NO_CONTENT)