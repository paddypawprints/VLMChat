"""
Authentication API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.core.database import get_db
from app.repositories.user_repository import UserRepository
from app.schemas.auth import (
    LoginRequest, OIDCLoginRequest, LoginResponse, LogoutResponse,
    UserCreate, UserResponse
)
from app.deps import create_session, destroy_session

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """Login with email and password (stubbed authentication)."""
    user_repo = UserRepository(db)
    
    # Check if user exists
    user = await user_repo.get_by_email(request.email)
    
    if not user:
        # Create new user for demo (stubbed authentication)
        user_data = UserCreate(
            email=request.email,
            name=request.email.split('@')[0] or 'User',
            provider='email'
        )
        user = await user_repo.create(user_data)
    
    # Create session
    session_id = create_session(user.id)
    
    return LoginResponse(
        user=UserResponse(
            id=user.id,
            email=user.email,
            name=user.name
        ),
        sessionId=session_id
    )


@router.post("/oidc-login", response_model=LoginResponse)
async def oidc_login(
    request: OIDCLoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """OIDC login (stubbed)."""
    user_repo = UserRepository(db)
    
    # Mock OIDC user data
    email = f"demo@{request.provider}.com"
    user = await user_repo.get_by_email(email)
    
    if not user:
        user_data = UserCreate(
            email=email,
            name='Demo User',
            provider=request.provider,
            provider_id='mock-id'
        )
        user = await user_repo.create(user_data)
    
    # Create session
    session_id = create_session(user.id)
    
    return LoginResponse(
        user=UserResponse(
            id=user.id,
            email=user.email,
            name=user.name
        ),
        sessionId=session_id
    )


@router.post("/logout", response_model=LogoutResponse)
async def logout(x_session_id: Optional[str] = Header(None)):
    """Logout and destroy session."""
    if x_session_id:
        destroy_session(x_session_id)
    return LogoutResponse()


@router.get("/session", response_model=UserResponse)
async def get_session(
    x_session_id: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db)
):
    """Get current session user."""
    if not x_session_id:
        raise HTTPException(status_code=401, detail="No session")
    
    from app.deps import session_store
    if x_session_id not in session_store:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    user_id = session_store[x_session_id]
    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(user_id)
    
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name
    )