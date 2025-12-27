"""
Dependency injection helpers.
"""
from fastapi import HTTPException, Header, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, Dict, Any
import uuid

from app.core.database import get_db
from app.repositories.user_repository import UserRepository
from app.schemas.auth import UserInDB


# In-memory session store (replace with Redis in production)
session_store: Dict[str, uuid.UUID] = {}


async def get_current_user(
    x_session_id: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db)
) -> Optional[UserInDB]:
    """Get current authenticated user from session."""
    if not x_session_id or x_session_id not in session_store:
        return None
    
    user_id = session_store[x_session_id]
    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(user_id)
    
    if not user:
        # Clean up invalid session
        session_store.pop(x_session_id, None)
        return None
    
    return UserInDB.model_validate(user)


async def require_auth(
    current_user: Optional[UserInDB] = Depends(get_current_user)
) -> UserInDB:
    """Require authentication."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return current_user


def create_session(user_id: uuid.UUID) -> str:
    """Create a new session for user."""
    import secrets
    session_id = secrets.token_urlsafe(32)
    session_store[session_id] = user_id
    return session_id


def destroy_session(session_id: str) -> None:
    """Destroy a session."""
    session_store.pop(session_id, None)