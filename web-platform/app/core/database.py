"""
Database configuration and session management.
"""
import uuid
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import DateTime, func, text
from datetime import datetime
from typing import AsyncGenerator

from app.core.config import get_settings


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass


# Database engine and session
settings = get_settings()
engine = create_async_engine(
    settings.database_url.replace("postgresql://", "postgresql+psycopg://"),
    echo=True
)
async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def create_db_and_tables():
    """Create database tables."""
    async with engine.begin() as conn:
        # Enable uuid-ossp extension for UUID generation
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\""))
        
        # Import all models to ensure they're registered
        from app.domain.models import User, Device, ChatMessage, AdminService  # noqa
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)