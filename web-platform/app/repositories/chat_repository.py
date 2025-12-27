"""
Chat repository for database operations.
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import Optional, List
import uuid

from app.domain.models import ChatMessage
from app.schemas.chat import ChatMessageCreate


class ChatRepository:
    """Chat repository."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_by_id(self, message_id: uuid.UUID) -> Optional[ChatMessage]:
        """Get chat message by ID."""
        result = await self.db.execute(select(ChatMessage).where(ChatMessage.id == message_id))
        return result.scalar_one_or_none()
    
    async def get_messages(self, user_id: uuid.UUID, device_id: Optional[str] = None) -> List[ChatMessage]:
        """Get chat messages for user and optionally device."""
        query = select(ChatMessage).where(ChatMessage.user_id == user_id)
        
        if device_id:
            query = query.where(ChatMessage.device_id == device_id)
        
        query = query.order_by(ChatMessage.created_at.asc())
        
        result = await self.db.execute(query)
        return list(result.scalars().all())
    
    async def create(self, message_data: ChatMessageCreate) -> ChatMessage:
        """Create a new chat message."""
        message = ChatMessage(
            user_id=message_data.user_id,
            device_id=message_data.device_id,
            role=message_data.role,
            content=message_data.content,
            images=message_data.images,
            debug=message_data.debug
        )
        self.db.add(message)
        await self.db.commit()
        await self.db.refresh(message)
        return message