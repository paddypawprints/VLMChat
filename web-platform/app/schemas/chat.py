"""
Chat-related Pydantic schemas.
"""
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid


class ChatMessageResponse(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    images: Optional[List[str]] = None
    debug: Optional[Dict[str, Any]] = None
    createdAt: datetime
    
    class Config:
        from_attributes = True


class ChatMessageCreate(BaseModel):
    user_id: uuid.UUID
    device_id: Optional[str] = None
    role: str
    content: str
    images: Optional[List[str]] = None
    debug: Optional[Dict[str, Any]] = None


class ChatMessageRequest(BaseModel):
    message: str
    deviceId: Optional[str] = None
    debug: Optional[str] = None  # Will be converted to boolean


class ChatResponse(BaseModel):
    userMessage: ChatMessageResponse
    aiMessage: ChatMessageResponse
    success: bool = True