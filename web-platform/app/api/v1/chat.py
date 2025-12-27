"""
Chat API endpoints.
"""
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import base64

from app.core.database import get_db
from app.repositories.chat_repository import ChatRepository
from app.services.chat_service import ChatService
from app.schemas.chat import ChatMessageResponse, ChatResponse
from app.schemas.auth import UserInDB
from app.deps import require_auth
from app.core.config import get_settings

router = APIRouter(prefix="/chat", tags=["chat"])


@router.get("/messages", response_model=List[ChatMessageResponse])
async def get_messages(
    deviceId: Optional[str] = None,
    current_user: UserInDB = Depends(require_auth),
    db: AsyncSession = Depends(get_db)
):
    """Get chat messages for user and optionally device."""
    chat_repo = ChatRepository(db)
    messages = await chat_repo.get_messages(current_user.id, deviceId)
    
    return [
        ChatMessageResponse(
            id=msg.id,
            role=msg.role,
            content=msg.content,
            images=msg.images,
            debug=msg.debug,
            createdAt=msg.created_at
        )
        for msg in messages
    ]


@router.post("/message", response_model=ChatResponse)
async def send_message(
    message: str = Form(...),
    deviceId: Optional[str] = Form(None),
    debug: Optional[str] = Form(None),
    images: List[UploadFile] = File(default=[]),
    current_user: UserInDB = Depends(require_auth),
    db: AsyncSession = Depends(get_db)
):
    """Send a chat message with optional images."""
    settings = get_settings()
    
    # Validate and process images
    image_data = []
    for image in images:
        # Check file size
        content = await image.read()
        if len(content) > settings.max_upload_size:
            raise HTTPException(status_code=413, detail=f"Image too large: {image.filename}")
        
        # Check file type
        if image.content_type not in settings.allowed_image_types:
            raise HTTPException(status_code=400, detail=f"Invalid image type: {image.content_type}")
        
        # Convert to base64 data URL
        image_b64 = base64.b64encode(content).decode()
        data_url = f"data:{image.content_type};base64,{image_b64}"
        image_data.append(data_url)
    
    # Initialize services
    chat_repo = ChatRepository(db)
    chat_service = ChatService(chat_repo)
    
    # Send message and get response
    debug_mode = debug == "true" if debug else False
    
    return await chat_service.send_message(
        user_id=current_user.id,
        message=message,
        device_id=deviceId,
        images=image_data if image_data else None,
        debug=debug_mode
    )