"""
Chat service with AI response generation.
"""
import asyncio
import random
from typing import List, Optional
import uuid

from app.repositories.chat_repository import ChatRepository
from app.schemas.chat import ChatMessageCreate, ChatMessageResponse, ChatResponse


class ChatService:
    """Chat service with mock AI responses."""
    
    def __init__(self, chat_repo: ChatRepository):
        self.chat_repo = chat_repo
    
    async def send_message(
        self, 
        user_id: uuid.UUID, 
        message: str, 
        device_id: Optional[str] = None,
        images: Optional[List[str]] = None,
        debug: bool = False
    ) -> ChatResponse:
        """Send a message and generate AI response."""
        
        # Create user message
        user_message_data = ChatMessageCreate(
            user_id=user_id,
            device_id=device_id,
            role="user",
            content=message,
            images=images or [],
            debug={"userInput": message, "timestamp": str(uuid.uuid4())} if debug else None
        )
        
        user_message = await self.chat_repo.create(user_message_data)
        
        # Simulate AI processing delay
        await asyncio.sleep(0.5 + random.random() * 1.5)
        
        # Generate mock AI response
        ai_response = self._generate_mock_response(message, images, debug)
        
        # Create AI message
        ai_message_data = ChatMessageCreate(
            user_id=user_id,
            device_id=device_id,
            role="assistant",
            content=ai_response["content"],
            images=[],
            debug=ai_response["debug"] if debug else None
        )
        
        ai_message = await self.chat_repo.create(ai_message_data)
        
        return ChatResponse(
            userMessage=ChatMessageResponse(
                id=user_message.id,
                role=user_message.role,
                content=user_message.content,
                images=user_message.images,
                debug=user_message.debug,
                createdAt=user_message.created_at
            ),
            aiMessage=ChatMessageResponse(
                id=ai_message.id,
                role=ai_message.role,
                content=ai_message.content,
                images=ai_message.images,
                debug=ai_message.debug,
                createdAt=ai_message.created_at
            )
        )
    
    def _generate_mock_response(self, message: str, images: Optional[List[str]] = None, debug: bool = False):
        """Generate a mock AI response."""
        responses = [
            "I'm processing your request on the edge device. The model is analyzing your input...",
            "Based on the data processed locally, here's what I found...",
            "Running inference on the edge hardware. This keeps your data private and secure.",
            "The edge AI model has completed processing. Here are the results...",
            "Processing complete. The advantage of edge computing is the low latency you're experiencing."
        ]
        
        content = random.choice(responses)
        
        if images:
            content += f" I can see you've shared {len(images)} image(s) with me."
        
        debug_info = None
        if debug:
            debug_info = {
                "systemPrompt": "You are a helpful AI assistant running on an edge device. Provide concise and accurate responses while highlighting the benefits of edge computing.",
                "modelInputs": {
                    "temperature": 0.7,
                    "max_tokens": 150,
                    "prompt_tokens": len(message),
                    "image_count": len(images) if images else 0
                },
                "modelOutputs": {
                    "tokens_generated": random.randint(20, 100),
                    "confidence": 0.85 + random.random() * 0.15
                },
                "processingTime": random.randint(200, 800),
                "device": {
                    "gpu_usage": random.randint(0, 80),
                    "memory_usage": random.randint(30, 70),
                    "temperature": random.randint(35, 55)
                }
            }
        
        return {"content": content, "debug": debug_info}