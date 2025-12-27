"""
Main API router combining all endpoints.
"""
from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.devices import router as devices_router
from app.api.v1.chat import router as chat_router
from app.api.v1.admin import router as admin_router

# Create main API router
api_router = APIRouter()

# Include all routers
api_router.include_router(auth_router)
api_router.include_router(devices_router)
api_router.include_router(chat_router)
api_router.include_router(admin_router)