#!/usr/bin/env python3
"""
FastAPI server startup script.
"""
import uvicorn
import asyncio
from app.main import app
from app.core.database import get_db
from app.repositories.device_repository import DeviceRepository
from app.services.device_service import DeviceService


async def initialize_data():
    """Initialize mock data."""
    async for db in get_db():
        device_repo = DeviceRepository(db)
        device_service = DeviceService(device_repo)
        await device_service.initialize_mock_devices()
        break


if __name__ == "__main__":
    # Initialize mock data
    asyncio.run(initialize_data())
    
    # Run the server
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )