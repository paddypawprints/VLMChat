#!/usr/bin/env python3
"""
Start FastAPI server for Independent Research.
"""
import uvicorn
import asyncio
from app.main import app
from app.core.database import get_db
from app.repositories.device_repository import DeviceRepository
from app.services.device_service import DeviceService


async def initialize_data():
    """Initialize mock data."""
    try:
        async for db in get_db():
            device_repo = DeviceRepository(db)
            device_service = DeviceService(device_repo)
            await device_service.initialize_mock_devices()
            print("✅ Mock devices initialized successfully")
            break
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize mock data: {e}")


if __name__ == "__main__":
    print("🚀 Starting Independent Research FastAPI Backend...")
    
    # Initialize mock data
    try:
        asyncio.run(initialize_data())
    except Exception as e:
        print(f"⚠️  Warning: Data initialization failed: {e}")
    
    # Run the server (disable reload when supervised)
    print("🌐 FastAPI server starting on http://0.0.0.0:8000")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )