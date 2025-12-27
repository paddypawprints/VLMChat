"""
Device management API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.core.database import get_db
from app.repositories.device_repository import DeviceRepository
from app.services.device_service import DeviceService
from app.schemas.devices import DeviceResponse, DeviceActionResponse, DeviceScanResponse
from app.schemas.auth import UserInDB
from app.deps import require_auth

router = APIRouter(prefix="/devices", tags=["devices"])


@router.get("", response_model=List[DeviceResponse])
async def get_devices(
    current_user: UserInDB = Depends(require_auth),
    db: AsyncSession = Depends(get_db)
):
    """Get user's devices."""
    device_repo = DeviceRepository(db)
    devices = await device_repo.get_user_devices(current_user.id)
    return [DeviceResponse.model_validate(device) for device in devices]


@router.post("/{device_id}/connect", response_model=DeviceActionResponse)
async def connect_device(
    device_id: str,
    current_user: UserInDB = Depends(require_auth),
    db: AsyncSession = Depends(get_db)
):
    """Connect to a device."""
    device_repo = DeviceRepository(db)
    device_service = DeviceService(device_repo)
    
    return await device_service.connect_device(device_id, current_user.id)


@router.post("/{device_id}/disconnect", response_model=DeviceActionResponse)
async def disconnect_device(
    device_id: str,
    current_user: UserInDB = Depends(require_auth),
    db: AsyncSession = Depends(get_db)
):
    """Disconnect from a device."""
    device_repo = DeviceRepository(db)
    device_service = DeviceService(device_repo)
    
    return await device_service.disconnect_device(device_id)


@router.post("/scan", response_model=DeviceScanResponse)
async def scan_devices(
    current_user: UserInDB = Depends(require_auth),
    db: AsyncSession = Depends(get_db)
):
    """Scan for available devices."""
    device_repo = DeviceRepository(db)
    device_service = DeviceService(device_repo)
    
    return await device_service.scan_for_devices(current_user.id)