"""
Admin API endpoints.
"""
import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.core.database import get_db
from app.repositories.device_repository import DeviceRepository
from app.repositories.admin_service_repository import AdminServiceRepository
from app.services.device_service import DeviceService
from app.schemas.devices import DeviceResponse, DeviceCreate, DeviceUpdate
from app.schemas.admin import AdminServiceResponse, AdminServiceCreate, AdminServiceUpdate
from app.schemas.auth import UserInDB
from app.deps import require_auth

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/devices", response_model=List[DeviceResponse])
async def admin_get_devices(
    current_user: UserInDB = Depends(require_auth),
    db: AsyncSession = Depends(get_db)
):
    """Get all devices (admin view)."""
    device_repo = DeviceRepository(db)
    devices = await device_repo.get_all()
    return [DeviceResponse.model_validate(device) for device in devices]


@router.post("/devices", response_model=DeviceResponse)
async def admin_create_device(
    device_data: DeviceCreate,
    current_user: UserInDB = Depends(require_auth),
    db: AsyncSession = Depends(get_db)
):
    """Create a new device (admin)."""
    device_repo = DeviceRepository(db)
    
    # Check if device already exists
    existing = await device_repo.get_by_id(device_data.id)
    if existing:
        raise HTTPException(status_code=400, detail="Device already exists")
    
    device = await device_repo.create(device_data)
    return DeviceResponse.model_validate(device)


@router.patch("/devices/{device_id}", response_model=DeviceResponse)
async def admin_update_device(
    device_id: str,
    device_data: DeviceUpdate,
    current_user: UserInDB = Depends(require_auth),
    db: AsyncSession = Depends(get_db)
):
    """Update a device (admin)."""
    device_repo = DeviceRepository(db)
    
    device = await device_repo.update(device_id, device_data)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    
    return DeviceResponse.model_validate(device)


@router.delete("/devices/{device_id}")
async def admin_delete_device(
    device_id: str,
    current_user: UserInDB = Depends(require_auth),
    db: AsyncSession = Depends(get_db)
):
    """Delete a device (admin)."""
    device_repo = DeviceRepository(db)
    
    deleted = await device_repo.delete(device_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Device not found")
    
    return {"message": "Device deleted successfully"}


@router.get("/services", response_model=List[AdminServiceResponse])
async def admin_get_services(
    current_user: UserInDB = Depends(require_auth),
    db: AsyncSession = Depends(get_db)
):
    """Get all admin services."""
    service_repo = AdminServiceRepository(db)
    services = await service_repo.get_all()
    return [AdminServiceResponse.model_validate(service) for service in services]


@router.post("/services", response_model=AdminServiceResponse)
async def admin_create_service(
    service_data: AdminServiceCreate,
    current_user: UserInDB = Depends(require_auth),
    db: AsyncSession = Depends(get_db)
):
    """Create a new admin service."""
    service_repo = AdminServiceRepository(db)
    service = await service_repo.create(service_data)
    return AdminServiceResponse.model_validate(service)


@router.patch("/services/{service_id}", response_model=AdminServiceResponse)
async def admin_update_service(
    service_id: str,
    service_data: AdminServiceUpdate,
    current_user: UserInDB = Depends(require_auth),
    db: AsyncSession = Depends(get_db)
):
    """Update an admin service."""
    service_repo = AdminServiceRepository(db)
    
    try:
        service_uuid = uuid.UUID(service_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid service ID format")
    
    service = await service_repo.update(service_uuid, service_data)
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")
    
    return AdminServiceResponse.model_validate(service)


@router.delete("/services/{service_id}")
async def admin_delete_service(
    service_id: str,
    current_user: UserInDB = Depends(require_auth),
    db: AsyncSession = Depends(get_db)
):
    """Delete an admin service."""
    service_repo = AdminServiceRepository(db)
    
    try:
        service_uuid = uuid.UUID(service_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid service ID format")
    
    deleted = await service_repo.delete(service_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Service not found")
    
    return {"message": "Service deleted successfully"}