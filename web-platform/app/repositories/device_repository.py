"""
Device repository for database operations.
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from typing import Optional, List
import uuid

from app.domain.models import Device
from app.schemas.devices import DeviceCreate, DeviceUpdate


class DeviceRepository:
    """Device repository."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_by_id(self, device_id: str) -> Optional[Device]:
        """Get device by ID."""
        result = await self.db.execute(select(Device).where(Device.id == device_id))
        return result.scalar_one_or_none()
    
    async def get_user_devices(self, user_id: uuid.UUID) -> List[Device]:
        """Get all devices for a user."""
        result = await self.db.execute(
            select(Device).where(Device.user_id == user_id)
        )
        return list(result.scalars().all())
    
    async def get_all(self) -> List[Device]:
        """Get all devices."""
        result = await self.db.execute(select(Device))
        return list(result.scalars().all())
    
    async def create(self, device_data: DeviceCreate) -> Device:
        """Create a new device."""
        device = Device(
            id=device_data.id,
            name=device_data.name,
            type=device_data.type,
            ip=device_data.ip,
            specs=device_data.specs
        )
        self.db.add(device)
        await self.db.commit()
        await self.db.refresh(device)
        return device
    
    async def update(self, device_id: str, device_data: DeviceUpdate) -> Optional[Device]:
        """Update device."""
        update_data = device_data.model_dump(exclude_unset=True)
        if update_data:
            result = await self.db.execute(
                update(Device)
                .where(Device.id == device_id)
                .values(**update_data)
                .returning(Device)
            )
            await self.db.commit()
            return result.scalar_one_or_none()
        else:
            return await self.get_by_id(device_id)
    
    async def update_status(self, device_id: str, status: str) -> Optional[Device]:
        """Update device status."""
        result = await self.db.execute(
            update(Device)
            .where(Device.id == device_id)
            .values(status=status)
            .returning(Device)
        )
        await self.db.commit()
        return result.scalar_one_or_none()
    
    async def assign_to_user(self, device_id: str, user_id: uuid.UUID) -> Optional[Device]:
        """Assign device to user."""
        result = await self.db.execute(
            update(Device)
            .where(Device.id == device_id)
            .values(user_id=user_id)
            .returning(Device)
        )
        await self.db.commit()
        return result.scalar_one_or_none()
    
    async def delete(self, device_id: str) -> bool:
        """Delete device."""
        device = await self.get_by_id(device_id)
        if device:
            await self.db.delete(device)
            await self.db.commit()
            return True
        return False