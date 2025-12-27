"""
AdminService repository for database operations.
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from typing import Optional, List
import uuid

from app.domain.models import AdminService
from app.schemas.admin import AdminServiceCreate, AdminServiceUpdate


class AdminServiceRepository:
    """AdminService repository."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_by_id(self, service_id: uuid.UUID) -> Optional[AdminService]:
        """Get admin service by ID."""
        result = await self.db.execute(select(AdminService).where(AdminService.id == service_id))
        return result.scalar_one_or_none()
    
    async def get_all(self) -> List[AdminService]:
        """Get all admin services."""
        result = await self.db.execute(select(AdminService))
        return list(result.scalars().all())
    
    async def create(self, service_data: AdminServiceCreate) -> AdminService:
        """Create a new admin service."""
        service = AdminService(
            name=service_data.name,
            type=service_data.type,
            endpoint=service_data.endpoint,
            status=service_data.status,
            config=service_data.config
        )
        self.db.add(service)
        await self.db.commit()
        await self.db.refresh(service)
        return service
    
    async def update(self, service_id: uuid.UUID, service_data: AdminServiceUpdate) -> Optional[AdminService]:
        """Update admin service."""
        update_data = service_data.model_dump(exclude_unset=True)
        if update_data:
            result = await self.db.execute(
                update(AdminService)
                .where(AdminService.id == service_id)
                .values(**update_data)
                .returning(AdminService)
            )
            await self.db.commit()
            return result.scalar_one_or_none()
        else:
            return await self.get_by_id(service_id)
    
    async def delete(self, service_id: uuid.UUID) -> bool:
        """Delete admin service."""
        service = await self.get_by_id(service_id)
        if service:
            await self.db.delete(service)
            await self.db.commit()
            return True
        return False