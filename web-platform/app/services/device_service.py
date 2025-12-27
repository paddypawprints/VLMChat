"""
Device service with stubbed device communication.
"""
import asyncio
import random
from typing import List
import uuid

from app.repositories.device_repository import DeviceRepository
from app.schemas.devices import DeviceCreate, DeviceActionResponse, DeviceScanResponse
from app.core.config import get_settings


class DeviceService:
    """Device service with stubbed device operations."""
    
    def __init__(self, device_repo: DeviceRepository):
        self.device_repo = device_repo
        self.settings = get_settings()
    
    async def connect_device(self, device_id: str, user_id: uuid.UUID) -> DeviceActionResponse:
        """Connect to a device (stubbed)."""
        # Simulate connection delay
        await asyncio.sleep(self.settings.device_connect_timeout)
        
        # Update device status
        device = await self.device_repo.update_status(device_id, "connected")
        if not device:
            return DeviceActionResponse(success=False, message="Device not found")
        
        # Assign device to user
        await self.device_repo.assign_to_user(device_id, user_id)
        
        return DeviceActionResponse(message="Device connected successfully")
    
    async def disconnect_device(self, device_id: str) -> DeviceActionResponse:
        """Disconnect from a device (stubbed)."""
        # Update device status
        device = await self.device_repo.update_status(device_id, "disconnected")
        if not device:
            return DeviceActionResponse(success=False, message="Device not found")
        
        return DeviceActionResponse(message="Device disconnected")
    
    async def scan_for_devices(self, user_id: uuid.UUID) -> DeviceScanResponse:
        """Scan for available devices (stubbed)."""
        # Simulate scanning delay
        await asyncio.sleep(self.settings.device_scan_timeout)
        
        # Get user's devices to return count
        devices = await self.device_repo.get_user_devices(user_id)
        
        return DeviceScanResponse(
            devices=len(devices),
            message="Scan completed"
        )
    
    async def initialize_mock_devices(self) -> None:
        """Initialize mock devices in the database."""
        mock_devices = [
            DeviceCreate(
                id="rpi-001",
                name="Raspberry Pi 4B",
                type="raspberry-pi",
                ip="192.168.1.100",
                specs={
                    "cpu": "ARM Cortex-A72",
                    "memory": "4GB RAM",
                    "temperature": 45,
                    "usage": 23
                }
            ),
            DeviceCreate(
                id="jetson-001",
                name="NVIDIA Jetson Nano",
                type="jetson",
                ip="192.168.1.101",
                specs={
                    "cpu": "ARM Cortex-A57",
                    "memory": "4GB RAM",
                    "temperature": 52,
                    "usage": 67
                }
            ),
            DeviceCreate(
                id="coral-001",
                name="Google Coral Dev Board",
                type="coral",
                ip="192.168.1.102",
                specs={
                    "cpu": "ARM Cortex-A53",
                    "memory": "1GB RAM",
                    "temperature": 38,
                    "usage": 12
                }
            )
        ]
        
        for device_data in mock_devices:
            existing = await self.device_repo.get_by_id(device_data.id)
            if not existing:
                await self.device_repo.create(device_data)