"""
Device-related Pydantic schemas.
"""
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
import uuid


class DeviceSpecs(BaseModel):
    cpu: str
    memory: str
    temperature: float
    usage: float


class DeviceResponse(BaseModel):
    id: str
    name: str
    type: str
    status: str
    ip: str
    specs: Optional[DeviceSpecs] = None
    user_id: Optional[uuid.UUID] = None
    last_seen: datetime
    
    class Config:
        from_attributes = True


class DeviceCreate(BaseModel):
    id: str
    name: str
    type: str
    ip: str
    specs: Optional[Dict[str, Any]] = None


class DeviceUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    ip: Optional[str] = None
    status: Optional[str] = None
    specs: Optional[Dict[str, Any]] = None


class DeviceActionResponse(BaseModel):
    success: bool = True
    message: str


class DeviceScanResponse(BaseModel):
    success: bool = True
    devices: int
    message: str