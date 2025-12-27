"""
Admin-related Pydantic schemas.
"""
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
import uuid


class AdminServiceResponse(BaseModel):
    id: uuid.UUID
    name: str
    type: str
    endpoint: Optional[str] = None
    status: str
    config: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class AdminServiceCreate(BaseModel):
    name: str
    type: str
    endpoint: Optional[str] = None
    status: str = "active"
    config: Optional[Dict[str, Any]] = None


class AdminServiceUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    endpoint: Optional[str] = None
    status: Optional[str] = None
    config: Optional[Dict[str, Any]] = None