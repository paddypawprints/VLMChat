"""
Authentication-related Pydantic schemas.
"""
from pydantic import BaseModel, EmailStr
from typing import Optional
import uuid
from datetime import datetime


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class OIDCLoginRequest(BaseModel):
    provider: str = "google"


class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    name: str


class LoginResponse(BaseModel):
    user: UserResponse
    sessionId: str


class LogoutResponse(BaseModel):
    success: bool = True


class UserCreate(BaseModel):
    email: EmailStr
    name: str
    provider: str = "email"
    provider_id: Optional[str] = None


class UserInDB(BaseModel):
    id: uuid.UUID
    email: str
    name: str
    provider: str
    provider_id: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True