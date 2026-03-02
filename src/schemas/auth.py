# pylint: disable=missing-module-docstring, missing-class-docstring, too-few-public-methods, import-error
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class UserResponse(BaseModel):
    id: UUID
    email: str
    role: str
    email_verified: bool = False
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None


class LoginRequest(BaseModel):
    """Schema for login request."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Schema for token response."""
    access_token: str
    token_type: str = "bearer"
    user: Optional["UserResponse"] = None


class UserCreate(BaseModel):
    email: EmailStr
    role: UserRole
    email_verified: bool
    oauth_provider: OAuthProvider
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None

    class Config:
        from_attributes = True
