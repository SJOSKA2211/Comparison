"""Authentication schemas."""

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class UserResponse(BaseModel):
    """User response schema."""
    """User response schema."""
    id: UUID
    email: str
    role: str
    email_verified: bool = False
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None

class LoginRequest(BaseModel):
    """Login request schema."""
    """Login request schema."""
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    """Token response schema."""
    """Token response schema."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 86400
    user: Optional[UserResponse] = None

class UserCreate(BaseModel):
    """User creation schema."""
    """User creation schema."""
    email: EmailStr
    password: str = Field(min_length=8)
    role: str = Field(default="trader", pattern="^(trader|researcher|admin)$")
