"""Authentication schemas."""
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    """Schema for user creation."""
    email: EmailStr
    password: str = Field(min_length=8)
    role: str = Field(default="trader")

    def model_post_init(self, __context):
        # Validate role
        if self.role not in ("trader", "researcher", "admin"):
            raise ValueError(f"Invalid role: {self.role}")

class LoginRequest(BaseModel):
    """Schema for login request."""
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    """Schema for token response."""
    access_token: str
    token_type: str = "bearer"
    user: Optional["UserResponse"] = None

class UserResponse(BaseModel):
    """Schema for user response."""
    id: UUID
    email: EmailStr
    role: str
    email_verified: bool
    oauth_provider: str
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None

    class Config:
        from_attributes = True
