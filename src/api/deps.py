from datetime import datetime
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db
from src.models.user import Session as UserSession
from src.models.user import User


async def get_current_user(request: Request, db: AsyncSession = Depends(get_db)) -> Optional[dict]:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None

    token = auth_header[7:]

    # We use ORM to validate session instead of PL/pgSQL function
    # Note: In a real system we'd use hashing for the token
    result = await db.execute(
        select(UserSession)
        .where(UserSession.token_hash == token)  # Simple match for MVP, would be hash in prod
        .where(UserSession.expires_at > datetime.utcnow())
    )
    session = result.scalars().first()

    if session:
        # Load user
        user_result = await db.execute(select(User).where(User.id == session.user_id))
        user = user_result.scalars().first()
        if user:
            return {
                "user_id": user.id,
                "email": user.email,
                "user_role": user.role,
                "email_verified": user.email_verified,
            }
    return None


def require_auth(user: Optional[dict] = Depends(get_current_user)) -> dict:
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_verified(user: dict = Depends(require_auth)) -> dict:
    if not user.get("email_verified", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Email not verified")
    return user


def require_role(*roles: str):
    def check_role(user: dict = Depends(require_verified)) -> dict:
        if user["user_role"] not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=f"Requires role: {', '.join(roles)}"
            )
        return user

    return check_role
