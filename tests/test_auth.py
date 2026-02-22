"""
BS-Opt Test Suite
Tests for authentication functions
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# Test User Models
# =============================================================================

class TestUserModels:
    """Tests for Pydantic user models"""

    def test_user_create_valid(self):
        from src.schemas.auth import UserCreate

        user = UserCreate(
            email="test@example.com",
            password="securepassword123",
            role="trader"
        )
        assert user.email == "test@example.com"
        assert user.role == "trader"

    def test_user_create_default_role(self):
        from src.schemas.auth import UserCreate

        user = UserCreate(email="test@example.com", password="password123")
        assert user.role == "trader"

    def test_user_create_invalid_email(self):
        from src.schemas.auth import UserCreate
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            UserCreate(email="not-an-email", password="password123")

    def test_user_create_short_password(self):
        from src.schemas.auth import UserCreate
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            UserCreate(email="test@example.com", password="short")

    def test_user_create_invalid_role(self):
        from src.schemas.auth import UserCreate
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            UserCreate(email="test@example.com", password="password123", role="invalid")


# =============================================================================
# Test Pricing Models
# =============================================================================

class TestPricingModels:
    """Tests for pricing request/response models"""

    def test_pricing_request_valid(self):
        from src.api.routers.pricing import PricingRequest

        req = PricingRequest(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.2,
            time_to_maturity=1.0,
            option_type="call"
        )
        assert req.spot == 100.0
        assert req.option_type == "call"

    def test_pricing_request_invalid_spot(self):
        from src.api.routers.pricing import PricingRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PricingRequest(
                spot=-100.0,  # Invalid: must be > 0
                strike=100.0,
                rate=0.05,
                volatility=0.2,
                time_to_maturity=1.0,
                option_type="call"
            )

    def test_pricing_request_invalid_option_type(self):
        from src.api.routers.pricing import PricingRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PricingRequest(
                spot=100.0,
                strike=100.0,
                rate=0.05,
                volatility=0.2,
                time_to_maturity=1.0,
                option_type="straddle"  # Invalid
            )


# =============================================================================
# Test OAuth URL Generation
# =============================================================================

class TestOAuthUrls:
    """Tests for OAuth URL handling"""

    def test_google_oauth_url_structure(self):
        """Verify Google OAuth URL structure"""
        from urllib.parse import urlencode

        GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
        params = {
            "client_id": "test-client-id",
            "redirect_uri": "http://localhost:8000/auth/google/callback",
            "response_type": "code",
            "scope": "openid email profile",
        }

        url = f"{GOOGLE_AUTH_URL}?{urlencode(params)}"

        assert "accounts.google.com" in url
        assert "client_id=test-client-id" in url
        assert "response_type=code" in url

    def test_github_oauth_url_structure(self):
        """Verify GitHub OAuth URL structure"""
        from urllib.parse import urlencode

        GITHUB_AUTH_URL = "https://github.com/login/oauth/authorize"
        params = {
            "client_id": "test-client-id",
            "redirect_uri": "http://localhost:8000/auth/github/callback",
            "scope": "read:user user:email",
        }

        url = f"{GITHUB_AUTH_URL}?{urlencode(params)}"

        assert "github.com" in url
        assert "client_id=test-client-id" in url


# =============================================================================
# Test Token Generation
# =============================================================================

class TestTokenGeneration:
    """Tests for token/secret generation"""

    def test_secure_token_generation(self):
        import secrets

        token = secrets.token_urlsafe(32)

        assert len(token) >= 32
        assert isinstance(token, str)

    def test_tokens_are_unique(self):
        import secrets

        tokens = [secrets.token_urlsafe(32) for _ in range(100)]

        assert len(set(tokens)) == 100  # All unique


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
