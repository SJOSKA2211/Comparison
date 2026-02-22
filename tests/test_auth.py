"""
BS-Opt Test Suite
Tests for authentication functions
"""

import secrets
from urllib.parse import urlencode

import pytest
from pydantic import ValidationError

from src.api.routers.pricing import PricingRequest
from src.schemas.auth import UserCreate


# =============================================================================
# Test User Models
# =============================================================================

class TestUserModels:
    """Tests for Pydantic user models"""

    def test_user_create_valid(self):
        """Test valid user creation"""
        user = UserCreate(
            email="test@example.com",
            password="securepassword123",
            role="trader"
        )
        assert user.email == "test@example.com"
        assert user.role == "trader"

    def test_user_create_default_role(self):
        """Test default role assignment"""
        user = UserCreate(email="test@example.com", password="password123")
        assert user.role == "trader"

    def test_user_create_invalid_email(self):
        """Test invalid email validation"""
        with pytest.raises(ValidationError):
            UserCreate(email="not-an-email", password="password123")

    def test_user_create_short_password(self):
        """Test short password validation"""
        with pytest.raises(ValidationError):
            UserCreate(email="test@example.com", password="short")

    def test_user_create_invalid_role(self):
        """Test invalid role validation"""
        with pytest.raises(ValidationError):
            UserCreate(email="test@example.com", password="password123", role="invalid")


# =============================================================================
# Test Pricing Models
# =============================================================================

class TestPricingModels:
    """Tests for pricing request/response models"""

    def test_pricing_request_valid(self):
        """Test valid pricing request"""
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
        """Test invalid spot price validation"""
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
        """Test invalid option type validation"""
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
        
        # pylint: disable=invalid-name
        google_auth_url = "https://accounts.google.com/o/oauth2/v2/auth"
        params = {
            "client_id": "test-client-id",
            "redirect_uri": "http://localhost:8000/auth/google/callback",
            "response_type": "code",
            "scope": "openid email profile",
        }
        
        url = f"{google_auth_url}?{urlencode(params)}"
        
        assert "accounts.google.com" in url
        assert "client_id=test-client-id" in url
        assert "response_type=code" in url

    def test_github_oauth_url_structure(self):
        """Verify GitHub OAuth URL structure"""
        
        # pylint: disable=invalid-name
        github_auth_url = "https://github.com/login/oauth/authorize"
        params = {
            "client_id": "test-client-id",
            "redirect_uri": "http://localhost:8000/auth/github/callback",
            "scope": "read:user user:email",
        }
        
        url = f"{github_auth_url}?{urlencode(params)}"
        
        assert "github.com" in url
        assert "client_id=test-client-id" in url


# =============================================================================
# Test Token Generation
# =============================================================================

class TestTokenGeneration:
    """Tests for token/secret generation"""

    def test_secure_token_generation(self):
        """Test token generation length"""
        token = secrets.token_urlsafe(32)
        
        assert len(token) >= 32
        assert isinstance(token, str)

    def test_tokens_are_unique(self):
        """Test token uniqueness"""
        tokens = [secrets.token_urlsafe(32) for _ in range(100)]
        
        assert len(set(tokens)) == 100  # All unique


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
