import pytest
from src.api.utils import get_password_hash, verify_password

def test_get_password_hash():
    password = "secret_password"
    hashed = get_password_hash(password)

    assert isinstance(hashed, str)
    assert hashed != password
    assert len(hashed) > 0

def test_get_password_hash_unique():
    password = "secret_password"
    hashed1 = get_password_hash(password)
    hashed2 = get_password_hash(password)

    # Bcrypt uses a salt, so hashes should be different
    assert hashed1 != hashed2

def test_verify_password_correct():
    password = "secret_password"
    hashed = get_password_hash(password)

    assert verify_password(password, hashed) is True

def test_verify_password_incorrect():
    password = "secret_password"
    wrong_password = "wrong_password"
    hashed = get_password_hash(password)

    assert verify_password(wrong_password, hashed) is False

@pytest.mark.parametrize("password", [
    "",
    "a" * 100,
    "!@#$%^&*()_+",
    " ",
    "12345678"
])
def test_password_edge_cases(password):
    hashed = get_password_hash(password)
    assert verify_password(password, hashed) is True
    assert hashed != password
