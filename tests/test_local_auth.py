import os
import sqlite3
import pytest
from fastapi.testclient import TestClient
import src.api.local_main

# Use a unique DB name for testing to avoid conflicts
TEST_DB = "test_bsopt_auth.db"

@pytest.fixture(scope="module")
def test_db():
    # Setup
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

    conn = sqlite3.connect(TEST_DB)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT,
        role TEXT DEFAULT 'trader',
        email_verified INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS sessions (
        user_id TEXT,
        token TEXT,
        expires_at TEXT
    );
    """)
    conn.close()

    # Monkeypatch the DATABASE_PATH in the module
    original_db_path = src.api.local_main.DATABASE_PATH
    src.api.local_main.DATABASE_PATH = TEST_DB

    yield TEST_DB

    # Teardown
    src.api.local_main.DATABASE_PATH = original_db_path
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

@pytest.fixture(scope="module")
def client(test_db):
    with TestClient(src.api.local_main.app) as c:
        yield c

def test_password_hashing_security(client):
    email = "security_test@example.com"
    password = "secure_password_123"

    # Register
    response = client.post("/auth/register", json={
        "email": email,
        "password": password,
        "role": "trader"
    })

    assert response.status_code == 200

    # Check the stored hash in the database
    conn = sqlite3.connect(TEST_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    password_hash = row[0]

    # Verify it is NOT SHA256 (64 hex chars)
    # And verify it IS bcrypt (starts with $2b$ or $2a$)
    is_sha256 = len(password_hash) == 64 and all(c in '0123456789abcdef' for c in password_hash)
    assert not is_sha256, "Password should NOT be stored as SHA256"
    assert password_hash.startswith("$2b$") or password_hash.startswith("$2a$"), "Password should be stored as BCrypt hash"

    # Test Login
    login_response = client.post("/auth/login", json={
        "email": email,
        "password": password
    })
    assert login_response.status_code == 200, "Login should succeed with correct password"

    # Test Login Failure
    fail_response = client.post("/auth/login", json={
        "email": email,
        "password": "wrong_password"
    })
    assert fail_response.status_code == 401, "Login should fail with incorrect password"
