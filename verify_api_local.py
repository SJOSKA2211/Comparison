# pylint: disable=C0114, C0116, C0301, W1514, E0401
# pylint: disable=missing-module-docstring, missing-function-docstring, import-error, broad-exception-caught

import requests

BASE_URL = "http://127.0.0.1:8000"

def check_health():
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=2)
        print(f"Health: {resp.status_code} {resp.json()}")
        return True
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_pricing():
    try:
        data = {
            "spot": 100, "strike": 100, "rate": 0.05,
            "volatility": 0.2, "time_to_maturity": 1, "option_type": "call"
        }
        # First register/login to get token if needed, but local_main might have a demo endpoint
        # Or just try pricing if auth is disabled for dev (it's not, require_auth is there)

        # Let's try auth
        reg_data = {"email": "test_integration@example.com", "password": "password123", "role": "trader"}
        resp = requests.post(f"{BASE_URL}/auth/register", json=reg_data, timeout=2)
        if resp.status_code == 409:
            login_data = {"email": "test_integration@example.com", "password": "password123"}
            resp = requests.post(f"{BASE_URL}/auth/login", json=login_data, timeout=2)

        if resp.status_code == 200:
            token = resp.json().get("access_token")
            headers = {"Authorization": f"Bearer {token}"}
            price_resp = requests.post(f"{BASE_URL}/pricing/black-scholes", json=data, headers=headers, timeout=2)
            print(f"Pricing: {price_resp.status_code} {price_resp.json()}")
        else:
            print(f"Auth failed: {resp.status_code} {resp.text}")

    except Exception as e:
        print(f"Pricing test failed: {e}")

if __name__ == "__main__":
    if check_health():
        test_pricing()
