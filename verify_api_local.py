"""Verify API local script."""

import requests

base_url = "http://127.0.0.1:8000"

def test_api():
    try:
        # Health check
        resp = requests.get(f"{base_url}/health", timeout=2)
        print(f"Health Check: {resp.status_code} {resp.json()}")

        # Pricing check (requires no auth in dev mode or we mock it)
        # Assuming local dev mode doesn't enforce strict auth for pricing or we use a demo endpoint
        # But wait, local_main has require_auth.
        # Let's try auth
        reg_data = {"email": "test_integration@example.com", "password": "password123", "role": "trader"}
        resp = requests.post(f"{base_url}/auth/register", json=reg_data, timeout=2)
        if resp.status_code == 409:
            print("User already exists, logging in...")
            login_data = {"email": "test_integration@example.com", "password": "password123"}
            resp = requests.post(f"{base_url}/auth/login", json=login_data, timeout=2)

        if resp.status_code == 200:
            token = resp.json().get("access_token")
            print("Auth successful")
            headers = {"Authorization": f"Bearer {token}"}

            data = {
                "spot": 100, "strike": 100, "rate": 0.05,
                "volatility": 0.2, "time_to_maturity": 1, "option_type": "call"
            }
            resp = requests.post(f"{base_url}/pricing/black-scholes", json=data, headers=headers, timeout=5)
            print(f"Pricing: {resp.status_code} {resp.text}")
        else:
            print(f"Auth failed: {resp.status_code} {resp.text}")

    except Exception as e:
        print(f"Pricing test failed: {e}")

if __name__ == "__main__":
    test_api()
