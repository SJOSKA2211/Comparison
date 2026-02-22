import requests
import sys

def test_api():
    base_url = "http://127.0.0.1:8000"
    try:
        # Health check
        resp = requests.get(f"{base_url}/health", timeout=2)
        print(f"Health Check: {resp.status_code} {resp.json()}")

        # Auth Register
        reg_data = {"email": "test_api@example.com", "password": "password123", "role": "trader"}
        resp = requests.post(f"{base_url}/auth/register", json=reg_data, timeout=2)
        if resp.status_code == 409:
            print("User already exists")
            # Login instead
            login_data = {"email": "test_api@example.com", "password": "password123"}
            resp = requests.post(f"{base_url}/auth/login", json=login_data, timeout=2)

        print(f"Auth: {resp.status_code}")
        token = resp.json().get("access_token")

        if token:
            headers = {"Authorization": f"Bearer {token}"}
            # Pricing check
            price_data = {
               "spot": 100, "strike": 100, "rate": 0.05,
               "volatility": 0.2, "time_to_maturity": 1, "option_type": "call"
            }
            resp = requests.post(f"{base_url}/pricing/black-scholes", json=price_data, headers=headers, timeout=10)
            print(f"Pricing: {resp.status_code} {resp.json().get('price')}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    with open("api_test_results.txt", "w") as f:
        sys.stdout = f
        test_api()
