import requests
from config import FMP_API_KEY, FMP_BASE_URL

def test_fmp_connection() -> None:
    print("TEST KEY:", FMP_API_KEY)

    url = f"{FMP_BASE_URL}/profile"
    params = {
        "symbol": "AAPL",
        "apikey": FMP_API_KEY,
    }

    response = requests.get(url, params=params, timeout=30)
    print("FINAL URL:", response.url)
    response.raise_for_status()

    data = response.json()
    print("Status code:", response.status_code)
    print(data[0] if data else "No data returned.")

if __name__ == "__main__":
    test_fmp_connection()