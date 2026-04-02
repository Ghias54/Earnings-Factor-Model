import sys
from pathlib import Path
import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import RAW_DATA_DIR, FMP_API_KEY, FMP_BASE_URL


# =========================
# SETTINGS
# =========================
ticker = "ABGSF"
REQUEST_TIMEOUT = 20
START_DATE = "2019-01-01"

print(f"Starting single-ticker shares history test for: {ticker}")


# =========================
# FETCH (QUARTERLY)
# =========================
def fetch_shares_history(ticker: str):
    url = f"{FMP_BASE_URL}/income-statement"

    try:
        response = requests.get(
            url,
            params={
                "symbol": ticker,
                "period": "quarter",   # 🔥 KEY CHANGE
                "limit": 200,          # get more history
                "apikey": FMP_API_KEY,
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, list) or len(data) == 0:
            return {"rows": [], "error": "empty response"}

        rows = []
        for item in data:
            shares = item.get("weightedAverageShsOutDil")
            date = item.get("date")

            if shares is None or date is None:
                continue

            rows.append({
                "ticker": ticker,
                "date": date,
                "sharesOutstanding": shares,
            })

        return {"rows": rows, "error": None}

    except Exception as e:
        return {"rows": [], "error": str(e)}


# =========================
# RUN
# =========================
result = fetch_shares_history(ticker)

print(f"Rows pulled (raw): {len(result['rows'])}")
print(f"Error: {result['error']}")


# =========================
# CLEAN
# =========================
df = pd.DataFrame(result["rows"])

if not df.empty:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sharesOutstanding"] = pd.to_numeric(df["sharesOutstanding"], errors="coerce")

    # 🔥 FILTER TO 2019+
    df = df[df["date"] >= START_DATE]

    df = df.dropna(subset=["date", "sharesOutstanding"])
    df = df[df["sharesOutstanding"] > 0]

    df = (
        df.sort_values(["ticker", "date"])
        .drop_duplicates(subset=["ticker", "date"], keep="last")
        .reset_index(drop=True)
    )


# =========================
# SAVE
# =========================
output_path = RAW_DATA_DIR / f"test_shares_history_{ticker}.csv"
df.to_csv(output_path, index=False)

print(f"\nSaved to: {output_path}")

if not df.empty:
    print("\nPreview:")
    print(df.head(10))

    print("\nDate range:")
    print(df["date"].min(), "->", df["date"].max())

    print("\nTotal cleaned rows:")
    print(len(df))
else:
    print("No data after filtering.")