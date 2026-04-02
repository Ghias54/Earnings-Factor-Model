import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

# allow import from project root
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, FMP_API_KEY, FMP_BASE_URL


# =========================
# SETTINGS
# =========================
MAX_WORKERS = 12
REQUEST_TIMEOUT = 20
SAVE_EVERY = 250
TARGET_FILE = "earnings_from_price_start.csv"
START_DATE = pd.Timestamp("2019-01-01")
LIMIT = 200


print("Starting shares history pull...")

# =========================
# LOAD ONLY NEEDED TICKERS
# =========================
earnings = pd.read_csv(PROCESSED_DATA_DIR / TARGET_FILE)

tickers = (
    earnings["ticker"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)

print(f"Total tickers from {TARGET_FILE}: {len(tickers)}")


# =========================
# SINGLE-TICKER FETCH
# =========================
def fetch_shares_history(ticker: str):
    url = f"{FMP_BASE_URL}/income-statement"

    try:
        response = requests.get(
            url,
            params={
                "symbol": ticker,
                "period": "quarter",
                "limit": LIMIT,
                "apikey": FMP_API_KEY,
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, list) or len(data) == 0:
            return {
                "ticker": ticker,
                "rows": [],
                "error": "empty or invalid response",
            }

        rows = []
        for item in data:
            shares = item.get("weightedAverageShsOutDil")
            date = item.get("date")

            if shares is None or date is None:
                continue

            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "sharesOutstanding": shares,
                }
            )

        if len(rows) == 0:
            return {
                "ticker": ticker,
                "rows": [],
                "error": "no weightedAverageShsOutDil values found",
            }

        return {
            "ticker": ticker,
            "rows": rows,
            "error": None,
        }

    except Exception as e:
        return {
            "ticker": ticker,
            "rows": [],
            "error": str(e),
        }


# =========================
# OUTPUT PATHS
# =========================
shares_path = RAW_DATA_DIR / "shares_history.csv"
errors_path = RAW_DATA_DIR / "shares_history_errors.csv"


# =========================
# PARALLEL PULL
# =========================
all_rows = []
all_errors = []

start_time = time.time()
processed = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(fetch_shares_history, ticker): ticker for ticker in tickers}

    for future in as_completed(futures):
        result = future.result()
        processed += 1

        if result["error"] is None:
            all_rows.extend(result["rows"])
        else:
            all_errors.append(
                {
                    "ticker": result["ticker"],
                    "error": result["error"],
                }
            )

        if processed <= 10:
            print(f'{result["ticker"]}: pulled {len(result["rows"])} rows | error={result["error"]}')

        if processed % 50 == 0:
            elapsed = time.time() - start_time
            print(
                f"Processed {processed}/{len(tickers)} | "
                f"Rows: {len(all_rows)} | Errors: {len(all_errors)} | "
                f"Elapsed: {elapsed:.1f}s"
            )

        if processed % SAVE_EVERY == 0:
            temp_df = pd.DataFrame(all_rows)
            temp_err_df = pd.DataFrame(all_errors)

            if not temp_df.empty:
                temp_df["date"] = pd.to_datetime(temp_df["date"], errors="coerce")
                temp_df["sharesOutstanding"] = pd.to_numeric(temp_df["sharesOutstanding"], errors="coerce")

                temp_df = temp_df.dropna(subset=["date", "sharesOutstanding"])
                temp_df = temp_df[temp_df["sharesOutstanding"] > 0]
                temp_df = temp_df[temp_df["date"] >= START_DATE]

                temp_df = (
                    temp_df.sort_values(["ticker", "date"])
                    .drop_duplicates(subset=["ticker", "date"], keep="last")
                    .reset_index(drop=True)
                )

            temp_df.to_csv(shares_path, index=False)
            temp_err_df.to_csv(errors_path, index=False)
            print(f"Checkpoint saved at {processed} tickers")


print("\nFinished API pull")


# =========================
# FINAL CLEAN
# =========================
shares_df = pd.DataFrame(all_rows)
errors_df = pd.DataFrame(all_errors)

if not shares_df.empty:
    shares_df["date"] = pd.to_datetime(shares_df["date"], errors="coerce")
    shares_df["sharesOutstanding"] = pd.to_numeric(shares_df["sharesOutstanding"], errors="coerce")

    shares_df = shares_df.dropna(subset=["date", "sharesOutstanding"])
    shares_df = shares_df[shares_df["sharesOutstanding"] > 0]
    shares_df = shares_df[shares_df["date"] >= START_DATE]

    shares_df = (
        shares_df.sort_values(["ticker", "date"])
        .drop_duplicates(subset=["ticker", "date"], keep="last")
        .reset_index(drop=True)
    )


# =========================
# SAVE FINAL FILES
# =========================
shares_df.to_csv(shares_path, index=False)
errors_df.to_csv(errors_path, index=False)


# =========================
# FINAL PRINTS
# =========================
elapsed = time.time() - start_time

print("\nDONE")
print(f"Saved shares file: {shares_path}")
print(f"Saved errors file: {errors_path}")
print(f"Total shares rows: {len(shares_df)}")
print(f"Total errors: {len(errors_df)}")
print(f"Total time: {elapsed:.1f}s")

if not shares_df.empty:
    print("\nSample shares data:")
    print(shares_df.head(10))

    print("\nDate range:")
    print(shares_df['date'].min(), "->", shares_df['date'].max())

if not errors_df.empty:
    print("\nSample errors:")
    print(errors_df.head(10))