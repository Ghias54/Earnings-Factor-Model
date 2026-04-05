import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import FMP_API_KEY, FMP_BASE_URL, RAW_DATA_DIR, PROCESSED_DATA_DIR

OUTPUT_FILE = RAW_DATA_DIR / "company_profiles.csv"
ERROR_FILE = RAW_DATA_DIR / "company_profiles_errors.csv"
SLEEP_SECONDS = 0.12
CHECKPOINT_EVERY = 100


def load_cleaned_companies() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "companies_cleaned.csv"
    return pd.read_csv(path)


def fetch_profile(session: requests.Session, ticker: str):
    url = f"{FMP_BASE_URL}/profile"
    params = {"symbol": ticker, "apikey": FMP_API_KEY}
    for attempt in range(3):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json(), None
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 * (attempt + 1))
                continue
            return None, f"HTTP {r.status_code}"
        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
            else:
                return None, str(e)
    return None, "Failed after retries"


def flatten_profile(ticker: str, data: Any) -> Optional[dict]:
    if not isinstance(data, list) or not data:
        return None
    row = data[0]
    if not isinstance(row, dict):
        return None
    out = {"ticker": ticker}
    for key in (
        "companyName",
        "cik",
        "exchange",
        "exchangeShortName",
        "sector",
        "industry",
        "country",
        "currency",
        "ipoDate",
        "isActivelyTrading",
        "isEtf",
        "isAdr",
        "marketCap",
        "price",
        "beta",
        "lastDividend",
    ):
        if key in row:
            out[key] = row.get(key)
    return out


def save_checkpoint(rows: list, errors: list) -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(ERROR_FILE, index=False)


def run() -> None:
    if not FMP_API_KEY:
        raise SystemExit("FMP_API_KEY missing in environment.")

    companies_df = load_cleaned_companies()
    tickers = companies_df["ticker"].dropna().astype(str).str.strip().tolist()

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    with requests.Session() as session:
        for i, ticker in enumerate(tickers, start=1):
            print(f"[{i}/{len(tickers)}] profile {ticker}")
            data, err = fetch_profile(session, ticker)
            if err:
                errors.append({"ticker": ticker, "error": err})
            else:
                flat = flatten_profile(ticker, data)
                if flat:
                    rows.append(flat)
                else:
                    errors.append({"ticker": ticker, "error": "Empty profile"})

            if i % CHECKPOINT_EVERY == 0:
                save_checkpoint(rows, errors)
                print(f"Checkpoint {i} | ok={len(rows)} err={len(errors)}")

            time.sleep(SLEEP_SECONDS)

    save_checkpoint(rows, errors)
    print(f"Done. Saved {len(rows)} profiles to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
