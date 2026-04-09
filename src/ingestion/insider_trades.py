from __future__ import annotations

import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from threading import Lock
from pathlib import Path

import pandas as pd
import requests

# Allow direct execution: `python src/ingestion/insider_trades.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    COMPANIES_CLEANED_FILE,
    FMP_API_KEY,
    FMP_BASE_URL,
    INSIDER_TRADES_ERRORS_FILE,
    INSIDER_TRADES_RAW_FILE,
    RAW_DATA_DIR,
)

SLEEP_SECONDS = 0.10
CHECKPOINT_EVERY = 100
PAGE_SIZE = 100
MAX_PAGES_PER_TICKER = 5
MAX_WORKERS = 12
MAX_CALLS_PER_MINUTE = 720  # keep buffer below user limit (750/min)


def load_cleaned_companies() -> pd.DataFrame:
    return pd.read_csv(COMPANIES_CLEANED_FILE)


class _RateLimiter:
    def __init__(self, calls_per_minute: int) -> None:
        self._interval = 60.0 / float(calls_per_minute)
        self._next_allowed = 0.0
        self._lock = Lock()

    def wait_turn(self) -> None:
        with self._lock:
            now = time.perf_counter()
            if now < self._next_allowed:
                sleep_for = self._next_allowed - now
            else:
                sleep_for = 0.0
            self._next_allowed = max(now, self._next_allowed) + self._interval
        if sleep_for > 0:
            time.sleep(sleep_for)


def _fetch_page(session: requests.Session, limiter: _RateLimiter, ticker: str, page: int) -> tuple[object | None, str | None]:
    url = f"{FMP_BASE_URL}/insider-trading/search"
    params = {"symbol": ticker, "page": page, "limit": PAGE_SIZE, "apikey": FMP_API_KEY}
    for attempt in range(3):
        try:
            limiter.wait_turn()
            resp = session.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json(), None
            if resp.status_code in (429, 500, 502, 503):
                time.sleep(2 * (attempt + 1))
                continue
            return None, f"HTTP {resp.status_code}"
        except Exception as e:  # noqa: BLE001
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
            else:
                return None, str(e)
    return None, "Failed after retries"


def fetch_insider_for_ticker(
    session: requests.Session,
    limiter: _RateLimiter,
    ticker: str,
) -> tuple[list[dict], str | None]:
    all_rows: list[dict] = []
    for page in range(MAX_PAGES_PER_TICKER):
        data, err = _fetch_page(session, limiter, ticker, page)
        if err:
            return [], err
        rows = standardize_rows(ticker, data)
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < PAGE_SIZE:
            break
    return all_rows, None


def _to_float(v: object) -> float | None:
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x):
        return None
    return float(x)


def _norm_date(v: object) -> object:
    d = pd.to_datetime(v, errors="coerce")
    if pd.isna(d):
        return None
    return d.date().isoformat()


def standardize_rows(ticker: str, data: object) -> list[dict]:
    if not isinstance(data, list):
        return []

    out: list[dict] = []
    for r in data:
        shares = _to_float(
            r.get("securitiesTransacted")
            or r.get("shares")
            or r.get("amount")
            or r.get("transactionShares")
        )
        price = _to_float(r.get("price") or r.get("transactionPrice"))
        txn_value = _to_float(
            r.get("securitiesTransactedValue")
            or r.get("transactionValue")
            or r.get("marketValue")
            or r.get("value")
        )
        if txn_value is None and shares is not None and price is not None:
            txn_value = shares * price

        out.append(
            {
                "ticker": ticker,
                "transaction_date": _norm_date(r.get("transactionDate") or r.get("transaction_date")),
                "filing_date": _norm_date(
                    r.get("filingDate")
                    or r.get("acceptedDate")
                    or r.get("reportedDate")
                    or r.get("filing_date")
                ),
                "reporting_owner_name": r.get("reportingName")
                or r.get("reportingOwnerName")
                or r.get("ownerName"),
                "reporting_owner_title": r.get("typeOfOwner")
                or r.get("reportingOwnerTitle")
                or r.get("officerTitle")
                or r.get("title"),
                "transaction_type": r.get("transactionType") or r.get("acquisitionOrDisposition"),
                "transaction_code": r.get("transactionCode") or r.get("code"),
                "shares": shares,
                "price": price,
                "transaction_value": txn_value,
                "ownership_type": r.get("ownershipType") or r.get("directOrIndirectOwnership"),
                "is_officer": r.get("officer") if "officer" in r else r.get("isOfficer"),
                "is_director": r.get("director") if "director" in r else r.get("isDirector"),
                "is_ten_percent_owner": r.get("tenPercentOwner")
                if "tenPercentOwner" in r
                else r.get("isTenPercentOwner"),
                "raw_json": str(r),
            }
        )
    return out


def save_checkpoint(rows: list[dict], errors: list[dict]) -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(INSIDER_TRADES_RAW_FILE, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(INSIDER_TRADES_ERRORS_FILE, index=False)


def dedupe_rows(df: pd.DataFrame) -> pd.DataFrame:
    key = [
        "ticker",
        "transaction_date",
        "filing_date",
        "reporting_owner_name",
        "transaction_code",
        "shares",
        "price",
    ]
    return df.drop_duplicates(subset=key, keep="last").reset_index(drop=True)


def run() -> None:
    tickers = (
        load_cleaned_companies()["ticker"].dropna().astype(str).str.strip().drop_duplicates().tolist()
    )
    total = len(tickers)
    print(f"Total cleaned tickers: {total}")

    rows: list[dict] = []
    errors: list[dict] = []
    found_tickers = 0
    empty_tickers = 0
    done = 0
    limiter = _RateLimiter(MAX_CALLS_PER_MINUTE)
    with requests.Session() as session, ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        in_flight: dict[Future, str] = {}
        idx = 0
        while idx < total or in_flight:
            while idx < total and len(in_flight) < MAX_WORKERS * 3:
                t = tickers[idx]
                idx += 1
                fut = ex.submit(fetch_insider_for_ticker, session, limiter, t)
                in_flight[fut] = t

            done_set, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for fut in done_set:
                t = in_flight.pop(fut)
                done += 1
                print(f"[{done}/{total}] Fetched insider trades for {t}")
                try:
                    ticker_rows, err = fut.result()
                except Exception as e:  # noqa: BLE001
                    ticker_rows, err = [], str(e)
                if err:
                    errors.append({"ticker": t, "error": err})
                else:
                    if ticker_rows:
                        rows.extend(ticker_rows)
                        found_tickers += 1
                    else:
                        empty_tickers += 1

                if done % CHECKPOINT_EVERY == 0:
                    save_checkpoint(rows, errors)
                    print(
                        f"Checkpoint at {done} tickers | rows: {len(rows)} | "
                        f"errors: {len(errors)} | with_data: {found_tickers} | empty: {empty_tickers}"
                    )
            if SLEEP_SECONDS > 0:
                time.sleep(SLEEP_SECONDS)

    if rows:
        out = dedupe_rows(pd.DataFrame(rows))
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out.to_csv(INSIDER_TRADES_RAW_FILE, index=False)
        print(f"Saved {len(out):,} rows to {INSIDER_TRADES_RAW_FILE}")
    else:
        print("No insider rows fetched.")
    if errors:
        pd.DataFrame(errors).to_csv(INSIDER_TRADES_ERRORS_FILE, index=False)
    print(f"Saved {len(errors):,} errors to {INSIDER_TRADES_ERRORS_FILE}")
    print(
        f"Ticker summary | with_data: {found_tickers:,} | "
        f"empty: {empty_tickers:,} | errors: {len(errors):,}"
    )


if __name__ == "__main__":
    run()
