"""
Fetch full daily OHLCV price history for every ticker in companies_cleaned.csv.

Uses 32 parallel threads with a 740 req/min sliding-window rate limiter
(safe under the 750/min FMP cap).  One API call per ticker.

Resumable: tickers already present in the output file are skipped.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import FMP_API_KEY, FMP_BASE_URL, RAW_DATA_DIR, PROCESSED_DATA_DIR

OUTPUT_FILE = RAW_DATA_DIR / "daily_prices_from_clean_universe.csv"
ERROR_FILE  = RAW_DATA_DIR / "daily_prices_from_clean_universe_errors.csv"

MAX_WORKERS = 32
RATE_LIMIT_CALLS_PER_MINUTE = 740   # safe under 750/min FMP cap
RATE_WINDOW_SECONDS = 60.0
CHECKPOINT_EVERY = 500
HEARTBEAT_EVERY  = 100

OUTPUT_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume"]


class SlidingWindowRateLimiter:
    def __init__(self, max_calls: int, window_seconds: float) -> None:
        self.max_calls = max_calls
        self.window = window_seconds
        self._lock = threading.Lock()
        self._times: list[float] = []

    def acquire(self) -> None:
        while True:
            wait = 0.0
            with self._lock:
                now = time.monotonic()
                cutoff = now - self.window
                self._times = [t for t in self._times if t > cutoff]
                if len(self._times) < self.max_calls:
                    self._times.append(now)
                    return
                wait = self.window - (now - self._times[0]) + 0.002
            time.sleep(max(wait, 0.01))


_limiter    = SlidingWindowRateLimiter(RATE_LIMIT_CALLS_PER_MINUTE, RATE_WINDOW_SECONDS)
_write_lock = threading.Lock()


def fetch_ticker(session: requests.Session, ticker: str) -> tuple[list[dict], str | None]:
    url    = f"{FMP_BASE_URL}/historical-price-eod/full"
    params = {"symbol": ticker, "apikey": FMP_API_KEY}

    for attempt in range(3):
        _limiter.acquire()
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json()
                if not isinstance(data, list) or not data:
                    return [], "No price data returned"
                rows = [
                    {
                        "ticker": ticker,
                        "date":   row.get("date"),
                        "open":   row.get("open"),
                        "high":   row.get("high"),
                        "low":    row.get("low"),
                        "close":  row.get("close"),
                        "volume": row.get("volume"),
                    }
                    for row in data
                ]
                return rows, None
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 * (attempt + 1))
                continue
            return [], f"HTTP {r.status_code}"
        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
            else:
                return [], str(e)

    return [], "Failed after retries"


def _append_csv(rows: list[dict], path: Path, columns: list[str]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows, columns=columns)
    with _write_lock:
        df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _append_errors(errors: list[dict], path: Path) -> None:
    if not errors:
        return
    df = pd.DataFrame(errors)
    with _write_lock:
        df.to_csv(path, mode="a", header=not path.exists(), index=False)


def get_completed_tickers() -> set[str]:
    """Return tickers already fetched OR confirmed to have no price data."""
    done: set[str] = set()
    if OUTPUT_FILE.exists():
        df = pd.read_csv(OUTPUT_FILE, usecols=["ticker"], on_bad_lines="skip", low_memory=False)
        df = df[df["ticker"] != "ticker"]
        done.update(df["ticker"].dropna().astype(str).str.strip().unique())
    if ERROR_FILE.exists():
        ef = pd.read_csv(ERROR_FILE, on_bad_lines="skip", low_memory=False)
        if "error" in ef.columns:
            permanent = ef[ef["error"].str.contains("No price data|HTTP 4", na=False)]
            done.update(permanent["ticker"].dropna().astype(str).str.strip().unique())
    return done


def run() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    companies = pd.read_csv(PROCESSED_DATA_DIR / "companies_cleaned.csv")
    all_tickers = companies["ticker"].dropna().astype(str).str.strip().tolist()

    done    = get_completed_tickers()
    tickers = [t for t in all_tickers if t not in done]

    print(f"Universe:   {len(all_tickers):,} tickers")
    print(f"Completed:  {len(done):,}")
    print(f"Remaining:  {len(tickers):,}")
    print(f"Workers:    {MAX_WORKERS}  |  Rate: {RATE_LIMIT_CALLS_PER_MINUTE}/min (~{len(tickers)/RATE_LIMIT_CALLS_PER_MINUTE:.0f} min)")
    print(f"Output:     {OUTPUT_FILE}")

    completed   = 0
    batch_rows:   list[dict] = []
    batch_errors: list[dict] = []
    total_rows    = 0

    with requests.Session() as session, ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_ticker, session, t): t for t in tickers}

        for future in as_completed(futures):
            ticker = futures[future]
            completed += 1
            try:
                rows, err = future.result()
                if err:
                    batch_errors.append({"ticker": ticker, "error": err})
                elif rows:
                    batch_rows.extend(rows)
                else:
                    batch_errors.append({"ticker": ticker, "error": "empty"})
            except Exception as e:
                batch_errors.append({"ticker": ticker, "error": str(e)})

            if completed % HEARTBEAT_EVERY == 0:
                pct = 100 * completed / len(tickers)
                print(f"  [{completed}/{len(tickers)}] {pct:.1f}%  rows_buf={len(batch_rows):,}  errors={len(batch_errors)}", flush=True)

            if completed % CHECKPOINT_EVERY == 0:
                _append_csv(batch_rows, OUTPUT_FILE, OUTPUT_COLUMNS)
                _append_errors(batch_errors, ERROR_FILE)
                total_rows += len(batch_rows)
                print(f"  Checkpoint: {completed} tickers, {total_rows:,} total rows written")
                batch_rows, batch_errors = [], []

    _append_csv(batch_rows, OUTPUT_FILE, OUTPUT_COLUMNS)
    _append_errors(batch_errors, ERROR_FILE)
    total_rows += len(batch_rows)
    print(f"\nDone. {completed:,} tickers processed, {total_rows:,} price rows written.")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
