"""
Fetch quarterly income statement, balance sheet, and cash flow for every ticker
in companies_cleaned.csv and store as a single flat CSV.

Each row = one (ticker, fiscal quarter).  Three FMP calls per ticker:
  /income-statement     → revenue, grossProfit, ebitda, netIncome, eps
  /balance-sheet-statement → totalAssets, totalStockholdersEquity, totalDebt, cash
  /cash-flow-statement  → operatingCashFlow, capitalExpenditure, freeCashFlow

Resumable: tickers already present in the output file are skipped.
Uses 32 parallel threads with a 700 req/min sliding-window rate limiter.
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

OUTPUT_FILE = RAW_DATA_DIR / "financial_statements_quarterly.csv"
ERROR_FILE  = RAW_DATA_DIR / "financial_statements_errors.csv"

MAX_WORKERS = 32
RATE_LIMIT_CALLS_PER_MINUTE = 740   # 3 calls/ticker → ~247 tickers/min (safe under 750/min cap)
RATE_WINDOW_SECONDS = 60.0
QUARTERS_LIMIT = 24          # ~6 years of history
CHECKPOINT_EVERY = 200
HEARTBEAT_EVERY = 50

OUTPUT_COLUMNS = [
    "ticker", "date", "filingDate", "period",
    # income statement
    "revenue", "grossProfit", "ebitda", "operatingIncome", "netIncome", "epsDiluted",
    # balance sheet
    "totalAssets", "totalStockholdersEquity", "totalDebt", "cashAndShortTermInvestments",
    # cash flow
    "operatingCashFlow", "capitalExpenditure", "freeCashFlow",
]


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


_limiter = SlidingWindowRateLimiter(RATE_LIMIT_CALLS_PER_MINUTE, RATE_WINDOW_SECONDS)
_write_lock = threading.Lock()


def _get(session: requests.Session, endpoint: str, ticker: str) -> list[dict] | None:
    """Single FMP GET with rate limiting and retries."""
    url = f"{FMP_BASE_URL}/{endpoint}"
    params = {"symbol": ticker, "period": "quarter", "limit": QUARTERS_LIMIT, "apikey": FMP_API_KEY}
    for attempt in range(3):
        _limiter.acquire()
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json()
                return data if isinstance(data, list) else None
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 * (attempt + 1))
                continue
            return None
        except Exception:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    return None


def fetch_ticker(session: requests.Session, ticker: str) -> tuple[list[dict], str | None]:
    """Fetch all three statements and merge by date. Bail early if no income statement."""
    inc = _get(session, "income-statement", ticker)
    if not inc:
        return [], f"{ticker}: no income statement"

    bal = _get(session, "balance-sheet-statement", ticker)
    cf  = _get(session, "cash-flow-statement",     ticker)

    inc_map = {r["date"]: r for r in inc}
    bal_map = {r["date"]: r for r in bal} if bal else {}
    cf_map  = {r["date"]: r for r in cf}  if cf  else {}

    rows: list[dict] = []
    for date, i in inc_map.items():
        b = bal_map.get(date, {})
        c = cf_map.get(date, {})
        rows.append({
            "ticker":   ticker,
            "date":     date,
            "filingDate": i.get("filingDate") or i.get("acceptedDate"),
            "period":   i.get("period"),
            # income statement
            "revenue":         i.get("revenue"),
            "grossProfit":     i.get("grossProfit"),
            "ebitda":          i.get("ebitda"),
            "operatingIncome": i.get("operatingIncome"),
            "netIncome":       i.get("netIncome") or i.get("bottomLineNetIncome"),
            "epsDiluted":      i.get("epsDiluted"),
            # balance sheet
            "totalAssets":              b.get("totalAssets"),
            "totalStockholdersEquity":  b.get("totalStockholdersEquity"),
            "totalDebt":                b.get("totalDebt"),
            "cashAndShortTermInvestments": b.get("cashAndShortTermInvestments"),
            # cash flow
            "operatingCashFlow":  c.get("netCashProvidedByOperatingActivities") or c.get("operatingCashFlow"),
            "capitalExpenditure": c.get("investmentsInPropertyPlantAndEquipment") or c.get("capitalExpenditure"),
            "freeCashFlow":       c.get("freeCashFlow"),
        })

    return rows, None


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
    """Return tickers already successfully fetched OR confirmed to have no data."""
    done: set[str] = set()
    if OUTPUT_FILE.exists():
        df = pd.read_csv(OUTPUT_FILE, usecols=["ticker"], on_bad_lines="skip", low_memory=False)
        done.update(df["ticker"].dropna().astype(str).str.strip().unique())
    # Also skip permanent failures (no income statement = no data on FMP, not a transient error)
    if ERROR_FILE.exists():
        ef = pd.read_csv(ERROR_FILE, on_bad_lines="skip", low_memory=False)
        if "error" in ef.columns:
            permanent = ef[ef["error"].str.contains("no income statement", na=False)]
            done.update(permanent["ticker"].dropna().astype(str).str.strip().unique())
    return done


def run() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    companies = pd.read_csv(PROCESSED_DATA_DIR / "companies_cleaned.csv")
    all_tickers = companies["ticker"].dropna().astype(str).str.strip().tolist()

    done = get_completed_tickers()
    tickers = [t for t in all_tickers if t not in done]

    print(f"Universe:   {len(all_tickers):,} tickers")
    print(f"Completed:  {len(done):,}")
    print(f"Remaining:  {len(tickers):,}")
    print(f"Workers:    {MAX_WORKERS}  |  Rate limit: {RATE_LIMIT_CALLS_PER_MINUTE}/min")
    print(f"Output:     {OUTPUT_FILE}")

    completed = 0
    batch_rows: list[dict] = []
    batch_errors: list[dict] = []
    total_rows = 0

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
                print(f"  [{completed}/{len(tickers)}] in-memory rows={len(batch_rows)}  errors={len(batch_errors)}", flush=True)

            if completed % CHECKPOINT_EVERY == 0:
                _append_csv(batch_rows, OUTPUT_FILE, OUTPUT_COLUMNS)
                _append_errors(batch_errors, ERROR_FILE)
                total_rows += len(batch_rows)
                print(f"  Checkpoint: {completed} tickers done, {total_rows} rows written")
                batch_rows, batch_errors = [], []

    _append_csv(batch_rows, OUTPUT_FILE, OUTPUT_COLUMNS)
    _append_errors(batch_errors, ERROR_FILE)
    total_rows += len(batch_rows)
    print(f"\nDone. {completed:,} tickers processed, {total_rows:,} quarterly rows written.")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
