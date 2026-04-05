import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import ENV_FILE_PATH, FMP_API_KEY, FMP_BASE_URL, RAW_DATA_DIR, PROCESSED_DATA_DIR

OUTPUT_FILE = RAW_DATA_DIR / "analyst_estimates_quarterly.csv"
ERROR_FILE = RAW_DATA_DIR / "analyst_estimates_errors.csv"

# Parallel workers (I/O bound); actual throughput is capped by the rate limiter below.
MAX_WORKERS = 32
# FMP cap: 750/min — stay safely under (each HTTP GET counts, including pagination + retries).
RATE_LIMIT_CALLS_PER_MINUTE = 700
RATE_WINDOW_SECONDS = 60.0

# Frequent console updates (no disk) so it never looks "frozen".
HEARTBEAT_EVERY = 25
# Full error-file + flush append buffer this often (not full CSV rewrite of all rows).
CHECKPOINT_EVERY = 500

OUTPUT_COLUMNS = [
    "ticker",
    "date",
    "symbol",
    "estimatedEpsAvg",
    "estimatedRevenueAvg",
    "numAnalysts",
]


class SlidingWindowRateLimiter:
    """Thread-safe limiter: at most `max_calls` per rolling `window_seconds`."""

    def __init__(self, max_calls: int, window_seconds: float) -> None:
        self.max_calls = max_calls
        self.window = window_seconds
        self._lock = threading.Lock()
        self._times: List[float] = []

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


def load_cleaned_companies() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "companies_cleaned.csv"
    if not path.is_file():
        raise SystemExit(
            f"Missing {path.resolve()}\n"
            "Create it by running: PYTHONPATH=. python src/ingestion/companies.py\n"
            "then: PYTHONPATH=. python src/processing/clean_companies.py\n"
            "Or copy DataSets/companies_cleaned.csv to data/processed/companies_cleaned.csv"
        )
    return pd.read_csv(path)


def load_previous_errors_df() -> pd.DataFrame:
    if not ERROR_FILE.is_file():
        return pd.DataFrame(columns=["ticker", "error"])
    try:
        return pd.read_csv(ERROR_FILE)
    except Exception:
        return pd.DataFrame(columns=["ticker", "error"])


def tickers_in_error_log(df: pd.DataFrame) -> Set[str]:
    if df.empty or "ticker" not in df.columns:
        return set()
    return set(df["ticker"].dropna().astype(str).str.strip())


def tickers_already_in_output() -> Set[str]:
    """Unique tickers already written to the main CSV (for resume without duplicates)."""
    if not OUTPUT_FILE.is_file() or OUTPUT_FILE.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(OUTPUT_FILE, usecols=["ticker"])
        return set(df["ticker"].dropna().astype(str).str.strip().unique())
    except Exception:
        return set()


def fetch_analyst_estimates(
    session: requests.Session,
    ticker: str,
    limiter: SlidingWindowRateLimiter,
) -> Tuple[Optional[List[Any]], Optional[str]]:
    """Quarterly analyst estimate rows (paginated). One rate-limit slot per HTTP GET."""
    all_rows: List[Any] = []
    page = 0
    limit = 100

    while True:
        url = f"{FMP_BASE_URL}/analyst-estimates"
        params = {
            "symbol": ticker,
            "period": "quarter",
            "page": page,
            "limit": limit,
            "apikey": FMP_API_KEY,
        }
        chunk: Any = None
        for attempt in range(3):
            limiter.acquire()
            try:
                r = session.get(url, params=params, timeout=45)
                if r.status_code == 200:
                    chunk = r.json()
                    if isinstance(chunk, dict):
                        chunk = [chunk]
                    break
                if r.status_code in (429, 500, 502, 503):
                    time.sleep(2 * (attempt + 1))
                    continue
                return None, f"HTTP {r.status_code}"
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                else:
                    return None, str(e)
        else:
            return None, "Failed after retries"

        if not isinstance(chunk, list) or len(chunk) == 0:
            break

        all_rows.extend(chunk)
        if len(chunk) < limit:
            break
        page += 1

    return all_rows, None


def normalize_rows(ticker: str, data: List[Any]) -> List[Dict[str, Any]]:
    rows = []
    if not isinstance(data, list):
        return rows
    for row in data:
        if not isinstance(row, dict):
            continue
        d = {
            "ticker": ticker,
            "date": row.get("date"),
            "symbol": row.get("symbol"),
            "estimatedEpsAvg": row.get("estimatedEpsAvg")
            or row.get("estimatedEps")
            or row.get("epsAvg"),
            "estimatedRevenueAvg": row.get("estimatedRevenueAvg")
            or row.get("estimatedRevenue")
            or row.get("revenueAvg"),
            "numAnalysts": row.get("numAnalysts")
            or row.get("numberAnalystEstimatedEps")
            or row.get("numberAnalysts"),
        }
        rows.append(d)
    return rows


def _init_output_file() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(OUTPUT_FILE, index=False)


def _append_estimate_rows(rows: List[dict]) -> None:
    """Append rows without rewriting the whole file (avoids multi-minute pauses)."""
    if not rows:
        return
    pd.DataFrame(rows).to_csv(OUTPUT_FILE, mode="a", header=False, index=False)


def _merge_and_save_errors(
    previous_errors: pd.DataFrame,
    new_errors: List[dict],
) -> None:
    if not new_errors:
        if not previous_errors.empty:
            previous_errors.to_csv(ERROR_FILE, index=False)
        return
    new_df = pd.DataFrame(new_errors)
    if previous_errors.empty:
        merged = new_df
    else:
        merged = pd.concat([previous_errors, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["ticker"], keep="last")
    merged.to_csv(ERROR_FILE, index=False)


def _process_one_ticker(
    ticker: str,
    limiter: SlidingWindowRateLimiter,
) -> Tuple[str, List[Dict[str, Any]], Optional[Dict[str, str]]]:
    """One Session per thread (requests.Session is not thread-safe across threads)."""
    session = requests.Session()
    data, err = fetch_analyst_estimates(session, ticker, limiter)
    if err:
        return ticker, [], {"ticker": ticker, "error": err}
    rows = normalize_rows(ticker, data or [])
    if not rows:
        return ticker, [], {"ticker": ticker, "error": "No estimate rows"}
    return ticker, rows, None


def run(
    *,
    fresh: bool = False,
    skip_errors: bool = True,
    refetch_existing: bool = False,
) -> None:
    if not FMP_API_KEY:
        raise SystemExit(
            "FMP_API_KEY is missing or empty.\n"
            f"Add this file (exact path): {ENV_FILE_PATH.resolve()}\n"
            "With one line (no quotes needed): FMP_API_KEY=your_key_here\n"
            f"File exists: {ENV_FILE_PATH.is_file()}"
        )

    print(f"Output file (absolute path): {OUTPUT_FILE.resolve()}")
    print(
        f"Parallel: max_workers={MAX_WORKERS}, rate cap≈{RATE_LIMIT_CALLS_PER_MINUTE} HTTP GETs / "
        f"{RATE_WINDOW_SECONDS:.0f}s (under 750/min).\n"
        "Progress: heartbeat every "
        f"{HEARTBEAT_EVERY} tickers; append checkpoint every {CHECKPOINT_EVERY} tickers.\n"
    )

    companies_df = load_cleaned_companies()
    all_from_universe = companies_df["ticker"].dropna().astype(str).str.strip().tolist()

    previous_errors_df = load_previous_errors_df()

    if fresh:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.unlink(missing_ok=True)
        ERROR_FILE.unlink(missing_ok=True)
        previous_errors_df = pd.DataFrame(columns=["ticker", "error"])
        skip_tickers: Set[str] = set()
        print("--fresh: cleared output + error log; fetching full universe.\n")
    else:
        skip_tickers = set()
        if skip_errors:
            n_err = tickers_in_error_log(previous_errors_df)
            skip_tickers |= n_err
            print(
                f"Skipping {len(n_err)} tickers listed in {ERROR_FILE.name} "
                "(use --no-skip-errors to retry them)."
            )
        if not refetch_existing:
            n_out = tickers_already_in_output()
            skip_tickers |= n_out
            print(
                f"Skipping {len(n_out)} tickers already present in {OUTPUT_FILE.name} "
                "(use --refetch-existing to fetch them again; may duplicate rows)."
            )
        print()

    tickers = [t for t in all_from_universe if t not in skip_tickers]
    n_skipped = len(all_from_universe) - len(tickers)
    print(f"Universe: {len(all_from_universe)} | to fetch: {len(tickers)} | skipped: {n_skipped}\n")

    if not tickers:
        print("Nothing to fetch. Use --fresh for a full run or --no-skip-errors / --refetch-existing.")
        return

    if fresh or not OUTPUT_FILE.is_file() or OUTPUT_FILE.stat().st_size == 0:
        _init_output_file()
    else:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    limiter = SlidingWindowRateLimiter(RATE_LIMIT_CALLS_PER_MINUTE, RATE_WINDOW_SECONDS)
    all_errors: List[dict] = []

    done = 0
    total = len(tickers)
    total_estimate_rows = 0
    pending_rows: List[dict] = []

    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {
            executor.submit(_process_one_ticker, t, limiter): t for t in tickers
        }
        for future in as_completed(future_to_ticker):
            done += 1
            ticker = future_to_ticker[future]
            try:
                _, rows, err = future.result()
            except Exception as e:
                all_errors.append({"ticker": ticker, "error": repr(e)})
            else:
                if err:
                    all_errors.append(err)
                else:
                    pending_rows.extend(rows)
                    total_estimate_rows += len(rows)

            if done % HEARTBEAT_EVERY == 0 or done == total:
                elapsed = time.monotonic() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(
                    f"[{done}/{total}] heartbeat | estimate_rows≈{total_estimate_rows} "
                    f"err={len(all_errors)} | {rate:.1f} tickers/s",
                    flush=True,
                )

            if done % CHECKPOINT_EVERY == 0 or done == total:
                print(
                    f"[{done}/{total}] writing append + errors (buffer={len(pending_rows)} rows)...",
                    flush=True,
                )
                _append_estimate_rows(pending_rows)
                pending_rows.clear()
                _merge_and_save_errors(previous_errors_df, all_errors)
                print(
                    f"[{done}/{total}] checkpoint | total_rows≈{total_estimate_rows} err={len(all_errors)}",
                    flush=True,
                )

    _merge_and_save_errors(previous_errors_df, all_errors)
    print(f"Done. Total estimate rows written this run ≈{total_estimate_rows} → {OUTPUT_FILE}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fetch quarterly analyst estimates (FMP). Remembers failed tickers for skip on rerun."
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Delete output + error CSV and fetch the full universe from scratch.",
    )
    p.add_argument(
        "--no-skip-errors",
        action="store_true",
        help="Do not skip tickers that appear in analyst_estimates_errors.csv (retry failures).",
    )
    p.add_argument(
        "--refetch-existing",
        action="store_true",
        help="Also fetch tickers that already have rows in the main CSV (can duplicate rows).",
    )
    args = p.parse_args()
    run(
        fresh=args.fresh,
        skip_errors=not args.no_skip_errors,
        refetch_existing=args.refetch_existing,
    )


if __name__ == "__main__":
    main()
