"""
Build a daily (ticker × trading date) panel: OHLCV plus factor/composite scores known as-of each day.

Scores from `composite_quant_scores.csv` are stamped on `earningsAnnouncementDate` (cross-section
at each report). For each calendar day we carry **backward** the latest score whose announcement
date is on or before that day — i.e. point-in-time: no look-ahead to the upcoming earnings row.

This is the right object for research and for backtests that enter **before** announcement: the
simulator joins composite on `buyDate` using the same rule (see `simulate_portfolio.load_and_merge`).

Universe / survivorship:
- Rows are every (ticker, date) present in `daily_prices_from_clean_universe.csv`. If that file only
  contains currently listed names, you still have survivorship bias at **ingestion**; this script
  does not drop names by a separate “live” list. Include delisted/history in raw prices to mitigate.

Output default: gzip-compressed CSV (large).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from config import COMPOSITE_SCORES_FILE, MASTER_DAILY_QUANT_PANEL_FILE, RAW_DATA_DIR

PRICES_FILE = RAW_DATA_DIR / "daily_prices_from_clean_universe.csv"
COMPOSITE_FILE = COMPOSITE_SCORES_FILE
DEFAULT_OUT = MASTER_DAILY_QUANT_PANEL_FILE


def load_prices(tickers: set[str] | None) -> pd.DataFrame:
    usecols = ["ticker", "date", "open", "high", "low", "close", "volume"]
    df = pd.read_csv(PRICES_FILE, usecols=lambda c: c in usecols, low_memory=False)
    df = df[df["ticker"] != "ticker"].copy()
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ticker", "date"]).sort_values(["ticker", "date"])
    if tickers is not None:
        df = df[df["ticker"].isin(tickers)].copy()
    return df


def load_composite() -> pd.DataFrame:
    df = pd.read_csv(COMPOSITE_FILE, low_memory=False)
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["earningsAnnouncementDate"] = pd.to_datetime(
        df["earningsAnnouncementDate"], errors="coerce"
    )
    df = df.dropna(subset=["ticker", "earningsAnnouncementDate"])
    df = df.sort_values(["ticker", "earningsAnnouncementDate"])
    df = df.rename(
        columns={"earningsAnnouncementDate": "composite_source_announcement_date"}
    )
    return df


def run(
    *,
    output_path: Path,
    tickers: set[str] | None,
    compression: str | None,
) -> None:
    if not PRICES_FILE.exists():
        raise SystemExit(f"Missing prices file: {PRICES_FILE}")
    if not COMPOSITE_FILE.exists():
        raise SystemExit(f"Missing {COMPOSITE_FILE}. Run build_quant_scores.py first.")

    print("Loading prices...")
    prices = load_prices(tickers)
    print(f"  Price rows: {len(prices):,}")

    print("Loading composite scores...")
    comp = load_composite()
    print(f"  Composite rows: {len(comp):,}")

    # Only attach scores for tickers we have prices for (and optional filter)
    if tickers is None:
        tix = set(prices["ticker"].unique())
    else:
        tix = tickers
    comp = comp[comp["ticker"].isin(tix)].copy()

    # merge_asof with `by` still requires valid ordering; per-ticker merges avoid
    # "left keys must be sorted" when dates are not globally monotonic across tickers.
    parts: list[pd.DataFrame] = []
    for t, pgrp in prices.groupby("ticker", sort=False):
        pgrp = pgrp.sort_values("date", kind="mergesort")
        cgrp = comp[comp["ticker"] == t].drop(columns=["ticker"]).sort_values(
            "composite_source_announcement_date", kind="mergesort"
        )
        if cgrp.empty:
            out = pgrp.copy()
            for col in comp.columns:
                if col == "ticker":
                    continue
                if col not in out.columns:
                    out[col] = np.nan
            parts.append(out)
            continue
        m = pd.merge_asof(
            pgrp,
            cgrp,
            left_on="date",
            right_on="composite_source_announcement_date",
            direction="backward",
        )
        parts.append(m)
    merged = pd.concat(parts, ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, compression=compression)
    print(f"Saved {len(merged):,} rows to {output_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output path (default: {DEFAULT_OUT})",
    )
    p.add_argument(
        "--tickers-file",
        type=Path,
        default=None,
        help="Optional newline or comma-separated tickers to restrict (smaller panel for tests).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    tickers: set[str] | None = None
    if args.tickers_file is not None:
        raw = args.tickers_file.read_text(encoding="utf-8", errors="replace")
        parts = [x.strip() for x in raw.replace(",", "\n").splitlines() if x.strip()]
        tickers = set(parts)
        print(f"Restricting to {len(tickers)} tickers from {args.tickers_file}")

    comp = "gzip" if str(args.output).endswith(".gz") else None
    run(output_path=args.output, tickers=tickers, compression=comp)


if __name__ == "__main__":
    main()
