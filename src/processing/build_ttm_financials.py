"""
Build TTM (trailing twelve months) financial metrics from financial_statements_quarterly.csv.

For each (ticker, earningsAnnouncementDate / anchorDate) event we:
  - Find all quarterly filings with filingDate <= anchorDate  (point-in-time, no look-ahead)
  - Sum the 4 most recent quarters for flow items (revenue, EBITDA, net income, FCF…)
  - Take the most-recent single quarter for stock items (total assets, equity, debt)

Output: data/processed/ttm_financials_enriched.csv
  One row per earnings event with TTM flow and latest-available stock metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

FS_FILE = RAW_DATA_DIR / "financial_statements_quarterly.csv"
EVENTS_FILE = PROCESSED_DATA_DIR / "valuation_features.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "ttm_financials_enriched.csv"

# Flow items: sum last 4 quarters
FLOW_COLS = ["revenue", "grossProfit", "ebitda", "operatingIncome", "netIncome",
             "operatingCashFlow", "freeCashFlow"]
# Stock items: use the latest available quarter
STOCK_COLS = ["totalAssets", "totalStockholdersEquity", "totalDebt",
              "cashAndShortTermInvestments"]


def run() -> None:
    if not FS_FILE.exists():
        print(f"Financial statements file not found: {FS_FILE}")
        print("Run src/ingestion/financial_statements.py first.")
        return

    print("Loading financial statements...")
    fs = pd.read_csv(FS_FILE, low_memory=False)
    fs["ticker"]      = fs["ticker"].astype(str).str.strip()
    fs["date"]        = pd.to_datetime(fs["date"],        errors="coerce")
    fs["filingDate"]  = pd.to_datetime(fs["filingDate"],  errors="coerce")
    for c in FLOW_COLS + STOCK_COLS:
        if c in fs.columns:
            fs[c] = pd.to_numeric(fs[c], errors="coerce")

    # Use date as fallback if filingDate is missing (some tickers lack filingDate)
    fs["effectiveDate"] = fs["filingDate"].fillna(fs["date"] + pd.Timedelta(days=60))
    fs = fs.dropna(subset=["ticker", "date"]).sort_values(["ticker", "effectiveDate"])
    print(f"  {len(fs):,} quarterly rows across {fs['ticker'].nunique():,} tickers")

    print("Loading earnings events...")
    ev = pd.read_csv(EVENTS_FILE, low_memory=False,
                     usecols=["ticker", "earningsAnnouncementDate", "anchorDate"])
    ev["ticker"]                  = ev["ticker"].astype(str).str.strip()
    ev["earningsAnnouncementDate"] = pd.to_datetime(ev["earningsAnnouncementDate"], errors="coerce")
    ev["anchorDate"]               = pd.to_datetime(ev["anchorDate"],               errors="coerce")
    ev = ev.dropna(subset=["ticker", "anchorDate"])
    print(f"  {len(ev):,} events across {ev['ticker'].nunique():,} tickers")

    # Build a dict of FS per ticker for fast lookup
    fs_by_ticker = {t: g.reset_index(drop=True) for t, g in fs.groupby("ticker")}

    out_rows: list[dict] = []
    ticker_groups = list(ev.groupby("ticker"))
    n = len(ticker_groups)

    for i, (ticker, grp) in enumerate(ticker_groups, 1):
        stmts = fs_by_ticker.get(ticker)
        if stmts is None or stmts.empty:
            for _, row in grp.iterrows():
                out_rows.append(_empty_row(ticker, row))
            continue

        for _, ev_row in grp.iterrows():
            anchor = ev_row["anchorDate"]
            # Only use filings available by anchor date
            avail = stmts[stmts["effectiveDate"] <= anchor].tail(4)
            out_rows.append(_compute_ttm(ticker, ev_row, avail))

        if i % 500 == 0 or i == n:
            print(f"  [{i}/{n}] tickers processed...", flush=True)

    out = pd.DataFrame(out_rows)
    out = out.sort_values(["ticker", "earningsAnnouncementDate"]).reset_index(drop=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(out):,} rows → {OUTPUT_FILE}")


def _compute_ttm(ticker: str, ev_row: pd.Series, avail: pd.DataFrame) -> dict:
    row: dict = {
        "ticker": ticker,
        "earningsAnnouncementDate": ev_row["earningsAnnouncementDate"],
        "anchorDate": ev_row["anchorDate"],
        "quarters_available": len(avail),
    }
    if avail.empty:
        for c in FLOW_COLS + STOCK_COLS:
            row[f"ttm_{c}"] = np.nan
        return row

    # TTM = sum of up to 4 quarters for flow items
    for c in FLOW_COLS:
        if c in avail.columns:
            vals = avail[c].dropna()
            row[f"ttm_{c}"] = vals.sum() if len(vals) >= 2 else np.nan
        else:
            row[f"ttm_{c}"] = np.nan

    # Latest quarter snapshot for stock items
    latest = avail.iloc[-1]
    for c in STOCK_COLS:
        row[f"ttm_{c}"] = latest.get(c, np.nan) if c in avail.columns else np.nan

    return row


def _empty_row(ticker: str, ev_row: pd.Series) -> dict:
    row: dict = {
        "ticker": ticker,
        "earningsAnnouncementDate": ev_row["earningsAnnouncementDate"],
        "anchorDate": ev_row["anchorDate"],
        "quarters_available": 0,
    }
    for c in FLOW_COLS + STOCK_COLS:
        row[f"ttm_{c}"] = np.nan
    return row


if __name__ == "__main__":
    run()
