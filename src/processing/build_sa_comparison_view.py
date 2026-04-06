"""
Build Seeking-Alpha-style comparison tables for a ticker.

Outputs (under data/processed):
1) sa_compare_<TICKER>_history.csv
   Columns: Date, Quant Rating, Quant Score, Valuation, Growth,
            Profitability, Momentum, EPS Rev.
2) sa_compare_<TICKER>_snapshot.csv
   Rows: Quant Rating / Quant Score / factor grades
   Columns: Now, 30d Ago, 6m Ago
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DATA_DIR

COMPOSITE_FILE = PROCESSED_DATA_DIR / "composite_quant_scores.csv"
FACTOR_FILES = {
    "Valuation": PROCESSED_DATA_DIR / "valuation_scores.csv",
    "Growth": PROCESSED_DATA_DIR / "growth_scores.csv",
    "Profitability": PROCESSED_DATA_DIR / "profitability_scores.csv",
    "Momentum": PROCESSED_DATA_DIR / "momentum_scores.csv",
    "EPS Rev.": PROCESSED_DATA_DIR / "revisions_scores.csv",
}


def quant_label_from_grade(grade: object) -> str:
    g = str(grade).strip() if pd.notna(grade) else ""
    if g in {"A+", "A"}:
        return "Strong Buy"
    if g in {"A-", "B+"}:
        return "Buy"
    if g in {"B", "B-", "C+", "C", "C-"}:
        return "Hold"
    if g in {"D+", "D"}:
        return "Sell"
    if g == "F":
        return "Strong Sell"
    return "N/A"


def _read_factor_grade(factor_name: str, ticker: str) -> pd.DataFrame:
    path = FACTOR_FILES[factor_name]
    grade_col = {
        "Valuation": "valuation_grade",
        "Growth": "growth_grade",
        "Profitability": "profitability_grade",
        "Momentum": "momentum_grade",
        "EPS Rev.": "revisions_grade",
    }[factor_name]
    df = pd.read_csv(path, usecols=["ticker", "earningsAnnouncementDate", grade_col], low_memory=False)
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df = df[df["ticker"] == ticker].copy()
    df["Date"] = pd.to_datetime(df["earningsAnnouncementDate"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df = df.rename(columns={grade_col: factor_name})
    return df[["Date", factor_name]]


def build_history(ticker: str) -> pd.DataFrame:
    comp = pd.read_csv(
        COMPOSITE_FILE,
        usecols=[
            "ticker",
            "earningsAnnouncementDate",
            "composite_rating",
            "composite_grade",
        ],
        low_memory=False,
    )
    comp["ticker"] = comp["ticker"].astype(str).str.strip()
    comp = comp[comp["ticker"] == ticker].copy()
    if comp.empty:
        return pd.DataFrame()

    comp["Date"] = pd.to_datetime(comp["earningsAnnouncementDate"], errors="coerce")
    comp = comp.dropna(subset=["Date"]).sort_values("Date")
    comp["Quant Score"] = pd.to_numeric(comp["composite_rating"], errors="coerce").round(2)
    comp["Quant Rating"] = comp["composite_grade"].map(quant_label_from_grade)
    out = comp[["Date", "Quant Rating", "Quant Score"]].copy()

    for factor in ("Valuation", "Growth", "Profitability", "Momentum", "EPS Rev."):
        f = _read_factor_grade(factor, ticker)
        out = out.merge(f, on="Date", how="left")

    out = out.sort_values("Date", ascending=False).reset_index(drop=True)
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    return out


def _pick_row_on_or_before(df: pd.DataFrame, target: pd.Timestamp) -> pd.Series | None:
    candidates = df[df["Date"] <= target]
    if candidates.empty:
        return None
    return candidates.iloc[-1]


def build_snapshot(history: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    h = history.copy()
    h["Date"] = pd.to_datetime(h["Date"], errors="coerce")
    h = h.dropna(subset=["Date"]).sort_values("Date")
    if h.empty:
        return pd.DataFrame()

    points = {
        "Now": as_of,
        "30d Ago": as_of - pd.Timedelta(days=30),
        "6m Ago": as_of - pd.Timedelta(days=182),
    }
    rows = [
        "Quant Rating",
        "Quant Score",
        "Valuation",
        "Growth",
        "Profitability",
        "Momentum",
        "EPS Rev.",
    ]
    out = pd.DataFrame({"Metric": rows})

    for col_name, dt in points.items():
        r = _pick_row_on_or_before(h, dt)
        if r is None:
            out[col_name] = "N/A"
            continue
        out[col_name] = [
            r.get("Quant Rating", "N/A"),
            r.get("Quant Score", "N/A"),
            r.get("Valuation", "N/A"),
            r.get("Growth", "N/A"),
            r.get("Profitability", "N/A"),
            r.get("Momentum", "N/A"),
            r.get("EPS Rev.", "N/A"),
        ]
    return out


def run(ticker: str, as_of: str | None) -> tuple[Path, Path]:
    t = ticker.strip().upper()
    history = build_history(t)
    if history.empty:
        raise SystemExit(f"No composite score history found for ticker: {t}")

    as_of_ts = pd.to_datetime(as_of, errors="coerce") if as_of else None
    if as_of_ts is None or pd.isna(as_of_ts):
        as_of_ts = pd.to_datetime(history["Date"]).max()

    snapshot = build_snapshot(history, as_of_ts)

    history_file = PROCESSED_DATA_DIR / f"sa_compare_{t}_history.csv"
    snapshot_file = PROCESSED_DATA_DIR / f"sa_compare_{t}_snapshot.csv"
    history.to_csv(history_file, index=False)
    snapshot.to_csv(snapshot_file, index=False)
    return history_file, snapshot_file


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Seeking-Alpha-style comparison exports.")
    p.add_argument("--ticker", required=True, help="Ticker symbol (e.g., RTX).")
    p.add_argument(
        "--as-of",
        default=None,
        help="Optional as-of date (YYYY-MM-DD). Defaults to latest date for ticker.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    h, s = run(args.ticker, args.as_of)
    print(f"Saved {h}")
    print(f"Saved {s}")


if __name__ == "__main__":
    main()
