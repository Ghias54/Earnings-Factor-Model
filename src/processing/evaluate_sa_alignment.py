"""
Evaluate alignment between local quant outputs and manually collected Seeking Alpha labels.

Input:
  data/processed/sa_benchmark.csv
Required columns:
  ticker,date,sa_quant_score,sa_quant_rating,sa_valuation,sa_growth,
  sa_profitability,sa_momentum,sa_revisions

Outputs:
  data/processed/sa_alignment_report.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DATA_DIR

BENCHMARK_FILE = PROCESSED_DATA_DIR / "sa_benchmark.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "sa_alignment_report.csv"


def _norm_grade(v: object) -> object:
    if pd.isna(v):
        return np.nan
    s = str(v).strip().upper()
    return s if s else np.nan


def _norm_quant_rating(v: object) -> object:
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower().replace("-", " ")
    mapping = {
        "strong buy": "Strong Buy",
        "buy": "Buy",
        "hold": "Hold",
        "sell": "Sell",
        "strong sell": "Strong Sell",
    }
    return mapping.get(s, np.nan)


def _our_quant_rating_from_grade(grade: object) -> object:
    g = _norm_grade(grade)
    if pd.isna(g):
        return np.nan
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
    return np.nan


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def load_our_scores() -> pd.DataFrame:
    comp = pd.read_csv(PROCESSED_DATA_DIR / "composite_quant_scores.csv", low_memory=False)
    val = pd.read_csv(
        PROCESSED_DATA_DIR / "valuation_scores.csv",
        usecols=["ticker", "earningsAnnouncementDate", "valuation_grade"],
        low_memory=False,
    )
    grw = pd.read_csv(
        PROCESSED_DATA_DIR / "growth_scores.csv",
        usecols=["ticker", "earningsAnnouncementDate", "growth_grade"],
        low_memory=False,
    )
    prof = pd.read_csv(
        PROCESSED_DATA_DIR / "profitability_scores.csv",
        usecols=["ticker", "earningsAnnouncementDate", "profitability_grade"],
        low_memory=False,
    )
    mom = pd.read_csv(
        PROCESSED_DATA_DIR / "momentum_scores.csv",
        usecols=["ticker", "earningsAnnouncementDate", "momentum_grade"],
        low_memory=False,
    )
    rev = pd.read_csv(
        PROCESSED_DATA_DIR / "revisions_scores.csv",
        usecols=["ticker", "earningsAnnouncementDate", "revisions_grade"],
        low_memory=False,
    )

    out = comp.merge(val, on=["ticker", "earningsAnnouncementDate"], how="left")
    out = out.merge(grw, on=["ticker", "earningsAnnouncementDate"], how="left")
    out = out.merge(prof, on=["ticker", "earningsAnnouncementDate"], how="left")
    out = out.merge(mom, on=["ticker", "earningsAnnouncementDate"], how="left")
    out = out.merge(rev, on=["ticker", "earningsAnnouncementDate"], how="left")

    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out["date"] = pd.to_datetime(out["earningsAnnouncementDate"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    out["our_quant_score"] = _safe_num(out["composite_rating"])
    out["our_quant_rating"] = out["composite_grade"].map(_our_quant_rating_from_grade)
    out["our_valuation"] = out["valuation_grade"].map(_norm_grade)
    out["our_growth"] = out["growth_grade"].map(_norm_grade)
    out["our_profitability"] = out["profitability_grade"].map(_norm_grade)
    out["our_momentum"] = out["momentum_grade"].map(_norm_grade)
    out["our_revisions"] = out["revisions_grade"].map(_norm_grade)
    return out[
        [
            "ticker",
            "date",
            "our_quant_score",
            "our_quant_rating",
            "our_valuation",
            "our_growth",
            "our_profitability",
            "our_momentum",
            "our_revisions",
        ]
    ]


def load_benchmark() -> pd.DataFrame:
    if not BENCHMARK_FILE.exists():
        raise SystemExit(f"Missing benchmark file: {BENCHMARK_FILE}")
    b = pd.read_csv(BENCHMARK_FILE, low_memory=False)
    required = [
        "ticker",
        "date",
        "sa_quant_score",
        "sa_quant_rating",
        "sa_valuation",
        "sa_growth",
        "sa_profitability",
        "sa_momentum",
        "sa_revisions",
    ]
    missing = [c for c in required if c not in b.columns]
    if missing:
        raise SystemExit(f"Missing benchmark columns: {missing}")

    b = b.copy()
    b["ticker"] = b["ticker"].astype(str).str.strip().str.upper()
    b["date"] = pd.to_datetime(b["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    b["sa_quant_score"] = _safe_num(b["sa_quant_score"])
    b["sa_quant_rating"] = b["sa_quant_rating"].map(_norm_quant_rating)
    for c in ["sa_valuation", "sa_growth", "sa_profitability", "sa_momentum", "sa_revisions"]:
        b[c] = b[c].map(_norm_grade)
    b = b.dropna(subset=["ticker", "date"]).copy()
    return b


def print_metrics(df: pd.DataFrame) -> None:
    print(f"Rows compared: {len(df):,}")
    if df.empty:
        print("No overlaps between benchmark and local scores.")
        return

    score_err = (df["our_quant_score"] - df["sa_quant_score"]).abs()
    print(f"Quant score MAE: {score_err.mean():.3f}")
    corr = df[["our_quant_score", "sa_quant_score"]].corr(method="pearson").iloc[0, 1]
    print(f"Quant score Pearson r: {corr:.3f}" if pd.notna(corr) else "Quant score Pearson r: N/A")
    rate_match = (df["our_quant_rating"] == df["sa_quant_rating"]).mean() * 100
    print(f"Quant rating exact match: {rate_match:.1f}%")

    for our_col, sa_col, label in [
        ("our_valuation", "sa_valuation", "Valuation"),
        ("our_growth", "sa_growth", "Growth"),
        ("our_profitability", "sa_profitability", "Profitability"),
        ("our_momentum", "sa_momentum", "Momentum"),
        ("our_revisions", "sa_revisions", "Revisions"),
    ]:
        m = (df[our_col] == df[sa_col]).mean() * 100
        print(f"{label} grade exact match: {m:.1f}%")


def run() -> None:
    ours = load_our_scores()
    bench = load_benchmark()
    merged = bench.merge(ours, on=["ticker", "date"], how="left")
    merged["quant_score_abs_error"] = (
        pd.to_numeric(merged["our_quant_score"], errors="coerce")
        - pd.to_numeric(merged["sa_quant_score"], errors="coerce")
    ).abs()
    merged["quant_rating_match"] = merged["our_quant_rating"] == merged["sa_quant_rating"]
    merged["valuation_match"] = merged["our_valuation"] == merged["sa_valuation"]
    merged["growth_match"] = merged["our_growth"] == merged["sa_growth"]
    merged["profitability_match"] = merged["our_profitability"] == merged["sa_profitability"]
    merged["momentum_match"] = merged["our_momentum"] == merged["sa_momentum"]
    merged["revisions_match"] = merged["our_revisions"] == merged["sa_revisions"]
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)

    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE}")
    print_metrics(merged.dropna(subset=["our_quant_score", "sa_quant_score"]))


if __name__ == "__main__":
    run()
