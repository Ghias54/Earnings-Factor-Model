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

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (
    COMPOSITE_SCORES_FILE,
    GROWTH_SCORES_FILE,
    MASTER_DAILY_QUANT_PANEL_FILE,
    MOMENTUM_SCORES_FILE,
    PROFITABILITY_SCORES_FILE,
    REVISIONS_SCORES_FILE,
    SA_ALIGNMENT_REPORT_FILE,
    SA_BENCHMARK_FILE,
    VALUATION_SCORES_FILE,
)

BENCHMARK_FILE = SA_BENCHMARK_FILE
OUTPUT_FILE = SA_ALIGNMENT_REPORT_FILE


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


def _dedupe_event_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure one row per (ticker, earningsAnnouncementDate) to prevent merge explosions.
    Keep last row after stable sort.
    """
    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out["earningsAnnouncementDate"] = pd.to_datetime(out["earningsAnnouncementDate"], errors="coerce")
    out = out.sort_values(["ticker", "earningsAnnouncementDate"], kind="mergesort")
    dups = out.duplicated(["ticker", "earningsAnnouncementDate"]).sum()
    if dups:
        print(f"Deduplicating event keys: dropped {dups:,} duplicate rows")
        out = out.drop_duplicates(["ticker", "earningsAnnouncementDate"], keep="last")
    return out


def load_our_scores() -> pd.DataFrame:
    comp = pd.read_csv(COMPOSITE_SCORES_FILE, low_memory=False)
    val = pd.read_csv(
        VALUATION_SCORES_FILE,
        usecols=["ticker", "earningsAnnouncementDate", "valuation_grade"],
        low_memory=False,
    )
    grw = pd.read_csv(
        GROWTH_SCORES_FILE,
        usecols=["ticker", "earningsAnnouncementDate", "growth_grade"],
        low_memory=False,
    )
    prof = pd.read_csv(
        PROFITABILITY_SCORES_FILE,
        usecols=["ticker", "earningsAnnouncementDate", "profitability_grade"],
        low_memory=False,
    )
    mom = pd.read_csv(
        MOMENTUM_SCORES_FILE,
        usecols=["ticker", "earningsAnnouncementDate", "momentum_grade"],
        low_memory=False,
    )
    rev = pd.read_csv(
        REVISIONS_SCORES_FILE,
        usecols=["ticker", "earningsAnnouncementDate", "revisions_grade"],
        low_memory=False,
    )

    comp = _dedupe_event_rows(comp)
    val = _dedupe_event_rows(val)
    grw = _dedupe_event_rows(grw)
    prof = _dedupe_event_rows(prof)
    mom = _dedupe_event_rows(mom)
    rev = _dedupe_event_rows(rev)

    out = comp.merge(val, on=["ticker", "earningsAnnouncementDate"], how="left", validate="one_to_one")
    out = out.merge(grw, on=["ticker", "earningsAnnouncementDate"], how="left", validate="one_to_one")
    out = out.merge(prof, on=["ticker", "earningsAnnouncementDate"], how="left", validate="one_to_one")
    out = out.merge(mom, on=["ticker", "earningsAnnouncementDate"], how="left", validate="one_to_one")
    out = out.merge(rev, on=["ticker", "earningsAnnouncementDate"], how="left", validate="one_to_one")

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


def _resolve_panel_path() -> Path:
    for p in (
        MASTER_DAILY_QUANT_PANEL_FILE,
        MASTER_DAILY_QUANT_PANEL_FILE.with_suffix(""),
    ):
        if p.is_file():
            return p
    raise SystemExit("No daily panel found. Build with build_master_daily_panel.py first.")


def load_our_scores_daily(bench: pd.DataFrame) -> pd.DataFrame:
    panel_path = _resolve_panel_path()
    keys = bench[["ticker", "date"]].drop_duplicates()
    tickers = set(keys["ticker"])
    usecols = [
        "ticker",
        "date",
        "valuation_score",
        "growth_score",
        "profitability_score",
        "revisions_score",
        "momentum_score",
        "composite_rating",
        "composite_grade",
    ]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(panel_path, usecols=usecols, chunksize=400_000, compression="infer", low_memory=False):
        chunk["ticker"] = chunk["ticker"].astype(str).str.strip().str.upper()
        chunk = chunk[chunk["ticker"].isin(tickers)]
        if chunk.empty:
            continue
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        hit = chunk.merge(keys, on=["ticker", "date"], how="inner")
        if not hit.empty:
            parts.append(hit)
    if not parts:
        return pd.DataFrame(columns=[
            "ticker","date","our_quant_score","our_quant_rating","our_valuation",
            "our_growth","our_profitability","our_momentum","our_revisions"
        ])
    out = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["ticker", "date"])
    out["our_quant_score"] = _safe_num(out["composite_rating"])
    out["our_quant_rating"] = out["composite_grade"].map(_our_quant_rating_from_grade)
    for score_col, out_col in [
        ("valuation_score", "our_valuation"),
        ("growth_score", "our_growth"),
        ("profitability_score", "our_profitability"),
        ("momentum_score", "our_momentum"),
        ("revisions_score", "our_revisions"),
    ]:
        rt = (_safe_num(out[score_col]) * 5).clip(lower=0.1).round(2)
        out[out_col] = rt.apply(_norm_grade)  # will pass through strings if any
        # Convert numeric rating -> grade bands
        out[out_col] = rt.apply(
            lambda x: (
                "A+" if pd.notna(x) and x >= 4.5 else
                "A" if pd.notna(x) and x >= 4.0 else
                "A-" if pd.notna(x) and x >= 3.67 else
                "B+" if pd.notna(x) and x >= 3.33 else
                "B" if pd.notna(x) and x >= 3.0 else
                "B-" if pd.notna(x) and x >= 2.67 else
                "C+" if pd.notna(x) and x >= 2.33 else
                "C" if pd.notna(x) and x >= 2.0 else
                "C-" if pd.notna(x) and x >= 1.67 else
                "D+" if pd.notna(x) and x >= 1.33 else
                "D" if pd.notna(x) and x >= 1.0 else
                "F" if pd.notna(x) else np.nan
            )
        )
    return out[[
        "ticker", "date", "our_quant_score", "our_quant_rating",
        "our_valuation", "our_growth", "our_profitability", "our_momentum", "our_revisions"
    ]]


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
    b = b.sort_values(["ticker", "date"], kind="mergesort")
    b = b.drop_duplicates(["ticker", "date"], keep="last")
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


def run(*, join: str) -> None:
    bench = load_benchmark()
    ours = load_our_scores_daily(bench) if join == "daily" else load_our_scores()
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

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE} (join={join})")
    print_metrics(merged.dropna(subset=["our_quant_score", "sa_quant_score"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate SA alignment")
    ap.add_argument("--join", choices=["daily", "earnings"], default="daily")
    args = ap.parse_args()
    run(join=args.join)
