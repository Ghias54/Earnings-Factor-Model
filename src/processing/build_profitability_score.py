import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC  = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))
from config import PROCESSED_DATA_DIR
from processing.scoring_utils import rating_to_grade, score_to_rating

INPUT_FILE  = PROCESSED_DATA_DIR / "profitability_features.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "profitability_scores.csv"

# Weighted metric config: (column, weight, ascending)
# Higher margin/return = better for all metrics
METRICS = [
    ("net_margin",      0.20, True),
    ("gross_margin",    0.20, True),
    ("ebitda_margin",   0.25, True),
    ("operating_margin",0.15, True),
    ("roa",             0.10, True),
    ("roe",             0.10, True),
]


def run() -> None:
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df["earningsAnnouncementDate"] = pd.to_datetime(df["earningsAnnouncementDate"], errors="coerce")

    # Rank within calendar quarter
    df["_quarter"] = df["earningsAnnouncementDate"].dt.to_period("Q")

    available_metrics = [(col, w, asc) for col, w, asc in METRICS if col in df.columns]
    if not available_metrics:
        raise SystemExit("No profitability metric columns found in input.")

    total_weight = sum(w for _, w, _ in available_metrics)

    score_cols: list[str] = []
    weights: list[float] = []

    for col, weight, ascending in available_metrics:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        rank_col = f"_{col}_rank"
        df[rank_col] = df.groupby("_quarter")[col].rank(pct=True, ascending=ascending)
        df.loc[df[col].isna(), rank_col] = np.nan
        score_cols.append(rank_col)
        weights.append(weight)

    # Weighted average of available ranks
    rank_df     = df[score_cols]
    weight_arr  = np.array(weights)
    notna_mask  = rank_df.notna()
    eff_weights = notna_mask.multiply(weight_arr)
    eff_total   = eff_weights.sum(axis=1)

    weighted_sum = (rank_df.fillna(0) * weight_arr).sum(axis=1)
    df["profitability_score"] = np.where(eff_total > 0, weighted_sum / eff_total, np.nan)
    df["profitability_component_count"] = notna_mask.sum(axis=1)

    df["profitability_rating"] = score_to_rating(df["profitability_score"])
    df["profitability_grade"]  = df["profitability_rating"].apply(rating_to_grade)

    # Drop temporary columns
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df):,} rows → {OUTPUT_FILE}")
    for col, _, _ in available_metrics:
        n = df[f"profitability_score"].notna().sum()
    print(f"  profitability_score coverage: {df['profitability_score'].notna().sum():,}/{len(df):,}")
    print(f"  avg component count: {df['profitability_component_count'].mean():.2f}")


if __name__ == "__main__":
    run()
