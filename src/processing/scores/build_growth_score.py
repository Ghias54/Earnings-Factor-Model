import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))
from config import GROWTH_FEATURES_FILE, GROWTH_SCORES_FILE
from processing.scoring_utils import rating_to_grade, score_to_rating

INPUT_FILE = GROWTH_FEATURES_FILE
OUTPUT_FILE = GROWTH_SCORES_FILE


def run() -> None:
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df["earningsAnnouncementDate"] = pd.to_datetime(
        df["earningsAnnouncementDate"], errors="coerce"
    )

    # Rank within calendar quarter so the cross-section is ~2,000 stocks, not ~30.
    df["_quarter"] = df["earningsAnnouncementDate"].dt.to_period("Q")

    # Weighted metrics: Revenue YoY=0.30, EPS YoY=0.25, EBITDA YoY=0.25, FCF YoY=0.20
    METRICS = [
        ("ttmRevenue_yoy",  0.30, 5.0),
        ("ttmEps_yoy",      0.25, 20.0),
        ("ttmEbitda_yoy",   0.25, 10.0),
        ("ttmFcf_yoy",      0.20, 10.0),
    ]

    score_cols: list[str] = []
    weights:    list[float] = []

    for col, weight, cap in METRICS:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col].abs() > cap, col] = np.nan
        rank_col = f"_{col}_rank"
        df[rank_col] = df.groupby("_quarter")[col].rank(pct=True, ascending=True)
        df.loc[df[col].isna(), rank_col] = np.nan
        score_cols.append(rank_col)
        weights.append(weight)

    rank_df    = df[score_cols]
    weight_arr = np.array(weights)
    notna_mask = rank_df.notna()
    eff_total  = notna_mask.multiply(weight_arr).sum(axis=1)

    weighted_sum = (rank_df.fillna(0) * weight_arr).sum(axis=1)
    df["growth_score"] = np.where(eff_total > 0, weighted_sum / eff_total, np.nan)
    df["growth_component_count"] = notna_mask.sum(axis=1)
    df.loc[df["growth_component_count"] == 0, "growth_score"] = np.nan

    df["growth_rating"] = score_to_rating(df["growth_score"])
    df["growth_grade"]  = df["growth_rating"].apply(rating_to_grade)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
