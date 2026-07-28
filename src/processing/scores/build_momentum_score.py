import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))
from config import MOMENTUM_FEATURES_FILE, MOMENTUM_SCORES_FILE
from processing.scoring_utils import rating_to_grade, score_to_rating

INPUT_FILE = MOMENTUM_FEATURES_FILE
OUTPUT_FILE = MOMENTUM_SCORES_FILE


def run() -> None:
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df["earningsAnnouncementDate"] = pd.to_datetime(
        df["earningsAnnouncementDate"], errors="coerce"
    )

    df["mom21"] = pd.to_numeric(df["mom21"], errors="coerce") if "mom21" in df.columns else np.nan
    df["mom63"] = pd.to_numeric(df["mom63"], errors="coerce")
    df["mom126"] = pd.to_numeric(df["mom126"], errors="coerce")
    df["mom252"] = pd.to_numeric(df["mom252"], errors="coerce") if "mom252" in df.columns else np.nan

    # Rank within calendar quarter so the cross-section is ~2,000 stocks, not ~30.
    df["_quarter"] = df["earningsAnnouncementDate"].dt.to_period("Q")

    # "Our Model": 1M/3M/6M/12M ranks over full quarter cross-section.
    df["mom21_score"] = df.groupby("_quarter")["mom21"].rank(pct=True, ascending=True)
    df["mom63_score"] = df.groupby("_quarter")["mom63"].rank(pct=True, ascending=True)
    df["mom126_score"] = df.groupby("_quarter")["mom126"].rank(pct=True, ascending=True)
    df["mom252_score"] = df.groupby("_quarter")["mom252"].rank(pct=True, ascending=True)

    # Equal blend across available horizons for stable coverage.
    score_parts  = pd.DataFrame({
        "s21": df["mom21_score"] * 0.25,
        "s63": df["mom63_score"] * 0.25,
        "s126": df["mom126_score"] * 0.25,
        "s252": df["mom252_score"] * 0.25,
    })
    weight_parts = pd.DataFrame({
        "w21": df["mom21_score"].notna() * 0.25,
        "w63": df["mom63_score"].notna() * 0.25,
        "w126": df["mom126_score"].notna() * 0.25,
        "w252": df["mom252_score"].notna() * 0.25,
    })
    total_weight = weight_parts.sum(axis=1)
    total_score  = score_parts.fillna(0).sum(axis=1)
    df["momentum_score"] = np.where(
        total_weight > 0, total_score / total_weight, np.nan
    )
    df["momentum_score"] = pd.Series(df["momentum_score"].astype(float))
    df.loc[total_weight == 0, "momentum_score"] = np.nan

    df["momentum_rating"] = score_to_rating(df["momentum_score"])
    df["momentum_grade"] = df["momentum_rating"].apply(rating_to_grade)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
