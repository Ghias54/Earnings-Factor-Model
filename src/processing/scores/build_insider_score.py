from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))
from config import INSIDER_FEATURES_FILE, INSIDER_SCORES_FILE
from processing.scoring_utils import rating_to_grade, score_to_rating


def run() -> None:
    df = pd.read_csv(INSIDER_FEATURES_FILE, low_memory=False)
    df["earningsAnnouncementDate"] = pd.to_datetime(df["earningsAnnouncementDate"], errors="coerce")
    df["_quarter"] = df["earningsAnnouncementDate"].dt.to_period("Q")

    numeric_cols = [
        "insider_buy_count_30d",
        "insider_sell_count_30d",
        "insider_buy_count_90d",
        "insider_sell_count_90d",
        "insider_net_value_30d",
        "insider_net_value_90d",
        "director_buy_count_90d",
        "ten_percent_owner_buy_count_90d",
        "ceo_buy_flag_90d",
        "cfo_buy_flag_90d",
        "ceo_cfo_buy_flag_90d",
        "large_insider_buy_flag_90d",
        "large_insider_sell_flag_90d",
        "informative_buy_flag",
        "informative_sell_flag",
    ]
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Build a simple directional signal then percentile-rank cross-sectionally by quarter.
    # Positive components: buys/informative buys/net buys. Negative: sells/informative sells.
    df["_insider_raw"] = (
        0.20 * df["insider_buy_count_30d"].fillna(0.0)
        + 0.15 * df["insider_buy_count_90d"].fillna(0.0)
        - 0.10 * df["insider_sell_count_30d"].fillna(0.0)
        - 0.05 * df["insider_sell_count_90d"].fillna(0.0)
        + 0.20 * (df["insider_net_value_30d"].fillna(0.0) / 1_000_000.0)
        + 0.10 * (df["insider_net_value_90d"].fillna(0.0) / 1_000_000.0)
        + 0.05 * df["ceo_buy_flag_90d"].fillna(0.0)
        + 0.05 * df["cfo_buy_flag_90d"].fillna(0.0)
        + 0.05 * df["ceo_cfo_buy_flag_90d"].fillna(0.0)
        + 0.05 * df["director_buy_count_90d"].fillna(0.0)
        + 0.05 * df["ten_percent_owner_buy_count_90d"].fillna(0.0)
        + 0.05 * df["large_insider_buy_flag_90d"].fillna(0.0)
        - 0.05 * df["large_insider_sell_flag_90d"].fillna(0.0)
        + 0.08 * df["informative_buy_flag"].fillna(0.0)
        - 0.08 * df["informative_sell_flag"].fillna(0.0)
    )

    df["insider_score"] = df.groupby("_quarter")["_insider_raw"].rank(pct=True, ascending=True)
    df.loc[df["_insider_raw"].isna(), "insider_score"] = np.nan
    df["insider_rating"] = score_to_rating(df["insider_score"])
    df["insider_grade"] = df["insider_rating"].apply(rating_to_grade)

    INSIDER_SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INSIDER_SCORES_FILE, index=False)
    print(f"Saved {len(df):,} rows to {INSIDER_SCORES_FILE}")


if __name__ == "__main__":
    run()
