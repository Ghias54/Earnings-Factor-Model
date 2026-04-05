import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))
from config import PROCESSED_DATA_DIR
from processing.scoring_utils import rating_to_grade, score_to_rating

INPUT_FILE = PROCESSED_DATA_DIR / "revisions_features.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "revisions_scores.csv"


def run() -> None:
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df["earningsAnnouncementDate"] = pd.to_datetime(
        df["earningsAnnouncementDate"], errors="coerce"
    )

    df["eps_surprise_pct"] = pd.to_numeric(df["eps_surprise_pct"], errors="coerce")

    # Rank within calendar quarter so the cross-section is ~2,000 stocks, not ~30.
    df["_quarter"] = df["earningsAnnouncementDate"].dt.to_period("Q")

    # Higher upward surprise = better (proxy for positive estimate revisions)
    df["revisions_score"] = df.groupby("_quarter")["eps_surprise_pct"].rank(
        pct=True, ascending=True
    )

    df.loc[df["eps_surprise_pct"].isna(), "revisions_score"] = np.nan

    df["revisions_rating"] = score_to_rating(df["revisions_score"])
    df["revisions_grade"] = df["revisions_rating"].apply(rating_to_grade)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
