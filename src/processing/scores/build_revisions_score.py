import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))
from config import REVISIONS_FEATURES_FILE, REVISIONS_SCORES_FILE
from processing.scoring_utils import rating_to_grade, score_to_rating

INPUT_FILE = REVISIONS_FEATURES_FILE
OUTPUT_FILE = REVISIONS_SCORES_FILE


def run() -> None:
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df["earningsAnnouncementDate"] = pd.to_datetime(
        df["earningsAnnouncementDate"], errors="coerce"
    )

    df["eps_surprise_pct"] = pd.to_numeric(df["eps_surprise_pct"], errors="coerce")
    if "eps_fwd_step_pct" in df.columns:
        df["eps_fwd_step_pct"] = pd.to_numeric(df["eps_fwd_step_pct"], errors="coerce")
    else:
        df["eps_fwd_step_pct"] = np.nan

    # Rank within calendar quarter so the cross-section is ~2,000 stocks, not ~30.
    df["_quarter"] = df["earningsAnnouncementDate"].dt.to_period("Q")

    # "Our Model": blend surprise + forward-step from one-snapshot estimate sequence.
    df["_surprise_rank"] = df.groupby("_quarter")["eps_surprise_pct"].rank(pct=True, ascending=True)
    df["_step_rank"] = df.groupby("_quarter")["eps_fwd_step_pct"].rank(pct=True, ascending=True)
    df.loc[df["eps_surprise_pct"].isna(), "_surprise_rank"] = np.nan
    df.loc[df["eps_fwd_step_pct"].isna(), "_step_rank"] = np.nan

    weights = pd.DataFrame({
        "w_surprise": df["_surprise_rank"].notna() * 0.60,
        "w_step": df["_step_rank"].notna() * 0.40,
    })
    numer = (
        df["_surprise_rank"].fillna(0) * 0.60
        + df["_step_rank"].fillna(0) * 0.40
    )
    denom = weights.sum(axis=1)
    df["revisions_score"] = np.where(denom > 0, numer / denom, np.nan)

    df["revisions_rating"] = score_to_rating(df["revisions_score"])
    df["revisions_grade"] = df["revisions_rating"].apply(rating_to_grade)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
