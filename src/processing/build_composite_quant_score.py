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

OUTPUT_FILE = PROCESSED_DATA_DIR / "composite_quant_scores.csv"

KEY = ["ticker", "earningsAnnouncementDate"]

FACTORS = [
    ("valuation_scores.csv", "valuation_score"),
    ("growth_scores.csv", "growth_score"),
    ("profitability_scores.csv", "profitability_score"),
    ("revisions_scores.csv", "revisions_score"),
    ("momentum_scores.csv", "momentum_score"),
]

# Composite blend weights (SA-like heuristic):
# Keep a smaller valuation weight and emphasize quality/trend buckets.
# Weights are renormalized across available factors row-by-row.
WEIGHTS = {
    "valuation_score": 0.15,
    "growth_score": 0.20,
    "profitability_score": 0.25,
    "revisions_score": 0.20,
    "momentum_score": 0.20,
}


def load_factor(path_name: str, score_col: str) -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / path_name
    if not path.exists():
        print(f"Optional missing {path}; {score_col} will be NaN.")
        return pd.DataFrame(columns=KEY + [score_col])
    df = pd.read_csv(path, low_memory=False)
    if score_col not in df.columns:
        print(f"No column {score_col} in {path}.")
        return pd.DataFrame(columns=KEY + [score_col])
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["earningsAnnouncementDate"] = pd.to_datetime(
        df["earningsAnnouncementDate"], errors="coerce"
    )
    return df[KEY + [score_col]].drop_duplicates(subset=KEY)


def run() -> None:
    base = load_factor("valuation_scores.csv", "valuation_score")
    if base.empty:
        raise SystemExit("valuation_scores.csv is required for composite.")

    merged = base.copy()
    for fname, col in FACTORS[1:]:
        part = load_factor(fname, col)
        if part.empty:
            merged[col] = np.nan
        else:
            merged = merged.merge(part, on=KEY, how="left")

    score_cols = [f[1] for f in FACTORS]
    present = merged[score_cols]
    merged["composite_component_count"] = present.notna().sum(axis=1)

    # Weighted blend on available factors.
    w = np.array([WEIGHTS.get(c, 0.0) for c in score_cols], dtype=float)
    w_df = present.notna().multiply(w, axis=1)
    w_sum = w_df.sum(axis=1)
    weighted_raw = (present.fillna(0.0) * w).sum(axis=1)
    merged["composite_raw_blend"] = np.where(w_sum > 0, weighted_raw / w_sum, np.nan)
    merged.loc[merged["composite_component_count"] == 0, "composite_raw_blend"] = np.nan

    # SA-style comparability improvement:
    # rank the blended composite cross-sectionally (by calendar quarter), then map to 0-5.
    merged["_quarter"] = merged["earningsAnnouncementDate"].dt.to_period("Q")
    merged["composite_score"] = merged.groupby("_quarter")["composite_raw_blend"].rank(
        pct=True, ascending=True
    )
    merged.loc[merged["composite_component_count"] == 0, "composite_score"] = np.nan
    merged["composite_rating"] = score_to_rating(merged["composite_score"])
    merged["composite_grade"] = merged["composite_rating"].apply(rating_to_grade)
    merged = merged.drop(columns=["_quarter"])

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(merged)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
