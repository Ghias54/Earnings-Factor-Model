import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))
from config import (
    COMPOSITE_SCORES_FILE,
    GROWTH_SCORES_FILE,
    INSIDER_SCORES_FILE,
    MOMENTUM_SCORES_FILE,
    PROFITABILITY_SCORES_FILE,
    REVISIONS_SCORES_FILE,
    VALUATION_SCORES_FILE,
)
from processing.scoring_utils import rating_to_grade, score_to_rating

OUTPUT_FILE = COMPOSITE_SCORES_FILE

KEY = ["ticker", "earningsAnnouncementDate"]

FACTORS = [
    (VALUATION_SCORES_FILE, "valuation_score"),
    (GROWTH_SCORES_FILE, "growth_score"),
    (PROFITABILITY_SCORES_FILE, "profitability_score"),
    (REVISIONS_SCORES_FILE, "revisions_score"),
    (MOMENTUM_SCORES_FILE, "momentum_score"),
    (INSIDER_SCORES_FILE, "insider_score"),
]

FACTOR_WEIGHTS = {
    "valuation_score": 0.20,
    "growth_score": 0.20,
    "profitability_score": 0.20,
    "revisions_score": 0.10,
    "momentum_score": 0.20,
    "insider_score": 0.10,
}


def load_factor(path: Path, score_col: str) -> pd.DataFrame:
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
    base = load_factor(VALUATION_SCORES_FILE, "valuation_score")
    if base.empty:
        raise SystemExit(f"{VALUATION_SCORES_FILE.name} is required for composite.")

    merged = base.copy()
    for fpath, col in FACTORS[1:]:
        part = load_factor(fpath, col)
        if part.empty:
            merged[col] = np.nan
        else:
            merged = merged.merge(part, on=KEY, how="left")

    score_cols = [f[1] for f in FACTORS]
    present = merged[score_cols]
    merged["composite_component_count"] = present.notna().sum(axis=1)

    weight_arr = np.array([FACTOR_WEIGHTS[c] for c in score_cols], dtype=float)
    notna_mask = present.notna()
    eff_w = notna_mask.multiply(weight_arr)
    numer = (present.fillna(0) * weight_arr).sum(axis=1)
    denom = eff_w.sum(axis=1)
    merged["composite_score"] = np.where(denom > 0, numer / denom, np.nan)
    merged["composite_rating"] = score_to_rating(merged["composite_score"])
    merged["composite_grade"] = merged["composite_rating"].apply(rating_to_grade)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(merged)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
