import sys
from pathlib import Path

import numpy as np
import pandas as pd

# allow import from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DATA_DIR

print("Starting valuation score build...")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(PROCESSED_DATA_DIR / "valuation_features.csv", low_memory=False)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# =========================
# DATE FORMAT
# =========================
df["earningsAnnouncementDate"] = pd.to_datetime(
    df["earningsAnnouncementDate"], errors="coerce"
)

# =========================
# SCORE EACH EARNINGS DATE CROSS-SECTION
# lower PE / PS = better
# =========================
# Rank within calendar quarter so the cross-section is ~2,000 stocks, not ~30.
df["_quarter"] = df["earningsAnnouncementDate"].dt.to_period("Q")

# Lower value = better for all valuation metrics → ascending=False
# Weighted: EV/EBITDA=0.30, P/E=0.25, P/S=0.20, P/FCF=0.15, P/B=0.10
METRICS = [
    ("pe",       0.25, False),
    ("ps",       0.20, False),
    ("ev_ebitda",0.30, False),
    ("p_fcf",    0.15, False),
    ("p_b",      0.10, False),
]

score_cols: list[str] = []
weights:    list[float] = []

for col, weight, ascending in METRICS:
    if col not in df.columns:
        continue
    df[col] = pd.to_numeric(df[col], errors="coerce")
    rank_col = f"_{col}_rank"
    df[rank_col] = df.groupby("_quarter")[col].rank(pct=True, ascending=ascending)
    df.loc[df[col].isna(), rank_col] = np.nan
    score_cols.append(rank_col)
    weights.append(weight)

# Keep legacy score columns for compatibility
if "_pe_rank" in df.columns:
    df["pe_score"] = df["_pe_rank"]
if "_ps_rank" in df.columns:
    df["ps_score"] = df["_ps_rank"]

rank_df    = df[score_cols]
weight_arr = np.array(weights)
notna_mask = rank_df.notna()
eff_weights = notna_mask.multiply(weight_arr)
eff_total   = eff_weights.sum(axis=1)

weighted_sum = (rank_df.fillna(0) * weight_arr).sum(axis=1)
df["valuation_score"] = np.where(eff_total > 0, weighted_sum / eff_total, np.nan)
df["valuation_component_count"] = notna_mask.sum(axis=1)
df.loc[df["valuation_component_count"] == 0, "valuation_score"] = np.nan

# =========================
# CONVERT TO 0-5 RATING
# keep decimals like 3.54
# =========================
df["valuation_rating"] = (df["valuation_score"] * 5).clip(lower=0.1).round(2)

# =========================
# OPTIONAL LETTER GRADE
# =========================
def rating_to_grade(x):
    if pd.isna(x):
        return np.nan
    if x >= 4.5:
        return "A+"
    elif x >= 4.0:
        return "A"
    elif x >= 3.67:
        return "A-"
    elif x >= 3.33:
        return "B+"
    elif x >= 3.0:
        return "B"
    elif x >= 2.67:
        return "B-"
    elif x >= 2.33:
        return "C+"
    elif x >= 2.0:
        return "C"
    elif x >= 1.67:
        return "C-"
    elif x >= 1.33:
        return "D+"
    elif x >= 1.0:
        return "D"
    else:
        return "F"

df["valuation_grade"] = df["valuation_rating"].apply(rating_to_grade)

# =========================
# DEBUG OUTPUT
# =========================
print("\nScore coverage:")
print("PE score not null:", df["pe_score"].notna().sum())
print("PS score not null:", df["ps_score"].notna().sum())
print("Valuation score not null:", df["valuation_score"].notna().sum())
print("Valuation rating not null:", df["valuation_rating"].notna().sum())

print("\nValuation score describe:")
print(df["valuation_score"].describe())

print("\nValuation rating describe:")
print(df["valuation_rating"].describe())

print("\nTop 10 valuation ratings:")
print(
    df[
        [
            "ticker",
            "earningsAnnouncementDate",
            "pe",
            "ps",
            "pe_score",
            "ps_score",
            "valuation_component_count",
            "valuation_score",
            "valuation_rating",
            "valuation_grade",
        ]
    ]
    .sort_values("valuation_rating", ascending=False)
    .head(10)
)

# =========================
# SAVE
# =========================
output_path = PROCESSED_DATA_DIR / "valuation_scores.csv"
df.to_csv(output_path, index=False)

print("\nDONE")
print(f"Saved valuation scores to: {output_path}")
print(f"Total rows: {len(df)}")