import sys
from pathlib import Path

import numpy as np
import pandas as pd

# allow import from project root
sys.path.append(str(Path(__file__).resolve().parents[2]))
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
df["pe_score"] = df.groupby("earningsAnnouncementDate")["pe"].rank(
    pct=True,
    ascending=False
)

df["ps_score"] = df.groupby("earningsAnnouncementDate")["ps"].rank(
    pct=True,
    ascending=False
)

# =========================
# COMBINE LIKE SEEKING ALPHA STYLE
# if both exist -> average both
# if only one exists -> use the one that exists
# if neither exists -> NaN
# =========================
df["valuation_component_count"] = df[["pe_score", "ps_score"]].notna().sum(axis=1)

df["valuation_score"] = np.where(
    df["pe_score"].notna() & df["ps_score"].notna(),
    (df["pe_score"] + df["ps_score"]) / 2,
    np.where(
        df["pe_score"].notna(),
        df["pe_score"],
        df["ps_score"]
    )
)

# make sure rows with no components stay NaN
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