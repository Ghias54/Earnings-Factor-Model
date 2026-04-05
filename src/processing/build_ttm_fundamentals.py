import sys
from pathlib import Path

import numpy as np
import pandas as pd

# allow import from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


print("Starting TTM fundamentals build...")

# =========================
# SETTINGS
# =========================
HISTORY_FILE = RAW_DATA_DIR / "earnings_from_clean_universe.csv"
TARGET_FILE = PROCESSED_DATA_DIR / "earnings_from_price_start.csv"


# =========================
# LOAD DATA
# =========================
history = pd.read_csv(HISTORY_FILE, low_memory=False)
target = pd.read_csv(TARGET_FILE, low_memory=False)

print(f"History rows loaded: {len(history)}")
print(f"Target rows loaded: {len(target)}")


# =========================
# KEEP ONLY NEEDED COLUMNS
# =========================
history = history[
    [
        "ticker",
        "earningsAnnouncementDate",
        "actualEps",
        "actualRevenue",
    ]
].copy()

target = target[
    [
        "ticker",
        "earningsAnnouncementDate",
    ]
].copy()


# =========================
# CLEAN TYPES
# =========================
history["ticker"] = history["ticker"].astype(str).str.strip()
target["ticker"] = target["ticker"].astype(str).str.strip()

history["earningsAnnouncementDate"] = pd.to_datetime(
    history["earningsAnnouncementDate"], errors="coerce"
)
target["earningsAnnouncementDate"] = pd.to_datetime(
    target["earningsAnnouncementDate"], errors="coerce"
)

history["actualEps"] = pd.to_numeric(history["actualEps"], errors="coerce")
history["actualRevenue"] = pd.to_numeric(history["actualRevenue"], errors="coerce")

history = history.dropna(subset=["ticker", "earningsAnnouncementDate"]).copy()
target = target.dropna(subset=["ticker", "earningsAnnouncementDate"]).copy()


# =========================
# REMOVE DUPLICATES
# =========================
history = history.drop_duplicates(
    subset=["ticker", "earningsAnnouncementDate"],
    keep="last"
).copy()

target = target.drop_duplicates(
    subset=["ticker", "earningsAnnouncementDate"],
    keep="last"
).copy()


# =========================
# SORT
# =========================
history = history.sort_values(
    ["ticker", "earningsAnnouncementDate"]
).reset_index(drop=True)


# =========================
# BUILD TTM FROM FULL HISTORY
# =========================
history["ttmEps"] = (
    history.groupby("ticker")["actualEps"]
    .rolling(window=4, min_periods=4)
    .sum()
    .reset_index(level=0, drop=True)
)

history["ttmRevenue"] = (
    history.groupby("ticker")["actualRevenue"]
    .rolling(window=4, min_periods=4)
    .sum()
    .reset_index(level=0, drop=True)
)

history["quarterCountSeen"] = history.groupby("ticker").cumcount() + 1


# =========================
# KEEP ONLY ROWS NEEDED FOR BACKTEST
# =========================
ttm = target.merge(
    history[
        [
            "ticker",
            "earningsAnnouncementDate",
            "actualEps",
            "actualRevenue",
            "ttmEps",
            "ttmRevenue",
            "quarterCountSeen",
        ]
    ],
    on=["ticker", "earningsAnnouncementDate"],
    how="left",
)

ttm = ttm.sort_values(
    ["ticker", "earningsAnnouncementDate"]
).reset_index(drop=True)


# =========================
# SAVE
# =========================
output_path = PROCESSED_DATA_DIR / "ttm_fundamentals.csv"
ttm.to_csv(output_path, index=False)


# =========================
# DEBUG OUTPUT
# =========================
print("\nDONE")
print(f"Saved TTM fundamentals to: {output_path}")
print(f"Total rows: {len(ttm)}")
print(f"TTM EPS not null: {ttm['ttmEps'].notna().sum()}")
print(f"TTM Revenue not null: {ttm['ttmRevenue'].notna().sum()}")

print("\nSample TTM data:")
print(ttm.head(12))

print("\nQuarter count describe:")
print(ttm["quarterCountSeen"].describe())

print("\nTTM EPS describe:")
print(ttm["ttmEps"].describe())

print("\nTTM Revenue describe:")
print(ttm["ttmRevenue"].describe())