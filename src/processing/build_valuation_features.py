import sys
from pathlib import Path

import numpy as np
import pandas as pd

# allow import from project root
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


print("Starting valuation feature build...")

# =========================
# SETTINGS
# =========================
MIN_PRICE = 2.0
MIN_MARKET_CAP = 50_000_000
MAX_SHARES_AGE_DAYS = 550

MIN_TTM_EPS_FOR_PE = 0.10
MAX_ABS_TTM_EPS = 1000
MAX_TTM_REVENUE = 1_000_000_000_000  # 1T

MAX_PE = 200
MAX_PS = 50


# =========================
# LOAD DATA
# =========================
earnings = pd.read_csv(PROCESSED_DATA_DIR / "earnings_from_price_start.csv")
prices = pd.read_csv(RAW_DATA_DIR / "daily_prices_from_clean_universe.csv", low_memory=False)
shares = pd.read_csv(RAW_DATA_DIR / "shares_history.csv")
ttm = pd.read_csv(PROCESSED_DATA_DIR / "ttm_fundamentals.csv")

print(f"Earnings rows loaded: {len(earnings)}")
print(f"Price rows loaded: {len(prices)}")
print(f"Shares rows loaded: {len(shares)}")
print(f"TTM rows loaded: {len(ttm)}")


# =========================
# KEEP ONLY NEEDED COLUMNS
# =========================
earnings = earnings[
    [
        "ticker",
        "earningsAnnouncementDate",
        "actualEps",
        "actualRevenue",
    ]
].copy()

prices = prices[
    [
        "ticker",
        "date",
        "close",
    ]
].copy()

shares = shares[
    [
        "ticker",
        "date",
        "sharesOutstanding",
    ]
].copy()

ttm = ttm[
    [
        "ticker",
        "earningsAnnouncementDate",
        "ttmEps",
        "ttmRevenue",
        "quarterCountSeen",
    ]
].copy()


# =========================
# FORMAT DATES
# =========================
earnings["earningsAnnouncementDate"] = pd.to_datetime(
    earnings["earningsAnnouncementDate"], errors="coerce"
)
prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
shares["date"] = pd.to_datetime(shares["date"], errors="coerce")
ttm["earningsAnnouncementDate"] = pd.to_datetime(
    ttm["earningsAnnouncementDate"], errors="coerce"
)


# =========================
# CLEAN NUMERIC FIELDS
# =========================
earnings["actualEps"] = pd.to_numeric(earnings["actualEps"], errors="coerce")
earnings["actualRevenue"] = pd.to_numeric(earnings["actualRevenue"], errors="coerce")
prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
shares["sharesOutstanding"] = pd.to_numeric(shares["sharesOutstanding"], errors="coerce")
ttm["ttmEps"] = pd.to_numeric(ttm["ttmEps"], errors="coerce")
ttm["ttmRevenue"] = pd.to_numeric(ttm["ttmRevenue"], errors="coerce")
ttm["quarterCountSeen"] = pd.to_numeric(ttm["quarterCountSeen"], errors="coerce")

earnings = earnings.dropna(subset=["ticker", "earningsAnnouncementDate"]).copy()
prices = prices.dropna(subset=["ticker", "date", "close"]).copy()
shares = shares.dropna(subset=["ticker", "date", "sharesOutstanding"]).copy()
ttm = ttm.dropna(subset=["ticker", "earningsAnnouncementDate"]).copy()

prices = prices[prices["close"] > 0].copy()
shares = shares[shares["sharesOutstanding"] > 0].copy()

earnings["ticker"] = earnings["ticker"].astype(str).str.strip()
prices["ticker"] = prices["ticker"].astype(str).str.strip()
shares["ticker"] = shares["ticker"].astype(str).str.strip()
ttm["ticker"] = ttm["ticker"].astype(str).str.strip()


# =========================
# CLEAN TTM OUTLIERS
# =========================
ttm.loc[ttm["ttmRevenue"] <= 0, "ttmRevenue"] = np.nan
ttm.loc[ttm["ttmRevenue"] > MAX_TTM_REVENUE, "ttmRevenue"] = np.nan
ttm.loc[ttm["ttmEps"].abs() > MAX_ABS_TTM_EPS, "ttmEps"] = np.nan

print("\nTTM usable counts after sanity filters:")
print(f"TTM EPS not null: {ttm['ttmEps'].notna().sum()}")
print(f"TTM Revenue not null: {ttm['ttmRevenue'].notna().sum()}")


# =========================
# MERGE EARNINGS + TTM
# exact merge on ticker + earnings date
# =========================
valuation = earnings.merge(
    ttm,
    on=["ticker", "earningsAnnouncementDate"],
    how="left",
)

print(f"\nRows after earnings + TTM merge: {len(valuation)}")
print(f"Missing TTM EPS rows: {valuation['ttmEps'].isna().sum()}")
print(f"Missing TTM Revenue rows: {valuation['ttmRevenue'].isna().sum()}")


# =========================
# RENAME FOR ASOF MERGES
# =========================
prices = prices.rename(columns={"date": "anchorDate", "close": "price"})
shares = shares.rename(columns={"date": "sharesDate"})


# =========================
# SORT FOR MERGE_ASOF
# =========================
valuation = valuation.sort_values(["earningsAnnouncementDate", "ticker"]).reset_index(drop=True)
prices = prices.sort_values(["anchorDate", "ticker"]).reset_index(drop=True)
shares = shares.sort_values(["sharesDate", "ticker"]).reset_index(drop=True)


# =========================
# MERGE PRIOR TRADING DAY PRICE
# =========================
valuation = pd.merge_asof(
    valuation,
    prices,
    left_on="earningsAnnouncementDate",
    right_on="anchorDate",
    by="ticker",
    direction="backward",
    allow_exact_matches=False,
)

print(f"\nRows after price merge: {len(valuation)}")
print(f"Missing price rows: {valuation['price'].isna().sum()}")

valuation = valuation.dropna(subset=["anchorDate", "price"]).copy()


# =========================
# MERGE LATEST SHARES SNAPSHOT
# =========================
valuation = valuation.sort_values(["anchorDate", "ticker"]).reset_index(drop=True)
shares = shares.sort_values(["sharesDate", "ticker"]).reset_index(drop=True)

valuation = pd.merge_asof(
    valuation,
    shares,
    left_on="anchorDate",
    right_on="sharesDate",
    by="ticker",
    direction="backward",
)

print(f"\nRows after shares merge: {len(valuation)}")
print(f"Missing shares rows: {valuation['sharesOutstanding'].isna().sum()}")


# =========================
# SHARES STALENESS
# =========================
valuation["sharesAgeDays"] = (valuation["anchorDate"] - valuation["sharesDate"]).dt.days

print("\nShares age before stale cutoff:")
print(valuation["sharesAgeDays"].describe())

stale_mask = valuation["sharesAgeDays"] > MAX_SHARES_AGE_DAYS
print(f"Rows with stale shares > {MAX_SHARES_AGE_DAYS} days: {stale_mask.sum()}")

valuation.loc[stale_mask, "sharesOutstanding"] = np.nan


# =========================
# CALCULATE MARKET CAP + RATIOS
# =========================
valuation["marketCap"] = valuation["price"] * valuation["sharesOutstanding"]
valuation["pe"] = valuation["price"] / valuation["ttmEps"]
valuation["ps"] = valuation["marketCap"] / valuation["ttmRevenue"]


# =========================
# BASE CLEANING
# =========================
valuation.loc[valuation["price"] < MIN_PRICE, ["marketCap", "pe", "ps"]] = np.nan
valuation.loc[valuation["marketCap"] < MIN_MARKET_CAP, ["pe", "ps"]] = np.nan

valuation.loc[valuation["ttmEps"] <= 0, "pe"] = np.nan
valuation.loc[valuation["ttmRevenue"] <= 0, "ps"] = np.nan

valuation.loc[valuation["ttmEps"] < MIN_TTM_EPS_FOR_PE, "pe"] = np.nan

valuation.loc[valuation["marketCap"] <= 0, "ps"] = np.nan
valuation = valuation.replace([np.inf, -np.inf], np.nan)


# =========================
# EXTREME OUTLIER FILTERS
# =========================
valuation.loc[valuation["pe"] > MAX_PE, "pe"] = np.nan
valuation.loc[valuation["ps"] > MAX_PS, "ps"] = np.nan

valuation.loc[valuation["pe"] <= 0, "pe"] = np.nan
valuation.loc[valuation["ps"] <= 0, "ps"] = np.nan


# =========================
# CLIPPED SCORE-FRIENDLY VERSIONS
# =========================
valuation["pe_clipped"] = valuation["pe"].clip(lower=0, upper=MAX_PE)
valuation["ps_clipped"] = valuation["ps"].clip(lower=0, upper=MAX_PS)


# =========================
# FINAL COLUMNS
# =========================
valuation = valuation[
    [
        "ticker",
        "earningsAnnouncementDate",
        "anchorDate",
        "sharesDate",
        "sharesAgeDays",
        "price",
        "sharesOutstanding",
        "marketCap",
        "actualEps",
        "actualRevenue",
        "ttmEps",
        "ttmRevenue",
        "quarterCountSeen",
        "pe",
        "ps",
        "pe_clipped",
        "ps_clipped",
    ]
]


# =========================
# SAVE
# =========================
output_path = PROCESSED_DATA_DIR / "valuation_features.csv"
valuation.to_csv(output_path, index=False)


# =========================
# DEBUG OUTPUT
# =========================
print("\nDONE")
print(f"Saved valuation features to: {output_path}")
print(f"Total rows: {len(valuation)}")

print("\nMissing values:")
print(f"Missing shares: {valuation['sharesOutstanding'].isna().sum()}")
print(f"Missing marketCap: {valuation['marketCap'].isna().sum()}")
print(f"Missing TTM EPS: {valuation['ttmEps'].isna().sum()}")
print(f"Missing TTM Revenue: {valuation['ttmRevenue'].isna().sum()}")
print(f"Missing PE: {valuation['pe'].isna().sum()}")
print(f"Missing PS: {valuation['ps'].isna().sum()}")

print("\nUsable counts:")
print(f"PE not null: {valuation['pe'].notna().sum()}")
print(f"PS not null: {valuation['ps'].notna().sum()}")
print(f"Both PE and PS not null: {valuation[['pe', 'ps']].notna().all(axis=1).sum()}")

print("\nSample valuation data:")
print(valuation.head(10))

print("\nPE describe:")
print(valuation["pe"].describe())

print("\nPS describe:")
print(valuation["ps"].describe())

print("\nShares age stats after merge:")
print(valuation["sharesAgeDays"].describe())