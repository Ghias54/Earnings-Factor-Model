"""
Build profitability features per earnings event.

Primary source: ttm_financials_enriched.csv (from financial_statements_quarterly.csv)
  → gross_margin, EBITDA_margin, operating_margin, net_margin, ROA, ROE, FCF_margin

Fallback (if enriched file unavailable): valuation_features.csv
  → net_margin, roe_proxy (as before)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DATA_DIR

VALUATION_FILE  = PROCESSED_DATA_DIR / "valuation_features.csv"
ENRICHED_FILE   = PROCESSED_DATA_DIR / "ttm_financials_enriched.csv"
OUTPUT_FILE     = PROCESSED_DATA_DIR / "profitability_features.csv"

MAX_MARGIN = 2.0   # clip margins beyond ±200%


def run() -> None:
    val = pd.read_csv(VALUATION_FILE, low_memory=False)
    val["earningsAnnouncementDate"] = pd.to_datetime(val["earningsAnnouncementDate"], errors="coerce")
    val["ticker"] = val["ticker"].astype(str).str.strip()
    if "anchorDate" in val.columns:
        val["anchorDate"] = pd.to_datetime(val["anchorDate"], errors="coerce")
        val = val.sort_values(["ticker", "earningsAnnouncementDate", "anchorDate"])
    else:
        val = val.sort_values(["ticker", "earningsAnnouncementDate"])
    val = val.drop_duplicates(["ticker", "earningsAnnouncementDate"], keep="last")

    if ENRICHED_FILE.exists():
        print("Using enriched TTM financials (gross margin, EBITDA, ROA, FCF)...")
        ttm = pd.read_csv(ENRICHED_FILE, low_memory=False)
        ttm["earningsAnnouncementDate"] = pd.to_datetime(ttm["earningsAnnouncementDate"], errors="coerce")
        ttm["ticker"] = ttm["ticker"].astype(str).str.strip()
        ttm = ttm.sort_values(["ticker", "earningsAnnouncementDate"])
        ttm = ttm.drop_duplicates(["ticker", "earningsAnnouncementDate"], keep="last")

        for c in ["ttm_revenue","ttm_grossProfit","ttm_ebitda","ttm_operatingIncome",
                  "ttm_netIncome","ttm_freeCashFlow","ttm_totalAssets",
                  "ttm_totalStockholdersEquity"]:
            if c in ttm.columns:
                ttm[c] = pd.to_numeric(ttm[c], errors="coerce")

        df = val.merge(
            ttm[["ticker","earningsAnnouncementDate",
                 "ttm_revenue","ttm_grossProfit","ttm_ebitda","ttm_operatingIncome",
                 "ttm_netIncome","ttm_freeCashFlow","ttm_totalAssets",
                 "ttm_totalStockholdersEquity"]],
            on=["ticker","earningsAnnouncementDate"], how="left", validate="one_to_one"
        )

        rev = df["ttm_revenue"]
        df["gross_margin"]     = np.where(rev.abs() > 1, df["ttm_grossProfit"]     / rev, np.nan)
        df["ebitda_margin"]    = np.where(rev.abs() > 1, df["ttm_ebitda"]          / rev, np.nan)
        df["operating_margin"] = np.where(rev.abs() > 1, df["ttm_operatingIncome"] / rev, np.nan)
        df["fcf_margin"]       = np.where(rev.abs() > 1, df["ttm_freeCashFlow"]    / rev, np.nan)

        # Net margin from financial statements (more accurate than EPS×shares/revenue)
        df["net_margin"] = np.where(rev.abs() > 1, df["ttm_netIncome"] / rev, np.nan)

        # ROA = net income / total assets
        df["roa"] = np.where(
            df["ttm_totalAssets"].abs() > 1,
            df["ttm_netIncome"] / df["ttm_totalAssets"], np.nan
        )

        # ROE = net income / equity (skip if negative equity)
        pos_equity = df["ttm_totalStockholdersEquity"] > 1
        df["roe"] = np.where(
            pos_equity,
            df["ttm_netIncome"] / df["ttm_totalStockholdersEquity"], np.nan
        )

        # Clip extremes
        for c in ["gross_margin","ebitda_margin","operating_margin","net_margin","fcf_margin","roa","roe"]:
            df.loc[df[c].abs() > MAX_MARGIN, c] = np.nan

        print(f"  gross_margin coverage: {df['gross_margin'].notna().sum():,}")
        print(f"  ebitda_margin coverage: {df['ebitda_margin'].notna().sum():,}")
        print(f"  roa coverage: {df['roa'].notna().sum():,}")

    else:
        print("Enriched TTM file not found — falling back to net_margin + roe_proxy from valuation_features.")
        df = val.copy()
        net_income = pd.to_numeric(df["ttmEps"], errors="coerce") * pd.to_numeric(df["sharesOutstanding"], errors="coerce")
        df["net_margin"] = np.where(
            pd.to_numeric(df["ttmRevenue"], errors="coerce").abs() > 1e-9,
            net_income / pd.to_numeric(df["ttmRevenue"], errors="coerce"), np.nan
        )
        df.loc[df["net_margin"].abs() > MAX_MARGIN, "net_margin"] = np.nan
        df["roe"] = np.where(
            pd.to_numeric(df["marketCap"], errors="coerce").abs() > 1e-9,
            net_income / pd.to_numeric(df["marketCap"], errors="coerce"), np.nan
        )
        df.loc[df["roe"].abs() > MAX_MARGIN, "roe"] = np.nan
        for c in ["gross_margin","ebitda_margin","operating_margin","fcf_margin","roa"]:
            df[c] = np.nan

    keep = ["ticker","earningsAnnouncementDate","anchorDate",
            "net_margin","gross_margin","ebitda_margin","operating_margin",
            "fcf_margin","roa","roe"]
    out = df[[c for c in keep if c in df.columns]].copy()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(out):,} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
