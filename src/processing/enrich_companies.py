"""Merge FMP `company_profiles` into `companies_cleaned` for sector, cap, exchange, etc."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

PROFILES_FILE = RAW_DATA_DIR / "company_profiles.csv"
INPUT_FILE = PROCESSED_DATA_DIR / "companies_cleaned.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "companies_enriched.csv"


def run() -> None:
    if not PROFILES_FILE.exists():
        print(f"No profiles file at {PROFILES_FILE}; copy cleaned only.")
        df = pd.read_csv(INPUT_FILE)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Wrote {OUTPUT_FILE} ({len(df)} rows) without enrichment.")
        return

    base = pd.read_csv(INPUT_FILE)
    prof = pd.read_csv(PROFILES_FILE, low_memory=False)

    base["ticker"] = base["ticker"].astype(str).str.strip()
    prof["ticker"] = prof["ticker"].astype(str).str.strip()

    prof = prof.drop_duplicates(subset=["ticker"], keep="last")

    merged = base.merge(prof, on="ticker", how="left", suffixes=("", "_prof"))

    if "companyName_prof" in merged.columns:
        merged = merged.drop(columns=["companyName_prof"], errors="ignore")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(merged)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
