import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (
    EARNINGS_FROM_PRICE_START_FILE,
    GROWTH_FEATURES_FILE,
    PROCESSED_FACTORS_DIR,
    RAW_DATA_DIR,
    TTM_FINANCIALS_ENRICHED_FILE,
)

HISTORY_FILE = RAW_DATA_DIR / "earnings_from_clean_universe.csv"
TARGET_FILE = EARNINGS_FROM_PRICE_START_FILE
OUTPUT_FILE = GROWTH_FEATURES_FILE


def run() -> None:
    history = pd.read_csv(HISTORY_FILE, low_memory=False)
    target = pd.read_csv(TARGET_FILE, low_memory=False)

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

    history = history.drop_duplicates(
        subset=["ticker", "earningsAnnouncementDate"], keep="last"
    )
    history = history.sort_values(["ticker", "earningsAnnouncementDate"])

    history["ttmEps"] = (
        history.groupby("ticker", group_keys=False)["actualEps"]
        .rolling(window=4, min_periods=4)
        .sum()
        .reset_index(level=0, drop=True)
    )
    history["ttmRevenue"] = (
        history.groupby("ticker", group_keys=False)["actualRevenue"]
        .rolling(window=4, min_periods=4)
        .sum()
        .reset_index(level=0, drop=True)
    )

    g = history.groupby("ticker", group_keys=False)
    prev_rev = g["ttmRevenue"].shift(4)
    prev_eps = g["ttmEps"].shift(4)

    history["ttmRevenue_yoy"] = np.where(
        prev_rev.abs() > 1e-9,
        (history["ttmRevenue"] - prev_rev) / prev_rev.abs(),
        np.nan,
    )
    history["ttmEps_yoy"] = np.where(
        prev_eps.abs() > 1e-9,
        (history["ttmEps"] - prev_eps) / prev_eps.abs(),
        np.nan,
    )

    keep = history[
        [
            "ticker",
            "earningsAnnouncementDate",
            "ttmRevenue",
            "ttmEps",
            "ttmRevenue_yoy",
            "ttmEps_yoy",
        ]
    ].copy()

    out = target.merge(keep, on=["ticker", "earningsAnnouncementDate"], how="left")

    # Enrich with EBITDA YoY and FCF YoY from financial statements (if available)
    ENRICHED_FILE = TTM_FINANCIALS_ENRICHED_FILE
    if ENRICHED_FILE.exists():
        print("Merging EBITDA YoY and FCF YoY from enriched TTM financials...")
        enr = pd.read_csv(ENRICHED_FILE, low_memory=False)
        enr["earningsAnnouncementDate"] = pd.to_datetime(enr["earningsAnnouncementDate"], errors="coerce")
        enr["ticker"] = enr["ticker"].astype(str).str.strip()
        for c in ["ttm_ebitda", "ttm_freeCashFlow"]:
            if c in enr.columns:
                enr[c] = pd.to_numeric(enr[c], errors="coerce")

        # Compute EBITDA YoY and FCF YoY using 4-quarter prior
        enr = enr.sort_values(["ticker", "earningsAnnouncementDate"])
        g2 = enr.groupby("ticker", group_keys=False)
        prev_ebitda = g2["ttm_ebitda"].shift(4)
        prev_fcf    = g2["ttm_freeCashFlow"].shift(4)

        enr["ttmEbitda_yoy"] = np.where(
            prev_ebitda.abs() > 1e4,
            (enr["ttm_ebitda"] - prev_ebitda) / prev_ebitda.abs(), np.nan
        )
        enr["ttmFcf_yoy"] = np.where(
            prev_fcf.abs() > 1e4,
            (enr["ttm_freeCashFlow"] - prev_fcf) / prev_fcf.abs(), np.nan
        )

        out = out.merge(
            enr[["ticker","earningsAnnouncementDate","ttmEbitda_yoy","ttmFcf_yoy"]],
            on=["ticker","earningsAnnouncementDate"], how="left"
        )
        print(f"  EBITDA YoY coverage: {out['ttmEbitda_yoy'].notna().sum():,}")
        print(f"  FCF YoY coverage:    {out['ttmFcf_yoy'].notna().sum():,}")
    else:
        out["ttmEbitda_yoy"] = np.nan
        out["ttmFcf_yoy"]    = np.nan

    PROCESSED_FACTORS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(out)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
