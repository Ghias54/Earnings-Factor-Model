"""
Build revisions features using EPS surprise (actual vs. consensus estimate at earnings).

The original approach tried to compute 90-day estimate changes using
analyst_estimates_quarterly.csv, but that file only contains ONE snapshot per
reporting period (the estimate as of the data pull date), not a time-series of
how estimates evolved.  Comparing the current quarter's estimate against the prior
quarter's estimate produces cross-quarter seasonal noise, not real revisions.

Instead we use EPS surprise % — a strong proxy for analyst estimate revisions
because large beats typically trigger upward revisions and vice versa.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DATA_DIR

VALUATION_FILE = PROCESSED_DATA_DIR / "valuation_features.csv"
EARNINGS_FILE  = PROCESSED_DATA_DIR / "earnings_events.csv"
OUTPUT_FILE    = PROCESSED_DATA_DIR / "revisions_features.csv"


def run() -> None:
    # Base: all earnings events with anchor dates (same universe as other factors)
    val = pd.read_csv(VALUATION_FILE, low_memory=False,
                      usecols=["ticker", "earningsAnnouncementDate", "anchorDate"])
    val["earningsAnnouncementDate"] = pd.to_datetime(val["earningsAnnouncementDate"], errors="coerce")
    val["anchorDate"]               = pd.to_datetime(val["anchorDate"],               errors="coerce")
    val["ticker"]                   = val["ticker"].astype(str).str.strip()

    # EPS surprise from earnings events
    if EARNINGS_FILE.exists():
        ev = pd.read_csv(EARNINGS_FILE, low_memory=False,
                         usecols=["ticker", "earningsAnnouncementDate", "actualEps", "estimatedEps"])
        ev["earningsAnnouncementDate"] = pd.to_datetime(ev["earningsAnnouncementDate"], errors="coerce")
        ev["ticker"]        = ev["ticker"].astype(str).str.strip()
        ev["actualEps"]     = pd.to_numeric(ev["actualEps"],    errors="coerce")
        ev["estimatedEps"]  = pd.to_numeric(ev["estimatedEps"], errors="coerce")

        # surprise % = (actual - estimated) / |estimated|; NaN if no estimate
        ev["eps_surprise_pct"] = np.where(
            ev["estimatedEps"].abs() > 1e-9,
            (ev["actualEps"] - ev["estimatedEps"]) / ev["estimatedEps"].abs(),
            np.nan,
        )
        # Cap extreme outliers
        ev.loc[ev["eps_surprise_pct"].abs() > 5, "eps_surprise_pct"] = np.nan

        ev = ev[["ticker", "earningsAnnouncementDate", "eps_surprise_pct"]].drop_duplicates(
            subset=["ticker", "earningsAnnouncementDate"]
        )
        out = val.merge(ev, on=["ticker", "earningsAnnouncementDate"], how="left")
    else:
        print("No earnings_events.csv found; eps_surprise_pct will be NaN.")
        out = val.copy()
        out["eps_surprise_pct"] = np.nan

    out = out.sort_values(["ticker", "earningsAnnouncementDate"]).reset_index(drop=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(out):,} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
