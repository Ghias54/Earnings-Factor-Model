"""
Build revisions features from the same "Our Model" primitives:
- EPS surprise % (actual vs estimate)
- Forward-curve step % from one snapshot sequence (estimated EPS vs prior estimate)
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

    # Revisions proxies from earnings events
    if EARNINGS_FILE.exists():
        ev = pd.read_csv(EARNINGS_FILE, low_memory=False,
                         usecols=["ticker", "earningsAnnouncementDate", "actualEps", "estimatedEps"])
        ev["earningsAnnouncementDate"] = pd.to_datetime(ev["earningsAnnouncementDate"], errors="coerce")
        ev["ticker"]        = ev["ticker"].astype(str).str.strip()
        ev["actualEps"] = pd.to_numeric(ev["actualEps"], errors="coerce")
        ev["estimatedEps"] = pd.to_numeric(ev["estimatedEps"], errors="coerce")

        # surprise % = (actual - estimated) / |estimated|; NaN if no estimate
        ev["eps_surprise_pct"] = np.where(
            ev["estimatedEps"].abs() > 1e-9,
            (ev["actualEps"] - ev["estimatedEps"]) / ev["estimatedEps"].abs(),
            np.nan,
        )
        # Cap extreme outliers
        ev.loc[ev["eps_surprise_pct"].abs() > 5, "eps_surprise_pct"] = np.nan

        # Forward curve step from one-snapshot estimate sequence:
        # compare current estimate to prior report's estimate for same ticker.
        ev = ev.sort_values(["ticker", "earningsAnnouncementDate"], kind="mergesort").reset_index(drop=True)
        prev_est = ev.groupby("ticker")["estimatedEps"].shift(1)
        ev["eps_fwd_step_pct"] = np.where(
            prev_est.abs() > 1e-9,
            (ev["estimatedEps"] - prev_est) / prev_est.abs(),
            np.nan,
        )
        ev.loc[ev["eps_fwd_step_pct"].abs() > 5, "eps_fwd_step_pct"] = np.nan

        ev = ev[["ticker", "earningsAnnouncementDate", "eps_surprise_pct", "eps_fwd_step_pct"]].drop_duplicates(
            subset=["ticker", "earningsAnnouncementDate"]
        )
        out = val.merge(ev, on=["ticker", "earningsAnnouncementDate"], how="left")
    else:
        print("No earnings_events.csv found; revisions features will be NaN.")
        out = val.copy()
        out["eps_surprise_pct"] = np.nan
        out["eps_fwd_step_pct"] = np.nan

    out = out.sort_values(["ticker", "earningsAnnouncementDate"]).reset_index(drop=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(out):,} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
