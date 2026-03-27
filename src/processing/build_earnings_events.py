import pandas as pd
from config import PROCESSED_DATA_DIR

INPUT_FILE = PROCESSED_DATA_DIR / "earnings_returns.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "earnings_events.csv"

df = pd.read_csv(INPUT_FILE)

keep_cols = [
    "ticker",
    "earningsAnnouncementDate",
    "actualEps",
    "estimatedEps",
    "actualRevenue",
    "estimatedRevenue",
    "lastUpdated",
    "anchorDate",
]

keep_cols = [col for col in keep_cols if col in df.columns]
events = df[keep_cols].copy()

events = events.drop_duplicates(subset=["ticker", "earningsAnnouncementDate"])

if "actualEps" in events.columns and "estimatedEps" in events.columns:
    events["epsSurprise"] = (
        (events["actualEps"] - events["estimatedEps"]) /
        events["estimatedEps"].replace(0, pd.NA).abs()
    ).round(4)
else:
    events["epsSurprise"] = pd.NA

if "actualRevenue" in events.columns and "estimatedRevenue" in events.columns:
    events["revenueSurprise"] = (
        (events["actualRevenue"] - events["estimatedRevenue"]) /
        events["estimatedRevenue"].replace(0, pd.NA).abs()
    ).round(4)
else:
    events["revenueSurprise"] = pd.NA

events["ticker"] = events["ticker"].astype(str).str.strip()

events = events.sort_values(["ticker", "earningsAnnouncementDate"]).reset_index(drop=True)

events.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print(f"Saved {len(events)} rows to {OUTPUT_FILE}")