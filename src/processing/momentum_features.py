import pandas as pd
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

EVENTS_FILE = PROCESSED_DATA_DIR / "earnings_events.csv"
PRICES_FILE = RAW_DATA_DIR / "daily_prices_from_clean_universe.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "momentum_features.csv"

# Load files
events = pd.read_csv(EVENTS_FILE)
prices = pd.read_csv(PRICES_FILE, usecols=["ticker", "date", "close"])

# Clean dates
events["anchorDate"] = pd.to_datetime(events["anchorDate"])
events["earningsAnnouncementDate"] = pd.to_datetime(events["earningsAnnouncementDate"])
prices["date"] = pd.to_datetime(prices["date"])

# Clean strings
events["ticker"] = events["ticker"].astype(str).str.strip()
prices["ticker"] = prices["ticker"].astype(str).str.strip()

# Clean prices
prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
prices = prices.dropna(subset=["ticker", "date", "close"])
prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

# Group once instead of filtering whole dataframe every event
price_groups = {ticker: group.reset_index(drop=True) for ticker, group in prices.groupby("ticker")}

rows = []

def calc_momentum_from_group(stock_prices, anchor_date, lookback_days):
    # only prices before anchor date
    hist = stock_prices[stock_prices["date"] < anchor_date]

    if len(hist) < lookback_days + 1:
        return pd.NA

    end_price = hist.iloc[-1]["close"]
    start_price = hist.iloc[-(lookback_days + 1)]["close"]

    if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
        return pd.NA

    return round((end_price / start_price) - 1, 4)

total_events = len(events)

for i, event in enumerate(events.itertuples(index=False), start=1):
    ticker = event.ticker
    anchor_date = event.anchorDate
    earnings_date = event.earningsAnnouncementDate

    stock_prices = price_groups.get(ticker)

    if stock_prices is None or stock_prices.empty:
        continue

    mom21 = calc_momentum_from_group(stock_prices, anchor_date, 21)
    mom63 = calc_momentum_from_group(stock_prices, anchor_date, 63)
    mom126 = calc_momentum_from_group(stock_prices, anchor_date, 126)

    rows.append({
        "ticker": ticker,
        "earningsAnnouncementDate": earnings_date.strftime("%Y-%m-%d"),
        "anchorDate": anchor_date.strftime("%Y-%m-%d"),
        "mom21": mom21,
        "mom63": mom63,
        "mom126": mom126,
    })

    if i % 1000 == 0:
        print(f"Processed {i:,} / {total_events:,} events...")

momentum = pd.DataFrame(rows)
momentum = momentum.sort_values(["ticker", "earningsAnnouncementDate"]).reset_index(drop=True)

momentum.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print(f"Saved {len(momentum):,} rows to {OUTPUT_FILE}")