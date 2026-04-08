from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Project root (so `python src/processing/earnings_returns.py` finds config)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (
    EARNINGS_FROM_PRICE_START_FILE,
    EARNINGS_BUY_DAYS_BEFORE_ANCHOR,
    EARNINGS_SELL_DAYS_AFTER_ANCHOR,
    EARNINGS_RETURNS_FILE,
    EARNINGS_RETURNS_SKIPPED_FILE,
    MIN_BUY_PRICE_FOR_TRADE,
    PROCESSED_FACTORS_DIR,
    RAW_DATA_DIR,
)


EARNINGS_FILE = EARNINGS_FROM_PRICE_START_FILE
PRICES_FILE = RAW_DATA_DIR / "daily_prices_from_clean_universe.csv"

TAKEN_OUTPUT_FILE = EARNINGS_RETURNS_FILE
SKIPPED_OUTPUT_FILE = EARNINGS_RETURNS_SKIPPED_FILE


def load_earnings() -> pd.DataFrame:
    df = pd.read_csv(EARNINGS_FILE)

    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["earningsAnnouncementDate"] = pd.to_datetime(
        df["earningsAnnouncementDate"], errors="coerce"
    )

    df = df.dropna(subset=["ticker", "earningsAnnouncementDate"]).copy()

    return df


def load_prices() -> pd.DataFrame:
    df = pd.read_csv(PRICES_FILE, on_bad_lines="skip", low_memory=False)

    # remove accidental header rows from append
    df = df[df["ticker"] != "ticker"].copy()

    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = df.dropna(subset=["ticker", "date", "close"]).copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    return df


def prepare_price_lookup(prices_df: pd.DataFrame) -> dict:
    price_lookup = {}

    for ticker, group in prices_df.groupby("ticker", sort=False):
        group = group.sort_values("date").reset_index(drop=True).copy()
        group["trade_index"] = range(len(group))
        price_lookup[ticker] = group

    return price_lookup


def process_earnings_event(
    event_row,
    price_lookup: dict,
    *,
    buy_days_before_anchor: int,
    sell_days_after_anchor: int,
    min_buy_price: float,
):
    ticker = event_row["ticker"]
    earnings_date = event_row["earningsAnnouncementDate"]

    base_taken = {
        "ticker": ticker,
        "earningsAnnouncementDate": earnings_date,
        "buyDaysBefore": buy_days_before_anchor,
        "sellDaysAfter": sell_days_after_anchor,
        "actualEps": event_row.get("actualEps"),
        "estimatedEps": event_row.get("estimatedEps"),
        "actualRevenue": event_row.get("actualRevenue"),
        "estimatedRevenue": event_row.get("estimatedRevenue"),
        "lastUpdated": event_row.get("lastUpdated"),
    }

    base_skipped = base_taken.copy()

    if ticker not in price_lookup:
        base_skipped["skipReason"] = "Ticker not found in prices"
        return None, base_skipped

    prices = price_lookup[ticker]

    # CRITICAL: ensure datetime dtype
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")

    event_pos = prices["date"].searchsorted(earnings_date, side="left")

    if event_pos <= 0:
        base_skipped["skipReason"] = "No trading day before earnings"
        return None, base_skipped

    anchor_index = event_pos - 1

    buy_index = anchor_index - buy_days_before_anchor
    sell_index = anchor_index + sell_days_after_anchor

    if buy_index < 0:
        base_skipped["skipReason"] = "Not enough days before earnings"
        return None, base_skipped

    if sell_index >= len(prices):
        base_skipped["skipReason"] = "Not enough days after earnings"
        return None, base_skipped

    buy_row = prices.iloc[buy_index]
    sell_row = prices.iloc[sell_index]

    buy_price = buy_row["close"]
    sell_price = sell_row["close"]

    if buy_price <= min_buy_price:
        base_skipped["skipReason"] = f"Buy price at or below ${min_buy_price:g}"
        return None, base_skipped

    if buy_row["volume"] < 100000:
        base_skipped["skipReason"] = "Low volume"
        return None, base_skipped

    if pd.isna(buy_price):
        base_skipped["skipReason"] = "Buy price missing"
        return None, base_skipped

    if pd.isna(sell_price):
        base_skipped["skipReason"] = "Sell price missing"
        return None, base_skipped

    if buy_price == 0:
        base_skipped["skipReason"] = "Buy price zero"
        return None, base_skipped

    trade = {
        **base_taken,
        "anchorDate": prices.iloc[anchor_index]["date"],
        "buyDate": buy_row["date"],
        "buyPrice": buy_price,
        "buyVolume": buy_row["volume"],
        "sellDate": sell_row["date"],
        "sellPrice": sell_price,
        "sellVolume": sell_row["volume"],
        "returnDecimal": (sell_price - buy_price) / buy_price,
        "returnPct": ((sell_price - buy_price) / buy_price) * 100,
    }

    return trade, None


def build_earnings_returns(
    earnings_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    buy_days_before_anchor: int | None = None,
    sell_days_after_anchor: int | None = None,
    min_buy_price: float | None = None,
):
    buy_days_before_anchor = (
        buy_days_before_anchor
        if buy_days_before_anchor is not None
        else EARNINGS_BUY_DAYS_BEFORE_ANCHOR
    )
    sell_days_after_anchor = (
        sell_days_after_anchor
        if sell_days_after_anchor is not None
        else EARNINGS_SELL_DAYS_AFTER_ANCHOR
    )
    min_buy_price = (
        min_buy_price if min_buy_price is not None else MIN_BUY_PRICE_FOR_TRADE
    )

    price_lookup = prepare_price_lookup(prices_df)

    taken_rows = []
    skipped_rows = []

    total = len(earnings_df)

    for i, (_, event_row) in enumerate(earnings_df.iterrows(), start=1):
        if i % 5000 == 0:
            print(f"Processed {i}/{total} earnings events...")

        trade_row, skipped_row = process_earnings_event(
            event_row,
            price_lookup,
            buy_days_before_anchor=buy_days_before_anchor,
            sell_days_after_anchor=sell_days_after_anchor,
            min_buy_price=min_buy_price,
        )

        if trade_row is not None:
            taken_rows.append(trade_row)
        else:
            skipped_rows.append(skipped_row)

    return pd.DataFrame(taken_rows), pd.DataFrame(skipped_rows)


def run(
    *,
    buy_days_before_anchor: int | None = None,
    sell_days_after_anchor: int | None = None,
    min_buy_price: float | None = None,
) -> None:
    PROCESSED_FACTORS_DIR.mkdir(parents=True, exist_ok=True)

    buy_days_before_anchor = (
        buy_days_before_anchor
        if buy_days_before_anchor is not None
        else EARNINGS_BUY_DAYS_BEFORE_ANCHOR
    )
    sell_days_after_anchor = (
        sell_days_after_anchor
        if sell_days_after_anchor is not None
        else EARNINGS_SELL_DAYS_AFTER_ANCHOR
    )
    min_buy_price = (
        min_buy_price if min_buy_price is not None else MIN_BUY_PRICE_FOR_TRADE
    )

    print("Loading earnings...")
    earnings_df = load_earnings()
    print(f"Earnings rows loaded: {len(earnings_df)}")

    print("Loading prices...")
    prices_df = load_prices()
    print(f"Price rows loaded: {len(prices_df)}")

    print(
        "Building trade dataset "
        f"(buy {buy_days_before_anchor} td before anchor, "
        f"sell {sell_days_after_anchor} td after anchor, "
        f"min buy price > ${min_buy_price:g})..."
    )
    taken_df, skipped_df = build_earnings_returns(
        earnings_df,
        prices_df,
        buy_days_before_anchor=buy_days_before_anchor,
        sell_days_after_anchor=sell_days_after_anchor,
        min_buy_price=min_buy_price,
    )

    taken_df.to_csv(TAKEN_OUTPUT_FILE, index=False)
    skipped_df.to_csv(SKIPPED_OUTPUT_FILE, index=False)

    print("\nDone.")
    print(f"Trades taken: {len(taken_df)}")
    print(f"Trades skipped: {len(skipped_df)}")

    if len(taken_df) > 0:
        print(f"Win rate: {(taken_df['returnDecimal'] > 0).mean() * 100:.2f}%")
        print(f"Average return: {taken_df['returnPct'].mean():.4f}%")
        print(f"Median return: {taken_df['returnPct'].median():.4f}%")

    if len(skipped_df) > 0:
        print("\nTop skip reasons:")
        print(skipped_df["skipReason"].value_counts().head(10))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build earnings_returns.csv: buy/sell windows are trading days "
            "relative to the last bar before earningsAnnouncementDate (anchor)."
        )
    )
    p.add_argument(
        "--buy-before",
        type=int,
        default=None,
        metavar="N",
        help=f"Trading days before anchor to buy (default: {EARNINGS_BUY_DAYS_BEFORE_ANCHOR})",
    )
    p.add_argument(
        "--sell-after",
        type=int,
        default=None,
        metavar="N",
        help=f"Trading days after anchor to sell (default: {EARNINGS_SELL_DAYS_AFTER_ANCHOR})",
    )
    p.add_argument(
        "--min-buy-price",
        type=float,
        default=None,
        metavar="P",
        help=f"Skip if buy close <= this (default: {MIN_BUY_PRICE_FOR_TRADE})",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        buy_days_before_anchor=args.buy_before,
        sell_days_after_anchor=args.sell_after,
        min_buy_price=args.min_buy_price,
    )