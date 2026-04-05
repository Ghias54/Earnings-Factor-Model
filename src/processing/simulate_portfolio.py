"""
Earnings-event portfolio backtest.

Methodology (research / simplified execution model):
- One candidate trade per (ticker, earningsAnnouncementDate) from earnings_returns.
- Merge momentum + composite quant scores **as of each buy date** (backward ``merge_asof`` on the
  last announced earnings snapshot on or before ``buyDate`` — avoids look-ahead vs merging on the
  same row's ``earningsAnnouncementDate`` when buying before announcement).
- **only enter** when quant is Buy or Strong Buy.
- **Strong Buy** and **Buy** map to the existing 0–5 `composite_rating` and letter grades
  produced by `build_composite_quant_score.py` (same scale as other factor scripts).
- Each calendar day: close matured trades first, then open new trades up to max_positions.
- New trades that day are the **highest** `composite_rating` first (no arbitrary CSV row order).
- Equal-weight allocation of **available cash** across **available slots** for that day’s openings.
- Round-trip transaction cost applied to each trade’s return.

This is not live execution (no slippage, borrow, partial fills, or exchange hours).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Project root (so `python src/processing/simulate_portfolio.py` finds config.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import MIN_BUY_PRICE_FOR_TRADE, PROCESSED_DATA_DIR

RETURNS_FILE = PROCESSED_DATA_DIR / "earnings_returns.csv"
MOMENTUM_FILE = PROCESSED_DATA_DIR / "momentum_features.csv"
EVENTS_FILE = PROCESSED_DATA_DIR / "earnings_events.csv"
COMPOSITE_FILE = PROCESSED_DATA_DIR / "composite_quant_scores.csv"

EQUITY_OUTPUT_FILE = PROCESSED_DATA_DIR / "equity_curve.csv"
EQUITY_CHART_FILE = PROCESSED_DATA_DIR / "equity_curve.png"
DRAWDOWN_CHART_FILE = PROCESSED_DATA_DIR / "drawdown_curve.png"
RETURNS_HIST_FILE = PROCESSED_DATA_DIR / "trade_return_histogram.png"

# --- Portfolio knobs ---
STARTING_CAPITAL = 10_000.0
MAX_POSITIONS = 10
TRANSACTION_COST_ROUND_TRIP = 0.002  # 0.20% round-trip per trade

# --- Quant filter: only Buy / Strong Buy ---
# Composite uses 0–5 `composite_rating` and letter grades (rating_to_grade scale).
# Strong Buy: letter A+ or A, or numeric >= 4.0
# Buy: letter A- or B+, or numeric in [3.33, 4.0)
# No trade at B or below.
STRONG_BUY_GRADES = frozenset({"A+", "A"})
BUY_GRADES = frozenset({"A-", "B+"})
STRONG_BUY_RATING_MIN = 4.0
BUY_RATING_MIN = 3.33  # bottom of B+ on this scale
DEFAULT_MIN_COMPOSITE_COMPONENTS = 3  # require enough factor coverage


@dataclass
class OpenPosition:
    ticker: str
    buy_date: pd.Timestamp
    sell_date: pd.Timestamp
    allocated: float
    final_value: float
    gross_return: float
    net_return: float


def quant_tier(composite_rating: float, composite_grade: object) -> str:
    """Map to strong_buy / buy / hold. Grades win if present; else use rating cutoffs."""
    g = composite_grade
    if pd.notna(g) and str(g).strip() in STRONG_BUY_GRADES:
        return "strong_buy"
    if pd.notna(g) and str(g).strip() in BUY_GRADES:
        return "buy"
    if pd.notna(composite_rating) and float(composite_rating) >= STRONG_BUY_RATING_MIN:
        return "strong_buy"
    if pd.notna(composite_rating) and float(composite_rating) >= BUY_RATING_MIN:
        return "buy"
    return "hold"


def load_and_merge(min_components: int, min_buy_price: float) -> pd.DataFrame:
    returns_df = pd.read_csv(RETURNS_FILE)
    momentum_df = pd.read_csv(MOMENTUM_FILE)
    events_df = pd.read_csv(EVENTS_FILE)

    for df in (returns_df, momentum_df, events_df):
        df["ticker"] = df["ticker"].astype(str).str.strip()
        df["earningsAnnouncementDate"] = pd.to_datetime(
            df["earningsAnnouncementDate"], errors="coerce"
        )

    df = returns_df.merge(
        momentum_df,
        on=["ticker", "earningsAnnouncementDate"],
        how="inner",
    )
    df = df.merge(
        events_df[["ticker", "earningsAnnouncementDate", "epsSurprise"]],
        on=["ticker", "earningsAnnouncementDate"],
        how="left",
    )

    df["buyDate"] = pd.to_datetime(df["buyDate"], errors="coerce")

    if not COMPOSITE_FILE.exists():
        raise FileNotFoundError(
            f"Missing {COMPOSITE_FILE}. Run: PYTHONPATH=. python build_quant_scores.py"
        )

    comp_df = pd.read_csv(COMPOSITE_FILE, low_memory=False)
    comp_df["ticker"] = comp_df["ticker"].astype(str).str.strip()
    comp_df["earningsAnnouncementDate"] = pd.to_datetime(
        comp_df["earningsAnnouncementDate"], errors="coerce"
    )
    comp_df = comp_df.dropna(subset=["ticker", "earningsAnnouncementDate"])
    comp_df = comp_df.rename(
        columns={"earningsAnnouncementDate": "composite_source_announcement_date"}
    )
    comp_df = comp_df.sort_values(["ticker", "composite_source_announcement_date"])

    score_cols = [
        c
        for c in comp_df.columns
        if c
        not in (
            "ticker",
            "composite_source_announcement_date",
        )
    ]
    comp_for_asof = comp_df[["ticker", "composite_source_announcement_date"] + score_cols]

    df = df.sort_values(["ticker", "buyDate"])
    # Per-ticker merge_asof: pandas can reject global `by=` when buyDate is not monotonic across tickers.
    merged_parts: list[pd.DataFrame] = []
    for t, dgrp in df.groupby("ticker", sort=False):
        dgrp = dgrp.sort_values("buyDate", kind="mergesort")
        cgrp = comp_for_asof[comp_for_asof["ticker"] == t].drop(columns=["ticker"]).sort_values(
            "composite_source_announcement_date", kind="mergesort"
        )
        if cgrp.empty:
            out = dgrp.copy()
            for col in comp_for_asof.columns:
                if col == "ticker":
                    continue
                if col not in out.columns:
                    out[col] = np.nan
            merged_parts.append(out)
            continue
        m = pd.merge_asof(
            dgrp,
            cgrp,
            left_on="buyDate",
            right_on="composite_source_announcement_date",
            direction="backward",
        )
        merged_parts.append(m)
    df = pd.concat(merged_parts, ignore_index=True)

    df["returnDecimal"] = pd.to_numeric(df["returnDecimal"], errors="coerce")
    df["buyPrice"] = pd.to_numeric(df["buyPrice"], errors="coerce")
    df["composite_rating"] = pd.to_numeric(df["composite_rating"], errors="coerce")
    if "composite_component_count" in df.columns:
        df["composite_component_count"] = pd.to_numeric(
            df["composite_component_count"], errors="coerce"
        )
    df["sellDate"] = pd.to_datetime(df["sellDate"], errors="coerce")

    df = df.dropna(
        subset=[
            "returnDecimal",
            "buyPrice",
            "buyDate",
            "sellDate",
            "composite_rating",
        ]
    ).copy()

    if "composite_component_count" in df.columns:
        df = df[df["composite_component_count"] >= min_components].copy()

    df = df[df["buyPrice"] > min_buy_price].copy()

    grades = (
        df["composite_grade"]
        if "composite_grade" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    df["quant_tier"] = [
        quant_tier(r, g) for r, g in zip(df["composite_rating"], grades)
    ]
    df = df[df["quant_tier"].isin(["strong_buy", "buy"])].copy()

    df["netReturn"] = (df["returnDecimal"] - TRANSACTION_COST_ROUND_TRIP).clip(lower=-1.0)

    # Best signals first when multiple trades share a buy date
    df = df.sort_values(
        ["buyDate", "composite_rating", "ticker"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    df = df.drop_duplicates(subset=["buyDate", "ticker"], keep="first")

    return df


def run_simulation(strategy_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategy_df = strategy_df.sort_values("buyDate").reset_index(drop=True)

    capital = STARTING_CAPITAL
    open_positions: list[OpenPosition] = []
    equity_rows: list[dict] = []

    all_dates = sorted(
        pd.unique(pd.concat([strategy_df["buyDate"], strategy_df["sellDate"]]))
    )

    for current_date in all_dates:
        still_open: list[OpenPosition] = []
        for pos in open_positions:
            if pos.sell_date <= current_date:
                capital += pos.final_value
            else:
                still_open.append(pos)
        open_positions = still_open

        todays = strategy_df[strategy_df["buyDate"] == current_date]
        available_slots = MAX_POSITIONS - len(open_positions)

        if available_slots > 0 and len(todays) > 0:
            trades_to_take = todays.head(available_slots)
            n_open = len(trades_to_take)
            allocation_each = capital / n_open if n_open > 0 else 0.0

            for _, row in trades_to_take.iterrows():
                if capital <= 0:
                    break
                alloc = min(allocation_each, capital)
                if alloc <= 0:
                    break
                final_value = alloc * (1.0 + float(row["netReturn"]))
                open_positions.append(
                    OpenPosition(
                        ticker=str(row["ticker"]),
                        buy_date=row["buyDate"],
                        sell_date=row["sellDate"],
                        allocated=alloc,
                        final_value=final_value,
                        gross_return=float(row["returnDecimal"]),
                        net_return=float(row["netReturn"]),
                    )
                )
                capital -= alloc

        total_equity = capital + sum(p.final_value for p in open_positions)
        equity_rows.append(
            {
                "date": current_date,
                "cash": capital,
                "open_positions": len(open_positions),
                "equity": total_equity,
            }
        )

    equity_df = pd.DataFrame(equity_rows).sort_values("date").reset_index(drop=True)
    equity_df["daily_return"] = equity_df["equity"].pct_change()
    equity_df["running_max"] = equity_df["equity"].cummax()
    equity_df["drawdown"] = (equity_df["equity"] / equity_df["running_max"]) - 1.0

    return equity_df, strategy_df


def plot_charts(equity_df: pd.DataFrame, strategy_df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(equity_df["date"], equity_df["equity"])
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(EQUITY_CHART_FILE, dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(equity_df["date"], equity_df["drawdown"])
    plt.title("Drawdown Curve")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(DRAWDOWN_CHART_FILE, dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.hist(strategy_df["netReturn"], bins=100)
    plt.title("Net Trade Return Distribution (Buy / Strong Buy only)")
    plt.xlabel("Net Trade Return")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RETURNS_HIST_FILE, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest earnings strategy with quant filter.")
    parser.add_argument(
        "--min-components",
        type=int,
        default=DEFAULT_MIN_COMPOSITE_COMPONENTS,
        help="Minimum composite factor count required to trade (default: 3).",
    )
    parser.add_argument(
        "--min-buy-price",
        type=float,
        default=None,
        help=f"Minimum buy price (strictly greater; default: {MIN_BUY_PRICE_FOR_TRADE} from config).",
    )
    args = parser.parse_args()

    min_components = max(1, args.min_components)
    min_buy_price = (
        float(args.min_buy_price)
        if args.min_buy_price is not None
        else float(MIN_BUY_PRICE_FOR_TRADE)
    )

    print("Loading and merging trades + composite quant scores...")
    strategy_df = load_and_merge(min_components, min_buy_price)
    print(
        f"Eligible trades (Buy/Strong Buy, ≥{min_components} factors, "
        f"price>${min_buy_price}): {len(strategy_df):,}"
    )
    if len(strategy_df) == 0:
        raise SystemExit(
            "No trades after filters. Try --min-components 1 or check composite data."
        )

    tier_counts = strategy_df["quant_tier"].value_counts()
    print("Quant tier counts:")
    print(tier_counts.to_string())
    print()

    equity_df, strategy_df = run_simulation(strategy_df)

    final_capital = float(equity_df.iloc[-1]["equity"])
    total_return = (final_capital / STARTING_CAPITAL) - 1.0
    max_drawdown = float(equity_df["drawdown"].min())

    print("=== PORTFOLIO RESULTS (Buy / Strong Buy only) ===")
    print(f"Strategy trades (rows): {len(strategy_df):,}")
    print(f"Starting capital: ${STARTING_CAPITAL:,.2f}")
    print(f"Max positions: {MAX_POSITIONS}")
    print(f"Round-trip cost / trade: {TRANSACTION_COST_ROUND_TRIP*100:.2f}%")
    print(f"Gross avg trade return: {strategy_df['returnDecimal'].mean()*100:.2f}%")
    print(f"Net avg trade return: {strategy_df['netReturn'].mean()*100:.2f}%")
    print(f"Final equity: ${final_capital:,.2f}")
    print(f"Total return: {total_return*100:.2f}%")
    print(f"Avg daily return (equity): {equity_df['daily_return'].mean()*100:.2f}%")
    print(f"Median daily return (equity): {equity_df['daily_return'].median()*100:.2f}%")
    print(f"Max drawdown: {max_drawdown*100:.2f}%")
    print()
    print(f"Buy date range: {strategy_df['buyDate'].min()} → {strategy_df['sellDate'].max()}")
    print(f"Unique buy days: {strategy_df['buyDate'].nunique():,}")

    equity_df.to_csv(EQUITY_OUTPUT_FILE, index=False)
    plot_charts(equity_df, strategy_df)

    print()
    print(f"Saved {EQUITY_OUTPUT_FILE}")
    print(f"Saved {EQUITY_CHART_FILE}, {DRAWDOWN_CHART_FILE}, {RETURNS_HIST_FILE}")


if __name__ == "__main__":
    main()
