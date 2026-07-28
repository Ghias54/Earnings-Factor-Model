"""Reusable earnings-event portfolio backtest for research dashboards and CLI runs."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))
from config import (
    COMPANIES_ENRICHED_FILE,
    COMPOSITE_SCORES_FILE,
    DRAWDOWN_CURVE_PNG_FILE,
    EQUITY_CURVE_FILE,
    EQUITY_CURVE_PNG_FILE,
    MIN_BUY_PRICE_FOR_TRADE,
    PORTFOLIO_TRADES_FILE,
    PROCESSED_BACKTEST_DIR,
    TRADE_HISTOGRAM_FILE,
    VALUATION_FEATURES_FILE,
)
from processing.backtest.earnings_returns import build_earnings_returns, load_earnings, load_prices

COMPOSITE_FILE = COMPOSITE_SCORES_FILE

EQUITY_OUTPUT_FILE = EQUITY_CURVE_FILE
EQUITY_CHART_FILE = EQUITY_CURVE_PNG_FILE
DRAWDOWN_CHART_FILE = DRAWDOWN_CURVE_PNG_FILE
RETURNS_HIST_FILE = TRADE_HISTOGRAM_FILE

# --- Portfolio knobs ---
STARTING_CAPITAL = 10_000.0
TRANSACTION_COST_ROUND_TRIP = 0.002  # 0.20% round-trip per trade

# --- Quant tiers (grades align with SA-style labels in build_sa_comparison_view.quant_label_from_grade) ---
STRONG_BUY_GRADES = frozenset({"A+", "A"})
BUY_GRADES = frozenset({"A-", "B+"})
HOLD_GRADES = frozenset({"B", "B-", "C+", "C", "C-"})
SELL_GRADES = frozenset({"D+", "D"})
STRONG_SELL_GRADES = frozenset({"F"})
# Fallback when composite_grade is missing: bands match processing.scoring_utils.rating_to_grade cutoffs
_STRONG_BUY_R_MIN = 4.0
_BUY_R_MIN = 3.33
_HOLD_R_MIN = 1.67
_SELL_R_MIN = 1.0
QUANT_TIER_IDS = frozenset({"strong_buy", "buy", "hold", "sell", "strong_sell"})
DEFAULT_MIN_COMPOSITE_COMPONENTS = 3


def _safe_dataframe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write CSV; on Windows lock (Excel, preview, etc.) fall back to a timestamped file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        df.to_csv(alt, index=False)
        print(f"Warning: could not write (file may be open elsewhere): {path}")
        print(f"  Wrote instead: {alt}")
        return alt


def _safe_savefig(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(path, dpi=150)
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        plt.savefig(alt, dpi=150)
        print(f"Warning: could not write (file may be open elsewhere): {path}")
        print(f"  Wrote instead: {alt}")
        return alt


def quant_tier(composite_rating: float, composite_grade: object) -> str:
    """Map composite score to strong_buy / buy / hold / sell / strong_sell. Grade wins when present."""
    g = composite_grade
    if pd.notna(g):
        gs = str(g).strip()
        if gs in STRONG_BUY_GRADES:
            return "strong_buy"
        if gs in BUY_GRADES:
            return "buy"
        if gs in HOLD_GRADES:
            return "hold"
        if gs in SELL_GRADES:
            return "sell"
        if gs in STRONG_SELL_GRADES:
            return "strong_sell"
    if pd.notna(composite_rating):
        r = float(composite_rating)
        if r >= _STRONG_BUY_R_MIN:
            return "strong_buy"
        if r >= _BUY_R_MIN:
            return "buy"
        if r >= _HOLD_R_MIN:
            return "hold"
        if r >= _SELL_R_MIN:
            return "sell"
        return "strong_sell"
    return "hold"


def resolve_quant_ratings(
    *,
    quant_rating_mode: str,
    quant_tiers: set[str] | None,
) -> set[str]:
    """Pick included tiers from explicit set or named mode."""
    rating_map: dict[str, set[str]] = {
        "buy": {"buy"},
        "strong_buy": {"strong_buy"},
        "both": {"buy", "strong_buy"},
        "hold": {"hold"},
        "sell": {"sell"},
        "strong_sell": {"strong_sell"},
        "bearish": {"sell", "strong_sell"},
        "all": set(QUANT_TIER_IDS),
    }
    if quant_tiers is not None and len(quant_tiers) > 0:
        bad = quant_tiers - QUANT_TIER_IDS
        if bad:
            raise ValueError(f"Invalid quant tier(s): {sorted(bad)}. Expected subset of {sorted(QUANT_TIER_IDS)}.")
        return set(quant_tiers)
    return set(rating_map.get(quant_rating_mode, {"buy", "strong_buy"}))


@lru_cache(maxsize=8)
def _load_trade_candidates_cached(days_before: int, days_after: int, min_buy_price: float) -> pd.DataFrame:
    earnings_df = load_earnings()
    prices_df = load_prices()
    taken_df, _ = build_earnings_returns(
        earnings_df,
        prices_df,
        buy_days_before_anchor=days_before,
        sell_days_after_anchor=days_after,
        min_buy_price=min_buy_price,
    )
    if taken_df.empty:
        return taken_df
    taken_df["ticker"] = taken_df["ticker"].astype(str).str.strip()
    taken_df["earningsAnnouncementDate"] = pd.to_datetime(taken_df["earningsAnnouncementDate"], errors="coerce")
    taken_df["buyDate"] = pd.to_datetime(taken_df["buyDate"], errors="coerce")
    taken_df["sellDate"] = pd.to_datetime(taken_df["sellDate"], errors="coerce")
    return taken_df


def _load_trade_candidates(days_before: int, days_after: int, min_buy_price: float) -> pd.DataFrame:
    # Copy so downstream filtering/mutation does not contaminate cached frames.
    return _load_trade_candidates_cached(days_before, days_after, float(min_buy_price)).copy()


@lru_cache(maxsize=1)
def _load_composite_for_asof() -> pd.DataFrame:
    if not COMPOSITE_FILE.exists():
        raise FileNotFoundError(f"Missing {COMPOSITE_FILE}. Run score pipeline first.")

    comp_df = pd.read_csv(COMPOSITE_FILE, low_memory=False)
    comp_df["ticker"] = comp_df["ticker"].astype(str).str.strip()
    comp_df["earningsAnnouncementDate"] = pd.to_datetime(comp_df["earningsAnnouncementDate"], errors="coerce")
    comp_df = comp_df.dropna(subset=["ticker", "earningsAnnouncementDate"])
    comp_df = comp_df.rename(columns={"earningsAnnouncementDate": "composite_source_announcement_date"})
    comp_df = comp_df.sort_values(["ticker", "composite_source_announcement_date"], kind="mergesort")
    return comp_df


def _attach_composite_asof(df: pd.DataFrame) -> pd.DataFrame:
    comp_for_asof = _load_composite_for_asof()
    score_cols = [
        c
        for c in comp_for_asof.columns
        if c not in ("ticker", "composite_source_announcement_date")
    ]
    comp_for_asof = comp_for_asof[["ticker", "composite_source_announcement_date"] + score_cols]

    df = df.sort_values(["ticker", "buyDate"], kind="mergesort")
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
    return pd.concat(merged_parts, ignore_index=True)


@lru_cache(maxsize=1)
def _load_market_cap_table() -> pd.DataFrame:
    if not VALUATION_FEATURES_FILE.exists():
        return pd.DataFrame(columns=["ticker", "earningsAnnouncementDate", "marketCap"])
    val = pd.read_csv(
        VALUATION_FEATURES_FILE,
        usecols=lambda c: c in {"ticker", "earningsAnnouncementDate", "marketCap"},
        low_memory=False,
    )
    val["ticker"] = val["ticker"].astype(str).str.strip()
    val["earningsAnnouncementDate"] = pd.to_datetime(val["earningsAnnouncementDate"], errors="coerce")
    val["marketCap"] = pd.to_numeric(val.get("marketCap"), errors="coerce")
    return val.drop_duplicates(["ticker", "earningsAnnouncementDate"], keep="last")


@lru_cache(maxsize=1)
def _load_sector_table() -> pd.DataFrame:
    if not COMPANIES_ENRICHED_FILE.exists():
        return pd.DataFrame(columns=["ticker", "sector"])
    comp = pd.read_csv(COMPANIES_ENRICHED_FILE, low_memory=False, usecols=lambda c: c in {"ticker", "sector"})
    comp["ticker"] = comp["ticker"].astype(str).str.strip()
    return comp.drop_duplicates(["ticker"], keep="last")


def _attach_optional_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    val = _load_market_cap_table()
    if not val.empty:
        out = out.merge(val, on=["ticker", "earningsAnnouncementDate"], how="left")

    comp = _load_sector_table()
    if not comp.empty:
        out = out.merge(comp, on="ticker", how="left")

    out["buyVolume"] = pd.to_numeric(out.get("buyVolume"), errors="coerce")
    out["dollar_volume"] = out["buyPrice"] * out["buyVolume"]
    return out


@lru_cache(maxsize=1)
def _load_price_path_for_stop_loss() -> dict[str, pd.DataFrame]:
    px = load_prices()[["ticker", "date", "low", "close", "volume"]].copy()
    px["ticker"] = px["ticker"].astype(str).str.strip()
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px["low"] = pd.to_numeric(px["low"], errors="coerce")
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px["volume"] = pd.to_numeric(px["volume"], errors="coerce")
    px = px.dropna(subset=["ticker", "date"]).sort_values(["ticker", "date"], kind="mergesort")
    return {t: g.reset_index(drop=True) for t, g in px.groupby("ticker", sort=False)}


def _apply_stop_loss(
    df: pd.DataFrame,
    *,
    stop_loss_pct: float | None,
    transaction_cost_round_trip: float,
) -> pd.DataFrame:
    out = df.copy()
    stop_pct = abs(float(stop_loss_pct)) if stop_loss_pct is not None else None
    out["stop_loss_pct"] = stop_pct if stop_pct is not None else np.nan
    out["stopped_out"] = False
    out["stop_out_date"] = pd.NaT
    out["days_to_stop"] = np.nan
    out["exit_reason"] = "normal exit"

    if stop_pct is None or stop_pct <= 0:
        return out

    # Assumption with daily bars: we trigger stop-loss if the day's LOW breaches stop level.
    # Since intraday sequencing is unavailable, we model execution at the stop level (not the close).
    price_map = _load_price_path_for_stop_loss()
    net_threshold = -stop_pct

    for i, r in out.iterrows():
        ticker = str(r["ticker"])
        buy_dt = pd.to_datetime(r["buyDate"], errors="coerce")
        planned_sell_dt = pd.to_datetime(r["sellDate"], errors="coerce")
        buy_price = pd.to_numeric(r["buyPrice"], errors="coerce")
        if pd.isna(buy_dt) or pd.isna(planned_sell_dt) or pd.isna(buy_price) or float(buy_price) <= 0:
            continue

        px = price_map.get(ticker)
        if px is None or px.empty:
            continue

        # Start checking from the day after entry since entry is modeled on buy-date close.
        window = px[(px["date"] > buy_dt) & (px["date"] <= planned_sell_dt)]
        if window.empty:
            continue

        stop_price = float(buy_price) * (1.0 + net_threshold + float(transaction_cost_round_trip))
        hit = window[(window["low"].notna()) & (window["low"] <= stop_price)]
        if hit.empty:
            continue

        h = hit.iloc[0]
        stop_dt = pd.to_datetime(h["date"], errors="coerce")
        out.at[i, "sellDate"] = stop_dt
        out.at[i, "sellPrice"] = stop_price
        out.at[i, "sellVolume"] = pd.to_numeric(h.get("volume"), errors="coerce")
        out.at[i, "returnDecimal"] = (stop_price - float(buy_price)) / float(buy_price)
        out.at[i, "returnPct"] = out.at[i, "returnDecimal"] * 100.0
        out.at[i, "stopped_out"] = True
        out.at[i, "stop_out_date"] = stop_dt
        out.at[i, "days_to_stop"] = max((stop_dt - buy_dt).days, 0)
        out.at[i, "exit_reason"] = "stop loss"

    return out


def build_strategy_trades(
    *,
    days_before: int,
    days_after: int,
    min_factors: int,
    min_price: float,
    min_composite_score: float,
    quant_ratings: set[str],
    top_n_per_day: int,
    transaction_cost_round_trip: float,
    stop_loss_pct: float | None = None,
    sector: str | None = None,
    min_market_cap: float | None = None,
    min_dollar_volume: float | None = None,
) -> pd.DataFrame:
    df = _load_trade_candidates(days_before, days_after, min_price)
    if df.empty:
        return df

    df = _attach_composite_asof(df)
    df = _attach_optional_metadata(df)
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
        df = df[df["composite_component_count"] >= min_factors].copy()
    df = df[df["buyPrice"] > min_price].copy()
    df = df[df["composite_rating"] >= min_composite_score].copy()

    grades = (
        df["composite_grade"]
        if "composite_grade" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    df["quant_tier"] = [
        quant_tier(r, g) for r, g in zip(df["composite_rating"], grades)
    ]
    df = df[df["quant_tier"].isin(quant_ratings)].copy()

    if sector and "sector" in df.columns:
        df = df[df["sector"].astype(str).str.strip().str.lower() == sector.strip().lower()].copy()
    if min_market_cap is not None and "marketCap" in df.columns:
        df = df[df["marketCap"] >= float(min_market_cap)].copy()
    if min_dollar_volume is not None and "dollar_volume" in df.columns:
        df = df[df["dollar_volume"] >= float(min_dollar_volume)].copy()

    df = _apply_stop_loss(
        df,
        stop_loss_pct=stop_loss_pct,
        transaction_cost_round_trip=float(transaction_cost_round_trip),
    )
    df["netReturn"] = (df["returnDecimal"] - float(transaction_cost_round_trip)).clip(lower=-1.0)
    df["realized_return"] = df["netReturn"]

    df = df.sort_values(
        ["buyDate", "composite_rating", "ticker"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    if top_n_per_day > 0:
        df = df.groupby("buyDate", group_keys=False).head(top_n_per_day)
    df = df.drop_duplicates(subset=["buyDate", "ticker"], keep="first").reset_index(drop=True)

    return df


def run_simulation(strategy_df: pd.DataFrame, *, starting_capital: float, max_positions: int) -> pd.DataFrame:
    strategy_df = strategy_df.sort_values("buyDate").reset_index(drop=True)

    capital = float(starting_capital)
    open_positions: list[dict] = []
    equity_rows: list[dict] = []

    all_dates = sorted(
        pd.unique(pd.concat([strategy_df["buyDate"], strategy_df["sellDate"]]))
    )

    for current_date in all_dates:
        still_open: list[dict] = []
        for pos in open_positions:
            if pos["sell_date"] <= current_date:
                capital += pos["final_value"]
            else:
                still_open.append(pos)
        open_positions = still_open

        todays = strategy_df[strategy_df["buyDate"] == current_date]
        available_slots = max_positions - len(open_positions)

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
                    {
                        "ticker": str(row["ticker"]),
                        "buy_date": row["buyDate"],
                        "sell_date": row["sellDate"],
                        "allocated": alloc,
                        "final_value": final_value,
                        "gross_return": float(row["returnDecimal"]),
                        "net_return": float(row["netReturn"]),
                    }
                )
                capital -= alloc

        total_equity = capital + sum(p["final_value"] for p in open_positions)
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

    return equity_df


def _yearly_returns(equity_df: pd.DataFrame) -> pd.DataFrame:
    x = equity_df[["date", "daily_return"]].dropna().copy()
    if x.empty:
        return pd.DataFrame(columns=["year", "yearly_return"])
    x["year"] = pd.to_datetime(x["date"]).dt.year
    yr = x.groupby("year")["daily_return"].apply(lambda s: (1.0 + s).prod() - 1.0).reset_index()
    yr = yr.rename(columns={"daily_return": "yearly_return"})
    return yr


def _monthly_heatmap(equity_df: pd.DataFrame) -> pd.DataFrame:
    x = equity_df[["date", "daily_return"]].dropna().copy()
    if x.empty:
        return pd.DataFrame()
    d = pd.to_datetime(x["date"])
    x["year"] = d.dt.year
    x["month"] = d.dt.month
    m = x.groupby(["year", "month"])["daily_return"].apply(lambda s: (1.0 + s).prod() - 1.0).reset_index()
    return m.pivot(index="year", columns="month", values="daily_return").sort_index()


def _metrics(trades: pd.DataFrame, equity_df: pd.DataFrame, *, starting_capital: float) -> dict:
    if equity_df.empty:
        return {}
    final_capital = float(equity_df.iloc[-1]["equity"])
    total_return = (final_capital / float(starting_capital)) - 1.0
    max_drawdown = float(equity_df["drawdown"].min()) if "drawdown" in equity_df.columns else np.nan
    daily = equity_df["daily_return"].dropna()
    tr = trades["netReturn"] if "netReturn" in trades.columns else pd.Series(dtype=float)
    stopped_mask = (
        trades["stopped_out"].astype(bool)
        if "stopped_out" in trades.columns
        else pd.Series(dtype=bool)
    )
    stopped_days = (
        pd.to_numeric(trades.loc[stopped_mask, "days_to_stop"], errors="coerce")
        if "days_to_stop" in trades.columns and len(stopped_mask)
        else pd.Series(dtype=float)
    )
    start_dt = pd.to_datetime(equity_df["date"].min())
    end_dt = pd.to_datetime(equity_df["date"].max())
    years = max((end_dt - start_dt).days / 365.25, 0.0)
    cagr = (final_capital / float(starting_capital)) ** (1 / years) - 1 if years > 0 and final_capital > 0 else np.nan

    return {
        "total_trades": int(len(trades)),
        "final_equity": final_capital,
        "total_return_pct": total_return * 100.0,
        "cagr_pct": cagr * 100.0 if pd.notna(cagr) else np.nan,
        "average_trade_return_pct": float(tr.mean() * 100.0) if len(tr) else np.nan,
        "median_trade_return_pct": float(tr.median() * 100.0) if len(tr) else np.nan,
        "win_rate_pct": float((tr > 0).mean() * 100.0) if len(tr) else np.nan,
        "max_drawdown_pct": max_drawdown * 100.0 if pd.notna(max_drawdown) else np.nan,
        "average_daily_return_pct": float(daily.mean() * 100.0) if len(daily) else np.nan,
        "median_daily_return_pct": float(daily.median() * 100.0) if len(daily) else np.nan,
        "stopped_out_count": int(stopped_mask.sum()) if len(stopped_mask) else 0,
        "stopped_out_pct": float(stopped_mask.mean() * 100.0) if len(stopped_mask) else 0.0,
        "avg_days_to_stop": float(stopped_days.mean()) if len(stopped_days) else np.nan,
        "median_days_to_stop": float(stopped_days.median()) if len(stopped_days) else np.nan,
    }


def run_portfolio_backtest(
    *,
    days_before: int = 5,
    days_after: int = 25,
    quant_rating_mode: str = "both",
    quant_tiers: set[str] | None = None,
    max_positions: int = 10,
    min_factors: int = 3,
    min_price: float = MIN_BUY_PRICE_FOR_TRADE,
    min_composite_score: float = 0.0,
    top_n_per_day: int = 0,
    transaction_cost_round_trip: float = TRANSACTION_COST_ROUND_TRIP,
    stop_loss_pct: float | None = None,
    starting_capital: float = STARTING_CAPITAL,
    sector: str | None = None,
    min_market_cap: float | None = None,
    min_dollar_volume: float | None = None,
) -> dict[str, object]:
    quant_ratings = resolve_quant_ratings(quant_rating_mode=quant_rating_mode, quant_tiers=quant_tiers)

    trades = build_strategy_trades(
        days_before=int(days_before),
        days_after=int(days_after),
        min_factors=max(1, int(min_factors)),
        min_price=float(min_price),
        min_composite_score=float(min_composite_score),
        quant_ratings=quant_ratings,
        top_n_per_day=max(0, int(top_n_per_day)),
        transaction_cost_round_trip=float(transaction_cost_round_trip),
        stop_loss_pct=stop_loss_pct,
        sector=sector,
        min_market_cap=min_market_cap,
        min_dollar_volume=min_dollar_volume,
    )

    if trades.empty:
        empty_eq = pd.DataFrame(columns=["date", "cash", "open_positions", "equity", "daily_return", "running_max", "drawdown"])
        return {
            "metrics": {},
            "equity_curve": empty_eq,
            "trades": trades,
            "daily_returns": pd.DataFrame(columns=["date", "daily_return"]),
            "yearly_returns": pd.DataFrame(columns=["year", "yearly_return"]),
            "monthly_returns_heatmap": pd.DataFrame(),
            "buy_vs_strong_buy": pd.DataFrame(),
            "params": {
                "days_before": days_before,
                "days_after": days_after,
                "quant_rating_mode": quant_rating_mode,
                "quant_tiers": sorted(quant_ratings),
                "max_positions": max_positions,
                "stop_loss_pct": stop_loss_pct,
            },
        }

    equity_df = run_simulation(trades, starting_capital=float(starting_capital), max_positions=int(max_positions))
    daily_returns = equity_df[["date", "daily_return"]].copy()
    yearly_returns = _yearly_returns(equity_df)
    monthly_heatmap = _monthly_heatmap(equity_df)
    metrics = _metrics(trades, equity_df, starting_capital=float(starting_capital))

    buy_vs = (
        trades.groupby("quant_tier")
        .agg(
            trade_count=("ticker", "count"),
            avg_net_return=("netReturn", "mean"),
            median_net_return=("netReturn", "median"),
            win_rate=("netReturn", lambda s: (s > 0).mean()),
        )
        .reset_index()
    )

    return {
        "metrics": metrics,
        "equity_curve": equity_df,
        "trades": trades,
        "daily_returns": daily_returns,
        "yearly_returns": yearly_returns,
        "monthly_returns_heatmap": monthly_heatmap,
        "buy_vs_strong_buy": buy_vs,
        "params": {
            "days_before": days_before,
            "days_after": days_after,
            "quant_rating_mode": quant_rating_mode,
            "quant_tiers": sorted(quant_ratings),
            "max_positions": max_positions,
            "min_factors": min_factors,
            "min_price": min_price,
            "min_composite_score": min_composite_score,
            "top_n_per_day": top_n_per_day,
            "transaction_cost_round_trip": transaction_cost_round_trip,
            "stop_loss_pct": stop_loss_pct,
            "starting_capital": starting_capital,
            "sector": sector,
            "min_market_cap": min_market_cap,
            "min_dollar_volume": min_dollar_volume,
        },
    }


def plot_charts(equity_df: pd.DataFrame, strategy_df: pd.DataFrame) -> tuple[Path, Path, Path]:
    PROCESSED_BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(equity_df["date"], equity_df["equity"])
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.tight_layout()
    p_eq = _safe_savefig(EQUITY_CHART_FILE)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(equity_df["date"], equity_df["drawdown"])
    plt.title("Drawdown Curve")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    p_dd = _safe_savefig(DRAWDOWN_CHART_FILE)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.hist(strategy_df["netReturn"], bins=100)
    plt.title("Net Trade Return Distribution")
    plt.xlabel("Net Trade Return")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    p_hist = _safe_savefig(RETURNS_HIST_FILE)
    plt.close()
    return p_eq, p_dd, p_hist


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest earnings strategy with quant filter.")
    parser.add_argument("--days-before", type=int, default=5, help="Trading days before anchor to enter.")
    parser.add_argument("--days-after", type=int, default=25, help="Trading days after anchor to exit.")
    parser.add_argument(
        "--quant-rating-mode",
        choices=[
            "buy",
            "strong_buy",
            "both",
            "hold",
            "sell",
            "strong_sell",
            "bearish",
            "all",
        ],
        default="both",
        help="Which quant rating tiers to include (ignored if --quant-tiers is set).",
    )
    parser.add_argument(
        "--quant-tiers",
        type=str,
        default=None,
        help="Comma-separated tiers overriding mode: strong_buy,buy,hold,sell,strong_sell.",
    )
    parser.add_argument("--max-positions", type=int, default=10, help="Maximum concurrent positions.")
    parser.add_argument(
        "--min-components",
        type=int,
        default=DEFAULT_MIN_COMPOSITE_COMPONENTS,
        help="Minimum composite factor count required to trade (default: 3).",
    )
    parser.add_argument("--min-composite-score", type=float, default=0.0, help="Minimum composite score (0-5).")
    parser.add_argument("--top-n-per-day", type=int, default=0, help="Take top N trades per buy day (0 = unlimited).")
    parser.add_argument(
        "--roundtrip-cost",
        type=float,
        default=TRANSACTION_COST_ROUND_TRIP,
        help="Round-trip transaction cost as decimal (0.002 = 0.2%).",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=None,
        help="Optional stop loss as decimal (0.10 means stop at -10% net return).",
    )
    parser.add_argument(
        "--min-buy-price",
        type=float,
        default=None,
        help=f"Minimum buy price (strictly greater; default: {MIN_BUY_PRICE_FOR_TRADE} from config).",
    )
    parser.add_argument("--sector", type=str, default=None, help="Optional sector filter if sector metadata exists.")
    parser.add_argument("--min-market-cap", type=float, default=None, help="Optional minimum market cap.")
    parser.add_argument("--min-dollar-volume", type=float, default=None, help="Optional minimum dollar volume.")
    args = parser.parse_args()

    min_buy_price = (
        float(args.min_buy_price)
        if args.min_buy_price is not None
        else float(MIN_BUY_PRICE_FOR_TRADE)
    )

    tier_override: set[str] | None = None
    if args.quant_tiers:
        tier_override = {t.strip() for t in args.quant_tiers.split(",") if t.strip()}

    try:
        result = run_portfolio_backtest(
            days_before=args.days_before,
            days_after=args.days_after,
            quant_rating_mode=args.quant_rating_mode,
            quant_tiers=tier_override,
            max_positions=max(1, int(args.max_positions)),
            min_factors=max(1, int(args.min_components)),
            min_price=min_buy_price,
            min_composite_score=float(args.min_composite_score),
            top_n_per_day=max(0, int(args.top_n_per_day)),
            transaction_cost_round_trip=float(args.roundtrip_cost),
            stop_loss_pct=float(args.stop_loss_pct) if args.stop_loss_pct is not None else None,
            sector=args.sector,
            min_market_cap=args.min_market_cap,
            min_dollar_volume=args.min_dollar_volume,
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e
    equity_df = result["equity_curve"]
    strategy_df = result["trades"]
    metrics = result["metrics"]
    if strategy_df.empty:
        raise SystemExit("No trades after filters.")

    qdesc = ", ".join(sorted(strategy_df["quant_tier"].unique())) if "quant_tier" in strategy_df.columns else ""
    print(f"=== PORTFOLIO RESULTS (quant tiers in book: {qdesc}) ===")
    print(f"Strategy trades (rows): {len(strategy_df):,}")
    print(f"Final equity: ${metrics.get('final_equity', np.nan):,.2f}")
    print(f"Total return: {metrics.get('total_return_pct', np.nan):.2f}%")
    if pd.notna(metrics.get("cagr_pct")):
        print(f"CAGR: {metrics['cagr_pct']:.2f}%")
    print(f"Gross avg trade return: {strategy_df['returnDecimal'].mean()*100:.2f}%")
    print(f"Net avg trade return: {metrics.get('average_trade_return_pct', np.nan):.2f}%")
    print(f"Median net trade return: {metrics.get('median_trade_return_pct', np.nan):.2f}%")
    print(f"Win rate: {metrics.get('win_rate_pct', np.nan):.2f}%")
    print(f"Max drawdown: {metrics.get('max_drawdown_pct', np.nan):.2f}%")
    print(
        f"Stopped out: {metrics.get('stopped_out_count', 0):,} "
        f"({metrics.get('stopped_out_pct', 0.0):.2f}%) | "
        f"Avg days to stop: {metrics.get('avg_days_to_stop', np.nan):.2f}"
    )
    print()
    print(f"Buy date range: {strategy_df['buyDate'].min()} to {strategy_df['sellDate'].max()}")
    print(f"Unique buy days: {strategy_df['buyDate'].nunique():,}")

    equity_path = _safe_dataframe_to_csv(equity_df, EQUITY_OUTPUT_FILE)
    trades_path = _safe_dataframe_to_csv(strategy_df, PORTFOLIO_TRADES_FILE)
    p_eq, p_dd, p_hist = plot_charts(equity_df, strategy_df)

    print()
    print(f"Saved {equity_path}")
    print(f"Saved {trades_path}")
    print(f"Saved {p_eq}, {p_dd}, {p_hist}")


if __name__ == "__main__":
    main()
