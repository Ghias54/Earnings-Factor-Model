"""Run locally: .\\.venv\\Scripts\\streamlit.exe run app.py"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from processing.backtest.simulate_portfolio import run_portfolio_backtest


st.set_page_config(page_title="Earnings Strategy Dashboard", layout="wide")
st.title("Earnings-Event Strategy Dashboard")
st.caption("Research-only local dashboard. Reuses existing portfolio simulation logic.")

with st.sidebar:
    st.header("Strategy Controls")
    days_before = st.number_input("Days before earnings entry", min_value=1, max_value=30, value=5, step=1)
    days_after = st.number_input("Days after earnings exit", min_value=1, max_value=60, value=25, step=1)
    _tier_labels = {
        "strong_buy": "Strong Buy",
        "buy": "Buy",
        "hold": "Hold",
        "sell": "Sell",
        "strong_sell": "Strong Sell",
    }
    _tier_order = ("strong_buy", "buy", "hold", "sell", "strong_sell")
    quant_tier_pick = st.multiselect(
        "Quant rating(s) to include",
        options=list(_tier_order),
        default=["strong_buy", "buy"],
        format_func=lambda k: _tier_labels[k],
        help="Select one or more tiers. Same simulation: long earnings-window trades filtered by composite quant at entry.",
    )
    max_positions = st.number_input("Max positions", min_value=1, max_value=100, value=10, step=1)
    min_factors = st.number_input("Minimum number of factors required", min_value=1, max_value=10, value=3, step=1)
    min_price = st.number_input("Minimum stock price", min_value=0.0, value=1.0, step=0.5)
    min_composite_score = st.number_input("Minimum composite score", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
    top_n_per_day = st.number_input("Top N trades per day by score (0 = no cap)", min_value=0, max_value=200, value=0, step=1)
    roundtrip_cost = st.number_input("Round-trip transaction cost (decimal)", min_value=0.0, max_value=0.1, value=0.002, step=0.0005, format="%.4f")

    st.subheader("Optional Filters")
    sector = st.text_input("Sector (exact match; leave blank for all)", value="")
    min_market_cap = st.number_input("Minimum market cap (0 = off)", min_value=0.0, value=0.0, step=10_000_000.0, format="%.0f")
    min_dollar_volume = st.number_input("Minimum dollar volume (0 = off)", min_value=0.0, value=0.0, step=1_000_000.0, format="%.0f")

    run_clicked = st.button("Run Backtest", type="primary")

if run_clicked:
    if not quant_tier_pick:
        st.warning("Select at least one quant tier.")
        st.stop()
    with st.spinner("Running backtest... this can take a bit depending on dataset size."):
        result = run_portfolio_backtest(
            days_before=int(days_before),
            days_after=int(days_after),
            quant_rating_mode="both",
            quant_tiers=set(quant_tier_pick),
            max_positions=int(max_positions),
            min_factors=int(min_factors),
            min_price=float(min_price),
            min_composite_score=float(min_composite_score),
            top_n_per_day=int(top_n_per_day),
            transaction_cost_round_trip=float(roundtrip_cost),
            sector=sector.strip() or None,
            min_market_cap=float(min_market_cap) if min_market_cap > 0 else None,
            min_dollar_volume=float(min_dollar_volume) if min_dollar_volume > 0 else None,
        )

    metrics: dict = result["metrics"]
    equity_df: pd.DataFrame = result["equity_curve"]
    trades_df: pd.DataFrame = result["trades"]
    daily_returns: pd.DataFrame = result["daily_returns"]
    yearly_returns: pd.DataFrame = result["yearly_returns"]
    monthly_heatmap: pd.DataFrame = result["monthly_returns_heatmap"]
    buy_vs: pd.DataFrame = result["buy_vs_strong_buy"]

    if not metrics:
        st.warning("No trades matched the current filters. Adjust settings and run again.")
        st.stop()

    st.subheader("Summary Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Trades", f"{metrics['total_trades']:,}")
    m2.metric("Final Equity", f"${metrics['final_equity']:,.2f}")
    m3.metric("Total Return", f"{metrics['total_return_pct']:.2f}%")
    m4.metric("CAGR", "N/A" if pd.isna(metrics["cagr_pct"]) else f"{metrics['cagr_pct']:.2f}%")
    m5.metric("Win Rate", f"{metrics['win_rate_pct']:.2f}%")

    m6, m7, m8, m9, m10 = st.columns(5)
    m6.metric("Avg Trade Return", f"{metrics['average_trade_return_pct']:.2f}%")
    m7.metric("Median Trade Return", f"{metrics['median_trade_return_pct']:.2f}%")
    m8.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
    m9.metric("Avg Daily Return", f"{metrics['average_daily_return_pct']:.3f}%")
    m10.metric("Median Daily Return", f"{metrics['median_daily_return_pct']:.3f}%")

    st.subheader("Visualizations")
    c1, c2 = st.columns(2)
    with c1:
        fig_eq = px.line(equity_df, x="date", y="equity", title="Equity Curve")
        st.plotly_chart(fig_eq, use_container_width=True)
    with c2:
        fig_dd = px.line(equity_df, x="date", y="drawdown", title="Drawdown")
        st.plotly_chart(fig_dd, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig_hist = px.histogram(trades_df, x="netReturn", nbins=80, title="Trade Return Histogram (Net)")
        st.plotly_chart(fig_hist, use_container_width=True)
    with c4:
        fig_scatter = px.scatter(
            trades_df,
            x="composite_rating",
            y="netReturn",
            color="quant_tier",
            title="Composite Score vs Trade Return",
            hover_data=["ticker", "buyDate", "sellDate"],
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        if not yearly_returns.empty:
            yearly_returns["year"] = yearly_returns["year"].astype(str)
            fig_year = px.bar(yearly_returns, x="year", y="yearly_return", title="Yearly Returns")
            st.plotly_chart(fig_year, use_container_width=True)
        else:
            st.info("Yearly returns unavailable.")
    with c6:
        if not monthly_heatmap.empty:
            hm = monthly_heatmap.copy()
            hm.columns = [str(c) for c in hm.columns]
            fig_hm = px.imshow(hm, aspect="auto", title="Monthly Returns Heatmap")
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.info("Monthly heatmap unavailable.")

    st.subheader("Performance by quant tier")
    if not buy_vs.empty:
        comp = buy_vs.copy()
        comp["avg_net_return_pct"] = comp["avg_net_return"] * 100.0
        comp["win_rate_pct"] = comp["win_rate"] * 100.0
        st.dataframe(comp, use_container_width=True)
        fig_buy = px.bar(comp, x="quant_tier", y=["avg_net_return_pct", "win_rate_pct"], barmode="group")
        st.plotly_chart(fig_buy, use_container_width=True)
    else:
        st.info("Comparison unavailable.")

    st.subheader("Trade Explorer")
    st.dataframe(trades_df, use_container_width=True, height=420)
    st.download_button(
        "Export trades to CSV",
        data=trades_df.to_csv(index=False).encode("utf-8"),
        file_name="dashboard_trades.csv",
        mime="text/csv",
    )
else:
    st.info("Set parameters in the sidebar and click Run Backtest.")
