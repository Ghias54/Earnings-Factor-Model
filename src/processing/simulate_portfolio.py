import pandas as pd
import matplotlib.pyplot as plt
from config import PROCESSED_DATA_DIR

RETURNS_FILE = PROCESSED_DATA_DIR / "earnings_returns.csv"
MOMENTUM_FILE = PROCESSED_DATA_DIR / "momentum_features.csv"
EVENTS_FILE = PROCESSED_DATA_DIR / "earnings_events.csv"

EQUITY_OUTPUT_FILE = PROCESSED_DATA_DIR / "equity_curve.csv"
EQUITY_CHART_FILE = PROCESSED_DATA_DIR / "equity_curve.png"
DRAWDOWN_CHART_FILE = PROCESSED_DATA_DIR / "drawdown_curve.png"
RETURNS_HIST_FILE = PROCESSED_DATA_DIR / "trade_return_histogram.png"

# -----------------------------
# SETTINGS
# -----------------------------
starting_capital = 10000
max_positions = 10

# Round-trip transaction cost per trade
# Example: 0.002 = 0.20%
transaction_cost = 0.002

# -----------------------------
# LOAD DATA
# -----------------------------
returns_df = pd.read_csv(RETURNS_FILE)
momentum_df = pd.read_csv(MOMENTUM_FILE)
events_df = pd.read_csv(EVENTS_FILE)

# Clean merge keys
for df in [returns_df, momentum_df, events_df]:
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["earningsAnnouncementDate"] = pd.to_datetime(df["earningsAnnouncementDate"])

# Merge in memory only
df = returns_df.merge(
    momentum_df,
    on=["ticker", "earningsAnnouncementDate"],
    how="inner"
)

df = df.merge(
    events_df[["ticker", "earningsAnnouncementDate", "epsSurprise"]],
    on=["ticker", "earningsAnnouncementDate"],
    how="left"
)

# Clean numeric columns
df["returnDecimal"] = pd.to_numeric(df["returnDecimal"], errors="coerce")
df["buyPrice"] = pd.to_numeric(df["buyPrice"], errors="coerce")
df["epsSurprise"] = pd.to_numeric(df["epsSurprise"], errors="coerce")
df["mom63"] = pd.to_numeric(df["mom63"], errors="coerce")

# Clean date columns
df["buyDate"] = pd.to_datetime(df["buyDate"])
df["sellDate"] = pd.to_datetime(df["sellDate"])

# Drop rows missing core fields
df = df.dropna(
    subset=[
        "returnDecimal",
        "buyPrice",
        "epsSurprise",
        "mom63",
        "buyDate",
        "sellDate",
    ]
).copy()

# -----------------------------
# APPLY COSTS
# -----------------------------
# Net return after round-trip costs
df["netReturn"] = df["returnDecimal"] - transaction_cost

# Optional safety floor to avoid impossible returns below -100%
df["netReturn"] = df["netReturn"].clip(lower=-1.0)

# -----------------------------
# STRATEGY FILTER
# -----------------------------
# Change this section whenever you want to test a different strategy

threshold = df["epsSurprise"].quantile(0.90)

strategy_df = df[
    (df["epsSurprise"] >= threshold) &
    (df["buyPrice"] > 5) &
    (df["mom63"] < 0)
].copy()

# Examples you can swap in later:
#
# strategy_df = df[df["epsSurprise"] > 0].copy()
#
# threshold = df["epsSurprise"].quantile(0.95)
# strategy_df = df[df["epsSurprise"] >= threshold].copy()
#
# threshold = df["epsSurprise"].quantile(0.95)
# strategy_df = df[
#     (df["epsSurprise"] >= threshold) &
#     (df["buyPrice"] > 5)
# ].copy()

# -----------------------------
# PORTFOLIO SIMULATION
# -----------------------------
strategy_df = strategy_df.sort_values("buyDate").reset_index(drop=True)

capital = starting_capital
open_positions = []
equity_curve = []

all_dates = sorted(pd.unique(pd.concat([strategy_df["buyDate"], strategy_df["sellDate"]])))

for current_date in all_dates:
    # 1. Close positions whose sell date is today or earlier
    still_open = []
    for pos in open_positions:
        if pos["sellDate"] <= current_date:
            capital += pos["final_value"]
        else:
            still_open.append(pos)
    open_positions = still_open

    # 2. Open new positions for today if slots are available
    todays_trades = strategy_df[strategy_df["buyDate"] == current_date]
    available_slots = max_positions - len(open_positions)

    if available_slots > 0 and len(todays_trades) > 0:
        trades_to_take = todays_trades.head(available_slots)

        allocation_per_trade = capital / available_slots if available_slots > 0 else 0

        for _, row in trades_to_take.iterrows():
            if capital <= 0:
                break

            actual_allocation = min(allocation_per_trade, capital)
            final_value = actual_allocation * (1 + row["netReturn"])

            open_positions.append({
                "ticker": row["ticker"],
                "buyDate": row["buyDate"],
                "sellDate": row["sellDate"],
                "allocated": actual_allocation,
                "final_value": final_value,
                "gross_return": row["returnDecimal"],
                "net_return": row["netReturn"],
            })

            capital -= actual_allocation

    # 3. Track total equity
    total_equity = capital + sum(pos["final_value"] for pos in open_positions)

    equity_curve.append({
        "date": current_date,
        "cash": capital,
        "open_positions": len(open_positions),
        "equity": total_equity,
    })

equity_df = pd.DataFrame(equity_curve).sort_values("date").reset_index(drop=True)

# -----------------------------
# STATS
# -----------------------------
final_capital = equity_df.iloc[-1]["equity"]
total_return = (final_capital / starting_capital) - 1

equity_df["daily_return"] = equity_df["equity"].pct_change()
avg_daily_return = equity_df["daily_return"].mean()
median_daily_return = equity_df["daily_return"].median()

equity_df["running_max"] = equity_df["equity"].cummax()
equity_df["drawdown"] = (equity_df["equity"] / equity_df["running_max"]) - 1
max_drawdown = equity_df["drawdown"].min()

gross_avg_trade_return = strategy_df["returnDecimal"].mean()
net_avg_trade_return = strategy_df["netReturn"].mean()

print("=== FIXED PORTFOLIO RESULTS ===")
print(f"Strategy trades: {len(strategy_df):,}")
print(f"Starting capital: ${starting_capital:,.2f}")
print(f"Max positions: {max_positions}")
print(f"Transaction cost per trade: {transaction_cost*100:.2f}%")
print(f"Gross avg trade return: {gross_avg_trade_return*100:.2f}%")
print(f"Net avg trade return: {net_avg_trade_return*100:.2f}%")
print(f"Final capital: ${final_capital:,.2f}")
print(f"Total return: {total_return*100:.2f}%")
print(f"Avg daily return: {avg_daily_return*100:.2f}%")
print(f"Median daily return: {median_daily_return*100:.2f}%")
print(f"Max drawdown: {max_drawdown*100:.2f}%")

# Save equity curve CSV
equity_df.to_csv(EQUITY_OUTPUT_FILE, index=False)

# -----------------------------
# VISUALIZATIONS
# -----------------------------

# 1. Equity curve
plt.figure(figsize=(12, 6))
plt.plot(equity_df["date"], equity_df["equity"])
plt.title("Equity Curve")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.tight_layout()
plt.savefig(EQUITY_CHART_FILE, dpi=150)
plt.close()

# 2. Drawdown curve
plt.figure(figsize=(12, 6))
plt.plot(equity_df["date"], equity_df["drawdown"])
plt.title("Drawdown Curve")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.grid(True)
plt.tight_layout()
plt.savefig(DRAWDOWN_CHART_FILE, dpi=150)
plt.close()

# 3. Histogram of NET trade returns for selected strategy
plt.figure(figsize=(12, 6))
plt.hist(strategy_df["netReturn"], bins=100)
plt.title("Net Trade Return Distribution")
plt.xlabel("Net Trade Return")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig(RETURNS_HIST_FILE, dpi=150)
plt.close()

print(f"Min buyDate: {strategy_df['buyDate'].min()}")
print(f"Max sellDate: {strategy_df['sellDate'].max()}")
print(f"Years covered: {(strategy_df['sellDate'].max() - strategy_df['buyDate'].min()).days / 365.25:.2f}")
print(f"Unique buy days: {strategy_df['buyDate'].nunique()}")
print(f"Average trades per buy day: {len(strategy_df) / strategy_df['buyDate'].nunique():.2f}")

print()
print(f"Saved equity curve CSV to {EQUITY_OUTPUT_FILE}")
print(f"Saved equity chart to {EQUITY_CHART_FILE}")
print(f"Saved drawdown chart to {DRAWDOWN_CHART_FILE}")
print(f"Saved return histogram to {RETURNS_HIST_FILE}")