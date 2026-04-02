import pandas as pd
from config import PROCESSED_DATA_DIR

RETURNS_FILE = PROCESSED_DATA_DIR / "earnings_returns.csv"
MOMENTUM_FILE = PROCESSED_DATA_DIR / "momentum_features.csv"
EVENTS_FILE = PROCESSED_DATA_DIR / "earnings_events.csv"

# -----------------------------
# SETTINGS
# -----------------------------
starting_capital = 10000
transaction_cost = 0.002

hold_periods = [3, 5, 10, 15]
max_positions_list = [5, 10, 20, 30]

# -----------------------------
# LOAD + CLEAN
# -----------------------------
returns_df = pd.read_csv(RETURNS_FILE)
momentum_df = pd.read_csv(MOMENTUM_FILE)
events_df = pd.read_csv(EVENTS_FILE)

for df in [returns_df, momentum_df, events_df]:
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["earningsAnnouncementDate"] = pd.to_datetime(df["earningsAnnouncementDate"])

df = returns_df.merge(momentum_df, on=["ticker", "earningsAnnouncementDate"])
df = df.merge(events_df[["ticker", "earningsAnnouncementDate", "epsSurprise"]],
              on=["ticker", "earningsAnnouncementDate"])

df["returnDecimal"] = pd.to_numeric(df["returnDecimal"], errors="coerce")
df["buyPrice"] = pd.to_numeric(df["buyPrice"], errors="coerce")
df["epsSurprise"] = pd.to_numeric(df["epsSurprise"], errors="coerce")
df["mom63"] = pd.to_numeric(df["mom63"], errors="coerce")

df["buyDate"] = pd.to_datetime(df["buyDate"])
df["sellDate"] = pd.to_datetime(df["sellDate"])

df = df.dropna(subset=["returnDecimal", "buyPrice", "epsSurprise", "buyDate", "sellDate"])

# Apply cost
df["netReturn"] = (df["returnDecimal"] - transaction_cost).clip(lower=-1)

# Strategy filter (keep constant for grid test)
threshold = df["epsSurprise"].quantile(0.90)

base_df = df[
    (df["epsSurprise"] >= threshold) &
    (df["buyPrice"] > 5)
].copy()

# -----------------------------
# SIMULATION FUNCTION
# -----------------------------
def run_simulation(data, max_positions):
    capital = starting_capital
    open_positions = []

    all_dates = sorted(pd.unique(pd.concat([data["buyDate"], data["sellDate"]])))

    equity = []

    for date in all_dates:
        # Close trades
        still_open = []
        for pos in open_positions:
            if pos["sellDate"] <= date:
                capital += pos["value"]
            else:
                still_open.append(pos)
        open_positions = still_open

        # Open trades
        todays = data[data["buyDate"] == date]
        slots = max_positions - len(open_positions)

        if slots > 0 and len(todays) > 0:
            trades = todays.head(slots)

            allocation = capital / slots

            for _, row in trades.iterrows():
                if capital <= 0:
                    break

                alloc = min(allocation, capital)
                value = alloc * (1 + row["netReturn"])

                open_positions.append({
                    "sellDate": row["sellDate"],
                    "value": value
                })

                capital -= alloc

        total_equity = capital + sum(p["value"] for p in open_positions)
        equity.append(total_equity)

    equity = pd.Series(equity)

    total_return = (equity.iloc[-1] / starting_capital) - 1
    daily_ret = equity.pct_change().mean()

    max_dd = ((equity / equity.cummax()) - 1).min()

    return total_return, daily_ret, max_dd


# -----------------------------
# GRID TEST
# -----------------------------
results = []

for hold in hold_periods:
    for max_pos in max_positions_list:

        # Adjust sell dates for hold period
        temp = base_df.copy()
        temp["sellDate"] = temp["buyDate"] + pd.to_timedelta(hold, unit="D")

        total_ret, daily_ret, dd = run_simulation(temp, max_pos)

        results.append({
            "hold_days": hold,
            "max_positions": max_pos,
            "total_return": total_ret,
            "avg_daily_return": daily_ret,
            "max_drawdown": dd
        })

results_df = pd.DataFrame(results)

# Sort by best return
results_df = results_df.sort_values("total_return", ascending=False)

print("\n=== GRID TEST RESULTS ===")
print(results_df)

# Save
results_df.to_csv(PROCESSED_DATA_DIR / "grid_results.csv", index=False)