import pandas as pd
from config import PROCESSED_DATA_DIR

RETURNS_FILE = PROCESSED_DATA_DIR / "earnings_returns.csv"
MOMENTUM_FILE = PROCESSED_DATA_DIR / "momentum_features.csv"
EVENTS_FILE = PROCESSED_DATA_DIR / "earnings_events.csv"

# Load data
returns_df = pd.read_csv(RETURNS_FILE)
momentum_df = pd.read_csv(MOMENTUM_FILE)
events_df = pd.read_csv(EVENTS_FILE)

# Clean keys
for df in [returns_df, momentum_df, events_df]:
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["earningsAnnouncementDate"] = pd.to_datetime(df["earningsAnnouncementDate"])

# Merge ALL (still in memory only)
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

print(f"Merged rows: {len(df)}\n")

# -------------------------
# 🔹 BASELINE
# -------------------------
print("=== BASELINE ===")
print(f"Avg return: {df['returnDecimal'].mean():.4f}")
print(f"Win rate: {(df['returnDecimal'] > 0).mean():.4f}\n")

# -------------------------
# 🔹 SURPRISE ONLY
# -------------------------
pos_surprise = df[df["epsSurprise"] > 0]
neg_surprise = df[df["epsSurprise"] <= 0]

print("=== EPS SURPRISE TEST ===")
print(f"Positive surprise avg return: {pos_surprise['returnDecimal'].mean():.4f}")
print(f"Positive surprise win rate: {(pos_surprise['returnDecimal'] > 0).mean():.4f}")
print()
print(f"Negative surprise avg return: {neg_surprise['returnDecimal'].mean():.4f}")
print(f"Negative surprise win rate: {(neg_surprise['returnDecimal'] > 0).mean():.4f}")
print()

# -------------------------
# 🔹 MOMENTUM + SURPRISE
# -------------------------
combo = df[
    (df["epsSurprise"] > 0) &
    (df["mom63"] < 0)
]

print("=== LOW MOMENTUM + POSITIVE SURPRISE ===")
print(f"Count: {len(combo)}")
print(f"Avg return: {combo['returnDecimal'].mean():.4f}")
print(f"Win rate: {(combo['returnDecimal'] > 0).mean():.4f}")
print()

# -------------------------
# 🔹 HIGH MOM + POSITIVE SURPRISE (compare)
# -------------------------
combo_high = df[
    (df["epsSurprise"] > 0) &
    (df["mom63"] > 0)
]

print("=== HIGH MOMENTUM + POSITIVE SURPRISE ===")
print(f"Count: {len(combo_high)}")
print(f"Avg return: {combo_high['returnDecimal'].mean():.4f}")
print(f"Win rate: {(combo_high['returnDecimal'] > 0).mean():.4f}")
print()

# -------------------------
# 🔹 EXTREME SURPRISE (top 20%)
# -------------------------
df = df.dropna(subset=["epsSurprise"])

threshold = df["epsSurprise"].quantile(0.8)

extreme = df[df["epsSurprise"] >= threshold]

print("=== TOP 20% EPS SURPRISE ===")
print(f"Count: {len(extreme)}")
print(f"Avg return: {extreme['returnDecimal'].mean():.4f}")
print(f"Win rate: {(extreme['returnDecimal'] > 0).mean():.4f}")
print()