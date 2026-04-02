import pandas as pd
from config import PROCESSED_DATA_DIR

RETURNS_FILE = PROCESSED_DATA_DIR / "earnings_returns.csv"
MOMENTUM_FILE = PROCESSED_DATA_DIR / "momentum_features.csv"
EVENTS_FILE = PROCESSED_DATA_DIR / "earnings_events.csv"

# Load data
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

# Drop rows missing core fields for testing
df = df.dropna(subset=["returnDecimal", "buyPrice", "epsSurprise"]).copy()

# Create clipped return version
df["clipped_return"] = df["returnDecimal"].clip(-0.20, 0.20)

print(f"Merged rows: {len(df):,}")
print()

def print_stats(name, test_df, return_col):
    test_df = test_df.dropna(subset=[return_col]).copy()

    count = len(test_df)
    if count == 0:
        print(f"=== {name} ===")
        print("Count: 0")
        print()
        return

    avg_return = test_df[return_col].mean()
    median_return = test_df[return_col].median()
    win_rate = (test_df[return_col] > 0).mean()

    print(f"=== {name} ===")
    print(f"Count: {count:,}")
    print(f"Avg return: {avg_return:.4f}")
    print(f"Median return: {median_return:.4f}")
    print(f"Win rate: {win_rate:.4f}")
    print()

# Thresholds from full cleaned dataset
top_20_threshold = df["epsSurprise"].quantile(0.80)
top_10_threshold = df["epsSurprise"].quantile(0.90)
top_5_threshold = df["epsSurprise"].quantile(0.95)

# Build test groups
positive_surprise = df[df["epsSurprise"] > 0]
top_20 = df[df["epsSurprise"] >= top_20_threshold]
top_10 = df[df["epsSurprise"] >= top_10_threshold]
top_5 = df[df["epsSurprise"] >= top_5_threshold]

positive_surprise_price = df[
    (df["epsSurprise"] > 0) &
    (df["buyPrice"] > 5)
]

top_10_price = df[
    (df["epsSurprise"] >= top_10_threshold) &
    (df["buyPrice"] > 5)
]

top_5_price = df[
    (df["epsSurprise"] >= top_5_threshold) &
    (df["buyPrice"] > 5)
]

low_mom_positive_surprise = df[
    (df["epsSurprise"] > 0) &
    (df["mom63"] < 0)
]

low_mom_top_10_price = df[
    (df["epsSurprise"] >= top_10_threshold) &
    (df["buyPrice"] > 5) &
    (df["mom63"] < 0)
]

# -------------------------
# ORIGINAL RETURNS
# -------------------------
print("#############################")
print("### ORIGINAL RETURN TESTS ###")
print("#############################")
print()

print_stats("BASELINE", df, "returnDecimal")
print_stats("POSITIVE EPS SURPRISE", positive_surprise, "returnDecimal")
print_stats("TOP 20% EPS SURPRISE", top_20, "returnDecimal")
print_stats("TOP 10% EPS SURPRISE", top_10, "returnDecimal")
print_stats("TOP 5% EPS SURPRISE", top_5, "returnDecimal")
print_stats("POSITIVE EPS SURPRISE + BUY PRICE > 5", positive_surprise_price, "returnDecimal")
print_stats("TOP 10% EPS SURPRISE + BUY PRICE > 5", top_10_price, "returnDecimal")
print_stats("TOP 5% EPS SURPRISE + BUY PRICE > 5", top_5_price, "returnDecimal")
print_stats("POSITIVE EPS SURPRISE + LOW MOM63", low_mom_positive_surprise, "returnDecimal")
print_stats("TOP 10% EPS SURPRISE + BUY PRICE > 5 + LOW MOM63", low_mom_top_10_price, "returnDecimal")

# -------------------------
# CLIPPED RETURNS
# -------------------------
print("############################")
print("### CLIPPED RETURN TESTS ###")
print("############################")
print()

print_stats("BASELINE", df, "clipped_return")
print_stats("POSITIVE EPS SURPRISE", positive_surprise, "clipped_return")
print_stats("TOP 20% EPS SURPRISE", top_20, "clipped_return")
print_stats("TOP 10% EPS SURPRISE", top_10, "clipped_return")
print_stats("TOP 5% EPS SURPRISE", top_5, "clipped_return")
print_stats("POSITIVE EPS SURPRISE + BUY PRICE > 5", positive_surprise_price, "clipped_return")
print_stats("TOP 10% EPS SURPRISE + BUY PRICE > 5", top_10_price, "clipped_return")
print_stats("TOP 5% EPS SURPRISE + BUY PRICE > 5", top_5_price, "clipped_return")
print_stats("POSITIVE EPS SURPRISE + LOW MOM63", low_mom_positive_surprise, "clipped_return")
print_stats("TOP 10% EPS SURPRISE + BUY PRICE > 5 + LOW MOM63", low_mom_top_10_price, "clipped_return")

print("=== THRESHOLDS USED ===")
print(f"Top 20% epsSurprise threshold: {top_20_threshold:.4f}")
print(f"Top 10% epsSurprise threshold: {top_10_threshold:.4f}")
print(f"Top 5% epsSurprise threshold: {top_5_threshold:.4f}")
print()

print("=== CLIPPED RETURN RULE ===")
print("Returns were clipped to the range [-0.20, 0.20].")
print("That means anything below -20% becomes -20%, and anything above +20% becomes +20%.")
print()