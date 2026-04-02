import pandas as pd
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

df = pd.read_csv(PROCESSED_DATA_DIR / "valuation_features.csv")


print("Number of Rows:", len(df))