import pandas as pd
from config import RAW_DATA_DIR

df = pd.read_csv(RAW_DATA_DIR / "")


print("Number of Rows:", len(df))