import requests
import pandas as pd

from config import FMP_API_KEY, FMP_BASE_URL, RAW_DATA_DIR


def fetch_companies() -> pd.DataFrame:
    url = f"{FMP_BASE_URL}/stock-list"
    params = {"apikey": FMP_API_KEY}

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()
    return pd.DataFrame(data)


def clean_companies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    print("Columns returned:")
    print(df.columns.tolist())

    df = df[["symbol", "companyName"]]

    df = df.rename(
        columns={
            "symbol": "ticker",
            "companyName": "companyName",
        }
    )

    df = df.dropna(subset=["ticker", "companyName"])
    df = df.drop_duplicates(subset=["ticker"])
    df = df.sort_values("ticker").reset_index(drop=True)

    return df


def save_companies(df: pd.DataFrame) -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DATA_DIR / "companies.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")


def run() -> None:
    df = fetch_companies()
    print(f"Fetched {len(df)} total rows from FMP")

    cleaned_df = clean_companies(df)
    print(f"Cleaned down to {len(cleaned_df)} rows")

    save_companies(cleaned_df)


if __name__ == "__main__":
    run()