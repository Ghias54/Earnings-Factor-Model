import re
import pandas as pd

from config import COMPANIES_CLEANED_FILE, PROCESSED_FACTORS_DIR, RAW_DATA_DIR


# Remove ETFs / funds
ETF_KEYWORDS = [
    "etf",
    "fund",
    "trust",
    "ishares",
    "spdr",
    "vanguard",
    "proshares",
    "invesco",
    "direxion",
    "ark ",
    "etn",
    "adr",
]

# Remove SPACs / junk / non-operating companies
BAD_KEYWORDS = [
    "acquisition",
    "capital",
    "holdings",
    "units",
    "rights",
    "warrants",
    "corp ii",
    "corp iii",
]


def load_raw_companies() -> pd.DataFrame:
    path = RAW_DATA_DIR / "companies.csv"
    return pd.read_csv(path)


def is_clean_ticker(ticker: str) -> bool:
    if not isinstance(ticker, str):
        return False

    ticker = ticker.strip()

    # Keep simple US-style tickers only: 1–5 uppercase letters
    return bool(re.fullmatch(r"[A-Z]{1,5}", ticker))


def is_valid_company(company_name: str) -> bool:
    if not isinstance(company_name, str):
        return False

    name = company_name.lower()

    # Remove ETFs / funds
    for keyword in ETF_KEYWORDS:
        if keyword in name:
            return False

    # Remove SPACs / junk
    for keyword in BAD_KEYWORDS:
        if keyword in name:
            return False

    return True


def clean_universe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    print(f"Starting rows: {len(df)}")

    # Remove nulls
    df = df.dropna(subset=["ticker", "companyName"])
    print(f"After dropping nulls: {len(df)}")

    # Keep only clean tickers
    df = df[df["ticker"].apply(is_clean_ticker)]
    print(f"After ticker filter: {len(df)}")

    # Remove ETFs, funds, SPACs, junk
    df = df[df["companyName"].apply(is_valid_company)]
    print(f"After ETF/SPAC filter: {len(df)}")

    # Remove duplicates
    df = df.drop_duplicates(subset=["ticker"])
    print(f"After removing duplicates: {len(df)}")

    # Sort
    df = df.sort_values("ticker").reset_index(drop=True)

    return df


def save_cleaned_universe(df: pd.DataFrame) -> None:
    PROCESSED_FACTORS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(COMPANIES_CLEANED_FILE, index=False)
    print(f"Saved cleaned universe to {COMPANIES_CLEANED_FILE}")


def run() -> None:
    df = load_raw_companies()
    cleaned_df = clean_universe(df)
    save_cleaned_universe(cleaned_df)


if __name__ == "__main__":
    run()