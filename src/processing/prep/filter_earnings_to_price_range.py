import pandas as pd

from config import EARNINGS_FROM_PRICE_START_FILE, PROCESSED_FACTORS_DIR, RAW_DATA_DIR


EARNINGS_FILE = RAW_DATA_DIR / "earnings_from_clean_universe.csv"
OUTPUT_FILE = EARNINGS_FROM_PRICE_START_FILE

PRICE_START_DATE = "2021-03-29"


def run() -> None:
    PROCESSED_FACTORS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(EARNINGS_FILE)

    df["earningsAnnouncementDate"] = pd.to_datetime(
        df["earningsAnnouncementDate"],
        errors="coerce",
    )

    df = df.dropna(subset=["earningsAnnouncementDate"]).copy()

    original_rows = len(df)

    filtered_df = df[
        df["earningsAnnouncementDate"] >= pd.Timestamp(PRICE_START_DATE)
    ].copy()

    filtered_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Price start date used: {PRICE_START_DATE}")
    print(f"Original earnings rows: {original_rows}")
    print(f"Filtered earnings rows: {len(filtered_df)}")
    print(f"Rows removed: {original_rows - len(filtered_df)}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    run()