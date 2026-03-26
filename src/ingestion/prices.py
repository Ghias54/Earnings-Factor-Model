import time
import requests
import pandas as pd

from config import FMP_API_KEY, FMP_BASE_URL, RAW_DATA_DIR, PROCESSED_DATA_DIR


OUTPUT_FILE = RAW_DATA_DIR / "daily_prices_from_clean_universe.csv"
ERROR_FILE = RAW_DATA_DIR / "daily_prices_from_clean_universe_errors.csv"

SLEEP_SECONDS = 0.05
CHECKPOINT_EVERY = 100


def load_cleaned_companies() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "companies_cleaned.csv"
    return pd.read_csv(path)


def get_completed_tickers(output_file) -> set:
    if not output_file.exists():
        return set()

    df = pd.read_csv(output_file, usecols=["ticker"], on_bad_lines="skip", low_memory=False)
    df = df[df["ticker"] != "ticker"].copy()

    return set(df["ticker"].dropna().astype(str).str.strip().unique())


def fetch_prices_for_ticker(session: requests.Session, ticker: str):
    url = f"{FMP_BASE_URL}/historical-price-eod/full"
    params = {
        "symbol": ticker,
        "apikey": FMP_API_KEY,
    }

    for attempt in range(3):
        try:
            response = session.get(url, params=params, timeout=30)

            if response.status_code == 200:
                return response.json(), None

            if response.status_code in [429, 500, 502, 503]:
                time.sleep(2 * (attempt + 1))
                continue

            return None, f"HTTP {response.status_code}"

        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
            else:
                return None, str(e)

    return None, "Failed after retries"


def standardize_price_rows(ticker: str, data):
    rows = []

    if not isinstance(data, list):
        return rows

    for row in data:
        rows.append(
            {
                "ticker": ticker,
                "date": row.get("date"),
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "volume": row.get("volume"),
            }
        )

    return rows


def append_rows_to_csv(rows: list, output_file) -> None:
    if not rows:
        return

    df = pd.DataFrame(rows)
    file_exists = output_file.exists()
    df.to_csv(output_file, mode="a", header=not file_exists, index=False)


def append_errors_to_csv(errors: list, error_file) -> None:
    if not errors:
        return

    df = pd.DataFrame(errors)
    file_exists = error_file.exists()
    df.to_csv(error_file, mode="a", header=not file_exists, index=False)


def run() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    companies_df = load_cleaned_companies()
    tickers = companies_df["ticker"].dropna().astype(str).str.strip().tolist()[:500]

    completed_tickers = get_completed_tickers(OUTPUT_FILE)
    original_total = len(tickers)
    tickers = [ticker for ticker in tickers if ticker not in completed_tickers]

    batch_prices = []
    batch_errors = []
    total_rows_written = 0
    total_errors_written = 0

    print(f"Total cleaned tickers: {original_total}")
    print(f"Already completed tickers: {len(completed_tickers)}")
    print(f"Remaining tickers to fetch: {len(tickers)}")

    with requests.Session() as session:
        for i, ticker in enumerate(tickers, start=1):
            print(f"[{i}/{len(tickers)}] Fetching prices for {ticker}")

            data, error = fetch_prices_for_ticker(session, ticker)

            if error is not None:
                batch_errors.append(
                    {
                        "ticker": ticker,
                        "error": error,
                    }
                )
            else:
                rows = standardize_price_rows(ticker, data)

                if rows:
                    batch_prices.extend(rows)
                else:
                    batch_errors.append(
                        {
                            "ticker": ticker,
                            "error": "No price data returned",
                        }
                    )

            if i % CHECKPOINT_EVERY == 0:
                append_rows_to_csv(batch_prices, OUTPUT_FILE)
                append_errors_to_csv(batch_errors, ERROR_FILE)

                total_rows_written += len(batch_prices)
                total_errors_written += len(batch_errors)

                print(
                    f"Checkpoint at {i} tickers | "
                    f"batch rows written: {len(batch_prices)} | "
                    f"total rows written this run: {total_rows_written} | "
                    f"errors written this run: {total_errors_written}"
                )

                batch_prices = []
                batch_errors = []

            time.sleep(SLEEP_SECONDS)

    append_rows_to_csv(batch_prices, OUTPUT_FILE)
    append_errors_to_csv(batch_errors, ERROR_FILE)

    total_rows_written += len(batch_prices)
    total_errors_written += len(batch_errors)

    print("\nDone.")
    print(f"Total price rows written this run: {total_rows_written}")
    print(f"Total errors written this run: {total_errors_written}")
    print(f"Prices file: {OUTPUT_FILE}")
    print(f"Errors file: {ERROR_FILE}")


if __name__ == "__main__":
    run()