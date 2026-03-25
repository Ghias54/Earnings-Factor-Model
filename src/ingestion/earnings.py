import time
import requests
import pandas as pd

from config import FMP_API_KEY, FMP_BASE_URL, RAW_DATA_DIR, PROCESSED_DATA_DIR


OUTPUT_FILE = RAW_DATA_DIR / "earnings_from_clean_universe.csv"
ERROR_FILE = RAW_DATA_DIR / "earnings_from_clean_universe_errors.csv"

SLEEP_SECONDS = 0.10
CHECKPOINT_EVERY = 100


def load_cleaned_companies() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "companies_cleaned.csv"
    return pd.read_csv(path)


def fetch_earnings_for_ticker(ticker: str):
    url = f"{FMP_BASE_URL}/earnings"
    params = {
        "symbol": ticker,
        "apikey": FMP_API_KEY,
    }

    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=30)

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


def standardize_earnings_rows(ticker: str, data):
    rows = []

    if not isinstance(data, list):
        return rows

    for row in data:
        rows.append(
            {
                "ticker": ticker,
                "earningsAnnouncementDate": row.get("date"),
                "actualEps": row.get("epsActual"),
                "estimatedEps": row.get("epsEstimated"),
                "actualRevenue": row.get("revenueActual"),
                "estimatedRevenue": row.get("revenueEstimated"),
                "lastUpdated": row.get("lastUpdated"),
            }
        )

    return rows


def save_checkpoint(events: list, errors: list) -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if events:
        pd.DataFrame(events).to_csv(OUTPUT_FILE, index=False)

    if errors:
        pd.DataFrame(errors).to_csv(ERROR_FILE, index=False)


def run() -> None:
    companies_df = load_cleaned_companies()
    tickers = companies_df["ticker"].dropna().astype(str).tolist()

    all_events = []
    all_errors = []

    total = len(tickers)
    print(f"Total cleaned tickers: {total}")

    for i, ticker in enumerate(tickers, start=1):  # test batch first
        print(f"[{i}/{total}] Fetching earnings for {ticker}")

        data, error = fetch_earnings_for_ticker(ticker)

        if error is not None:
            all_errors.append(
                {
                    "ticker": ticker,
                    "error": error,
                }
            )
        else:
            rows = standardize_earnings_rows(ticker, data)

            if rows:
                all_events.extend(rows)
            else:
                all_errors.append(
                    {
                        "ticker": ticker,
                        "error": "No earnings data returned",
                    }
                )

        if i % CHECKPOINT_EVERY == 0:
            save_checkpoint(all_events, all_errors)
            print(
                f"Checkpoint at {i} tickers | "
                f"events: {len(all_events)} | errors: {len(all_errors)}"
            )

        time.sleep(SLEEP_SECONDS)

    save_checkpoint(all_events, all_errors)

    print("\nDone.")
    print(f"Total events saved: {len(all_events)}")
    print(f"Total errors saved: {len(all_errors)}")
    print(f"Earnings file: {OUTPUT_FILE}")
    print(f"Errors file: {ERROR_FILE}")


if __name__ == "__main__":
    run()