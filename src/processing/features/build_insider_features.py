from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import EARNINGS_FROM_PRICE_START_FILE, INSIDER_FEATURES_FILE, INSIDER_TRADES_RAW_FILE, PROCESSED_FACTORS_DIR

LARGE_TXN_VALUE = 100_000.0
INFORMATIVE_TXN_VALUE = 50_000.0
UNINFORMATIVE_CODES = {"A", "G", "M", "F", "I", "J", "C"}
BUY_CODES = {"P"}
SELL_CODES = {"S"}


def _norm_bool(v: object) -> int:
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return 1
    return 0


def _prepare_base() -> pd.DataFrame:
    base = pd.read_csv(EARNINGS_FROM_PRICE_START_FILE, low_memory=False)
    base["ticker"] = base["ticker"].astype(str).str.strip()
    base["earningsAnnouncementDate"] = pd.to_datetime(base["earningsAnnouncementDate"], errors="coerce")
    base = base.dropna(subset=["ticker", "earningsAnnouncementDate"]).copy()
    return base.sort_values(["ticker", "earningsAnnouncementDate"]).reset_index(drop=True)


def _prepare_insider() -> pd.DataFrame:
    if not INSIDER_TRADES_RAW_FILE.exists():
        return pd.DataFrame(
            columns=[
                "ticker",
                "effective_date",
                "is_buy",
                "is_sell",
                "buy_value",
                "sell_value",
                "ceo_buy",
                "cfo_buy",
                "director_buy",
                "tenpct_buy",
                "large_buy",
                "large_sell",
                "informative_buy",
                "informative_sell",
            ]
        )

    df = pd.read_csv(INSIDER_TRADES_RAW_FILE, low_memory=False)
    if df.empty:
        return df

    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["transaction_date"] = pd.to_datetime(df.get("transaction_date"), errors="coerce")
    df["filing_date"] = pd.to_datetime(df.get("filing_date"), errors="coerce")
    # Backtest-safe: use filing date when available (publicly known timestamp); fallback to transaction date.
    df["effective_date"] = df["filing_date"].where(df["filing_date"].notna(), df["transaction_date"])
    df["transaction_value"] = pd.to_numeric(df.get("transaction_value"), errors="coerce")
    df["transaction_code_norm"] = (
        df.get("transaction_code", pd.Series("", index=df.index)).astype(str).str.upper().str.strip()
    )
    txt = (
        df.get("transaction_type", pd.Series("", index=df.index)).astype(str).str.lower().fillna("")
        + " "
        + df.get("reporting_owner_title", pd.Series("", index=df.index)).astype(str).str.lower().fillna("")
    )
    title = df.get("reporting_owner_title", pd.Series("", index=df.index)).astype(str).str.lower()
    is_officer = df.get("is_officer", pd.Series(0, index=df.index)).map(_norm_bool) == 1
    is_director = df.get("is_director", pd.Series(0, index=df.index)).map(_norm_bool) == 1
    is_tenpct = df.get("is_ten_percent_owner", pd.Series(0, index=df.index)).map(_norm_bool) == 1

    code = df["transaction_code_norm"]
    is_buy = code.isin(BUY_CODES) | txt.str.contains("purchase", na=False)
    is_sell = code.isin(SELL_CODES) | txt.str.contains("sale|sell", na=False)

    df["is_buy"] = is_buy.astype(int)
    df["is_sell"] = is_sell.astype(int)
    df["buy_value"] = np.where(is_buy, df["transaction_value"].fillna(0.0), 0.0)
    df["sell_value"] = np.where(is_sell, df["transaction_value"].fillna(0.0), 0.0)

    role_is_meaningful = is_officer | is_director | is_tenpct | title.str.contains("ceo|chief executive", na=False) | title.str.contains("cfo|chief financial", na=False)
    uninformative = code.isin(UNINFORMATIVE_CODES) | txt.str.contains("award|grant|compensation|vesting|option", na=False)

    ceo = title.str.contains("ceo|chief executive", na=False)
    cfo = title.str.contains("cfo|chief financial", na=False)
    df["ceo_buy"] = (is_buy & ceo).astype(int)
    df["cfo_buy"] = (is_buy & cfo).astype(int)
    df["director_buy"] = (is_buy & is_director).astype(int)
    df["tenpct_buy"] = (is_buy & is_tenpct).astype(int)
    df["large_buy"] = (is_buy & (df["buy_value"] >= LARGE_TXN_VALUE)).astype(int)
    df["large_sell"] = (is_sell & (df["sell_value"] >= LARGE_TXN_VALUE)).astype(int)

    df["informative_buy"] = (
        is_buy
        & role_is_meaningful
        & (~uninformative)
        & (df["buy_value"] >= INFORMATIVE_TXN_VALUE)
    ).astype(int)
    df["informative_sell"] = (
        is_sell
        & role_is_meaningful
        & (~uninformative)
        & (df["sell_value"] >= INFORMATIVE_TXN_VALUE)
    ).astype(int)

    df = df.dropna(subset=["ticker", "effective_date"]).copy()
    return df.sort_values(["ticker", "effective_date"]).reset_index(drop=True)


def _window_sum(prefix: np.ndarray, l: int, r: int) -> float:
    if r <= l:
        return 0.0
    return float(prefix[r] - prefix[l])


def _compute_group_features(base_g: pd.DataFrame, ins_g: pd.DataFrame) -> pd.DataFrame:
    out = base_g.copy()
    dates = ins_g["effective_date"].to_numpy(dtype="datetime64[ns]")
    if len(dates) == 0:
        for c in [
            "insider_buy_count_30d",
            "insider_sell_count_30d",
            "insider_buy_count_90d",
            "insider_sell_count_90d",
            "insider_net_value_30d",
            "insider_net_value_90d",
            "ceo_buy_flag_90d",
            "cfo_buy_flag_90d",
            "ceo_cfo_buy_flag_90d",
            "director_buy_count_90d",
            "ten_percent_owner_buy_count_90d",
            "large_insider_buy_flag_90d",
            "large_insider_sell_flag_90d",
            "informative_buy_flag",
            "informative_sell_flag",
        ]:
            out[c] = 0.0
        return out

    arrs = {
        "buy": ins_g["is_buy"].to_numpy(dtype=float),
        "sell": ins_g["is_sell"].to_numpy(dtype=float),
        "buy_val": ins_g["buy_value"].to_numpy(dtype=float),
        "sell_val": ins_g["sell_value"].to_numpy(dtype=float),
        "ceo_buy": ins_g["ceo_buy"].to_numpy(dtype=float),
        "cfo_buy": ins_g["cfo_buy"].to_numpy(dtype=float),
        "director_buy": ins_g["director_buy"].to_numpy(dtype=float),
        "tenpct_buy": ins_g["tenpct_buy"].to_numpy(dtype=float),
        "large_buy": ins_g["large_buy"].to_numpy(dtype=float),
        "large_sell": ins_g["large_sell"].to_numpy(dtype=float),
        "inf_buy": ins_g["informative_buy"].to_numpy(dtype=float),
        "inf_sell": ins_g["informative_sell"].to_numpy(dtype=float),
    }
    pref = {k: np.concatenate([[0.0], np.cumsum(v)]) for k, v in arrs.items()}

    asof_dates = out["earningsAnnouncementDate"].to_numpy(dtype="datetime64[ns]")
    d30 = np.timedelta64(30, "D")
    d90 = np.timedelta64(90, "D")

    vals: dict[str, list[float]] = {k: [] for k in [
        "insider_buy_count_30d", "insider_sell_count_30d", "insider_buy_count_90d", "insider_sell_count_90d",
        "insider_net_value_30d", "insider_net_value_90d", "ceo_buy_flag_90d", "cfo_buy_flag_90d",
        "ceo_cfo_buy_flag_90d", "director_buy_count_90d", "ten_percent_owner_buy_count_90d",
        "large_insider_buy_flag_90d", "large_insider_sell_flag_90d", "informative_buy_flag", "informative_sell_flag",
    ]}

    for d in asof_dates:
        r = int(np.searchsorted(dates, d, side="left"))  # strictly historical; excludes same-day filings
        l30 = int(np.searchsorted(dates, d - d30, side="left"))
        l90 = int(np.searchsorted(dates, d - d90, side="left"))

        buy30 = _window_sum(pref["buy"], l30, r)
        sell30 = _window_sum(pref["sell"], l30, r)
        buy90 = _window_sum(pref["buy"], l90, r)
        sell90 = _window_sum(pref["sell"], l90, r)
        net30 = _window_sum(pref["buy_val"], l30, r) - _window_sum(pref["sell_val"], l30, r)
        net90 = _window_sum(pref["buy_val"], l90, r) - _window_sum(pref["sell_val"], l90, r)
        ceo90 = _window_sum(pref["ceo_buy"], l90, r)
        cfo90 = _window_sum(pref["cfo_buy"], l90, r)
        dir90 = _window_sum(pref["director_buy"], l90, r)
        ten90 = _window_sum(pref["tenpct_buy"], l90, r)
        lb90 = _window_sum(pref["large_buy"], l90, r)
        ls90 = _window_sum(pref["large_sell"], l90, r)
        ib90 = _window_sum(pref["inf_buy"], l90, r)
        is90 = _window_sum(pref["inf_sell"], l90, r)

        vals["insider_buy_count_30d"].append(buy30)
        vals["insider_sell_count_30d"].append(sell30)
        vals["insider_buy_count_90d"].append(buy90)
        vals["insider_sell_count_90d"].append(sell90)
        vals["insider_net_value_30d"].append(net30)
        vals["insider_net_value_90d"].append(net90)
        vals["ceo_buy_flag_90d"].append(1.0 if ceo90 > 0 else 0.0)
        vals["cfo_buy_flag_90d"].append(1.0 if cfo90 > 0 else 0.0)
        vals["ceo_cfo_buy_flag_90d"].append(1.0 if (ceo90 > 0 and cfo90 > 0) else 0.0)
        vals["director_buy_count_90d"].append(dir90)
        vals["ten_percent_owner_buy_count_90d"].append(ten90)
        vals["large_insider_buy_flag_90d"].append(1.0 if lb90 > 0 else 0.0)
        vals["large_insider_sell_flag_90d"].append(1.0 if ls90 > 0 else 0.0)
        vals["informative_buy_flag"].append(1.0 if ib90 > 0 else 0.0)
        vals["informative_sell_flag"].append(1.0 if is90 > 0 else 0.0)

    for k, v in vals.items():
        out[k] = v
    return out


def run() -> None:
    base = _prepare_base()
    insider = _prepare_insider()
    print(f"Base rows: {len(base):,}")
    print(f"Insider rows: {len(insider):,}")

    parts: list[pd.DataFrame] = []
    for t, bg in base.groupby("ticker", sort=False):
        ig = insider[insider["ticker"] == t] if not insider.empty else insider
        parts.append(_compute_group_features(bg, ig))

    out = pd.concat(parts, ignore_index=True) if parts else base.copy()
    out = out.sort_values(["ticker", "earningsAnnouncementDate"]).reset_index(drop=True)

    PROCESSED_FACTORS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(INSIDER_FEATURES_FILE, index=False)
    print(f"Saved {len(out):,} rows to {INSIDER_FEATURES_FILE}")


if __name__ == "__main__":
    run()
