from pathlib import Path
import os
import re
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
_ENV_PATH = BASE_DIR / ".env"
# Exposed so scripts can print the path in error messages
ENV_FILE_PATH = _ENV_PATH


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse KEY=value lines; UTF-8 BOM safe; strips quotes and whitespace."""
    if not path.is_file():
        return {}
    try:
        raw = path.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if val and (val[0] == val[-1]) and val[0] in "\"'":
            val = val[1:-1]
        if "#" in val and not (val.startswith('"') or val.startswith("'")):
            val = val.split("#", 1)[0].strip()
        out[key] = val
    return out


# Load .env: dotenv first, then manual parse (handles BOM / odd editors).
load_dotenv(_ENV_PATH, override=True)
if not os.getenv("FMP_API_KEY"):
    load_dotenv(Path.cwd() / ".env", override=True)
if not os.getenv("FMP_API_KEY"):
    for k, v in _parse_env_file(_ENV_PATH).items():
        if k == "FMP_API_KEY" and v:
            os.environ["FMP_API_KEY"] = v
            break
if not os.getenv("FMP_API_KEY"):
    for k, v in _parse_env_file(Path.cwd() / ".env").items():
        if k == "FMP_API_KEY" and v:
            os.environ["FMP_API_KEY"] = v
            break

# API — env / .env take precedence; fallback only if unset (avoid committing real keys to public repos)
FMP_API_KEY = (os.getenv("FMP_API_KEY") or "").strip() or None
if not FMP_API_KEY:
    FMP_API_KEY = "cdX5NTapRqNthZvrvfUpif3QRHpv0DUN"
FMP_BASE_URL = "https://financialmodelingprep.com/stable"

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Earnings event trade geometry (trading days vs last trading day before announcement).
# Buy: `buy_days_before_anchor` sessions before that anchor; sell: `sell_days_after_anchor`
# sessions after the anchor (anchor = last bar strictly before earningsAnnouncementDate).
EARNINGS_BUY_DAYS_BEFORE_ANCHOR = 5
EARNINGS_SELL_DAYS_AFTER_ANCHOR = 25
# Trades require buy price strictly above this (same idea in earnings_returns + simulate_portfolio).
MIN_BUY_PRICE_FOR_TRADE = 1.0

# Daily panel: prices + point-in-time composite (see build_master_daily_panel.py)
MASTER_DAILY_QUANT_PANEL_FILE = PROCESSED_DATA_DIR / "master_daily_quant_panel.csv.gz"
