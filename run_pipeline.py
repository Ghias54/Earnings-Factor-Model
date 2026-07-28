#!/usr/bin/env python3
"""
Run the processing stack in order (no API ingestion here).

Ingestion (run separately when refreshing raw data); full chain with correct
ordering (filter_earnings before shares_history):
  python3 scripts/run_ingestion.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PROCESSING_STEPS = [
    "src/processing/prep/filter_earnings_to_price_range.py",
    "src/processing/prep/build_ttm_fundamentals.py",
    "src/processing/prep/build_ttm_financials.py",
    "src/processing/features/build_valuation_features.py",
    "src/processing/scores/build_valuation_score.py",
    "src/processing/backtest/earnings_returns.py",
    "src/processing/prep/build_earnings_events.py",
    "src/processing/features/momentum_features.py",
    "src/processing/features/build_growth_features.py",
    "src/processing/scores/build_growth_score.py",
    "src/processing/features/build_profitability_features.py",
    "src/processing/scores/build_profitability_score.py",
    "src/processing/features/build_revisions_features.py",
    "src/processing/features/build_insider_features.py",
    "src/processing/scores/build_revisions_score.py",
    "src/processing/scores/build_insider_score.py",
    "src/processing/scores/build_momentum_score.py",
    "src/processing/scores/build_composite_quant_score.py",
    "src/processing/backtest/build_master_daily_panel.py",
    "src/processing/backtest/simulate_portfolio.py",
]


def main() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    for rel in PROCESSING_STEPS:
        script = ROOT / rel
        if not script.exists():
            print(f"Skip missing script: {script}")
            continue
        print(f"\n=== Running {rel} ===\n")
        r = subprocess.run([sys.executable, str(script)], cwd=str(ROOT), env=env)
        if r.returncode != 0:
            raise SystemExit(r.returncode)
    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
