#!/usr/bin/env python3
"""
Fetch raw data from FMP in dependency order, then run the pre-processing steps
that depend on it (company cleaning, earnings filtering, company enrichment).

Order:
  companies → clean_companies → prices → earnings → filter_earnings_to_price_range
  → shares_history → company_profiles → enrich_companies → analyst_estimates

Usage (from repo root):
  python3 run_ingestion.py
  python3 run_ingestion.py --from src/ingestion/prices.py   # resume at a step
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    "src/ingestion/companies.py",
    "src/processing/prep/clean_companies.py",
    "src/ingestion/insider_trades.py",
    "src/ingestion/prices.py",
    "src/ingestion/earnings.py",
    "src/processing/prep/filter_earnings_to_price_range.py",
    "src/ingestion/shares_history.py",
    "src/ingestion/company_profiles.py",
    "src/processing/prep/enrich_companies.py",
    "src/ingestion/analyst_estimates.py",
    "src/ingestion/financial_statements.py",   # income stmt + balance sheet + cash flow
]


def run_step(rel: str) -> int:
    script = ROOT / rel
    if not script.is_file():
        print(f"[SKIP] {rel} — file not found", file=sys.stderr)
        return 0
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.call([sys.executable, str(script)], cwd=str(ROOT), env=env)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--from",
        dest="from_step",
        metavar="REL_PATH",
        help="Start at this step (e.g. src/ingestion/prices.py).",
    )
    args = p.parse_args()

    steps = STEPS
    if args.from_step:
        rel = args.from_step.replace("\\", "/").strip()
        if rel not in STEPS:
            print(f"Unknown step {rel!r}. Valid steps:\n" + "\n".join(f"  {s}" for s in STEPS),
                  file=sys.stderr)
            raise SystemExit(2)
        steps = STEPS[STEPS.index(rel):]

    for rel in steps:
        print(f"\n=== {rel} ===\n", flush=True)
        code = run_step(rel)
        if code != 0:
            print(f"\nStep failed: {rel}", file=sys.stderr)
            raise SystemExit(code)

    print(f"\nIngestion complete ({len(steps)} step(s)).")


if __name__ == "__main__":
    main()
