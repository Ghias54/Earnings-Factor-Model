#!/usr/bin/env python3
"""
Post-ingestion runner: waits for financial_statements ingestion to finish,
then fetches prices for the full universe, then runs the processing pipeline.

Usage (runs automatically after calling this script):
  python3 run_after_ingestion.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

FS_LOG = Path("/tmp/fin_stmt_ingest.log")
FS_OUT = ROOT / "data" / "raw" / "financial_statements_quarterly.csv"


def run_step(rel: str, label: str | None = None) -> int:
    label = label or rel
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.call([sys.executable, str(ROOT / rel)], cwd=str(ROOT), env=env)


def wait_for_financial_statements() -> None:
    """Block until the background financial_statements.py process finishes."""
    print("Waiting for financial_statements ingestion to complete...")
    print("(Check progress: tail -f /tmp/fin_stmt_ingest.log)\n")

    dot_count = 0
    while True:
        # Check if the log file says Done
        if FS_LOG.exists():
            text = FS_LOG.read_text(errors="replace")
            if "Done." in text and "tickers processed" in text:
                lines = [l for l in text.splitlines() if "Done." in l or "tickers processed" in l]
                print(f"\nFinancial statements complete: {lines[-1] if lines else 'Done'}")
                break

        # Heartbeat every 30s
        time.sleep(30)
        dot_count += 1
        rows = 0
        if FS_OUT.exists():
            try:
                with open(FS_OUT) as f:
                    rows = sum(1 for _ in f) - 1
            except Exception:
                pass
        print(f"  [{dot_count * 30}s elapsed] {rows:,} rows written so far...", flush=True)


def summarise_errors() -> None:
    """Print a brief error summary across all raw data files."""
    print("\n" + "="*60)
    print("  Error summary (data/raw/*_errors.csv)")
    print("="*60)
    error_files = sorted((ROOT / "data" / "raw").glob("*errors*.csv"))
    for f in error_files:
        try:
            import pandas as pd
            df = pd.read_csv(f)
            n = len(df)
            if n > 0:
                top = df["error"].value_counts().head(3).to_dict() if "error" in df.columns else {}
                print(f"  {f.name}: {n:,} errors")
                for err, cnt in top.items():
                    short = err[:60]
                    print(f"    {cnt:>5}x  {short}")
        except Exception as e:
            print(f"  {f.name}: could not read ({e})")
    print()


def main() -> None:
    # Step 1: wait for financial statements
    wait_for_financial_statements()

    summarise_errors()

    # Step 2: fetch prices for full universe (resumes from where it left off)
    code = run_step("src/ingestion/prices.py", "Fetching prices for full universe (~21,585 tickers)")
    if code != 0:
        print(f"prices.py exited with code {code} — check errors and re-run if needed.")

    # Step 3: summarise errors again after prices
    summarise_errors()

    # Step 4: rebuild the full processing pipeline
    code = run_step("run_pipeline.py", "Running full processing pipeline")
    if code != 0:
        print(f"run_pipeline.py exited with code {code}.")
        raise SystemExit(code)

    print("\n✓ All steps complete.")


if __name__ == "__main__":
    main()
