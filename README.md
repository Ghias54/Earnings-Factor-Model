# Earnings Event Trading

A quantitative earnings-event trading model that aims to improve win rate and consistency by filtering out low-quality setups.

## Strategy

This project builds a 5-factor quant model based on:

- Valuation
- Growth
- Profitability
- Momentum
- Revisions (proxy-based)

## Roadmap

### Phase 1
Build historical data warehouse:
- Earnings data
- Price data
- Financial statements
- Analyst estimates

### Phase 2
Build quant scoring system

### Phase 3
Research strategy performance

### Phase 4
Backtest trading strategy

### Phase 5
Live screening and automation

## Tech Stack
- Python
- Financial Modeling Prep API

## Data layout

- Raw downloads: `data/raw/` (gitignored)
- Processed features and scores: `data/processed/` (gitignored)
- Set `FMP_API_KEY` in `.env` at the project root for ingestion scripts.

## Pipeline

**Processing (after raw CSVs exist under `data/raw/` and processed inputs are available):**

```bash
PYTHONPATH=. python run_pipeline.py
```

The runner sets `PYTHONPATH` for child scripts so `config` imports resolve.

**Typical order**

1. Ingest: `src/ingestion/companies.py` → `src/processing/clean_companies.py` → prices, earnings, shares, optional `company_profiles.py` / `analyst_estimates.py`
2. `run_pipeline.py`: filter earnings → TTM → valuation features/scores → earnings returns → events → momentum features → growth/profitability/revisions/momentum scores → composite → `simulate_portfolio.py`

**Factor outputs**

| File | Role |
|------|------|
| `data/processed/valuation_scores.csv` | Valuation (PE/PS ranks) |
| `data/processed/growth_scores.csv` | YoY TTM revenue/EPS growth ranks |
| `data/processed/profitability_scores.csv` | Net margin + ROE proxy ranks |
| `data/processed/revisions_scores.csv` | EPS estimate revision (needs `data/raw/analyst_estimates_quarterly.csv`) |
| `data/processed/momentum_scores.csv` | 63d momentum rank |
| `data/processed/composite_quant_scores.csv` | Mean of available factor scores |

`src/processing/simulate_portfolio.py` defaults to a **composite** filter (`FILTER_MODE = "composite"`). If `composite_quant_scores.csv` is missing, it falls back to the legacy EPS-surprise + negative-momentum rule. Revisions stay NaN until analyst estimates are ingested; the composite still averages the other factors.