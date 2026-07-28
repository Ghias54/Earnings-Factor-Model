# Earnings Factor Model

A quantitative research framework that studies whether the price volatility around
corporate earnings announcements can be traded systematically — and, more
specifically, whether layering a multi-factor scoring model on top of a naive
"trade every earnings event" strategy improves consistency rather than just chasing
the largest moves.

> Research project — not financial advice. All results are hypothetical and derived
> from historical data.

## The idea

Earnings announcements are among the most reliable recurring sources of single-stock
volatility. A naive strategy that trades every event captures that volatility but
also inherits all of its noise: the win rate hovers near a coin flip and drawdowns
are brutal. The hypothesis this project tests is that most of that noise comes from
low-quality setups, and that ranking events by fundamental and quantitative factors —
then trading only the higher-conviction subset — should raise the win rate and
smooth the equity curve, even at the cost of taking fewer trades.

The entire pipeline exists to test that hypothesis end to end and to make the
trade-offs inspectable through an interactive dashboard.

## Data ingestion

Raw data is pulled from the Financial Modeling Prep API and cached to local CSV, so
the expensive network step runs once and every downstream experiment reads from
disk. Separate ingestion modules cover the distinct data domains the model needs:

- Company universe and profiles (sector, market cap)
- Earnings announcement dates and surprises
- Daily split-adjusted prices
- Shares-outstanding history (for point-in-time valuation)
- Analyst estimates and revisions
- Financial statements (for TTM fundamentals)
- Insider transactions

Splitting ingestion by domain keeps each API concern isolated and makes it cheap to
refresh one dataset without re-pulling everything.

## Factor construction and scoring

Each earnings event is scored across six factors — valuation, growth, profitability,
momentum, analyst revisions, and insider activity. Rather than mixing raw metrics on
incompatible scales, each factor is converted to a **cross-sectional percentile
rank**, so a stock is measured against its peers at that point in time rather than
against an absolute threshold. Percentile scores are then mapped onto a 0–5 rating
scale for interpretability.

The six factor scores are blended into a single **composite quant score** as an
average of the available components. The pipeline explicitly tracks how many factors
were present for each event, so a name scored on two factors is never silently
treated as equivalent to one scored on all six — missing data reduces the component
count instead of being imputed away.

## Point-in-time discipline (avoiding look-ahead)

The most common way an earnings backtest lies to you is by leaking future
information into past decisions. Two design choices guard against it:

- **As-of factor attachment.** Composite scores are joined to trades with a
  backward-looking `merge_asof`, so each trade only ever sees the most recent score
  that existed *on or before* its entry date — never a score computed from data that
  hadn't been published yet.
- **Anchor-relative windows.** Entry and exit are indexed off an anchor tied to the
  announcement (a configurable number of trading days before entry and after exit),
  computed on split-adjusted price series, so the trade timing is defined
  consistently across thousands of events without hand-picking dates.

## Backtesting

The backtest runs a portfolio-level simulation rather than averaging isolated trade
returns. Capital carries forward day by day, positions compete for a finite number
of slots, and the simulation models:

- Configurable entry/exit windows relative to the earnings anchor
- Factor filters (minimum composite score, minimum number of factors present,
  quant-rating tiers) to isolate the high-conviction subset
- Position limits and per-day trade caps
- Round-trip transaction costs
- Optional stop-loss exits

Because capital and open positions are path-dependent, the simulation is inherently
sequential; the hot paths were later optimized (pre-grouping trade candidates by date
and by ticker) to keep full-universe runs fast without altering results.

## Validation

Model output isn't taken on faith. An evaluation step compares the locally computed
factor scores and composite ratings against an external reference set of manually
collected labels, producing an alignment report. This is a sanity check that the
factor construction is behaving as intended rather than quietly diverging from a
known baseline.

## Interactive dashboard

A Streamlit dashboard exposes the full strategy surface — entry/exit windows, factor
filters, position limits, transaction costs, and stop-loss — and recomputes the
backtest live, rendering the equity curve, drawdown, trade distribution, and summary
metrics (total return, CAGR, win rate, max drawdown) on each run.

The dashboard is self-hosted on a private server and reachable over a private
Tailscale network rather than exposed publicly; a live link can be shared on request.

## Tech stack

Python · pandas · NumPy · Streamlit · Plotly · Financial Modeling Prep API

## Disclaimer

For educational and research purposes only. Not financial advice. All strategies are
hypothetical, based on historical data, and carry no guarantee of future performance.

## Author

Rehan Ghias — https://github.com/Ghias54
