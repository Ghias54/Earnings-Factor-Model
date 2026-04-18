# Earnings Factor Model

A quantitative research project designed to analyze and backtest trading strategies around earnings events using multi-factor filtering.

## Overview

This project explores whether earnings-driven volatility can be systematically traded by combining fundamental and quantitative factors. The goal is not just to capture large winners, but to improve consistency by filtering out low-quality setups.

The model evaluates how different factors impact post-earnings price behavior and tests strategies across a broad universe of stocks.

## Objectives

- Analyze stock price reactions to earnings announcements  
- Measure the impact of earnings surprises on short-term returns  
- Develop a multi-factor model to filter high-probability trades  
- Backtest strategies across thousands of historical earnings events  
- Improve risk-adjusted returns through systematic filtering  

## Key Features

- **Earnings Event Analysis**  
  Collects and processes historical earnings data to study pre- and post-earnings price movements  

- **Multi-Factor Model**  
  Incorporates factors such as:
  - Valuation (P/E, P/S)
  - Growth
  - Profitability
  - Momentum
  - Analyst revisions  

- **Backtesting Framework**  
  Simulates trading strategies based on:
  - Entry timing (before/after earnings)
  - Exit windows (e.g., 1–10 days after earnings)
  - Factor-based filtering  

- **Performance Metrics**  
  Evaluates strategies using:
  - Average return  
  - Median return  
  - Win rate  
  - Trade distribution  

## Tech Stack

- Python  
- Pandas  
- Requests  
- CSV-based data storage  

## Data Pipeline

- Pulls historical financial and price data via APIs  
- Cleans and structures datasets for analysis  
- Stores processed data locally for efficient backtesting  
- Avoids repeated API calls by maintaining reusable datasets  

## Strategy Approach

The core strategy focuses on identifying high-probability trades around earnings by:

1. Ranking stocks based on factor scores  
2. Selecting top-performing subsets (e.g., top 5–10%)  
3. Entering positions relative to earnings timing  
4. Holding for a defined period post-earnings  
5. Comparing performance across different filters  

## Example Insights

- Raw earnings strategies can produce inconsistent results  
- Filtering trades using factor models significantly improves win rate  
- Limiting position count can enhance overall performance  
- Volume and liquidity filters help reduce noise  

## Future Improvements

- Integrate additional datasets (insider trading, options data, implied volatility)  
- Expand factor model to include more advanced signals  
- Build a visualization dashboard for strategy comparison  
- Optimize execution logic and portfolio constraints  

## Disclaimer

This project is for educational and research purposes only. It is not financial advice. All strategies presented are hypothetical and based on historical data.

## Author

Rehan Ghias  
GitHub: https://github.com/Ghias54
