# 📊 Portfolio Risk Dashboard

This project is a portfolio risk analysis dashboard built using Python and Streamlit. It helps users analyze the performance and risk of a stock portfolio using historical stock prices from Yahoo Finance.

The dashboard allows users to compare different stocks, view portfolio performance, calculate risk metrics, and generate a simple portfolio report.

## Features

- Select multiple stocks for portfolio analysis
- Add custom stock tickers
- Equal weight and custom weight portfolio allocation
- Portfolio value calculation
- Total return and annualized return
- Annualized volatility
- Sharpe Ratio
- Value at Risk (VaR)
- Maximum Drawdown
- Portfolio allocation pie chart
- Portfolio vs benchmark comparison
- Rolling volatility
- Correlation matrix
- Download portfolio report as CSV

## Dataset

The stock price data is collected using the **Yahoo Finance API (yfinance)**.

The dashboard includes the following companies by default:

- Apple (AAPL)
- Amazon (AMZN)
- Google (GOOGL)
- Microsoft (MSFT)
- NVIDIA (NVDA)
- JPMorgan Chase (JPM)
- Johnson & Johnson (JNJ)
- Procter & Gamble (PG)
- Exxon Mobil (XOM)

## Tools Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- yfinance


## Dashboard Includes

- Portfolio Summary
- Stock Price Trends
- Portfolio Allocation
- Portfolio vs Benchmark
- Individual Stock Performance
- Annualized Volatility
- Rolling Volatility
- Correlation Heatmap
- Portfolio Report


## How to Run

Install the required libraries

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Project Structure

```
portfolio-risk-dashboard/
│
├── app.py
├── requirements.txt
└── README.md
```


## Future Improvements

Some features I would like to add in the future:

- Portfolio optimization
- Monte Carlo simulation
- Efficient Frontier
- CAPM (Beta and Alpha)
- Plotly interactive charts

## Author

Shail Singh

M.Sc. Statistics

Interested in Data Science, Machine Learning, and Risk Analytics.
