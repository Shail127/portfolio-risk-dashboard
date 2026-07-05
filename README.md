# Portfolio Risk Analysis Dashboard

## About the Project
Portfolio Risk Dashboard is an interactive financial analytics application built using Python and Streamlit. It allows users to analyze the performance and risk of a stock portfolio using historical market data from Yahoo Finance.

The dashboard includes portfolio performance metrics, risk measures, benchmark comparison, and interactive visualizations.

##Features
Select multiple US stocks
Add custom stock tickers
Equal-weight and custom-weight portfolios
Portfolio value tracking
Total and annualized returns
Annualized volatility
Sharpe Ratio
Value at Risk (VaR 95%)
Maximum Drawdown
Portfolio allocation visualization
Benchmark comparison (S&P 500, NASDAQ, Dow Jones)
Rolling volatility analysis
Correlation heatmap
Individual stock performance
Downloadable portfolio report (CSV)

##Dataset

Historical daily stock prices are collected using Yahoo Finance (yfinance).

Default companies included:
Apple (AAPL)
Amazon (AMZN)
Google (GOOGL)
Johnson & Johnson (JNJ)
JPMorgan Chase (JPM)
Microsoft (MSFT)
NVIDIA (NVDA)
Procter & Gamble (PG)
Exxon Mobil (XOM)

##Technologies Used
Python
Streamlit
Pandas
NumPy
Matplotlib
Seaborn
Yahoo Finance API (yfinance)

##Performance Metrics
The dashboard computes:
Portfolio Return
Portfolio Value
Annualized Return
Annualized Volatility
Sharpe Ratio
Value at Risk (VaR 95%)
Maximum Drawdown

##Visualizations
Stock Price Trends
Portfolio Allocation
Portfolio vs Benchmark
Cumulative Portfolio Return
Rolling Volatility
Correlation Matrix
Annualized Volatility
Individual Stock Performance

##Installation
pip install -r requirements.txt

##Run the Project
streamlit run app.py

##Future Improvements
Portfolio Optimization (Maximum Sharpe Ratio)
Efficient Frontier
Monte Carlo Portfolio Simulation
CAPM (Beta & Alpha)
Expected Shortfall (CVaR)
Interactive Plotly Visualizations
