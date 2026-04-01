import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Portfolio Risk Dashboard")

# selecting time period
period = st.radio("Select Period", ["1mo", "3mo", "6mo", "1y"], horizontal=True)

# stock list
stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]

st.write("Loading data...")

data = yf.download(stocks, period=period)

if data.empty:
    st.write("Data not loaded")
else:
    data = data["Close"]

    # calculating returns
    returns = data.pct_change().dropna()

    # portfolio return (equal weight)
    portfolio_return = returns.mean(axis=1)

    # basic metrics
    total_return = (1 + portfolio_return).prod() - 1
    daily_return = portfolio_return.iloc[-1]

    sharpe = (portfolio_return.mean() / portfolio_return.std()) * np.sqrt(252)
    var95 = np.percentile(portfolio_return, 5)

    # assuming initial investment
    initial = 100000
    value = initial * (1 + total_return)

    st.subheader("Portfolio Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Portfolio Value", f"₹{value:,.0f}")
    c2.metric("Total Return", f"{total_return:.2%}")
    c3.metric("Daily Return", f"{daily_return:.2%}")
    c4.metric("VaR (95%)", f"{var95:.2%}")

    st.subheader("Stock Prices")
    st.line_chart(data)

    # volatility
    vol = returns.std() * np.sqrt(252)

    st.subheader("Volatility")
    st.bar_chart(vol)
    # trying small change
    # rolling volatility
    st.subheader("Rolling Volatility (30 days)")
    roll_vol = returns.rolling(30).std() * np.sqrt(252)
    st.line_chart(roll_vol)

    # correlation
    st.subheader("Correlation Matrix")
    corr = returns.corr()

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax)
    st.pyplot(fig)

    # simple report button
    if st.button("Generate Report"):
        st.write("Portfolio Value:", value)
        st.write("Total Return:", total_return)
        st.write("Sharpe Ratio:", sharpe)
