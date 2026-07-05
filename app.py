import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Portfolio Risk Dashboard", layout="wide")
st.title("📊 Portfolio Risk Dashboard")

# ---------------- Sidebar ----------------
st.sidebar.header("Portfolio Settings")

available_stocks = [
    "AAPL","AMZN","GOOGL","JNJ","JPM",
    "MSFT","NVDA","PG","XOM"
]

stocks = st.sidebar.multiselect(
    "Select Stocks",
    options=available_stocks,
    default=available_stocks
)

custom = st.sidebar.text_input("Custom Ticker (Optional)")
if custom:
    custom = custom.strip().upper()
    if custom not in stocks:
        stocks.append(custom)

period = st.sidebar.radio(
    "Period",
    ["1mo","3mo","6mo","1y","2y","5y"],
    horizontal=True
)

weighting = st.sidebar.radio(
    "Weighting",
    ["Equal Weight","Custom Weight"]
)

initial = st.sidebar.number_input(
    "Initial Investment ($)",
    min_value=1000,
    value=10000,
    step=1000
)

risk_free = st.sidebar.number_input(
    "Risk Free Rate (%)",
    value=4.0,
    min_value=0.0
)/100

benchmark = st.sidebar.selectbox(
    "Benchmark",
    [
        "^GSPC (S&P500)",
        "^IXIC (NASDAQ)",
        "^DJI (Dow Jones)",
        "None"
    ]
)

if len(stocks)==0:
    st.warning("Select at least one stock.")
    st.stop()

weights=None
if weighting=="Custom Weight":
    st.sidebar.subheader("Portfolio Weights")
    raw={}
    for s in stocks:
        raw[s]=st.sidebar.slider(s,0,100,int(100/len(stocks)))
    total=sum(raw.values())
    if total==0:
        st.sidebar.error("Weights cannot be zero.")
    else:
        weights={k:v/total for k,v in raw.items()}

@st.cache_data(ttl=3600)
def load_prices(tickers,period):
    df=yf.download(tickers,period=period,progress=False,auto_adjust=False)
    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns,pd.MultiIndex):
        if "Adj Close" in df.columns.levels[0]:
            df=df["Adj Close"]
        else:
            df=df["Close"]
    else:
        df=df[["Close"]]
        df.columns=tickers

    return df.dropna(how="all")

prices=load_prices(stocks,period)

if prices.empty:
    st.error("Unable to download data.")
    st.stop()

prices=prices.dropna(axis=1,how="all")
returns=prices.pct_change().dropna()

if weighting=="Custom Weight" and weights:
    w=pd.Series({c:weights.get(c,0) for c in returns.columns})
    portfolio=(returns*w).sum(axis=1)
else:
    portfolio=returns.mean(axis=1)

total_return=(1+portfolio).prod()-1
annual_return=portfolio.mean()*252
annual_vol=portfolio.std()*np.sqrt(252)
sharpe=(annual_return-risk_free)/annual_vol if annual_vol>0 else np.nan
var95=np.percentile(portfolio,5)

cum=(1+portfolio).cumprod()
running=cum.cummax()
drawdown=(cum-running)/running
max_dd=drawdown.min()

value=initial*(1+total_return)

st.subheader("Portfolio Summary")
c1,c2,c3,c4,c5=st.columns(5)
c1.metric("Portfolio Value",f"${value:,.2f}")
c2.metric("Total Return",f"{total_return:.2%}")
c3.metric("Sharpe Ratio",f"{sharpe:.2f}")
c4.metric("VaR (95%)",f"{var95:.2%}")
c5.metric("Max Drawdown",f"{max_dd:.2%}")

if benchmark!="None":
    symbol=benchmark.split()[0]
    bm=load_prices([symbol],period)
    if not bm.empty:
        bm_ret=bm.iloc[:,0].pct_change().dropna()
        compare=pd.DataFrame({
            "Portfolio":(1+portfolio).cumprod(),
            "Benchmark":(1+bm_ret).cumprod()
        }).dropna()
        st.subheader("Portfolio vs Benchmark")

fig = px.line(
    compare,
    x=compare.index,
    y=compare.columns,
    title="Portfolio vs Benchmark"
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Cumulative Return",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Portfolio Allocation")
if weighting=="Custom Weight" and weights:
    alloc=pd.Series(weights)
else:
    alloc=pd.Series(np.repeat(1/len(returns.columns),len(returns.columns)),index=returns.columns)

fig,ax=plt.subplots(figsize=(5,5))
ax.pie(alloc,labels=alloc.index,autopct="%1.1f%%",startangle=90)
st.pyplot(fig)

st.subheader("Stock Prices")

fig = px.line(
    prices,
    x=prices.index,
    y=prices.columns,
    title="Stock Prices"
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price ($)",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Cumulative Portfolio Return")

cum_return = (1 + portfolio).cumprod()

fig = px.line(
    x=cum_return.index,
    y=cum_return.values,
    labels={"x": "Date", "y": "Portfolio Value"},
    title="Cumulative Portfolio Return"
)

fig.update_layout(hovermode="x unified")

st.plotly_chart(fig, use_container_width=True)

st.subheader("Individual Stock Performance")

performance = (1 + returns).cumprod()

fig = px.line(
    performance,
    x=performance.index,
    y=performance.columns,
    title="Individual Stock Performance"
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Growth",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Annualized Volatility")
vol=returns.std()*np.sqrt(252)
st.bar_chart(vol)

st.subheader("Rolling Volatility (30 Days)")

roll = returns.rolling(30).std() * np.sqrt(252)

fig = px.line(
    roll,
    x=roll.index,
    y=roll.columns,
    title="Rolling Volatility"
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Volatility",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Portfolio Drawdown")
st.area_chart(drawdown)

st.subheader("Correlation Matrix")
fig,ax=plt.subplots(figsize=(8,6))
sns.heatmap(returns.corr(),annot=True,cmap="coolwarm",ax=ax)
st.pyplot(fig)

perf=(1+returns).prod()-1
a,b=st.columns(2)
a.metric("Best Performing Stock",perf.idxmax(),f"{perf.max():.2%}")
b.metric("Worst Performing Stock",perf.idxmin(),f"{perf.min():.2%}")

stats=pd.DataFrame({
    "Annual Return":returns.mean()*252,
    "Annual Volatility":returns.std()*np.sqrt(252),
    "Sharpe Ratio":(returns.mean()*252)/(returns.std()*np.sqrt(252))
})
st.subheader("Stock Statistics")
st.dataframe(stats.style.format("{:.2%}",subset=["Annual Return","Annual Volatility"]).format("{:.2f}",subset=["Sharpe Ratio"]))

report=pd.DataFrame({
    "Metric":[
        "Initial Investment","Portfolio Value","Total Return",
        "Annual Return","Annual Volatility","Sharpe Ratio",
        "VaR 95%","Max Drawdown"
    ],
    "Value":[
        f"${initial:,.2f}",
        f"${value:,.2f}",
        f"{total_return:.2%}",
        f"{annual_return:.2%}",
        f"{annual_vol:.2%}",
        f"{sharpe:.2f}",
        f"{var95:.2%}",
        f"{max_dd:.2%}"
    ]
})

st.subheader("Portfolio Report")
st.dataframe(report,hide_index=True)

csv=report.to_csv(index=False).encode()
st.download_button(
    "Download CSV Report",
    csv,
    "portfolio_report.csv",
    "text/csv"
)
