import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit App

st.set_page_config(
    page_title="FinSights AI",
    page_icon="📊",
    layout="wide"
)

# Stock list

STOCK_LIST = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Tesla (TSLA)": "TSLA",
    "Meta (META)": "META",
    "Netflix (NFLX)": "NFLX",
    "NVIDIA (NVDA)": "NVDA",
    "Infosys (INFY)": "INFY",
    "TCS (TCS.NS)": "TCS.NS",
    "Reliance (RELIANCE.NS)": "RELIANCE.NS",
    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
    "Wipro (WIPRO.NS)": "WIPRO.NS",
    "ICICI Bank (ICICIBANK.NS)": "ICICIBANK.NS",
    "SBI (SBIN.NS)": "SBIN.NS",
    "Bharti Airtel (BHARTIARTL.NS)": "BHARTIARTL.NS",
    "HUL (HINDUNILVR.NS)": "HINDUNILVR.NS",
    "ITC (ITC.NS)": "ITC.NS",
    "Asian Paints (ASIANPAINT.NS)": "ASIANPAINT.NS",
    "Bajaj Finance (BAJFINANCE.NS)": "BAJFINANCE.NS",
}

# SIDEBAR

st.sidebar.title("📊 FinSights AI")
st.sidebar.markdown("Smart Stock Prediction using AI")

st.sidebar.subheader("🔍 Select Stock")

stock_names = list(STOCK_LIST.keys())
selected_stock = st.sidebar.selectbox(
    "Choose a company",
    options=stock_names,
    index=stock_names.index("Infosys (INFY)")
)

custom_ticker=st.sidebar.text_input(
    "or enter a custom ticker",
    placeholder="e.g., GOOGL, TATAMOTORS.NS"
)

if custom_ticker.strip():
    ticker = custom_ticker.strip().upper()
else:
    ticker = STOCK_LIST[selected_stock]

st.sidebar.markdown("---")

period = st.sidebar.selectbox(
    "📅 Time Period",
    options=["6mo", "1y", "2y", "5y"],
    index=1
)

forecast_days = st.sidebar.slider(
    "📅 Forecast Days",
    min_value=7,
    max_value=90,
    value=30,
    step=7
)

#Data Fetching (With Caching)

@st.cache_data(ttl=300)
def fetch_stock_data(ticker, period):
    data = yf.download(ticker, period=period)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
        data = data.loc[:, ~data.columns.duplicated()]
        
    return data

#Call the function

data = fetch_stock_data(ticker, period)

if data.empty:
    st.error(f"No data found for ticker '{ticker}'. Please check the symbol.")
    st.stop()

df = data.reset_index()

#Main Layout

main_col, chat_col = st.columns([7, 3])

with main_col:
    st.title(f" {ticker} - Stock Analysis")

    tab1, tab2, tab3 = st.tabs([
        "📈 Price Forecast",
        "🎯 Predict Direction",
        "📰 News & Sentiment"
    ])

    # Tab Placeholders

    with tab1:
        st.header("📈 Price Forecast")
        st.info("🚧 Prophet forecasting will be added next (Section 1)")

    with tab2:
            st.header("🎯 Predict Direction")
            st.info("🚧 Randomforest prediction will be added next (Section 2)")

    with tab3:
            st.header("📰 News & Sentiment")
            st.info("🚧 Sentiment Analysis will be added in Phase 1")

with chat_col:
    st.subheader("💬 Ask FinSights-AI")
    st.caption(f"Ask anything about {ticker}")
    st.info("🚧 AI chat will be added in Phase 3")
