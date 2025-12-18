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

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="Finsights AI", layout="wide")
st.title("📊 Finsights AI – Stock Market Trend Analysis")

# Sidebar
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., INFY, AAPL, TSLA)", "INFY")
period = st.sidebar.selectbox("Select Period", ["6mo", "1y", "2y", "5y"], index=1)
forecast_days = st.sidebar.slider("Forecast Days", min_value=7, max_value=90, value=30, step=7)
mode = st.sidebar.radio("Choose Analysis", ["📈 Prophet Forecasting", "🌲 RandomForest Classification"])

# ----------------------------
# Fetch Data
# ----------------------------
st.write(f"Fetching data for **{ticker}**...")
data = yf.download(ticker, period=period)

if data.empty:
    st.error("No data found for this ticker.")
    st.stop()

df = data.reset_index()

# Debugging: show df before using Prophet/ML
st.write("🔎 Raw Data Sample:", df.head())
st.write("Available Columns:", df.columns.tolist())

# ----------------------------
# Mode 1: Prophet Forecasting
# ----------------------------
if mode == "📈 Prophet Forecasting":
    st.subheader("📈 Prophet Forecasting")

    # Handle column naming safely
    if "Close" in df.columns:
        df_prophet = df.rename(columns={"Date": "ds", "Close": "y"})
    elif "Adj Close" in df.columns:
        df_prophet = df.rename(columns={"Date": "ds", "Adj Close": "y"})
    else:
        st.error("⚠️ No 'Close' or 'Adj Close' column found in Yahoo Finance data!")
        st.stop()

    # Keep only required columns
    df_prophet = df_prophet[["ds", "y"]].copy()

    # Convert datatypes
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"], errors="coerce")
    df_prophet["y"] = pd.to_numeric(df_prophet["y"], errors="coerce")

    # Drop bad rows
    df_prophet = df_prophet.dropna(subset=["ds", "y"]).reset_index(drop=True)

    # Debugging check
    st.write("✅ Cleaned Data for Prophet:", df_prophet.head())
    st.write("Cleaned dtypes:", df_prophet.dtypes)

    # Train Prophet
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(df_prophet)

    # Forecast
    future = prophet_model.make_future_dataframe(periods=forecast_days, freq="D")
    forecast = prophet_model.predict(future)

    # Debugging
    st.write("🔎 Forecast Sample:", forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].head())
    st.write("Forecast shape:", forecast.shape)

    if forecast.shape[0] > 0:
        # Plot with confidence intervals
        fig = px.line(forecast, x="ds", y="yhat", title=f"{ticker} Stock Forecast (Next {forecast_days} Days)")
        fig.add_scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines",
                        name="Upper Bound", line=dict(dash="dot", color="gray"))
        fig.add_scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines",
                        name="Lower Bound", line=dict(dash="dot", color="red"))
        st.plotly_chart(fig, use_container_width=True)

        # Actual vs Predicted
        fig2, ax = plt.subplots(figsize=(10,5))
        ax.plot(df_prophet['ds'], df_prophet['y'], label="Actual", color="blue")
        ax.plot(forecast['ds'], forecast['yhat'], label="Predicted", color="orange")
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                        alpha=0.2, color="gray")
        ax.legend()
        st.pyplot(fig2)
    else:
        st.error("⚠️ Forecast DataFrame is empty. Check input data.")

# ----------------------------
# Mode 2: RandomForest Classification
# ----------------------------
elif mode == "🌲 RandomForest Classification":
    st.subheader("🌲 RandomForest Classification – Predict UP/DOWN")

    # Make sure Close/Adj Close exists
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})

    # Create features
    df["Return"] = df["Close"].pct_change()
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["Volatility"] = df["Return"].rolling(window=10).std()
    df["Target"] = (df["Return"] > 0).astype(int)

    df = df.dropna()

    features = ["Return", "SMA_5", "SMA_10", "Volatility"]
    X = df[features]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.write(f"✅ RandomForest Accuracy: **{acc:.2f}**")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig3, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"], ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig3)

    # Classification Report
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Feature Importance
    st.subheader("🔑 Feature Importance")
    importance = pd.DataFrame({"Feature": features, "Importance": clf.feature_importances_})
    fig4 = px.bar(importance, x="Feature", y="Importance", title="RandomForest Feature Importance")
    st.plotly_chart(fig4, use_container_width=True)
