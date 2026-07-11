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
from newsapi import NewsApiClient
from textblob import TextBlob
from datetime import datetime, timedelta
import google.generativeai as genai

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

#Setup Gemini LLM
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model=genai.GenerativeModel("gemini-2.0-flash")
    llm_available=True
except Exception:
    llm_available=False

def generative_ai_analysis(prompt_text):
    try:
        response=gemini_model.generate_content(prompt_text)
        return response.text
    except Exception as e:
        return f"Analysis Unavailable: {e}"

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
        
        if "Close" in df.columns:
            df_prophet=df.rename(columns={"Date":"ds","Close":"y"})

        elif "Adj Close" in df.columns:
            df_prophet=df.rename(columns={"Date":"ds", "Adj Close":"y"})
        else:
            st.error("⚠️ No Price Column Found")
            st.stop()

        df_prophet = df_prophet[["ds", "y"]].copy()
        df_prophet["ds"]= pd.to_datetime(df_prophet["ds"])
        df_prophet["y"]= pd.to_numeric(df_prophet["y"], errors="coerce")
        df_prophet=df_prophet.dropna()

        #Train Prophet Model
        model = Prophet(daily_seasonality=True) #create model (detects daily patterns)
        model.fit(df_prophet) #train on historical data

        #Make Future Predictions
        future = model.make_future_dataframe(periods=forecast_days)
        forecast=model.predict(future)

        #Chart-1: Interactive forecast with confidence bands
        fig=px.line(forecast, x="ds", y="yhat", title=f"{ticker} - {forecast_days}-Day Forecast")
        fig.add_scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper Bound", line=dict(dash="dot", color="gray")) #upper confidence bound
        fig.add_scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Bound",  line=dict(dash="dot", color="red")) #lower confidence bound
        st.plotly_chart(fig, use_container_width=True) #render the chart

        #Chart-2: Actual vs Forecasted Values
        fig2, ax = plt.subplots(figsize=(10, 5)) #create matplotlib figure
        ax.plot(df_prophet["ds"], df_prophet["y"], label="Actual", color="blue") #actual values
        ax.plot(forecast["ds"], forecast["yhat"], label="Predicted", color="orange") #predicted values
        ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],  alpha=0.2, color="gray") #confidence intervals
        ax.set_title(f"{ticker} — Actual vs Predicted")
        ax.legend()
        st.pyplot(fig2)

        #AI Analysis
        if llm_available:
            st.markdown("---")
            st.subheader("🔮 AI Forecast Analysis")
            with st.spinner("Generating AI insights..."):
                last_price=df_prophet["y"].iloc[-1]
                predicted_price=forecast["yhat"].iloc[-1]
                direction="UP" if predicted_price > last_price else "DOWN"

                prompt= f"""You are a financial analyst. Analyze this stock data briefly: Stock: {ticker} Current Price: {last_price:.2f} 
                {forecast_days}-Day Forecast: {predicted_price:.2f} ({direction})
                Confidence Range: {forecast['yhat_lower'].iloc[-1]:.2f} to{forecast['yhat_upper'].iloc[-1]:.2f}

                Give a 3-4 sentence market outlook. Mention the trend, confidence, and risks.
                Add disclaimer: This is not financial advice."""

                ai_response=generative_ai_analysis(prompt)
                st.markdown(ai_response)

    with tab2:
        st.header("🎯 Predict Direction")

        #Ensure close column exists
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "Close"})
        
        #Engineer Features
        df["Return"] = df["Close"].pct_change() #daily % change in prices
        df["SMA_5"] = df["Close"].rolling(window=5).mean() #5-day moving avg
        df["SMA_10"] = df["Close"].rolling(window=10).mean() #10 day moving avg
        df["Volatility"] = df["Return"].rolling(window=10).std() # 10 day volatility (risk)
        df["Target"] = (df["Return"]>0).astype(int) # 1 = price went UP, 0 = DOWN

        df_clean=df.dropna() #remove rows with NaN (first 10 rows)

        #Split into features(X) and target(Y)
        features=["Return", "SMA_5","SMA_10","Volatility"]
        X = df_clean[features] #input columns
        y = df_clean["Target"] 


        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        #Train RandomForest
        clf=RandomForestClassifier(n_estimators=100, random_state=42) #100 decision trees
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test) #make predictions on test data

        #Show Accuracy
        acc = accuracy_score(y_test, y_pred)
        st.metric("Model Accuracy", f"{acc:.0%}")

        #Show Confusion Matrix
        cm = confusion_matrix(y_test, y_pred) #TP/FP/TN/FN counts
        fig3, ax=plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Down","UP"], yticklabels=["Down","UP"], ax=ax) #blue heatmap with numbers
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{ticker} - Confusion Matrix")
        st.pyplot(fig3)
        
        # Classification Report
        st.subheader("🎯 Classification Report")
        st.text(classification_report(y_test, y_pred)) #precision, recall, f1-score

        #Feature Importance Chart
        st.subheader("Feature Importance")
        importance=pd.DataFrame({
            "Feature": features,
            "Importance":clf.feature_importances_
        }).sort_values("Importance", ascending=False)

        fig4=px.bar(importance, x="Importance", y="Feature", orientation="h", title="What Drives the Prediction?", color="Importance", color_continuous_scale="Blues")
        st.plotly_chart(fig4, use_container_width=True)

        #AI Analysis
        if llm_available:
            st.markdown("---")
            st.subheader("🤖 AI Analysis")
            with st.spinner("Generating AI insights..."):
                top_feature=importance.iloc[0]["Feature"]

                prompt = f"""You are a financial analyst. Analyze this prediction briefly: 
                Stock: {ticker}
                Model Prediction: {'UP' if y_pred[-1]==1 else 'DOWN'}
                Model Accuracy: {acc:.0%}
                Top Feature: {top_feature}
                Features Used: Return, SMA_5, SMA_10, Volatility

                Give a 3-4 sentence analysis. Menton model confidence, what drives the prediction, and limitations.
                Add disclaimer: This is not financial advice."""

                ai_response=generative_ai_analysis(prompt)
                st.markdown(ai_response)
                

    with tab3:
        st.header("📰 News & Sentiment")

        #Get API Keys
        try:
            news_api_key = st.secrets["NEWS_API_KEY"]
        except Exception:
            st.warning("Add NEWS_API_KEY in .streamlit/secrets.toml to enable this feature.")
            st.stop()

        #Fetch News Article
        @st.cache_data(ttl=3600)
        def fetch_news(query):
            newsapi=NewsApiClient(api_key=news_api_key)
            today=datetime.now()
            month_ago=today-timedelta(days=28)
            articles=newsapi.get_everything(
                q=query, #search term (company name)
                from_param=month_ago.strftime("%Y-%m-%d"), #start date
                to=today.strftime("%Y-%m-%d"), #end date
                language="en", #English only
                sort_by="publishedAt", #most recent first
                page_size=30 #top 30 articles
            )
            return articles.get("articles",[])

        #Score sentiment for each headline
        def analyze_sentiment(articles):
            results=[]
            for article in articles:
                title=article.get("title", "")
                if not title or title == "[Removed]":
                    continue
                score=TextBlob(title).sentiment.polarity
                results.append({
                    "Date": article.get("publishedAt", "")[:10],
                    "Headline": title,
                    "Source": article.get("source", {}).get("name", "Unknown"),
                    "Sentiment": round(score, 3)
                })
            return pd.DataFrame(results)

        #Get company name from ticker for search
        company_name=selected_stock.split("(")[0].strip()

        with st.spinner("Fetching News..."):
            articles=fetch_news(company_name)

        if not articles:
            st.warning(f"No news found for {company_name}.")
        else:
            sentiment_df=analyze_sentiment(articles)

            if sentiment_df.empty:
                st.warning("Could not analyze sentiment.")
            else:
                #Avg Sentiment metric
                avg_sentiment=sentiment_df["Sentiment"].mean()
                sentiment_label="Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment< -0.05 else "Neutral"

                col1, col2, col3=st.columns(3)
                col1.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
                col2.metric("Mood", sentiment_label)
                col3.metric("Articles Analyzed", len(sentiment_df))

                #Sentiment timeline chart
                daily_sentiment=sentiment_df.groupby("Date")["Sentiment"].mean().reset_index()
                fig5=px.line(daily_sentiment, x="Date", y="Sentiment", title=f"{company_name} - Daily News Sentiment")
                fig5.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig5, use_container_width=True)

                #Headline table with color
                st.subheader("Recent Headlines")
                def color_sentiment(val):
                    if val>0.05:
                        return "background-color: #1b5e20; color: white"
                    elif val<-0.05:
                        return "background-color: #b71c1c; color: white"
                    return ""

                styled_df = sentiment_df[["Date", "Headline", "Source", "Sentiment"]]
                st.dataframe(
                    styled_df.style.map(color_sentiment, subset=["Sentiment"]),
                    use_container_width=True,
                    height=400
                )
            

with chat_col:
    st.subheader("💬 Ask FinSights-AI")
    st.caption(f"Ask anything about {ticker}")
    st.info("🚧 AI chat will be added in Phase 3")
