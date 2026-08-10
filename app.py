import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
import plotly.express as px
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from newsapi import NewsApiClient
from textblob import TextBlob
from datetime import datetime, timedelta
from groq import Groq
from auth import sign_in, sign_up, get_oauth_url, logout, get_current_user
from database import save_chat, get_chat_history, log_query, get_watchlist, add_to_watchlist, remove_from_watchlist
import time

st.set_page_config(
    page_title="FinSights AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

from styles import apply_theme, render_stock_header

try:
    from agents import run_crew
    crew_available = True
except Exception:
    crew_available = False

try:
    from validator import validate_response
    validator_available = True
except Exception as e:
    validator_available = False

apply_theme()

def render_login_page():
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; min-height: 80vh;">
        <div style="background: rgba(255,255,255,0.03); backdrop-filter: blur(20px); border: 1px solid rgba(99,102,241,0.2); border-radius: 24px; padding: 3rem; max-width: 450px; width: 100%;">
            <div style="text-align: center; margin-bottom: 2rem;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">📊</div>
                <div style="font-size: 1.8rem; font-weight: 800; background: linear-gradient(135deg, #818cf8, #6366f1, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">FinSights AI</div>
                <div style="color: #64748b; font-size: 0.9rem; margin-top: 0.3rem;">Smart Stock Analysis powered by AI</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        auth_mode = st.radio("", ["Sign In", "Create Account"], horizontal=True, label_visibility="collapsed")

        if auth_mode == "Sign In":
            email = st.text_input("Email", placeholder="you@example.com", key="login_email")
            password = st.text_input("Password", type="password", placeholder="••••••••", key="login_pass")

            if st.button("Sign In", use_container_width=True, key="signin_btn"):
                if email and password:
                    result = sign_in(email, password)
                    if result["success"]:
                        st.session_state.user = result["user"]
                        st.session_state.user_id = result["user"].id
                        st.session_state.user_email = result["user"].email
                        st.rerun()
                    else:
                        st.error(f"Login failed: {result['error']}")
                else:
                    st.warning("Please enter email and password.")

        else:
            email = st.text_input("Email", placeholder="you@example.com", key="signup_email")
            password = st.text_input("Password", type="password", placeholder="Min 6 characters", key="signup_pass")
            confirm = st.text_input("Confirm Password", type="password", placeholder="••••••••", key="signup_confirm")

            if st.button("Create Account", use_container_width=True, key="signup_btn"):
                if not email or not password:
                    st.warning("Please fill all fields.")
                elif password != confirm:
                    st.error("Passwords don't match.")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    result = sign_up(email, password)
                    if result["success"]:
                        st.success("Account created! Check your email to verify, then sign in.")
                    else:
                        st.error(f"Signup failed: {result['error']}")

        st.markdown("---")
        st.markdown("<div style='text-align:center; color:#64748b; font-size:0.8rem;'>Or continue with</div>", unsafe_allow_html=True)

        oauth_col1, oauth_col2 = st.columns(2)
        with oauth_col1:
            google_url = get_oauth_url("google")
            if google_url:
                st.link_button("🔵 Google", google_url, use_container_width=True)
        with oauth_col2:
            github_url = get_oauth_url("github")
            if github_url:
                st.link_button("⚫ GitHub", github_url, use_container_width=True)

# ─── Auth Gate ───
user = get_current_user()
is_logged_in = user is not None

STOCK_LIST = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Tesla (TSLA)": "TSLA",
    "Meta (META)": "META",
    "Netflix (NFLX)": "NFLX",
    "NVIDIA (NVDA)": "NVDA",
    "Infosys (INFY.NS)": "INFY.NS",
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

st.sidebar.title("📊 FinSights AI")
st.sidebar.markdown("Smart Stock Prediction using AI")

# ─── Top-right Auth Bar ───
auth_left, auth_right = st.columns([8, 2])
with auth_right:
    if is_logged_in:
        with st.popover("👤 " + st.session_state.user_email.split("@")[0]):
            st.markdown(f"**{st.session_state.user_email}**")
            st.markdown("---")
            if st.button("🚪 Sign Out", use_container_width=True, key="top_logout"):
                logout()
                st.rerun()
    else:
        with st.popover("🔐 Sign In"):
            auth_mode = st.radio("Mode", ["Sign In", "Create Account"], horizontal=True, key="top_auth_mode")

            if auth_mode == "Sign In":
                email = st.text_input("Email", placeholder="you@example.com", key="top_email")
                password = st.text_input("Password", type="password", placeholder="••••••••", key="top_pass")
                if st.button("Sign In", use_container_width=True, key="top_signin"):
                    if email and password:
                        result = sign_in(email, password)
                        if result["success"]:
                            st.session_state.user = result["user"]
                            st.session_state.user_id = result["user"].id
                            st.session_state.user_email = result["user"].email
                            st.rerun()
                        else:
                            st.error(result["error"])
                    else:
                        st.warning("Enter email and password.")
            else:
                email = st.text_input("Email", placeholder="you@example.com", key="top_signup_email")
                password = st.text_input("Password", type="password", placeholder="Min 6 chars", key="top_signup_pass")
                confirm = st.text_input("Confirm Password", type="password", key="top_signup_confirm")
                if st.button("Create Account", use_container_width=True, key="top_signup"):
                    if not email or not password:
                        st.warning("Fill all fields.")
                    elif password != confirm:
                        st.error("Passwords don't match.")
                    elif len(password) < 6:
                        st.error("Min 6 characters.")
                    else:
                        result = sign_up(email, password)
                        if result["success"]:
                            st.success("Account created! Check email, then sign in.")
                        else:
                            st.error(result["error"])

            st.markdown("---")
            st.markdown("<div style='text-align:center; color:#64748b; font-size:0.8rem;'>Or continue with</div>", unsafe_allow_html=True)
            oauth_c1, oauth_c2 = st.columns(2)
            with oauth_c1:
                google_url = get_oauth_url("google")
                if google_url:
                    st.link_button("Google", google_url, use_container_width=True)
            with oauth_c2:
                github_url = get_oauth_url("github")
                if github_url:
                    st.link_button("GitHub", github_url, use_container_width=True)


st.sidebar.subheader("🔍 Select Stock")

stock_names = list(STOCK_LIST.keys())
selected_stock = st.sidebar.selectbox(
    "Choose a company",
    options=stock_names,
    index=stock_names.index("Infosys (INFY.NS)")
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

#Fetch stock info (company name)
@st.cache_data(ttl=3600)
def get_stock_info(symbol):
    try:
        info=yf.Ticker(symbol).info
        return info.get("shortName", symbol)
    except Exception:
        return symbol

stock_name=get_stock_info(ticker)

#Calculate price metrics
current_price=df["Close"].dropna().iloc[-1]
prev_price=df["Close"].dropna().iloc[-2] if len(df)>1 else current_price
price_change=current_price - prev_price
pct_change=(price_change / prev_price) * 100

#Setup Groq LLM
try:
    groq_client=Groq(api_key=st.secrets["GROQ_API_KEY"])
    llm_available=True
except Exception:
    llm_available=False

@st.cache_data(ttl=600)
def generative_ai_analysis(prompt_text):
    try:
        response=groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Analysis Unavailable: {e}"

main_col, chat_col = st.columns([7, 3])

with main_col:
    #Stock Header
    render_stock_header(ticker, stock_name, current_price, price_change, pct_change)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Price Forecast",
        "🎯 Predict Direction",
        "📰 News & Sentiment",
        "🤖 Agent Workflows"
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
        st.plotly_chart(fig, width='stretch') #render the chart

        #Chart-2: Actual vs Forecasted Values
        fig2, ax = plt.subplots(figsize=(10, 5)) #create matplotlib figure
        ax.plot(df_prophet["ds"], df_prophet["y"], label="Actual", color="blue") #actual values
        ax.plot(forecast["ds"], forecast["yhat"], label="Predicted", color="orange") #predicted values
        ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],  alpha=0.2, color="gray") #confidence intervals
        ax.set_title(f"{ticker} — Actual vs Predicted")
        ax.legend()
        st.pyplot(fig2)

        #AI Analysis
        if not is_logged_in:
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px solid rgba(99,102,241,0.15);">
                <div style="font-size: 2rem;">🔒</div>
                <div style="font-weight: 600; color: #f1f5f9; margin: 0.5rem 0;">AI Forecast Analysis</div>
                <div style="color: #64748b; font-size: 0.85rem;">Sign in to unlock AI-powered insights</div>
            </div>
            """, unsafe_allow_html=True)
        elif llm_available:
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
        df["SMA_20"] = df["Close"].rolling(window=20).mean() # 20 day moving avg
        df["Volatility"] = df["Return"].rolling(window=10).std() # 10 day volatility (risk)
        df["RSI"] = 100-(100/(1+(df["Close"].diff().clip(lower=0).rolling(14).mean()/df["Close"].diff().clip(upper=0).abs().rolling(14).mean())))
        df["Target"] = (df["Return"].shift(-1)>0).astype(int) # 1 = price went UP, 0 = DOWN

        df_clean=df.dropna() #remove rows with NaN (first 10 rows)

        #Split into features(X) and target(Y)
        features=["Return", "SMA_5","SMA_10","SMA_20","Volatility","RSI"]
        X = df_clean[features] #input columns
        y = df_clean["Target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        #Train XGBoost
        clf=XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="logloss") #100 decision trees
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
        st.plotly_chart(fig4, width='stretch')

        #AI Analysis
        if not is_logged_in:
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px solid rgba(99,102,241,0.15);">
                <div style="font-size: 2rem;">🔒</div>
                <div style="font-weight: 600; color: #f1f5f9; margin: 0.5rem 0;">AI Prediction Analysis</div>
                <div style="color: #64748b; font-size: 0.85rem;">Sign in to unlock AI-powered insights</div>
            </div>
            """, unsafe_allow_html=True)
        elif llm_available:
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

                Give a 3-4 sentence analysis. Mention model confidence, what drives the prediction, and limitations.
                Add disclaimer: This is not financial advice."""

                ai_response=generative_ai_analysis(prompt)
                st.markdown(ai_response)
                

    with tab3:
        if not is_logged_in:
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🔒</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #f1f5f9;">Sign in to unlock News & Sentiment</div>
                <div style="color: #64748b; margin-top: 0.5rem;">Get real-time news analysis and sentiment scoring</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.header("📰 News & Sentiment")

            try:
                news_api_key = st.secrets["NEWS_API_KEY"]
            except Exception:
                st.warning("Add NEWS_API_KEY in .streamlit/secrets.toml to enable this feature.")
                st.stop()

            @st.cache_data(ttl=3600)
            def fetch_news(query):
                newsapi=NewsApiClient(api_key=news_api_key)
                today=datetime.now()
                month_ago=today-timedelta(days=28)
                articles=newsapi.get_everything(
                    q=query,
                    from_param=month_ago.strftime("%Y-%m-%d"),
                    to=today.strftime("%Y-%m-%d"),
                    language="en",
                    sort_by="publishedAt",
                    page_size=30
                )
                return articles.get("articles",[])

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
                    avg_sentiment=sentiment_df["Sentiment"].mean()
                    sentiment_label="Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment< -0.05 else "Neutral"

                    col1, col2, col3=st.columns(3)
                    col1.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
                    col2.metric("Mood", sentiment_label)
                    col3.metric("Articles Analyzed", len(sentiment_df))

                    daily_sentiment=sentiment_df.groupby("Date")["Sentiment"].mean().reset_index()
                    fig5=px.line(daily_sentiment, x="Date", y="Sentiment", title=f"{company_name} - Daily News Sentiment")
                    fig5.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig5, width='stretch')

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
                        width='stretch',
                        height=400
                    )
            
    with tab4:
        if not is_logged_in:
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🔒</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #f1f5f9;">Sign in to unlock Agent Workflow</div>
                <div style="color: #64748b; margin-top: 0.5rem;">See how our AI agents collaborate to analyze stocks</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.header("🤖 Agent Workflow")
            st.markdown("How our multi-agent system processes your queries:")

            # Visual pipeline
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin: 2rem 0;">
                <div style="background: linear-gradient(135deg, #1e3a5f, #2d5a8e); padding: 20px; border-radius: 12px; flex: 1; min-width: 200px; text-align: center;">
                    <div style="font-size: 2rem;">🔍</div>
                    <div style="font-weight: 700; color: #60a5fa; margin: 8px 0;">Researcher</div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">Gathers stock metrics, price data, SMA, volatility, RSI</div>
                </div>
                <div style="font-size: 1.5rem; color: #475569;">→</div>
            <div style="background: linear-gradient(135deg, #1e3a5f, #2d5a8e); padding: 20px; border-radius: 12px; flex: 1; min-width: 200px; text-align: center;">
                <div style="font-size: 2rem;">📰</div>
                <div style="font-weight: 700; color: #60a5fa; margin: 8px 0;">News Analyst</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">Analyzes headlines, sentiment scores, market mood</div>
            </div>
            <div style="font-size: 1.5rem; color: #475569;">→</div>
            <div style="background: linear-gradient(135deg, #1e3a5f, #2d5a8e); padding: 20px; border-radius: 12px; flex: 1; min-width: 200px; text-align: center;">
                <div style="font-size: 2rem;">🧠</div>
                <div style="font-weight: 700; color: #60a5fa; margin: 8px 0;">Strategist</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">Synthesizes everything into actionable market analysis</div>
            </div>
            <div style="font-size: 1.5rem; color: #475569;">→</div>
                <div style="background: linear-gradient(135deg, #14532d, #166534); padding: 20px; border-radius: 12px; flex: 1; min-width: 200px; text-align: center;">
                    <div style="font-size: 2rem;">✅</div>
                    <div style="font-weight: 700; color: #4ade80; margin: 8px 0;">Validator</div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">Scores quality (0-1.0), retries if below 0.7</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # System status
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Agents Active", "3" if crew_available else "Fallback")
            with col2:
                st.metric("Validator", "✅ Active" if validator_available else "⚠️ Unavailable")
            with col3:
                st.metric("LLM Provider", "Groq (LLaMA-3.3-70B)")

            # Architecture details
            with st.expander("🔧 Technical Architecture"):
                st.markdown("""
                **Multi-Agent Pipeline (CrewAI)**
                - 3 specialized agents run sequentially via Groq API
                - Each agent has a unique role, goal, and backstory
                - Output flows: Researcher → News Analyst → Strategist

                **Self-Correction Loop (LLM-as-Judge)**
                - Validator scores every response on 3 criteria: Relevancy, Specificity, Completeness
                - Threshold: 0.7/1.0 — below this triggers a retry
                - Max 2 retries with feedback injected into the prompt
                - Ensures consistent, high-quality financial analysis

                **Fallback System**
                - If CrewAI fails to load → direct Groq API call with system prompt
                - If Validator fails → response shown without scoring
                - App never crashes — graceful degradation at every layer
                """)

with chat_col:
    #Chat Header
    st.markdown(f"""
    <div class="chat-header">
        <div class="chat-header-title">💬 Ask FinSights-AI</div>
        <div class="chat-header-sub"><span class="pulse-dot"></span> Analyzing {ticker} · {stock_name}</div>
    </div>
    """, unsafe_allow_html=True)

    if not is_logged_in:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem; color: #64748b;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🔒</div>
            <div style="font-size: 0.9rem; font-weight: 500;">Sign in to chat with FinSights AI</div>
            <div style="font-size: 0.75rem; margin-top: 0.3rem; color: #475569;">Get personalized stock analysis</div>
        </div>
        """, unsafe_allow_html=True)
    elif not llm_available:
        st.warning("Add GROQ_API_KEY to enable chat.")
    else:
        
        #Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages=[]

        #Scrollable chat container (fixed height)
        chat_container=st.container(height=500)

        with chat_container:
            if not st.session_state.messages:
                st.markdown("""
                <div style="text-align: center; padding: 3rem 1rem; color: #64748b;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🤖</div>
                    <div style="font-size: 0.9rem; font-weight: 500;">Ask me anything about {}</div>
                    <div style="font-size: 0.75rem; margin-top: 0.3rem; color: #475569;">Forecasts · Predictions · News · Trends</div>
                </div>
                """.format(ticker), unsafe_allow_html=True)
            else:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

        #Clear history button
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat", key="clear_chat"):
                st.session_state.messages=[]
                st.rerun()

        #Chat input
        if prompt := st.chat_input("Ask about forecast, news, trends..."):

            st.session_state.messages.append({"role": "user", "content": prompt})

            #Build RAG context (live data into prompt)
            latest_price=df["Close"].dropna().iloc[-1]
            if pd.isna(latest_price):
                latest_price=current_price
            
            #AI Response
            try:
                stock_context = f"""Stock: {ticker} ({stock_name})
                Price: ${latest_price:.2f} | Period: {period}
                Change: {price_change:.2f} ({pct_change:.2f}%)"""

                max_retries = 2
                for attempt in range(max_retries + 1):
                    if crew_available:
                        full_response = run_crew(prompt, stock_context)
                    else:
                        response = groq_client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {"role": "system", "content": "You are FinSights AI, a professional financial analyst. Analyze stocks using the data provided. Give specific, data-backed insights. Never refuse to analyze."},
                                {"role": "user", "content": f"{stock_context}\n\nQuestion: {prompt}"}
                            ],
                            temperature=0.7,
                            max_tokens=500
                        )
                        full_response = response.choices[0].message.content

                    if validator_available and attempt < max_retries:
                        validation = validate_response(prompt, full_response)
                        if validation["pass"]:
                            full_response += f"\n\n*✅ Quality Score: {validation['score']}/1.0*"
                            break
                        prompt = f"{prompt}\n\nPrevious answer was weak ({validation['feedback']}). Be more specific and data-driven."
                    else:
                        break
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Chat unavailable: {e}"})

            st.rerun()