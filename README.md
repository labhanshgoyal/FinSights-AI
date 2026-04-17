# 📊 FinSights AI — Stock Market Trend Analysis & Forecasting

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Prophet](https://img.shields.io/badge/Prophet-Forecasting-0057B7?style=for-the-badge)
![RandomForest](https://img.shields.io/badge/Random_Forest-Classifier-228B22?style=for-the-badge)
![yFinance](https://img.shields.io/badge/yFinance-Live_Data-6C63FF?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

**An AI-powered stock market analysis platform that combines time-series forecasting (Prophet) with ML-based trend classification (Random Forest) — live data, interactive dashboards, all in a Streamlit app.**

[🚀 Quick Start](#quick-start) · [📁 Project Structure](#project-structure) · [📐 Data Schema](#data-schema) · [🔁 Workflow](#application-workflow) · [📓 Notebook](#notebook)

</div>

---

## 📑 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Data Schema](#data-schema)
- [Feature Engineering](#feature-engineering)
- [Application Workflow](#application-workflow)
- [Analysis Modes](#analysis-modes)
  - [Mode 1 — Prophet Forecasting](#mode-1--prophet-forecasting)
  - [Mode 2 — RandomForest Classification](#mode-2--randomforest-classification)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Technologies Used](#technologies-used)
- [Quick Start](#quick-start)
- [App Usage Guide](#app-usage-guide)
- [Visualizations](#visualizations)
- [Future Work](#future-work)
- [Contributing](#contributing)

---

## Overview

**FinSights AI** is an interactive, AI-powered financial analytics web application built with **Streamlit**. It fetches live stock data using **Yahoo Finance (yFinance)**, and offers two intelligent analysis modes:

1. **📈 Prophet Forecasting** — Time-series forecasting of future stock prices with confidence intervals using Meta's Prophet model.
2. **🌲 Random Forest Classification** — ML-based prediction of whether a stock will move **UP** or **DOWN** on any given day, using engineered technical indicators as features.

Users can analyze any publicly traded stock (NSE/BSE/NYSE/NASDAQ) by entering its ticker symbol directly in the sidebar, with full control over the analysis period and forecast horizon.

---

## Key Features

| Feature | Description |
|---|---|
| 🔴 **Live Data Ingestion** | Real-time OHLCV stock data via `yfinance` |
| 📈 **Prophet Time-Series Forecasting** | Predicts stock price N days into the future with upper/lower bounds |
| 🌲 **Random Forest Classification** | Predicts UP/DOWN movement using technical indicators |
| 📊 **Interactive Plotly Charts** | Zoomable, hoverable forecast and feature importance charts |
| 🎛️ **Streamlit Sidebar Controls** | Ticker input, time period selector, forecast days slider, mode toggle |
| 🔑 **Feature Importance Analysis** | Visualizes which technical indicators drive model predictions |
| 🧪 **Confusion Matrix & Classification Report** | Full model evaluation breakdown |
| 💾 **Saved Model Support** | Pre-trained `random_forest_model.pkl` for instant predictions |
| 📰 **News & Sentiment Ready** | `newsapi-python` and `TextBlob` in requirements for NLP extensions |

---

## Project Structure

```
FinSights-AI/
│
├── app (1).py                    # Main Streamlit application (full pipeline)
├── FinSights.ipynb               # Jupyter Notebook (exploratory analysis & model training)
├── random_forest_model.pkl       # Pre-trained Random Forest model (serialized)
├── requirements.txt              # Python dependencies
├── stock_data.csv                # Primary stock dataset (historical OHLCV)
├── stock_data (1).csv            # Secondary/alternate stock dataset
└── README.md                     # Project documentation
```

### File Roles

| File | Role |
|---|---|
| `app (1).py` | Streamlit web app — UI, data fetching, model training & prediction |
| `FinSights.ipynb` | Notebook for EDA, prototyping, and model development |
| `random_forest_model.pkl` | Serialized trained classifier for direct inference |
| `stock_data.csv` | Offline historical stock dataset for testing without API |
| `requirements.txt` | All pip dependencies for reproducible environment setup |

---

## Data Sources

### Live Data (Primary)
Stock data is fetched in real-time via the **Yahoo Finance API** (`yfinance`):

```python
import yfinance as yf
data = yf.download(ticker, period=period)
# Supported periods: 6mo, 1y, 2y, 5y
# Supported tickers: Any valid Yahoo Finance ticker (INFY, AAPL, TSLA, RELIANCE.NS, etc.)
```

### Offline Data (Fallback)
`stock_data.csv` and `stock_data (1).csv` provide historical OHLCV data for offline use and notebook exploration.

---

## Data Schema

### Raw OHLCV Data (from yFinance / CSV)

```
stock_data.csv
│
├── Date          → DateTime    Index — trading date
├── Open          → Float       Opening price of the session
├── High          → Float       Highest price during the session
├── Low           → Float       Lowest price during the session
├── Close         → Float       Closing price of the session  ← Primary feature
├── Adj Close     → Float       Dividend/split-adjusted closing price
└── Volume        → Integer     Number of shares traded
```

### Column Types Summary

| Column | dtype | Description |
|---|---|---|
| `Date` | `datetime64` | Trading date index |
| `Open` | `float64` | Session open price |
| `High` | `float64` | Session high price |
| `Low` | `float64` | Session low price |
| `Close` | `float64` | Session close price ← used in models |
| `Adj Close` | `float64` | Adjusted close (fallback for Close) |
| `Volume` | `int64` | Number of shares traded |

### Engineered Features (for Random Forest)

```
Derived Feature Columns
│
├── Return        → Float    Daily percentage change in Close price
│                            formula: Close.pct_change()
│
├── SMA_5         → Float    5-day Simple Moving Average of Close
│                            formula: Close.rolling(window=5).mean()
│
├── SMA_10        → Float    10-day Simple Moving Average of Close
│                            formula: Close.rolling(window=10).mean()
│
├── Volatility    → Float    10-day rolling standard deviation of Return
│                            formula: Return.rolling(window=10).std()
│
└── Target        → Binary   1 = Price went UP (Return > 0)
                             0 = Price went DOWN or flat (Return ≤ 0)
```

### Prophet Input Schema

```
Prophet DataFrame
│
├── ds            → DateTime    Date column (renamed from "Date")
└── y             → Float       Target value (renamed from "Close" or "Adj Close")
```

---

## Feature Engineering

The Random Forest model is trained on four engineered technical indicators derived from raw price data:

| Feature | Window | Formula | Signal Type |
|---|---|---|---|
| `Return` | 1-day | `Close.pct_change()` | Momentum |
| `SMA_5` | 5-day | `Close.rolling(5).mean()` | Short-term Trend |
| `SMA_10` | 10-day | `Close.rolling(10).mean()` | Medium-term Trend |
| `Volatility` | 10-day | `Return.rolling(10).std()` | Risk / Uncertainty |

**Target Variable:** `Target = 1` if next-day return is positive (`Close` goes UP), else `0`.

---

## Application Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                      USER — STREAMLIT SIDEBAR                        │
│                                                                       │
│   Ticker Input    →   Period Selector   →   Forecast Slider          │
│   (e.g. INFY)         (6mo/1y/2y/5y)        (7–90 days)             │
│                                                                       │
│                    Mode: Prophet  OR  RandomForest                   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  LIVE DATA FETCHING  (yfinance)                      │
│                                                                       │
│   yf.download(ticker, period) → raw OHLCV DataFrame                 │
│   Validate: Close column present? (fallback to Adj Close)           │
│   Show: raw data sample + column list                                │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
               ┌────────────────┴────────────────┐
               │                                 │
               ▼                                 ▼
┌──────────────────────────┐       ┌──────────────────────────────────┐
│  MODE 1: PROPHET         │       │  MODE 2: RANDOM FOREST           │
│  FORECASTING             │       │  CLASSIFICATION                  │
│                          │       │                                  │
│  1. Rename cols → ds, y  │       │  1. Compute Return               │
│  2. Cast dtypes          │       │  2. Compute SMA_5, SMA_10        │
│  3. Drop NaN rows        │       │  3. Compute Volatility           │
│  4. Fit Prophet model    │       │  4. Assign binary Target         │
│  5. Make future df       │       │  5. Drop NaN rows                │
│     (N forecast days)    │       │  6. Train/Test split (80/20)     │
│  6. Predict → forecast   │       │  7. Fit RandomForest(100 trees)  │
│  7. Plot: forecast line  │       │  8. Predict on test set          │
│     + CI upper/lower     │       │  9. Accuracy score               │
│  8. Plot: actual vs      │       │  10. Confusion Matrix heatmap    │
│     predicted overlay    │       │  11. Classification Report       │
│                          │       │  12. Feature Importance chart    │
└──────────────────────────┘       └──────────────────────────────────┘
               │                                 │
               └────────────────┬────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│               INTERACTIVE VISUALIZATIONS                             │
│                                                                       │
│  Prophet:  Plotly forecast chart  +  Matplotlib actual vs predicted │
│  RF:       Seaborn confusion matrix  +  Plotly feature importance   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Analysis Modes

### Mode 1 — 📈 Prophet Forecasting

Meta's **Prophet** is an additive time-series forecasting model designed for daily business data with strong seasonality patterns.

**Internal flow:**

```
Historical Close Prices
         │
         ▼
   Prophet Model
   ├── Trend Component         (long-term price direction)
   ├── Weekly Seasonality      (day-of-week effects)
   ├── Daily Seasonality       (intraday patterns, if enabled)
   └── Holidays (optional)
         │
         ▼
   Forecast Output
   ├── yhat           → Best-estimate predicted price
   ├── yhat_upper     → Upper 80% confidence bound  (dotted gray)
   └── yhat_lower     → Lower 80% confidence bound  (dotted red)
```

**Key charts produced:**
- Plotly interactive forecast line + confidence interval ribbon
- Matplotlib actual price vs predicted price with shaded band

---

### Mode 2 — 🌲 RandomForest Classification

A **Random Forest Classifier** trained on engineered technical indicators to predict the **direction** (UP or DOWN) of the next trading day's close price.

**Internal flow:**

```
Raw OHLCV DataFrame
         │
         ▼
Feature Engineering
   Return = Close.pct_change()
   SMA_5  = Close.rolling(5).mean()
   SMA_10 = Close.rolling(10).mean()
   Volatility = Return.rolling(10).std()
         │
         ▼
Target Creation
   Target = 1 if Return > 0 else 0
         │
         ▼
Train/Test Split  (80% / 20%, time-ordered — no shuffle)
         │
         ▼
RandomForestClassifier(n_estimators=100, random_state=42)
         │
         ▼
Evaluation: Accuracy · Confusion Matrix · Classification Report
         │
         ▼
Feature Importance Visualization (Plotly bar chart)
```

---

## Model Architecture

### Prophet

| Parameter | Value |
|---|---|
| Model Type | Additive Time-Series |
| Daily Seasonality | Enabled |
| Forecast Horizon | 7–90 days (user-selected) |
| Input Format | `ds` (date), `y` (close price) |
| Output | `yhat`, `yhat_lower`, `yhat_upper` |

### Random Forest Classifier

| Parameter | Value |
|---|---|
| Algorithm | Random Forest (Bagging Ensemble) |
| Number of Trees | `n_estimators=100` |
| Random Seed | `random_state=42` |
| Train/Test Split | 80% / 20% (chronological) |
| Input Features | `Return`, `SMA_5`, `SMA_10`, `Volatility` |
| Target | `1` (UP) / `0` (DOWN) |
| Saved Model File | `random_forest_model.pkl` |

---

## Evaluation Metrics

### Random Forest

| Metric | Description |
|---|---|
| **Accuracy** | Overall % of correct UP/DOWN predictions |
| **Precision** | Of all predicted UPs — how many were truly UP |
| **Recall** | Of all actual UPs — how many were correctly caught |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **Confusion Matrix** | TP / FP / TN / FN heatmap |

### Confusion Matrix

```
                  Predicted DOWN    Predicted UP
Actual DOWN     [ True Negative  |  False Positive ]
Actual UP       [ False Negative |  True Positive  ]
```

### Feature Importance (Approximate Ranking)

```
Feature Importance — Random Forest
────────────────────────────────────────────
1. Volatility     ████████████████████  (highest — captures risk/uncertainty)
2. Return         ████████████████
3. SMA_5          ████████████
4. SMA_10         ██████████           (lowest — smoother, less reactive signal)
```

---

## Technologies Used

| Library | Version | Purpose |
|---|---|---|
| `streamlit` | latest | Web app framework & interactive UI |
| `yfinance` | latest | Live stock price data from Yahoo Finance |
| `prophet` | latest | Time-series forecasting (Meta/Facebook) |
| `scikit-learn` | ≥1.0 | RandomForest, train/test split, metrics |
| `plotly` | latest | Interactive charts and visualizations |
| `pandas` | ≥1.3 | Data wrangling and feature engineering |
| `numpy` | ≥1.21 | Numerical computations |
| `matplotlib` | ≥3.4 | Static chart rendering |
| `seaborn` | ≥0.11 | Confusion matrix heatmap |
| `ta` | latest | Extended technical analysis indicators |
| `newsapi-python` | latest | Financial news fetching (NLP pipeline) |
| `textblob` | latest | Sentiment analysis on news headlines |
| `flask` | latest | REST API backend (extension) |
| `pyngrok` | latest | Public tunneling for Colab deployment |

---

## Quick Start

### Prerequisites

- Python 3.8+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/labhanshgoyal/FinSights-AI.git
cd FinSights-AI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Prophet Note:** If Prophet installation fails, try:
> ```bash
> conda install -c conda-forge prophet
> # or
> pip install pystan==2.19.1.1 && pip install prophet
> ```

### 3. Run the Streamlit App

```bash
streamlit run "app (1).py"
```

App opens at → `http://localhost:8501`

### 4. (Optional) Explore the Notebook

```bash
jupyter notebook FinSights.ipynb
```

---

## App Usage Guide

Use the **left sidebar** to configure your analysis session:

```
┌────────────────────────────┐
│        SIDEBAR             │
│                            │
│  Stock Ticker              │
│  ┌──────────────────────┐  │
│  │  INFY  (default)     │  │
│  └──────────────────────┘  │
│  Try: AAPL · TSLA · MSFT  │
│       RELIANCE.NS          │
│       HDFCBANK.NS          │
│                            │
│  Period                    │
│  [ 6mo | 1y | 2y | 5y ]   │
│                            │
│  Forecast Days             │
│  ──●──────────── (7–90)    │
│                            │
│  Choose Analysis           │
│  ○ 📈 Prophet Forecasting  │
│  ● 🌲 RandomForest         │
└────────────────────────────┘
```

| Control | Options | Default |
|---|---|---|
| Stock Ticker | Any valid Yahoo Finance symbol | `INFY` |
| Period | `6mo` / `1y` / `2y` / `5y` | `1y` |
| Forecast Days | `7` to `90` (step 7) | `30` |
| Mode | Prophet / RandomForest | — |

---

## Visualizations

### Prophet Mode

| Chart | Library | Description |
|---|---|---|
| **Forecast Line + CI Ribbon** | Plotly | Future price with upper (gray) and lower (red) confidence dotted bounds |
| **Actual vs Predicted Overlay** | Matplotlib | Historical actuals in blue, predictions in orange, shaded confidence band |

### RandomForest Mode

| Chart | Library | Description |
|---|---|---|
| **Confusion Matrix Heatmap** | Seaborn | Color-coded TP/FP/TN/FN grid with UP/DOWN axis labels |
| **Feature Importance Bar Chart** | Plotly | Ranked horizontal bar chart of all 4 input features |

---

## Notebook

`FinSights.ipynb` covers the full development workflow:

```
Step 1  → Import libraries and configure environment
Step 2  → Load offline stock CSV (stock_data.csv)
Step 3  → Exploratory Data Analysis (price trends, volume, returns)
Step 4  → Feature engineering (Return, SMA_5, SMA_10, Volatility, Target)
Step 5  → Prophet model: fit, forecast, visualize
Step 6  → Random Forest: train, evaluate, confusion matrix
Step 7  → Export trained model → random_forest_model.pkl
Step 8  → Final visualizations and insights
```

---

## Future Work

- [ ] **Sentiment Analysis** — Integrate `newsapi-python` + `TextBlob` for news-driven buy/sell signals
- [ ] **Extended Indicators** — Add RSI, MACD, Bollinger Bands via the `ta` library as RF features
- [ ] **LSTM / Transformer** — Replace Prophet with a deep learning time-series model for higher accuracy
- [ ] **Multi-Ticker Comparison** — Compare multiple tickers on the same dashboard
- [ ] **Portfolio Tracker** — Aggregate analysis across a user-defined portfolio
- [ ] **Backtesting Engine** — Simulate strategy P&L on historical data using model predictions
- [ ] **Deploy on Streamlit Cloud** — One-click public deployment via `streamlit.io`
- [ ] **SHAP Explainability** — Per-prediction SHAP waterfall plots for transparent ML decisions
- [ ] **Price Alerts** — Email/Slack notifications triggered by model-detected signals

---

## Contributing

Contributions are welcome! Here's how:

```bash
# 1. Fork the repository on GitHub
# 2. Create your branch
git checkout -b feature/your-feature-name

# 3. Commit your changes
git commit -m "feat: add your feature description"

# 4. Push to your fork
git push origin feature/your-feature-name

# 5. Open a Pull Request against main
```

Please ensure any new features are accompanied by clear inline comments.

---

## Author

**Labhansh Goyal**

[![GitHub](https://img.shields.io/badge/GitHub-labhanshgoyal-181717?style=flat&logo=github)](https://github.com/labhanshgoyal)

---

> ⚠️ **Disclaimer:** FinSights AI is built for **educational and research purposes only**. Stock market predictions are inherently uncertain and past performance does not guarantee future results. Nothing in this project constitutes financial advice. Always consult a qualified financial advisor before making investment decisions.

---

<div align="center">

*If you found this project useful, consider giving it a ⭐ on GitHub!*

</div>
