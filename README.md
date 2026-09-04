# 📈 QuantAI - Machine Learning Stock Analysis & Prediction Platform

An end-to-end Machine Learning system for stock market analysis, technical indicator feature engineering, multi-model ensemble forecasting, algorithmic backtesting simulation, and real-time interactive web visualization.

---

## 🌟 Key Features

1. **Automated Data Processing & Market Ingestion** (`ml_engine/data_fetcher.py`)
   - Fetches historical daily stock data via `yfinance` API.
   - Robust offline mode with high-precision Geometric Brownian Motion market data generator for seamless testing without rate limits.

2. **Advanced Feature Engineering & Technical Analysis** (`ml_engine/feature_engineering.py`)
   - **Trend Indicators**: SMA (10, 20, 50, 200), EMA (12, 26)
   - **Oscillators & Momentum**: Relative Strength Index (RSI 14), MACD (Line, Signal, Histogram), Stochastic Oscillator (%K, %D), Rate of Change (ROC 10)
   - **Volatility & Bands**: Bollinger Bands (Upper, Lower, Width, %B), Average True Range (ATR 14)
   - **Volume Dynamics**: On-Balance Volume (OBV), Volume SMAs, Volume Spikes
   - **Log Return & Volatility Lags**: Multi-period log returns and 20-day rolling annualized volatility.

3. **Multi-Model Machine Learning Suite** (`ml_engine/models.py`)
   - **Random Forest Regressor**: 150 decision trees for non-linear feature interactions and feature importance extraction.
   - **Gradient Boosting Regressor**: Optimized gradient boosting trees for sequential return optimization.
   - **Ridge Regression**: Regularized linear baseline model.
   - **Multi-Layer Perceptron (MLP Neural Net)**: Sequence-aware neural network for non-linear temporal dynamics.
   - **Ensemble Predictor**: Weighted combination of top models for robust multi-day forecasting.
   - **Evaluation Breakdown**: RMSE, MAE, R² score, and Directional Accuracy (%).

4. **Algorithmic Trading Backtesting Engine** (`ml_engine/backtester.py`)
   - Simulates trade execution driven by ML model signal confidence.
   - Calculates **Strategy Total ROI (%)**, **Buy & Hold ROI (%)**, **Sharpe Ratio**, **Sortino Ratio**, **Max Drawdown (%)**, **Win Rate (%)**, and **Profit Factor**.

5. **Modern Glassmorphism Web Dashboard** (`server.py` + `public/`)
   - Built with Vanilla CSS, Chart.js, and HTTP REST API.
   - Dark theme styling with glowing status badges, interactive candlestick/line charts, technical indicator overlays, RSI/MACD sub-charts, feature importance ranking, multi-day prediction tables with 90% confidence bounds, and equity curve comparison.

---

## 🚀 Quick Start Guide

### Prerequisites
- Node.js (for web dashboard)
- Python 3.10+ (managed automatically via `uv`)

### 1. Installation

Using `uv` (recommended):
```bash
# Clone or navigate to project directory
cd e:/ml

# Create virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt
```

### 2. Running the ML Pipeline via CLI

Run the full machine learning training and prediction pipeline directly from your terminal:
```bash
# Analyze Apple Inc. (AAPL)
.venv\Scripts\python -m ml_engine.pipeline --symbol AAPL --period 2y

# Analyze NVIDIA (NVDA)
.venv\Scripts\python -m ml_engine.pipeline --symbol NVDA --period 1y
```

### 3. Launching the Interactive Web Dashboard Server

Start the backend web server:
```bash
.venv\Scripts\python server.py
```

Then open your browser and navigate to:
[http://localhost:8000](http://localhost:8000)

---

## 🏗️ Project Architecture

```
e:/ml/
├── ml_engine/                # Core Machine Learning & Quantitative Engine
│   ├── __init__.py
│   ├── data_fetcher.py        # Market data fetcher & synthetic generator fallback
│   ├── feature_engineering.py # Technical indicators & dataset preprocessing
│   ├── models.py             # Model suite (RF, GB, Ridge, MLP, Ensemble)
│   ├── backtester.py         # Trading backtest & portfolio performance metrics
│   └── pipeline.py           # End-to-end ML pipeline runner
│
├── public/                   # Web Dashboard Frontend
│   ├── index.html            # Single page application structure
│   ├── style.css             # Glassmorphism dark theme & CSS design system
│   └── app.js                # Chart.js visualizations & API integration
│
├── pyproject.toml            # Project dependencies & uv configuration
├── requirements.txt          # Python packages list
├── server.py                 # REST API server & static file host
└── README.md                 # System documentation
```

---

## 📊 REST API Endpoints

- `GET /api/stocks`: List available popular assets.
- `GET /api/analyze?symbol={SYMBOL}&period={PERIOD}`: Run full ML analysis pipeline, returning technical indicators, model performance metrics, feature importances, 30-day price path predictions, and backtest results.
- `GET /api/health`: Server status endpoint.

---

## ⚖️ License & Disclaimer

This software is for educational and research purposes only. Stock market trading involves substantial risk of loss and is not suitable for every investor. Past performance of machine learning models or backtests is no guarantee of future results.
"# StockmarketPredection" 
"# StockmarketPredection" 
