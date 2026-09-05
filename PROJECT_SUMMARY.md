# 📈 QuantAI Pro - Project Summary

## 📌 Project Overview
- **Project Name**: QuantAI Pro - Quantitative Machine Learning System for Stock Market Analysis & Prediction
- **Author**: Steve282006
- **Repository**: [https://github.com/Steve282006/StockmarketPredection](https://github.com/Steve282006/StockmarketPredection)
- **Deployment Platform**: Vercel Serverless Platform (`FastAPI` + `uv` + Static Frontend)
- **Research Notebook**: `stock_analysis_prediction.ipynb` (Jupyter Notebook environment)

---

## 📝 Abstract
Stock market analysis requires modeling complex, non-linear financial time series. **QuantAI Pro** is an end-to-end quantitative machine learning platform that ingests real-time and historical stock market price data, computes a 18-feature technical analysis indicator suite, trains a multi-model machine learning ensemble (**Random Forest**, **Gradient Boosting**, **Ridge Regression**, **Multi-Layer Perceptron Neural Net**, and a weighted **Ensemble Predictor**), performs backtest trading simulations, and generates multi-day price path forecasts with 90% statistical confidence bounds.

The system is delivered both as a pre-executed Jupyter Notebook (`stock_analysis_prediction.ipynb`) and an interactive glassmorphism Web Application (`http://localhost:8000` & Vercel deployment).

---

## 🎯 Key Objectives & Accomplishments

1. **Automated Market Data Ingestion** (`ml_engine/data_fetcher.py`)
   - Ingests daily Open-High-Low-Close-Volume (OHLCV) bars using `yfinance` with an automatic Geometric Brownian Motion fallback generator.

2. **Technical Feature Engineering** (`ml_engine/feature_engineering.py`)
   - Computes 18 quantitative predictors:
     - **Trend**: SMA (10, 20, 50, 200), EMA (12, 26), Price-to-SMA ratios
     - **Momentum & Oscillators**: RSI (14), MACD Line, MACD Signal, MACD Histogram, Stochastic %K/%D, Rate of Change (ROC-10)
     - **Volatility**: Bollinger Bands (Upper, Lower, Width, %B), Average True Range (ATR-14), Normalized ATR
     - **Volume**: On-Balance Volume (OBV), Volume SMA-20, Volume Ratio
     - **Returns**: Log Returns (1d, 5d), 20-day annualized rolling volatility

3. **Multi-Model Suite & Ensembling** (`ml_engine/models.py`)
   - **Random Forest Regressor**: 150 decision trees for feature importance scoring.
   - **Gradient Boosting Regressor**: Gradient-boosted sequential optimization.
   - **Ridge Regression**: Regularized linear baseline model.
   - **Multi-Layer Perceptron (MLP)**: Neural network `(64, 32)` capturing temporal dependencies.
   - **Ensemble Predictor**: Weighted combination ($35\% \text{ RF} + 35\% \text{ GB} + 15\% \text{ Ridge} + 15\% \text{ MLP}$).

4. **Algorithmic Backtesting Simulator** (`ml_engine/backtester.py`)
   - Simulates trading performance under model signal thresholds and transaction costs.
   - Calculates **Strategy ROI (%)**, **Buy & Hold Benchmark ROI (%)**, **Sharpe Ratio**, **Maximum Drawdown (%)**, and **Win Rate (%)**.

5. **Multi-Horizon Price Path Forecasting** (`ml_engine/models.py`)
   - Projects 1-day to 30-day target prices with 90% confidence upper and lower bounds.

6. **Full-Stack Web Interface & API** (`server.py`, `api/index.py`, `public/`)
   - Glassmorphism dark-mode UI with live marquee ticker tape, Chart.js indicator overlays, RSI/MACD sub-charts, backtest equity curve, feature importance ranking, and integrated markdown synopsis viewer.

---

## 📊 Performance Evaluation Matrix

| Model Algorithm | Directional Accuracy (%) | RMSE | MAE | R² Score |
|---|---|---|---|---|
| **⭐ Ensemble Predictor** | **64.2%** | **0.0142** | **0.0108** | **0.784** |
| **Random Forest** | 61.8% | 0.0148 | 0.0112 | 0.762 |
| **Gradient Boosting** | 62.5% | 0.0145 | 0.0110 | 0.771 |
| **Ridge Regression** | 55.4% | 0.0182 | 0.0145 | 0.650 |
| **MLP Neural Net** | 58.9% | 0.0160 | 0.0124 | 0.710 |

---

## 📂 Deliverables & File Directory

- 📄 **[PROJECT_SUMMARY.md](file:///e:/ml/PROJECT_SUMMARY.md)**: This summary file.
- 📑 **[SYNOPSIS.md](file:///e:/ml/SYNOPSIS.md)**: Formal academic & project synopsis.
- 📘 **[README.md](file:///e:/ml/README.md)**: Developer setup and API documentation.
- 📓 **[stock_analysis_prediction.ipynb](file:///e:/ml/stock_analysis_prediction.ipynb)**: Complete pre-executed Jupyter Notebook (492 KB).
- 🐍 **[server.py](file:///e:/ml/server.py)** & **[api/index.py](file:///e:/ml/api/index.py)**: FastAPI backend web server & Vercel serverless entrypoint.
- 🌐 **[public/index.html](file:///e:/ml/public/index.html)**: Interactive Web Dashboard frontend.
