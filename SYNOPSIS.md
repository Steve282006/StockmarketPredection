# PROJECT SYNOPSIS

## 📌 Project Title
**QuantAI: Quantitative Machine Learning System for Stock Market Analysis, Multi-Model Forecasting, and Algorithmic Backtesting**

---

## 📝 Abstract
Financial stock market prediction is a complex time-series challenge characterized by non-linear dynamics, noise, and volatile regime shifts. This project implements an end-to-end quantitative machine learning pipeline designed to analyze historical financial bar data, extract multi-dimensional technical indicators, train a suite of machine learning models (**Random Forest**, **Gradient Boosting**, **Ridge Regression**, **Multi-Layer Perceptron Neural Networks**, and a weighted **Ensemble Predictor**), and generate multi-day price path forecasts accompanied by 90% statistical confidence bounds. 

Furthermore, the platform incorporates an algorithmic trading backtesting engine to evaluate strategy returns (Cumulative ROI, Sharpe Ratio, Maximum Drawdown %, Win Rate) against benchmark buy-and-hold strategies. The system is delivered both as a pre-executed **Jupyter Notebook** (`stock_analysis_prediction.ipynb`) for quantitative research and an interactive dark-mode **Web Dashboard** (`server.py` + HTML5/CSS3/Chart.js).

---

## 🎯 Objectives & Scope

### Key Objectives:
1. **Automated Market Ingestion**: Ingest daily Open-High-Low-Close-Volume (OHLCV) stock data using `yfinance` with an automatic Geometric Brownian Motion fallback generator.
2. **Feature Engineering**: Compute 18 technical indicators covering trend, momentum, volatility, and volume dynamics (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic %K/%D, OBV, Log Returns, and Volatility).
3. **Multi-Model Suite**: Train baseline linear, tree-based ensemble, and neural network algorithms to predict 1-day log returns and directional price movement.
4. **Model Performance Evaluation**: Quantify model accuracy using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Coefficient of Determination ($R^2$), and Directional Accuracy (%).
5. **Algorithmic Strategy Backtesting**: Simulate trade execution based on model signal confidence thresholds under realistic transaction cost constraints.
6. **Multi-Horizon Price Forecasting**: Predict future price trajectories across 1-day to 30-day horizons with confidence intervals.
7. **Dual Interface**: Provide both a quantitative Jupyter Notebook environment and an interactive full-stack web application.

---

## 🏗️ System Architecture & Workflow

```
[ Stock Market Data Ingestion ]
          │
          ▼
[ Technical Feature Engineering ]
  • Trend: SMA (10/20/50/200), EMA (12/26)
  • Momentum: RSI (14), MACD, Stochastic, ROC
  • Volatility: Bollinger Bands, ATR (14)
  • Volume: OBV, Volume Ratio
          │
          ▼
[ Data Preprocessing & Chronological Split ]
  • Chronological 80/20 Train-Test Partitioning
  • Feature Standardization (StandardScaler)
          │
          ▼
[ Multi-Model Machine Learning Training ]
  ├── Random Forest Regressor
  ├── Gradient Boosting Regressor
  ├── Ridge Regression
  ├── MLP Neural Network
  └── ⭐ Weighted Ensemble Predictor
          │
          ▼
[ Evaluation & Strategy Backtest ]
  ├── RMSE, MAE, R², Directional Accuracy (%)
  └── Backtest Equity Curve, Sharpe Ratio, Max Drawdown
          │
          ▼
[ Output Delivery ]
  ├── Interactive Jupyter Notebook (.ipynb)
  └── Glassmorphism Web Dashboard (http://localhost:8000)
```

---

## ⚙️ Technical Indicator Feature Matrix

| Indicator Category | Feature Name | Description / Formula |
|---|---|---|
| **Trend** | `SMA_10`, `SMA_20`, `SMA_50`, `SMA_200` | Moving average price levels across short, medium, and long term. |
| **Trend Ratios** | `Price_to_SMA_X` | Ratio of current price relative to moving averages. |
| **Oscillators** | `RSI_14` | Relative Strength Index (14-day) measuring momentum gain/loss ratio. |
| **MACD** | `MACD_Line`, `MACD_Signal`, `MACD_Hist` | Exponential moving average convergence/divergence lines and histogram. |
| **Volatility** | `BB_Upper`, `BB_Lower`, `BB_Width`, `BB_PctB` | Bollinger Bands (20-day, 2 std dev) measuring volatility expansion/compression. |
| **Volatility** | `ATR_14`, `Normalized_ATR` | Average True Range measuring absolute daily price movement range. |
| **Stochastic** | `Stoch_K`, `Stoch_D` | Stochastic Oscillator measuring relative close position within 14-day high/low. |
| **Volume** | `OBV`, `Volume_Ratio` | On-Balance Volume and ratio relative to 20-day volume SMA. |
| **Returns** | `Log_Return_1d`, `Log_Return_5d`, `Volatility_20` | Daily/5-day log returns and 20-day annualized rolling volatility. |

---

## 🤖 Machine Learning Algorithms

1. **Random Forest Regressor**: Ensemble of 150 decision trees capturing non-linear indicator interactions and feature importance rankings.
2. **Gradient Boosting Regressor**: Boosting algorithm iteratively minimizing square error loss on complex temporal patterns.
3. **Ridge Regression**: Regularized linear model serving as an interpretable baseline benchmark.
4. **Multi-Layer Perceptron (MLP)**: Deep feedforward neural network `(64, 32)` learning non-linear feature representations.
5. **Weighted Ensemble Predictor**: Combines top tree-based and neural predictions ($35\% \text{ RF} + 35\% \text{ GB} + 15\% \text{ Ridge} + 15\% \text{ MLP}$) for enhanced out-of-sample generalization.

---

## 💼 Algorithmic Backtesting Simulator

The backtesting engine models trading performance over unseen test periods:
- **Buy Signal Threshold**: Triggered when predicted 1-day return $> +0.15\%$.
- **Sell Signal Threshold**: Triggered when predicted 1-day return $< -0.15\%$.
- **Transaction Costs**: Fixed $0.1\%$ fee per execution.
- **Computed Metrics**:
  - **Strategy Total ROI (%)** vs **Buy & Hold Benchmark ROI (%)**
  - **Sharpe Ratio** (Annualized risk-adjusted return ratio)
  - **Maximum Drawdown (%)** (Peak-to-trough capital decline)
  - **Win Rate (%)** & **Profit Factor**

---

## 💻 Tech Stack & Tools

- **Core Programming**: Python 3.11+
- **Environment Management**: `uv` package manager
- **Data & ML**: `pandas`, `numpy`, `scikit-learn`, `yfinance`, `joblib`, `scipy`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`, `Chart.js`
- **Notebook Environment**: `Jupyter Notebook` (`stock_analysis_prediction.ipynb`)
- **Web App**: Python HTTP Server REST API + Vanilla CSS (Glassmorphism dark theme)

---

## 📈 Summary of Output Deliverables

1. **Jupyter Notebook File**: `stock_analysis_prediction.ipynb`
2. **Web Dashboard Server**: Running at `http://localhost:8000` via `server.py`.
3. **ML Engine Library**: Modular python files under `ml_engine/`.
4. **Documentation**: `README.md`, `SYNOPSIS.md`, and `walkthrough.md`.
