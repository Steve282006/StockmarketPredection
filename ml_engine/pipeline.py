import json
import logging
import argparse
import pandas as pd
from typing import Dict, Any

from .data_fetcher import fetch_stock_data
from .feature_engineering import add_technical_indicators, prepare_ml_dataset
from .models import StockPredictorSuite
from .backtester import BacktestEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_full_ml_pipeline(
    symbol: str = "AAPL",
    period: str = "2y",
    forecast_days: int = 30
) -> Dict[str, Any]:
    """
    Executes complete end-to-end stock machine learning pipeline:
    Data fetch -> Feature engineering -> Model training & evaluation -> Backtest -> Multi-day forecast.
    """
    symbol = symbol.upper().strip()
    logger.info(f"--- Starting Stock ML Pipeline for Symbol: {symbol} ---")
    
    # 1. Data Fetching
    raw_df = fetch_stock_data(symbol, period=period)
    
    # 2. Feature Engineering & Dataset Preparation
    X_train, X_test, y_train, y_test, feature_names = prepare_ml_dataset(raw_df)
    
    # 3. Model Suite Fitting & Evaluation
    predictor = StockPredictorSuite()
    predictor.fit(X_train, y_train)
    evaluation_metrics = predictor.evaluate(X_test, y_test)
    
    # 4. Feature Importances
    feature_importances = predictor.get_feature_importances()
    
    # 5. Model Backtest Simulation
    df_with_indicators = add_technical_indicators(raw_df)
    test_df = df_with_indicators.iloc[len(X_train):].copy()
    
    # Predict returns on test set using ensemble model
    X_test_scaled = predictor.scaler.transform(X_test)
    ensemble_test_preds = predictor._predict_ensemble_scaled(X_test_scaled)
    
    backtester = BacktestEngine(initial_capital=10000.0)
    backtest_results = backtester.run_backtest(test_df, ensemble_test_preds)
    
    # 6. Future Multi-Day Predictions
    current_price = float(raw_df["Close"].iloc[-1])
    future_predictions = predictor.predict_next_days(
        latest_features=X_test, 
        current_price=current_price, 
        days=forecast_days
    )
    
    # 7. Format Historical Price Series for Charting (last 250 bars)
    historical_chart_df = df_with_indicators.tail(250).reset_index()
    historical_data = []
    for _, row in historical_chart_df.iterrows():
        historical_data.append({
            "date": str(row["Date"])[:10],
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
            "sma_20": round(float(row["SMA_20"]), 2) if pd.notnull(row.get("SMA_20")) else None,
            "sma_50": round(float(row["SMA_50"]), 2) if pd.notnull(row.get("SMA_50")) else None,
            "rsi_14": round(float(row["RSI_14"]), 2) if pd.notnull(row.get("RSI_14")) else None,
            "macd_line": round(float(row["MACD_Line"]), 2) if pd.notnull(row.get("MACD_Line")) else None,
            "macd_signal": round(float(row["MACD_Signal"]), 2) if pd.notnull(row.get("MACD_Signal")) else None,
            "bb_upper": round(float(row["BB_Upper"]), 2) if pd.notnull(row.get("BB_Upper")) else None,
            "bb_lower": round(float(row["BB_Lower"]), 2) if pd.notnull(row.get("BB_Lower")) else None
        })
        
    latest_metrics = evaluation_metrics.get("Ensemble", {})
    summary_signal = "BUY" if latest_metrics.get("Directional_Accuracy", 50) > 52 and future_predictions[0]["expected_return_pct"] > 0 else "HOLD"
    if future_predictions[0]["expected_return_pct"] < -0.5 and latest_metrics.get("Directional_Accuracy", 50) > 52:
        summary_signal = "SELL"
        
    result = {
        "symbol": symbol,
        "current_price": current_price,
        "signal": summary_signal,
        "latest_date": str(raw_df.index[-1])[:10],
        "metrics": evaluation_metrics,
        "feature_importances": feature_importances,
        "future_predictions": future_predictions,
        "backtest": backtest_results,
        "historical_data": historical_data
    }
    
    logger.info(f"--- Pipeline complete for {symbol}. Signal: {summary_signal} ---")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock ML Pipeline CLI")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--period", type=str, default="2y", help="Historical data period")
    args = parser.parse_args()
    
    res = run_full_ml_pipeline(symbol=args.symbol, period=args.period)
    print(json.dumps({k: v for k, v in res.items() if k != "historical_data"}, indent=2))
