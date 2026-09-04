import numpy as np
import pandas as pd
from typing import Tuple, List

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical analysis indicators and feature representations for stock price data.
    """
    data = df.copy()
    
    # 1. Moving Averages
    for window in [10, 20, 50, 200]:
        data[f"SMA_{window}"] = data["Close"].rolling(window=window, min_periods=1).mean()
        data[f"Price_to_SMA_{window}"] = data["Close"] / (data[f"SMA_{window}"] + 1e-8)
        
    data["EMA_12"] = data["Close"].ewm(span=12, adjust=False, min_periods=1).mean()
    data["EMA_26"] = data["Close"].ewm(span=26, adjust=False, min_periods=1).mean()
    
    # 2. MACD
    data["MACD_Line"] = data["EMA_12"] - data["EMA_26"]
    data["MACD_Signal"] = data["MACD_Line"].ewm(span=9, adjust=False, min_periods=1).mean()
    data["MACD_Hist"] = data["MACD_Line"] - data["MACD_Signal"]
    
    # 3. Relative Strength Index (RSI - 14)
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    data["RSI_14"] = 100 - (100 / (1 + rs))
    
    # 4. Bollinger Bands (20, 2)
    data["BB_Middle"] = data["SMA_20"]
    bb_std = data["Close"].rolling(window=20, min_periods=1).std().fillna(0)
    data["BB_Upper"] = data["BB_Middle"] + (2 * bb_std)
    data["BB_Lower"] = data["BB_Middle"] - (2 * bb_std)
    data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / (data["BB_Middle"] + 1e-8)
    data["BB_PctB"] = (data["Close"] - data["BB_Lower"]) / (data["BB_Upper"] - data["BB_Lower"] + 1e-8)
    
    # 5. Average True Range (ATR - 14)
    high_low = data["High"] - data["Low"]
    high_cp = (data["High"] - data["Close"].shift(1)).abs()
    low_cp = (data["Low"] - data["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    data["ATR_14"] = true_range.rolling(window=14, min_periods=1).mean()
    data["Normalized_ATR"] = data["ATR_14"] / (data["Close"] + 1e-8)
    
    # 6. Stochastic Oscillator (14, 3)
    low_14 = data["Low"].rolling(window=14, min_periods=1).min()
    high_14 = data["High"].rolling(window=14, min_periods=1).max()
    data["Stoch_K"] = 100 * ((data["Close"] - low_14) / (high_14 - low_14 + 1e-8))
    data["Stoch_D"] = data["Stoch_K"].rolling(window=3, min_periods=1).mean()
    
    # 7. Volume Dynamics & OBV
    data["Volume_SMA_20"] = data["Volume"].rolling(window=20, min_periods=1).mean()
    data["Volume_Ratio"] = data["Volume"] / (data["Volume_SMA_20"] + 1e-8)
    
    obv_change = np.sign(data["Close"].diff()) * data["Volume"]
    data["OBV"] = obv_change.cumsum().fillna(0)
    data["OBV_SMA_10"] = data["OBV"].rolling(window=10, min_periods=1).mean()
    
    # 8. Returns & Volatility
    data["Log_Return_1d"] = np.log(data["Close"] / data["Close"].shift(1)).fillna(0)
    data["Log_Return_5d"] = np.log(data["Close"] / data["Close"].shift(5)).fillna(0)
    data["Volatility_20"] = data["Log_Return_1d"].rolling(window=20, min_periods=1).std().fillna(0) * np.sqrt(252)
    data["ROC_10"] = (((data["Close"] - data["Close"].shift(10)) / (data["Close"].shift(10) + 1e-8)) * 100).fillna(0)
    
    # 9. Machine Learning Target Variables
    data["Target_1d_Return"] = (data["Close"].shift(-1) - data["Close"]) / data["Close"]
    data["Target_5d_Return"] = (data["Close"].shift(-5) - data["Close"]) / data["Close"]
    data["Target_1d_Price"] = data["Close"].shift(-1)
    data["Target_5d_Price"] = data["Close"].shift(-5)
    data["Target_Direction"] = (data["Target_1d_Return"] > 0).astype(int)
    
    return data

def get_feature_columns() -> List[str]:
    """
    Returns the list of input predictor features used for ML model training.
    """
    return [
        "Price_to_SMA_10", "Price_to_SMA_20", "Price_to_SMA_50", "Price_to_SMA_200",
        "MACD_Line", "MACD_Signal", "MACD_Hist",
        "RSI_14", "BB_Width", "BB_PctB",
        "Normalized_ATR", "Stoch_K", "Stoch_D",
        "Volume_Ratio", "Log_Return_1d", "Log_Return_5d",
        "Volatility_20", "ROC_10"
    ]

def prepare_ml_dataset(
    df: pd.DataFrame, 
    target_col: str = "Target_1d_Return",
    test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Prepares train/test split preserving temporal chronological order.
    """
    data = add_technical_indicators(df)
    features = get_feature_columns()
    
    # Drop rows with NaN in features or target
    clean_df = data.dropna(subset=features + [target_col]).copy()
    
    split_idx = int(len(clean_df) * (1 - test_ratio))
    
    X_train = clean_df.iloc[:split_idx][features]
    X_test = clean_df.iloc[split_idx:][features]
    
    y_train = clean_df.iloc[:split_idx][target_col]
    y_test = clean_df.iloc[split_idx:][target_col]
    
    return X_train, X_test, y_train, y_test, features

if __name__ == "__main__":
    from data_fetcher import fetch_stock_data
    raw = fetch_stock_data("AAPL", period="1y")
    feat_df = add_technical_indicators(raw)
    print(f"Feature dataframe columns: {len(feat_df.columns)}")
    X_tr, X_te, y_tr, y_te, feats = prepare_ml_dataset(raw)
    print(f"Train shape: {X_tr.shape}, Test shape: {X_te.shape}")
