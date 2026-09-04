import datetime
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_stock_data(symbol: str = "AAPL", period_days: int = 500) -> pd.DataFrame:
    """
    Generates realistic synthetic stock market daily bar data using Geometric Brownian Motion
    with volatility clustering and dynamic trend regimes. Used as a reliable fallback.
    """
    np.random.seed(hash(symbol) % (2**32 - 1))
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=int(period_days * 1.5))
    
    # Generate business days
    dates = pd.bdate_range(start=start_date, end=end_date)[-period_days:]
    n = len(dates)
    
    # Initial base prices by ticker family
    base_prices = {
        "NVDA": 120.0, "AAPL": 220.0, "TSLA": 250.0, "MSFT": 440.0,
        "GOOGL": 180.0, "AMZN": 185.0, "META": 500.0, "SPY": 550.0, "BTC-USD": 65000.0
    }
    s0 = base_prices.get(symbol.upper(), 150.0)
    
    # GBM with regime shifts & drift
    drift = 0.0004  # Annualized positive expectation ~10%
    volatility = 0.018 if symbol.upper() not in ["TSLA", "BTC-USD"] else 0.035
    
    returns = np.random.normal(drift, volatility, n)
    
    # Add momentum & autocorrelation
    for i in range(1, n):
        returns[i] += 0.08 * returns[i-1]
        
    price_series = s0 * np.exp(np.cumsum(returns))
    
    # Generate OHLVC bars
    data = []
    for i in range(n):
        close_price = price_series[i]
        daily_vol = close_price * (volatility * np.random.uniform(0.5, 1.5))
        open_price = close_price + np.random.uniform(-0.4, 0.4) * daily_vol
        high_price = max(open_price, close_price) + abs(np.random.uniform(0.1, 1.2) * daily_vol)
        low_price = min(open_price, close_price) - abs(np.random.uniform(0.1, 1.2) * daily_vol)
        volume = int(np.random.uniform(15_000_000, 85_000_000) * (1.0 + abs(returns[i]) * 10))
        
        data.append({
            "Date": dates[i],
            "Open": round(float(open_price), 2),
            "High": round(float(high_price), 2),
            "Low": round(float(low_price), 2),
            "Close": round(float(close_price), 2),
            "Adj Close": round(float(close_price), 2),
            "Volume": volume
        })
        
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df

def fetch_stock_data(symbol: str = "AAPL", period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetches real historical stock data using yfinance, falling back to synthetic generator if necessary.
    """
    symbol = symbol.upper().strip()
    period_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
    days = period_map.get(period, 730)
    
    try:
        import yfinance as yf
        logger.info(f"Downloading market data for {symbol} (period={period})...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty or len(df) < 30:
            logger.warning(f"yfinance returned empty/insufficient data for {symbol}. Using synthetic generator.")
            return generate_synthetic_stock_data(symbol=symbol, period_days=days)
            
        # Reset index and clean column names
        df = df.reset_index()
        if "Date" not in df.columns and "Datetime" in df.columns:
            df.rename(columns={"Datetime": "Date"}, inplace=True)
            
        # Select standard columns
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column {col} in downloaded data")
                
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]
            
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df.set_index("Date", inplace=True)
        
        # Drop duplicates and sort index
        df = df[~df.index.duplicated(keep='first')].sort_index()
        
        # Ensure numerical types
        cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        df.dropna(subset=["Close"], inplace=True)
        
        logger.info(f"Successfully loaded {len(df)} bars for {symbol}.")
        return df

    except Exception as e:
        logger.warning(f"Failed to fetch live data for {symbol} via yfinance ({e}). Using synthetic generator.")
        return generate_synthetic_stock_data(symbol=symbol, period_days=days)

if __name__ == "__main__":
    test_df = fetch_stock_data("AAPL", period="1y")
    print(f"Loaded AAPL dataset shape: {test_df.shape}")
    print(test_df.tail())
