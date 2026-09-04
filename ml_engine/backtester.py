import numpy as np
import pandas as pd
from typing import Dict, Any, List

class BacktestEngine:
    """
    Simulates algorithmic trading execution based on ML model signal predictions
    and computes portfolio performance metrics vs Buy & Hold benchmark.
    """
    def __init__(self, initial_capital: float = 10000.0, transaction_fee: float = 0.001):
        self.initial_capital = initial_capital
        self.transaction_fee = transaction_fee

    def run_backtest(
        self, 
        df: pd.DataFrame, 
        predictions: np.ndarray, 
        buy_threshold: float = 0.0015,
        sell_threshold: float = -0.0015
    ) -> Dict[str, Any]:
        """
        Runs backtest simulation on test timeframe dataframe.
        """
        data = df.copy().reset_index()
        n = len(data)
        
        cash = self.initial_capital
        position = 0  # Number of shares owned
        portfolio_values = []
        buy_hold_values = []
        trades: List[Dict[str, Any]] = []
        
        initial_price = float(data.iloc[0]["Close"])
        buy_hold_shares = self.initial_capital / initial_price
        
        for i in range(n):
            current_date = str(data.iloc[i]["Date"])[:10]
            close_price = float(data.iloc[i]["Close"])
            signal = float(predictions[i]) if i < len(predictions) else 0.0
            
            # Trading logic
            if signal > buy_threshold and position == 0:
                # Buy Long
                num_shares = (cash * (1 - self.transaction_fee)) / close_price
                cost = num_shares * close_price
                fee = cost * self.transaction_fee
                cash -= (cost + fee)
                position = num_shares
                trades.append({
                    "date": current_date,
                    "action": "BUY",
                    "price": round(close_price, 2),
                    "shares": round(num_shares, 4),
                    "cost": round(cost + fee, 2)
                })
            elif signal < sell_threshold and position > 0:
                # Sell / Exit Long
                revenue = position * close_price
                fee = revenue * self.transaction_fee
                cash += (revenue - fee)
                trades.append({
                    "date": current_date,
                    "action": "SELL",
                    "price": round(close_price, 2),
                    "shares": round(position, 4),
                    "revenue": round(revenue - fee, 2)
                })
                position = 0
                
            # Current portfolio value
            current_port_val = cash + (position * close_price)
            bh_val = buy_hold_shares * close_price
            
            portfolio_values.append({
                "date": current_date,
                "strategy": round(current_port_val, 2),
                "buy_and_hold": round(bh_val, 2)
            })

        # Final metrics computation
        final_strategy_val = portfolio_values[-1]["strategy"]
        final_bh_val = portfolio_values[-1]["buy_and_hold"]
        
        strategy_roi = ((final_strategy_val - self.initial_capital) / self.initial_capital) * 100
        buy_hold_roi = ((final_bh_val - self.initial_capital) / self.initial_capital) * 100
        
        val_series = pd.Series([pv["strategy"] for pv in portfolio_values])
        daily_returns = val_series.pct_change().dropna()
        
        # Sharpe Ratio (Annualized)
        rf_daily = 0.04 / 252.0  # 4% risk free rate
        excess_returns = daily_returns - rf_daily
        sharpe_ratio = float((excess_returns.mean() / (excess_returns.std() + 1e-8)) * np.sqrt(252))
        
        # Max Drawdown
        cum_max = val_series.cummax()
        drawdown = (val_series - cum_max) / cum_max
        max_drawdown = float(drawdown.min() * 100)
        
        # Win Rate & Profit Factor
        winning_trades = 0
        total_gains = 0.0
        total_losses = 0.0
        trade_pairs = len(trades) // 2
        
        for t in range(0, len(trades) - 1, 2):
            if trades[t]["action"] == "BUY" and trades[t+1]["action"] == "SELL":
                pnl = trades[t+1]["revenue"] - trades[t]["cost"]
                if pnl > 0:
                    winning_trades += 1
                    total_gains += pnl
                else:
                    total_losses += abs(pnl)
                    
        win_rate = (winning_trades / max(1, trade_pairs)) * 100 if trade_pairs > 0 else 0.0
        profit_factor = (total_gains / (total_losses + 1e-8)) if total_losses > 0 else (total_gains if total_gains > 0 else 1.0)
        
        return {
            "initial_capital": self.initial_capital,
            "final_portfolio_value": round(final_strategy_val, 2),
            "strategy_roi_pct": round(strategy_roi, 2),
            "buy_hold_roi_pct": round(buy_hold_roi, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "total_trades": len(trades),
            "equity_curve": portfolio_values,
            "trades": trades[-20:]  # Return last 20 trades
        }

if __name__ == "__main__":
    bt = BacktestEngine()
    print("Backtest engine initialized successfully.")
