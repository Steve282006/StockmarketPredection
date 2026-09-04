import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class StockPredictorSuite:
    """
    Multi-model suite for stock price return prediction, model evaluation,
    feature importance scoring, and future horizon forecasting.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.models: Dict[str, Any] = {
            "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_split=5, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=120, learning_rate=0.03, max_depth=4, random_state=42),
            "RidgeRegression": Ridge(alpha=2.0),
            "MLPNeuralNet": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, alpha=0.01, random_state=42)
        }
        self.trained = False
        self.feature_names: List[str] = []
        self.metrics: Dict[str, Dict[str, float]] = {}
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Fits all predictor models and scales features.
        """
        self.feature_names = list(X_train.columns)
        X_scaled = self.scaler.fit_transform(X_train)
        
        for name, model in self.models.items():
            model.fit(X_scaled, y_train)
            
        self.trained = True

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Evaluates models on test data calculating RMSE, MAE, R2, and Directional Accuracy.
        """
        if not self.trained:
            raise ValueError("Suite has not been trained yet. Call fit() first.")
            
        X_scaled = self.scaler.transform(X_test)
        results = {}
        
        for name, model in self.models.items():
            preds = model.predict(X_scaled)
            
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))
            
            # Directional accuracy (% of correct gain/loss signs)
            correct_direction = np.sign(preds) == np.sign(y_test.values)
            dir_acc = float(np.mean(correct_direction) * 100)
            
            results[name] = {
                "RMSE": round(rmse, 6),
                "MAE": round(mae, 6),
                "R2": round(r2, 4),
                "Directional_Accuracy": round(dir_acc, 2)
            }
            
        # Add Ensemble Model evaluation
        ensemble_preds = self._predict_ensemble_scaled(X_scaled)
        ensemble_rmse = float(np.sqrt(mean_squared_error(y_test, ensemble_preds)))
        ensemble_mae = float(mean_absolute_error(y_test, ensemble_preds))
        ensemble_r2 = float(r2_score(y_test, ensemble_preds))
        ensemble_dir_acc = float(np.mean(np.sign(ensemble_preds) == np.sign(y_test.values)) * 100)
        
        results["Ensemble"] = {
            "RMSE": round(ensemble_rmse, 6),
            "MAE": round(ensemble_mae, 6),
            "R2": round(ensemble_r2, 4),
            "Directional_Accuracy": round(ensemble_dir_acc, 2)
        }
        
        self.metrics = results
        return results

    def _predict_ensemble_scaled(self, X_scaled: np.ndarray) -> np.ndarray:
        preds_list = []
        weights = {"RandomForest": 0.35, "GradientBoosting": 0.35, "RidgeRegression": 0.15, "MLPNeuralNet": 0.15}
        
        for name, model in self.models.items():
            preds_list.append(weights[name] * model.predict(X_scaled))
            
        return np.sum(preds_list, axis=0)

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Returns average feature importances from tree-based models.
        """
        if not self.trained:
            return {}
            
        rf_imp = self.models["RandomForest"].feature_importances_
        gb_imp = self.models["GradientBoosting"].feature_importances_
        avg_imp = (rf_imp + gb_imp) / 2.0
        
        # Normalize to percentage
        total = np.sum(avg_imp)
        if total > 0:
            avg_imp = (avg_imp / total) * 100
            
        imp_dict = {feat: round(float(imp), 2) for feat, imp in zip(self.feature_names, avg_imp)}
        # Sort descending
        return dict(sorted(imp_dict.items(), key=lambda item: item[1], reverse=True))

    def predict_next_days(
        self, 
        latest_features: pd.DataFrame, 
        current_price: float, 
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Generates multi-day price path predictions with upper and lower confidence intervals.
        """
        if not self.trained:
            raise ValueError("Model suite not trained.")
            
        X_scaled = self.scaler.transform(latest_features.iloc[[-1]])
        
        # Predict 1-day log return expectation
        expected_daily_return = float(self._predict_ensemble_scaled(X_scaled)[0])
        
        # Historical volatility scaling
        volatility = float(latest_features.iloc[-1].get("Volatility_20", 0.25)) / np.sqrt(252)
        volatility = max(volatility, 0.008)
        
        predictions = []
        sim_price = current_price
        
        for d in range(1, days + 1):
            # Compound expected return with subtle mean reversion decay
            decay_factor = 0.95 ** (d - 1)
            daily_step_return = expected_daily_return * decay_factor
            sim_price = sim_price * (1 + daily_step_return)
            
            # Confidence interval calculation (+/- z * vol * sqrt(days))
            std_bound = 1.645 * volatility * np.sqrt(d) * sim_price  # 90% confidence interval
            upper_bound = sim_price + std_bound
            lower_bound = max(0.1, sim_price - std_bound)
            
            predictions.append({
                "day": d,
                "predicted_price": round(sim_price, 2),
                "upper_bound": round(upper_bound, 2),
                "lower_bound": round(lower_bound, 2),
                "expected_return_pct": round(((sim_price - current_price) / current_price) * 100, 2)
            })
            
        return predictions

    def save(self, filepath: str) -> None:
        """Saves model suite state to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "models": self.models,
            "trained": self.trained,
            "feature_names": self.feature_names,
            "metrics": self.metrics
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> "StockPredictorSuite":
        """Loads model suite state from disk."""
        data = joblib.load(filepath)
        suite = cls()
        suite.scaler = data["scaler"]
        suite.models = data["models"]
        suite.trained = data["trained"]
        suite.feature_names = data["feature_names"]
        suite.metrics = data["metrics"]
        return suite
