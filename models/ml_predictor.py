import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MarketPredictor:
    """ML model for agricultural market price prediction"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression()
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.is_trained = False
        self.feature_names = []
        
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train multiple models and select the best performing one"""
        if len(X) < 10:
            raise ValueError("Insufficient data for training. Need at least 10 samples.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model_scores = {}
        
        # Train and evaluate each model
        for name, model in self.models.items():
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                model_scores[name] = {
                    'mae': mae,
                    'mse': mse,
                    'r2': r2,
                    'model': model
                }
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        if not model_scores:
            raise ValueError("No models could be trained successfully")
        
        # Select best model based on R² score
        self.best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['r2'])
        self.best_model = model_scores[self.best_model_name]['model']
        self.is_trained = True
        
        return model_scores
    
    def predict_prices(self, features: np.ndarray) -> np.ndarray:
        """Make price predictions using the trained model"""
        if not self.is_trained or self.best_model is None:
            raise ValueError("Model must be trained before making predictions")
        
        features_scaled = self.scaler.transform(features)
        predictions = self.best_model.predict(features_scaled)
        return predictions
    
    def predict_future_prices(self, last_features: np.ndarray, days: int = 3) -> List[float]:
        """Predict prices for the next few days"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        current_features = last_features.copy()
        
        for day in range(days):
            # Predict next price
            pred_price = self.predict_prices(current_features.reshape(1, -1))[0]
            predictions.append(float(pred_price))
            
            # Update features for next prediction (simple approach)
            # In a real implementation, you'd update all time-dependent features
            if len(current_features) >= 3:  # Assuming last 3 features are price lags
                current_features[-3] = current_features[-2]  # price_lag_7 = price_lag_1
                current_features[-2] = current_features[-1]  # price_lag_1 = current price
                current_features[-1] = pred_price  # current price = prediction
        
        return predictions
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from the trained model"""
        if not self.is_trained or self.best_model is None:
            return {}
        
        if hasattr(self.best_model, 'feature_importances_'):
            feature_names = [
                'day_of_year', 'month', 'quarter', 'price_ma_7', 'price_ma_30',
                'volatility', 'price_lag_1', 'price_lag_7', 'price_lag_30'
            ]
            
            importance_dict = dict(zip(feature_names, self.best_model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {}
    
    def calculate_prediction_confidence(self, features: np.ndarray) -> float:
        """Calculate confidence score for predictions"""
        if not self.is_trained:
            return 0.0
        
        try:
            # Use ensemble variance as confidence measure
            predictions = []
            for model in self.models.values():
                if hasattr(model, 'predict'):
                    features_scaled = self.scaler.transform(features)
                    pred = model.predict(features_scaled)
                    predictions.append(pred[0])
            
            if len(predictions) > 1:
                variance = np.var(predictions)
                confidence = max(0.0, min(1.0, 1.0 - (variance / np.mean(predictions))))
                return float(confidence)
            
        except Exception:
            pass
        
        return 0.5  # Default confidence
    
    def generate_market_signals(self, current_price: float, predicted_prices: List[float]) -> Dict:
        """Generate trading signals based on predictions"""
        if not predicted_prices:
            return {"signal": "HOLD", "strength": 0}
        
        tomorrow_price = predicted_prices[0]
        price_change = ((tomorrow_price - current_price) / current_price) * 100
        
        if price_change > 5:
            signal = "STRONG_BUY"
            strength = min(100, abs(price_change) * 10)
        elif price_change > 2:
            signal = "BUY"
            strength = min(80, abs(price_change) * 15)
        elif price_change < -5:
            signal = "STRONG_SELL"
            strength = min(100, abs(price_change) * 10)
        elif price_change < -2:
            signal = "SELL"
            strength = min(80, abs(price_change) * 15)
        else:
            signal = "HOLD"
            strength = 100 - abs(price_change) * 10
        
        return {
            "signal": signal,
            "strength": round(strength, 1),
            "price_change_percent": round(price_change, 2),
            "recommendation": self._get_recommendation(signal, price_change)
        }
    
    def _get_recommendation(self, signal: str, price_change: float) -> str:
        """Get human-readable recommendation based on signal"""
        recommendations = {
            "STRONG_BUY": f"Strong upward trend expected (+{abs(price_change):.1f}%). Consider buying for better margins.",
            "BUY": f"Moderate price increase expected (+{abs(price_change):.1f}%). Good time to buy.",
            "HOLD": f"Stable prices expected ({price_change:+.1f}%). Hold current position.",
            "SELL": f"Price decline expected ({price_change:+.1f}%). Consider selling soon.",
            "STRONG_SELL": f"Significant price drop expected ({price_change:+.1f}%). Sell immediately if possible."
        }
        return recommendations.get(signal, "Monitor market conditions closely.")
