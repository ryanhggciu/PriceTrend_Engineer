import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Tuple, Optional

class DataProcessor:
    """Handles data processing and preparation for ML models"""
    
    def __init__(self):
        self.price_history = {}
        
    def generate_historical_data(self, crop: str, location: str, days: int = 365) -> pd.DataFrame:
        """
        Generate realistic historical price data based on seasonal patterns
        This simulates real market data with proper seasonal variations
        """
        np.random.seed(hash(crop + location) % 2**32)
        
        # Base price varies by crop type
        base_prices = {
            "Rice": 2500, "Wheat": 2200, "Maize": 1800, "Cotton": 5500,
            "Sugarcane": 350, "Groundnut": 5000, "Tomato": 2000, "Onion": 1500,
            "Potato": 1200, "Mango": 4000, "Apple": 8000, "Chickpea": 6000
        }
        
        base_price = base_prices.get(crop, 3000)
        
        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        prices = []
        for i, date in enumerate(dates):
            # Seasonal variation (higher prices in off-season)
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
            
            # Random market fluctuation
            random_factor = 1 + np.random.normal(0, 0.1)
            
            # Trend factor (slight inflation over time)
            trend_factor = 1 + (i / len(dates)) * 0.05
            
            # Location factor (some states have higher/lower prices)
            location_multipliers = {
                "Maharashtra": 1.1, "Punjab": 1.05, "Kerala": 1.2, "Bihar": 0.9,
                "Uttar Pradesh": 0.95, "West Bengal": 1.0, "Tamil Nadu": 1.1
            }
            location_factor = location_multipliers.get(location, 1.0)
            
            price = base_price * seasonal_factor * random_factor * trend_factor * location_factor
            prices.append(max(price, base_price * 0.5))  # Minimum price floor
        
        df = pd.DataFrame({
            'date': dates,
            'price': prices,
            'crop': crop,
            'location': location
        })
        
        # Add technical indicators
        df['price_ma_7'] = df['price'].rolling(window=7).mean()
        df['price_ma_30'] = df['price'].rolling(window=30).mean()
        df['volatility'] = df['price'].rolling(window=30).std()
        df['price_change'] = df['price'].pct_change()
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for ML model training"""
        df = df.dropna()
        
        # Feature engineering
        df['day_of_year'] = pd.to_datetime(df['date']).dt.dayofyear
        df['month'] = pd.to_datetime(df['date']).dt.month
        df['quarter'] = pd.to_datetime(df['date']).dt.quarter
        
        # Technical features
        df['price_lag_1'] = df['price'].shift(1)
        df['price_lag_7'] = df['price'].shift(7)
        df['price_lag_30'] = df['price'].shift(30)
        
        # Select features for model
        feature_columns = [
            'day_of_year', 'month', 'quarter', 'price_ma_7', 'price_ma_30',
            'volatility', 'price_lag_1', 'price_lag_7', 'price_lag_30'
        ]
        
        # Remove rows with NaN values after lag features
        df_clean = df.dropna()
        
        if len(df_clean) == 0:
            raise ValueError("Not enough data for feature preparation")
        
        X = df_clean[feature_columns].values
        y = df_clean['price'].values
        
        return X, y
    
    def calculate_market_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate various market metrics for analysis"""
        if df.empty:
            return {}
        
        latest_price = df['price'].iloc[-1]
        previous_price = df['price'].iloc[-2] if len(df) > 1 else latest_price
        
        # Calculate metrics
        metrics = {
            'current_price': round(latest_price, 2),
            'previous_price': round(previous_price, 2),
            'price_change': round(latest_price - previous_price, 2),
            'price_change_percent': round(((latest_price - previous_price) / previous_price) * 100, 2) if previous_price != 0 else 0,
            'avg_price_30d': round(df['price'].tail(30).mean(), 2),
            'max_price_30d': round(df['price'].tail(30).max(), 2),
            'min_price_30d': round(df['price'].tail(30).min(), 2),
            'volatility_30d': round(df['price'].tail(30).std(), 2),
            'trend': 'Bullish' if latest_price > df['price'].tail(10).mean() else 'Bearish'
        }
        
        return metrics
    
    def get_seasonal_insights(self, crop: str) -> Dict:
        """Get seasonal insights for specific crops"""
        seasonal_patterns = {
            "Rice": {
                "peak_season": "November-February",
                "lean_season": "July-September",
                "harvest_months": ["November", "December", "April", "May"]
            },
            "Wheat": {
                "peak_season": "March-June",
                "lean_season": "September-November",
                "harvest_months": ["March", "April", "May"]
            },
            "Cotton": {
                "peak_season": "October-February",
                "lean_season": "June-August",
                "harvest_months": ["October", "November", "December"]
            },
            "Tomato": {
                "peak_season": "December-March",
                "lean_season": "June-September",
                "harvest_months": ["December", "January", "February", "March"]
            },
            "Onion": {
                "peak_season": "March-June",
                "lean_season": "September-November",
                "harvest_months": ["March", "April", "May", "November", "December"]
            }
        }
        
        return seasonal_patterns.get(crop, {
            "peak_season": "Varies by region",
            "lean_season": "Varies by region",
            "harvest_months": ["Depends on variety"]
        })
