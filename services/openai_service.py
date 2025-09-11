import json
import os
from typing import Dict, List, Optional
from openai import OpenAI

class OpenAIAnalysisService:
    """Service for AI-powered market analysis using OpenAI"""
    
    def __init__(self):
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "default_openai_key"))
        self.model = "gpt-5"
    
    def analyze_market_trends(self, crop: str, location: str, market_data: Dict, predictions: List[float]) -> Dict:
        """Generate comprehensive market trend analysis"""
        try:
            prompt = f"""
            As an agricultural market expert, analyze the following data for {crop} in {location}:
            
            Current Market Data:
            - Current Price: ₹{market_data.get('current_price', 0)} per quintal
            - Yesterday Price: ₹{market_data.get('yesterday_price', 0)} per quintal
            - Weekly High: ₹{market_data.get('weekly_high', 0)} per quintal
            - Weekly Low: ₹{market_data.get('weekly_low', 0)} per quintal
            - Volume Traded: {market_data.get('volume_traded', 0)} tonnes
            - Current Trend: {market_data.get('trend', 'Stable')}
            
            Price Predictions (next 3 days): {predictions}
            
            Provide a comprehensive market analysis in JSON format with these fields:
            - overall_sentiment: (Bullish/Bearish/Neutral)
            - key_insights: (array of 3-4 key market insights)
            - risk_factors: (array of potential risks)
            - opportunities: (array of market opportunities)
            - short_term_outlook: (1-2 week outlook)
            - medium_term_outlook: (1-3 month outlook)
            - recommended_actions: (array of actionable recommendations)
            - confidence_score: (0-100, your confidence in this analysis)
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert agricultural market analyst with deep knowledge of Indian commodity markets, seasonal patterns, and price dynamics."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            analysis = json.loads(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            print(f"OpenAI analysis error: {str(e)}")
            return self._get_fallback_analysis(crop, market_data, predictions)
    
    def get_farming_recommendations(self, crop: str, location: str, market_analysis: Dict) -> Dict:
        """Get AI-powered farming and business recommendations"""
        try:
            prompt = f"""
            Based on the market analysis for {crop} in {location}, provide farming and business recommendations:
            
            Market Sentiment: {market_analysis.get('overall_sentiment', 'Neutral')}
            Key Insights: {market_analysis.get('key_insights', [])}
            
            Provide recommendations in JSON format with these fields:
            - planting_recommendations: (best practices for current season)
            - harvest_timing: (optimal harvest timing advice)
            - storage_advice: (post-harvest storage recommendations)
            - selling_strategy: (when and how to sell for best prices)
            - risk_mitigation: (ways to reduce market risks)
            - alternative_crops: (suggestions for crop diversification)
            - technology_adoption: (relevant agricultural technologies)
            - financial_planning: (budgeting and investment advice)
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an agricultural advisor specializing in Indian farming practices, crop management, and market strategies."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.4
            )
            
            recommendations = json.loads(response.choices[0].message.content)
            return recommendations
            
        except Exception as e:
            print(f"OpenAI recommendations error: {str(e)}")
            return self._get_fallback_recommendations(crop, location)
    
    def analyze_price_factors(self, crop: str, location: str, market_data: Dict) -> Dict:
        """Analyze factors affecting crop prices"""
        try:
            prompt = f"""
            Analyze the price factors affecting {crop} in {location} given this market data:
            
            Current Price: ₹{market_data.get('current_price', 0)}
            Price Trend: {market_data.get('trend', 'Stable')}
            Volume: {market_data.get('volume_traded', 0)} tonnes
            
            Identify and explain the key factors in JSON format:
            - supply_factors: (factors affecting supply)
            - demand_factors: (factors affecting demand)
            - seasonal_impact: (how seasons affect prices)
            - government_policies: (relevant policy impacts)
            - external_factors: (weather, global markets, etc.)
            - regional_factors: (location-specific factors)
            - quality_factors: (factors affecting crop quality and price)
            - market_dynamics: (trading patterns and market behavior)
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a commodity market analyst specializing in agricultural price dynamics and market factors in India."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            factors = json.loads(response.choices[0].message.content)
            return factors
            
        except Exception as e:
            print(f"OpenAI factors analysis error: {str(e)}")
            return self._get_fallback_factors(crop)
    
    def generate_market_summary(self, crop: str, location: str, all_data: Dict) -> str:
        """Generate a concise market summary"""
        try:
            prompt = f"""
            Create a concise market summary for {crop} in {location}:
            
            Market Data: {all_data.get('market_data', {})}
            Predictions: {all_data.get('predictions', [])}
            Analysis: {all_data.get('analysis', {})}
            
            Write a brief, professional summary (2-3 sentences) that a farmer can quickly understand.
            Focus on: current price situation, immediate outlook, and one key recommendation.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are writing for farmers who need clear, actionable market insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"OpenAI summary error: {str(e)}")
            return f"Market analysis for {crop} in {location} shows current trading activity with standard price movements. Monitor market conditions for optimal selling opportunities."
    
    def _get_fallback_analysis(self, crop: str, market_data: Dict, predictions: List[float]) -> Dict:
        """Fallback analysis when OpenAI is unavailable"""
        current_price = market_data.get('current_price', 0)
        yesterday_price = market_data.get('yesterday_price', current_price)
        
        price_change = ((current_price - yesterday_price) / yesterday_price * 100) if yesterday_price > 0 else 0
        
        sentiment = "Bullish" if price_change > 2 else "Bearish" if price_change < -2 else "Neutral"
        
        return {
            "overall_sentiment": sentiment,
            "key_insights": [
                f"Current {crop} price is ₹{current_price} per quintal",
                f"Price change: {price_change:+.1f}% from yesterday",
                f"Market showing {sentiment.lower()} sentiment",
                f"Trading volume indicates {'active' if market_data.get('volume_traded', 0) > 1000 else 'moderate'} market participation"
            ],
            "risk_factors": [
                "Weather dependency for crop quality",
                "Market volatility due to seasonal factors",
                "Government policy changes affecting pricing"
            ],
            "opportunities": [
                "Growing demand for quality produce",
                "Technology adoption for better yields",
                "Government support schemes available"
            ],
            "short_term_outlook": f"Prices expected to remain {sentiment.lower()} in the coming weeks",
            "medium_term_outlook": f"Seasonal patterns suggest moderate price movements for {crop}",
            "recommended_actions": [
                "Monitor daily price movements",
                "Consider market timing for sales",
                "Focus on quality improvement"
            ],
            "confidence_score": 75
        }
    
    def _get_fallback_recommendations(self, crop: str, location: str) -> Dict:
        """Fallback recommendations when OpenAI is unavailable"""
        return {
            "planting_recommendations": f"Follow optimal planting practices for {crop} in {location} climate",
            "harvest_timing": "Monitor crop maturity and market prices for optimal harvest timing",
            "storage_advice": "Use proper storage facilities to maintain quality and reduce post-harvest losses",
            "selling_strategy": "Consider market trends and seasonal patterns when planning sales",
            "risk_mitigation": "Diversify crops and consider crop insurance options",
            "alternative_crops": "Explore complementary crops suitable for your region",
            "technology_adoption": "Consider modern farming techniques and equipment",
            "financial_planning": "Maintain proper records and plan for seasonal cash flows"
        }
    
    def _get_fallback_factors(self, crop: str) -> Dict:
        """Fallback price factors when OpenAI is unavailable"""
        return {
            "supply_factors": ["Harvest timing", "Crop yield", "Storage capacity"],
            "demand_factors": ["Consumer demand", "Export requirements", "Industrial usage"],
            "seasonal_impact": f"Seasonal variations typical for {crop} cultivation and harvesting",
            "government_policies": ["MSP policies", "Export-import regulations", "Subsidies"],
            "external_factors": ["Weather conditions", "Global commodity prices", "Transportation costs"],
            "regional_factors": [f"Local market conditions in the region", "Infrastructure availability"],
            "quality_factors": ["Crop quality standards", "Grading and certification"],
            "market_dynamics": ["Trading patterns", "Market participation", "Price discovery"]
        }
