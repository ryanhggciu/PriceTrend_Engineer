# AgriMarket AI - Agricultural Market Analysis

## Overview

AgriMarket AI is a comprehensive agricultural market analysis platform designed to help farmers and agricultural stakeholders make informed decisions. The system provides real-time market data, price predictions using machine learning, AI-powered market analysis, and government scheme recommendations. Built with Streamlit for the frontend, the application integrates multiple data sources and AI services to deliver actionable insights for Indian agricultural markets.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with wide layout configuration
- **UI Components**: Interactive dashboards with Plotly visualizations for market trends and price charts
- **Accessibility**: Text-to-speech functionality for content accessibility
- **Caching**: Streamlit resource caching for service initialization to improve performance

### Backend Architecture
- **Service-Oriented Design**: Modular service architecture with clear separation of concerns
  - `DataService`: Handles market data fetching with fallback mechanisms
  - `OpenAIAnalysisService`: Provides AI-powered market analysis
  - `GovernmentSchemeService`: Manages government scheme recommendations
  - `DataProcessor`: Handles data processing and preparation for ML models
- **ML Pipeline**: `MarketPredictor` class implementing ensemble learning with multiple algorithms (Random Forest, Gradient Boosting, Linear Regression)
- **Data Processing**: Comprehensive historical data generation with seasonal patterns and market fluctuations

### Machine Learning Components
- **Ensemble Approach**: Multiple ML models (Random Forest, Gradient Boosting, Linear Regression) with automatic best model selection
- **Feature Engineering**: Time-series features, seasonal patterns, and market indicators
- **Model Evaluation**: Cross-validation with multiple metrics (MAE, MSE, R²)
- **Prediction Pipeline**: Short-term price forecasting (3-day predictions) with confidence scoring

### Data Management
- **Multi-Source Integration**: Government APIs (data.gov.in), weather APIs, and market data sources
- **Caching Strategy**: Time-based cache with expiry for API responses
- **Fallback Mechanisms**: Realistic data simulation when APIs are unavailable
- **Data Validation**: Comprehensive error handling and data quality checks

### AI Integration
- **OpenAI Integration**: GPT-5 model for comprehensive market analysis
- **Structured Outputs**: JSON-formatted AI responses for consistent data processing
- **Market Intelligence**: Sentiment analysis, risk assessment, and opportunity identification
- **Confidence Scoring**: AI-generated confidence levels for analysis reliability

## External Dependencies

### AI and ML Services
- **OpenAI API**: GPT-5 model integration for market analysis and insights generation
- **Scikit-learn**: Machine learning algorithms and preprocessing utilities
- **NumPy/Pandas**: Data manipulation and numerical computing

### Visualization and UI
- **Streamlit**: Web application framework and hosting platform
- **Plotly**: Interactive charts and data visualization components
- **Streamlit Components**: Custom UI components and integrations

### Data Sources
- **Government APIs**: 
  - Indian Government Open Data Portal (data.gov.in) for agricultural commodity data
  - Agricultural market price APIs for real-time pricing information
- **Weather APIs**: OpenWeatherMap integration for weather-based market factors
- **Market Data**: Multiple agricultural market data providers with fallback mechanisms

### Configuration Management
- **Environment Variables**: Secure API key management for external services
- **Static Data**: Predefined crop categories, Indian states mapping, and government schemes database
- **Settings Configuration**: Centralized configuration management for API endpoints and application settings

### Data Processing
- **Time Series Analysis**: Historical data processing with seasonal pattern recognition
- **Statistical Libraries**: Advanced data processing and feature engineering capabilities
- **Model Persistence**: Joblib for ML model serialization and deployment