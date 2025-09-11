import os

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "default_openai_key")
AGRICULTURAL_DATA_API_KEY = os.getenv("AGRICULTURAL_DATA_API_KEY", "default_agri_key")

# API Endpoints
CROP_PRICE_API_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"

# Crop categories and government schemes mapping
CROP_CATEGORIES = {
    "Cereals": ["Rice", "Wheat", "Maize", "Barley", "Millet"],
    "Pulses": ["Chickpea", "Lentil", "Black Gram", "Green Gram", "Pigeon Pea"],
    "Cash Crops": ["Cotton", "Sugarcane", "Jute", "Tobacco"],
    "Oilseeds": ["Groundnut", "Mustard", "Sunflower", "Sesame", "Safflower"],
    "Spices": ["Turmeric", "Coriander", "Cumin", "Fenugreek", "Black Pepper"],
    "Fruits": ["Mango", "Apple", "Banana", "Orange", "Grapes"],
    "Vegetables": ["Tomato", "Onion", "Potato", "Cabbage", "Cauliflower"]
}

# Indian states for location selection
INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
]
