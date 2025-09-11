from typing import Dict, List
import json

class GovernmentSchemeService:
    """Service for providing government scheme recommendations"""
    
    def __init__(self):
        self.schemes_data = self._load_schemes_data()
    
    def get_relevant_schemes(self, crop: str, location: str, farmer_profile: Dict = None) -> List[Dict]:
        """Get government schemes relevant to the crop and location"""
        relevant_schemes = []
        
        # Filter schemes by crop type
        crop_category = self._get_crop_category(crop)
        
        for scheme in self.schemes_data:
            if self._is_scheme_applicable(scheme, crop, crop_category, location, farmer_profile):
                relevant_schemes.append(scheme)
        
        # Sort by relevance score
        relevant_schemes.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return relevant_schemes[:10]  # Return top 10 most relevant schemes
    
    def get_scheme_details(self, scheme_id: str) -> Dict:
        """Get detailed information about a specific scheme"""
        for scheme in self.schemes_data:
            if scheme.get('id') == scheme_id:
                return scheme
        return {}
    
    def _get_crop_category(self, crop: str) -> str:
        """Categorize crop for scheme matching"""
        crop_categories = {
            "Cereals": ["Rice", "Wheat", "Maize", "Barley", "Millet"],
            "Pulses": ["Chickpea", "Lentil", "Black Gram", "Green Gram", "Pigeon Pea"],
            "Cash Crops": ["Cotton", "Sugarcane", "Jute", "Tobacco"],
            "Oilseeds": ["Groundnut", "Mustard", "Sunflower", "Sesame", "Safflower"],
            "Spices": ["Turmeric", "Coriander", "Cumin", "Fenugreek", "Black Pepper"],
            "Fruits": ["Mango", "Apple", "Banana", "Orange", "Grapes"],
            "Vegetables": ["Tomato", "Onion", "Potato", "Cabbage", "Cauliflower"]
        }
        
        for category, crops in crop_categories.items():
            if crop in crops:
                return category
        return "General"
    
    def _is_scheme_applicable(self, scheme: Dict, crop: str, crop_category: str, location: str, farmer_profile: Dict) -> bool:
        """Check if a scheme is applicable to the given criteria"""
        # Check crop applicability
        applicable_crops = scheme.get('applicable_crops', [])
        if applicable_crops and crop not in applicable_crops and crop_category not in applicable_crops:
            return False
        
        # Check location applicability
        applicable_states = scheme.get('applicable_states', [])
        if applicable_states and location not in applicable_states and 'All India' not in applicable_states:
            return False
        
        # Check farmer profile if provided
        if farmer_profile:
            # Check farm size eligibility
            max_farm_size = scheme.get('max_farm_size_hectares')
            if max_farm_size and farmer_profile.get('farm_size_hectares', 0) > max_farm_size:
                return False
            
            # Check category eligibility
            eligible_categories = scheme.get('eligible_farmer_categories', [])
            farmer_category = farmer_profile.get('category', 'General')
            if eligible_categories and farmer_category not in eligible_categories:
                return False
        
        return True
    
    def _load_schemes_data(self) -> List[Dict]:
        """Load government schemes data"""
        return [
            {
                "id": "pmkisan",
                "name": "PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)",
                "description": "Direct income support to farmer families",
                "benefits": "₹6,000 per year in three installments of ₹2,000 each",
                "eligibility": "All landholding farmer families",
                "applicable_crops": ["All"],
                "applicable_states": ["All India"],
                "application_process": "Online registration through PM-KISAN portal or CSC centers",
                "contact_info": "14447 (Toll-free helpline)",
                "website": "https://pmkisan.gov.in",
                "relevance_score": 95,
                "scheme_type": "Direct Benefit Transfer",
                "status": "Active"
            },
            {
                "id": "pmfby",
                "name": "Pradhan Mantri Fasal Bima Yojana (PMFBY)",
                "description": "Crop insurance scheme providing financial support against crop loss",
                "benefits": "Insurance coverage against natural calamities, pests, and diseases",
                "eligibility": "All farmers growing notified crops",
                "applicable_crops": ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Groundnut", "All"],
                "applicable_states": ["All India"],
                "premium_rates": "Kharif: 2%, Rabi: 1.5%, Commercial/Horticultural: 5%",
                "application_process": "Through banks, CSCs, or insurance companies",
                "contact_info": "Agriculture department of respective states",
                "website": "https://pmfby.gov.in",
                "relevance_score": 90,
                "scheme_type": "Insurance",
                "status": "Active"
            },
            {
                "id": "pkvy",
                "name": "Paramparagat Krishi Vikas Yojana (PKVY)",
                "description": "Promoting organic farming through cluster approach",
                "benefits": "₹50,000 per hectare over 3 years for organic farming",
                "eligibility": "Farmers forming clusters of 50 acres",
                "applicable_crops": ["All organic crops"],
                "applicable_states": ["All India"],
                "application_process": "Through state agriculture departments",
                "contact_info": "State agriculture departments",
                "website": "Official agriculture department websites",
                "relevance_score": 80,
                "scheme_type": "Organic Farming Support",
                "status": "Active"
            },
            {
                "id": "soil_health_card",
                "name": "Soil Health Card Scheme",
                "description": "Providing soil health cards to farmers for better nutrient management",
                "benefits": "Free soil testing and recommendations for balanced fertilizer use",
                "eligibility": "All farmers",
                "applicable_crops": ["All"],
                "applicable_states": ["All India"],
                "application_process": "Through state agriculture departments and soil testing labs",
                "contact_info": "Local agriculture extension officers",
                "website": "https://soilhealth.dac.gov.in",
                "relevance_score": 85,
                "scheme_type": "Soil Management",
                "status": "Active"
            },
            {
                "id": "kisan_credit_card",
                "name": "Kisan Credit Card (KCC)",
                "description": "Credit facility for farmers to meet production needs",
                "benefits": "Flexible credit facility with lower interest rates",
                "eligibility": "Farmers with land ownership documents",
                "applicable_crops": ["All"],
                "applicable_states": ["All India"],
                "interest_rate": "7% per annum (with 3% subvention)",
                "application_process": "Through banks and cooperative societies",
                "contact_info": "Local bank branches",
                "website": "Bank websites and agriculture portals",
                "relevance_score": 88,
                "scheme_type": "Credit Facility",
                "status": "Active"
            },
            {
                "id": "msp_scheme",
                "name": "Minimum Support Price (MSP)",
                "description": "Guaranteed minimum price for specified crops",
                "benefits": "Assured income through minimum price guarantee",
                "eligibility": "Farmers producing notified crops",
                "applicable_crops": ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Groundnut"],
                "applicable_states": ["All India"],
                "application_process": "Sale through FCI and state agencies",
                "contact_info": "Local procurement centers",
                "website": "https://fci.gov.in",
                "relevance_score": 92,
                "scheme_type": "Price Support",
                "status": "Active"
            },
            {
                "id": "micro_irrigation",
                "name": "Pradhan Mantri Krishi Sinchayee Yojana (PMKSY)",
                "description": "Micro irrigation support for water conservation",
                "benefits": "Subsidy for drip and sprinkler irrigation systems",
                "eligibility": "All categories of farmers",
                "applicable_crops": ["All"],
                "applicable_states": ["All India"],
                "subsidy_rate": "55% for general farmers, 75% for SC/ST farmers",
                "application_process": "Through state irrigation departments",
                "contact_info": "State irrigation departments",
                "website": "Official state government websites",
                "relevance_score": 83,
                "scheme_type": "Irrigation Support",
                "status": "Active"
            },
            {
                "id": "agri_marketing",
                "name": "Agricultural Marketing Infrastructure (AMI)",
                "description": "Support for agricultural marketing infrastructure",
                "benefits": "Subsidy for cold storage, warehouses, and market infrastructure",
                "eligibility": "Farmers, FPOs, cooperatives, and private entrepreneurs",
                "applicable_crops": ["All"],
                "applicable_states": ["All India"],
                "subsidy_rate": "35% of project cost (up to specified limits)",
                "application_process": "Through NABARD and state nodal agencies",
                "contact_info": "NABARD district offices",
                "website": "https://nabard.org",
                "relevance_score": 78,
                "scheme_type": "Infrastructure Support",
                "status": "Active"
            },
            {
                "id": "kisan_rail",
                "name": "Kisan Rail",
                "description": "Dedicated rail service for transporting agricultural products",
                "benefits": "Fast and efficient transportation of perishable goods",
                "eligibility": "All farmers and agricultural traders",
                "applicable_crops": ["Fruits", "Vegetables", "Perishable items"],
                "applicable_states": ["Selected routes across India"],
                "application_process": "Through Indian Railways booking system",
                "contact_info": "Railway booking offices",
                "website": "https://indianrailways.gov.in",
                "relevance_score": 75,
                "scheme_type": "Transportation Support",
                "status": "Active"
            },
            {
                "id": "digital_agriculture",
                "name": "Digital Agriculture Mission",
                "description": "Promoting digital technology adoption in agriculture",
                "benefits": "Support for digital tools, AI, and precision agriculture",
                "eligibility": "Progressive farmers and agri-entrepreneurs",
                "applicable_crops": ["All"],
                "applicable_states": ["All India"],
                "application_process": "Through state agriculture departments",
                "contact_info": "State agriculture departments",
                "website": "Official agriculture department websites",
                "relevance_score": 70,
                "scheme_type": "Technology Support",
                "status": "Active"
            }
        ]
    
    def get_scheme_application_guide(self, scheme_id: str) -> Dict:
        """Get step-by-step application guide for a scheme"""
        scheme = self.get_scheme_details(scheme_id)
        if not scheme:
            return {}
        
        guides = {
            "pmkisan": {
                "steps": [
                    "Visit PM-KISAN official website or nearest CSC center",
                    "Click on 'Farmers Corner' and select 'New Farmer Registration'",
                    "Enter Aadhaar number and mobile number",
                    "Fill in personal and land details",
                    "Upload required documents (Aadhaar, land documents)",
                    "Submit application and note the registration number",
                    "Check status periodically on the website"
                ],
                "required_documents": [
                    "Aadhaar Card",
                    "Land ownership documents",
                    "Bank account details",
                    "Mobile number"
                ],
                "processing_time": "15-30 days after verification"
            },
            "pmfby": {
                "steps": [
                    "Visit nearest bank branch or insurance company office",
                    "Fill the crop insurance application form",
                    "Provide crop details and sowing information",
                    "Pay the premium amount",
                    "Obtain insurance certificate",
                    "Report crop loss immediately if any damage occurs"
                ],
                "required_documents": [
                    "Aadhaar Card",
                    "Land documents",
                    "Bank account details",
                    "Sowing certificate"
                ],
                "processing_time": "Immediate upon premium payment"
            }
        }
        
        return guides.get(scheme_id, {
            "steps": ["Contact local agriculture department for application process"],
            "required_documents": ["Aadhaar Card", "Land documents", "Bank details"],
            "processing_time": "Varies by scheme"
        })
