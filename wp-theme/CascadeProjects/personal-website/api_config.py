"""
API-avainten ja mallien hallinta (VAIN KEHITYSYMPÄRISTÖÖN)
=======================================================

VAROITUS: Tämä on väliaikainen ratkaisu API-avainten hallintaan.
ÄLÄ käytä tätä tuotantoympäristössä!
"""

from typing import Dict, Optional
import warnings
from user_messages import motivate_user

# API-avainten määrittelyt (VAIN KEHITYSYMPÄRISTÖÖN)
API_KEYS: Dict[str, str] = {
    # OpenAI API-avaimet
    "gpt_4_turbo": "sk-4turb0123456789abcdefghijklmnopqrstuvwxyz",
    "gpt_3_5_turbo": "sk-35tur0123456789abcdefghijklmnopqrstuvwxyz",
    
    # Hugging Face API-avain
    "starcoder": "hf_starc0123456789abcdefghijklmnopqrstuvwxyz",
    
    # Anthropic API-avain
    "claude_opus": "sk-ant-0123456789abcdefghijklmnopqrstuvwxyz",
    
    # AutoGPT API-avain
    "autogpt": "ag-auto0123456789abcdefghijklmnopqrstuvwxyz"
}

# Mallien tunnukset ja tehtävätyypit
MODEL_CONFIG = {
    # GPT-4: Monimutkaiset analyysit ja tekstintuotto
    "gpt4": {
        "model_id": "gpt-4",
        "tasks": [
            "complex_analysis",
            "detailed_text_generation",
            "deep_analysis",
            "advanced_reasoning",
            "creative_writing"
        ]
    },
    
    # GPT-3.5: Nopeat ja yksinkertaiset tehtävät
    "gpt35": {
        "model_id": "gpt-3.5-turbo",
        "tasks": [
            "quick_responses",
            "simple_tasks",
            "basic_analysis",
            "chat_completion",
            "text_correction"
        ]
    },
    
    # Starcoder: Koodin generointi ja optimointi
    "starcoder": {
        "model_id": "bigcode/starcoder",
        "tasks": [
            "code_generation",
            "code_optimization",
            "code_review",
            "bug_fixing",
            "code_completion"
        ]
    },
    
    # Claude: Pitkät keskustelut ja kontekstuaalinen analyysi
    "claude": {
        "model_id": "claude-2.1",
        "tasks": [
            "contextual_analysis",
            "long_conversation",
            "document_analysis",
            "research_synthesis"
        ]
    }
}

class APIConfig:
    """API-avainten ja mallien hallinta (väliaikainen kehitysympäristön ratkaisu)"""
    
    def __init__(self):
        """Alustaa APIConfig:in ja näyttää kehitysympäristövaroituksen"""
        warnings.warn(
            "\nVAROITUS: Käytät väliaikaista API-avainten hallintaa!\n"
            "Tämä ratkaisu on tarkoitettu VAIN kehitysympäristöön.\n"
            "ÄLÄ käytä tätä ratkaisua tuotantoympäristössä.\n",
            RuntimeWarning
        )
        self._validate_keys()
    
    def _validate_keys(self) -> None:
        """Validoi API-avainten muodon"""
        # Tarkista OpenAI avaimet
        for key_name in ["gpt_4_turbo", "gpt_3_5_turbo"]:
            if not API_KEYS.get(key_name, "").startswith("sk-"):
                warnings.warn(f"{key_name} avaimen tulee alkaa 'sk-'")
        
        # Tarkista Hugging Face avain
        if not API_KEYS.get("starcoder", "").startswith("hf_"):
            warnings.warn("Starcoder avaimen tulee alkaa 'hf_'")
        
        # Tarkista Anthropic avain
        if not API_KEYS.get("claude_opus", "").startswith("sk-ant-"):
            warnings.warn("Claude avaimen tulee alkaa 'sk-ant-'")
        
        print("API-avaimet validoitu onnistuneesti")
        motivate_user()
    
    def get_api_key(self, service: str) -> str:
        """Hakee API-avaimen. Heittää varoituksen jos avainta ei löydy."""
        key_mapping = {
            'gpt4': 'gpt_4_turbo',
            'gpt3': 'gpt_3_5_turbo',
            'claude': 'claude_opus',
            'starcoder': 'starcoder',
            'autogpt': 'autogpt'
        }
        
        if service not in key_mapping:
            warnings.warn(f"Tuntematon palvelu: {service}")
            return ""
            
        mapped_service = key_mapping[service]
        if mapped_service not in API_KEYS:
            warnings.warn(f"API-avainta ei löydy palvelulle: {service}")
            return ""
            
        return API_KEYS[mapped_service]
    
    def get_model_for_task(self, task_type: str) -> Optional[Dict[str, str]]:
        """Hakee sopivan mallin tehtävätyypille"""
        for model_name, config in MODEL_CONFIG.items():
            if task_type in config["tasks"]:
                return {
                    "model": model_name,
                    "model_id": config["model_id"]
                }
        return None
    
    def list_models(self) -> None:
        """Listaa kaikki mallit ja niiden tehtävätyypit"""
        print("\nMallit ja tehtävätyypit:")
        for model_name, config in MODEL_CONFIG.items():
            print(f"\n{model_name.upper()} ({config['model_id']}):")
            for task in config["tasks"]:
                print(f"  - {task}")
        
        motivate_user()
    
    def list_keys(self) -> None:
        """Listaa kaikki API-avaimet (näyttää vain alkuosan)"""
        print("\nAPI-avaimet:")
        for key_name, value in API_KEYS.items():
            # Näytä vain avainten alkuosa turvallisuussyistä
            print(f"{key_name}: {value[:8]}...")
        
        print("\nMallien tunnukset:")
        for model_name, config in MODEL_CONFIG.items():
            print(f"{model_name}: {config['model_id']}")
        
        motivate_user()

# Globaali instanssi
api_config = APIConfig()
