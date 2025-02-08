import os
from dataclasses import dataclass
from typing import Dict, Optional
from user_messages import motivate_user
from api_keys import key_manager

@dataclass
class ModelConfig:
    """Mallin konfiguraatiotiedot"""
    model_id: str
    api_key: str
    max_tokens: int = 2000
    temperature: float = 0.7

class ModelManager:
    def __init__(self):
        self._load_model_configs()
        
    def _load_model_configs(self):
        """Lataa mallien konfiguraatiot KeyManagerista"""
        self.models: Dict[str, ModelConfig] = {
            "gpt4": ModelConfig(
                model_id=key_manager.get_key("gpt4_model"),
                api_key=key_manager.get_key("openai"),
                max_tokens=4000,
                temperature=0.7
            ),
            "gpt35": ModelConfig(
                model_id=key_manager.get_key("gpt35_model"),
                api_key=key_manager.get_key("openai"),
                max_tokens=4000,
                temperature=0.8
            ),
            "starcoder": ModelConfig(
                model_id=key_manager.get_key("starcoder_model"),
                api_key=key_manager.get_key("huggingface"),
                max_tokens=2048,
                temperature=0.5
            )
        }
        
        # Tarkista puuttuvat API-avaimet
        missing_keys = []
        for model_name, config in self.models.items():
            if not config.api_key:
                missing_keys.append(f"{model_name} (API key)")
            if not config.model_id:
                missing_keys.append(f"{model_name} (Model ID)")
        
        if missing_keys:
            print(f"Varoitus: Puuttuvat konfiguraatiot: {', '.join(missing_keys)}")
        else:
            print("Kaikki mallien konfiguraatiot ladattu onnistuneesti.")
            motivate_user()
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Hae mallin konfiguraatio"""
        if model_name not in self.models:
            print(f"Varoitus: Mallia {model_name} ei löydy konfiguraatiosta")
            return None
        return self.models[model_name]
    
    def update_model_config(self, model_name: str, **kwargs):
        """Päivitä mallin konfiguraatiota"""
        if model_name not in self.models:
            raise ValueError(f"Mallia {model_name} ei löydy")
            
        config = self.models[model_name]
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Virheellinen konfiguraatioparametri: {key}")
        
        print(f"Mallin {model_name} konfiguraatio päivitetty")
        motivate_user()
    
    def list_available_models(self):
        """Listaa kaikki saatavilla olevat mallit ja niiden konfiguraatiot"""
        print("\nSaatavilla olevat mallit:")
        for model_name, config in self.models.items():
            print(f"\n{model_name}:")
            print(f"  - Model ID: {config.model_id}")
            print(f"  - Max Tokens: {config.max_tokens}")
            print(f"  - Temperature: {config.temperature}")
            # API-avainta ei tulosteta turvallisuussyistä
        motivate_user()
