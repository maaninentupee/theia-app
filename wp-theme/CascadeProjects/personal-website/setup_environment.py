"""
Ympäristön asetukset ja API-avainten hallinta.
Tämä tiedosto vastaa:
1. API-avainten validoinnista
2. Konfiguraatiotiedostojen luonnista ja päivityksestä
3. Agenttien asetusten hallinnasta
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Konfiguroi lokitus
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnvironmentSetup:
    """Hallinnoi ympäristön asetuksia ja API-avaimia"""
    
    def __init__(self):
        """Alusta ympäristön asetukset"""
        self.env_file = ".env"
        self.config_file = "config.json"
        self.agent_config_file = "agent_config.json"
        self.required_dirs = ["logs", "cache", "temp"]
        
        # Lataa ympäristömuuttujat
        load_dotenv()
        
        # Luo tarvittavat hakemistot
        self._create_directories()
    
    def _create_directories(self):
        """Luo tarvittavat hakemistot"""
        for dir_name in self.required_dirs:
            os.makedirs(dir_name, exist_ok=True)
            logger.info(f"Hakemisto varmistettu: {dir_name}")
    
    def check_and_create_env(self) -> bool:
        """
        Tarkista ja luo .env tiedosto jos ei ole olemassa
        
        Returns:
            bool: True jos onnistui, False jos virhe
        """
        try:
            if not os.path.exists(self.env_file):
                with open(self.env_file, "w", encoding='utf-8') as f:
                    f.write("""# API Keys
# OpenAI API avain (pakollinen)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Hugging Face API avain (pakollinen)
HUGGINGFACE_API_KEY=hf-your-huggingface-api-key-here

# Anthropic API avain (valinnainen)
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here

# Mallikohtaiset asetukset
GPT4_MODEL=gpt-4
GPT35_MODEL=gpt-3.5-turbo
STARCODER_MODEL=bigcode/starcoder
CLAUDE_MODEL=claude-3-opus-20240229
CLAUDE_BATCH_API_URL=https://api.anthropic.com/v1/batch

# Claude Batch API asetukset
CLAUDE_BATCH_MAX_TOKENS=4096
CLAUDE_BATCH_TEMPERATURE=0.7
CLAUDE_BATCH_TOP_P=0.9
CLAUDE_BATCH_MAX_RETRIES=3
CLAUDE_BATCH_TIMEOUT=300
""")
                logger.info(f"{self.env_file} luotu. Lisää API-avaimet tiedostoon.")
            else:
                logger.info(f"{self.env_file} on jo olemassa.")
            
            return True
        
        except Exception as e:
            logger.error(f"Virhe .env tiedoston luonnissa: {str(e)}")
            return False
    
    def update_config(self) -> bool:
        """
        Päivitä config.json tiedosto
        
        Returns:
            bool: True jos onnistui, False jos virhe
        """
        try:
            config_data = {
                "api_keys": {
                    "openai": "${OPENAI_API_KEY}",
                    "huggingface": "${HUGGINGFACE_API_KEY}",
                    "anthropic": "${ANTHROPIC_API_KEY}"
                },
                "anthropic": {
                    "enabled": True,
                    "default_model": "claude-3-opus-20240229",
                    "batch_mode": True,
                    "batch_limit": 15,
                    "models": {
                        "claude-3-opus": {
                            "id": "claude-3-opus-20240229",
                            "context_window": 200000,
                            "max_tokens": 4096,
                            "supports_batch": True,
                            "batch_settings": {
                                "max_batch_size": 15,
                                "max_concurrent_batches": 3,
                                "batch_timeout": 300
                            }
                        }
                    },
                    "rate_limits": {
                        "requests_per_minute": 50,
                        "tokens_per_minute": 100000,
                        "batch_cooldown": 1.0,
                        "concurrent_requests": 5
                    },
                    "retry_settings": {
                        "max_retries": 3,
                        "initial_delay": 1,
                        "max_delay": 10,
                        "backoff_factor": 2
                    }
                },
                "models": {
                    "gpt_4": os.getenv("GPT4_MODEL", "gpt-4"),
                    "gpt_3_5": os.getenv("GPT35_MODEL", "gpt-3.5-turbo"),
                    "starcoder": os.getenv("STARCODER_MODEL", "bigcode/starcoder"),
                    "claude": os.getenv("CLAUDE_MODEL", "claude-3-opus-20240229")
                },
                "claude_batch": {
                    "api_url": os.getenv("CLAUDE_BATCH_API_URL"),
                    "max_tokens": int(os.getenv("CLAUDE_BATCH_MAX_TOKENS", "4096")),
                    "temperature": float(os.getenv("CLAUDE_BATCH_TEMPERATURE", "0.7")),
                    "top_p": float(os.getenv("CLAUDE_BATCH_TOP_P", "0.9")),
                    "max_retries": int(os.getenv("CLAUDE_BATCH_MAX_RETRIES", "3")),
                    "timeout": int(os.getenv("CLAUDE_BATCH_TIMEOUT", "300")),
                    "batch_size": 10,
                    "concurrent_requests": 5,
                    "rate_limit": {
                        "requests_per_minute": 50,
                        "tokens_per_minute": 100000
                    }
                },
                "task_allocation": {
                    "priority_order": ["claude-3-opus", "gpt_4", "starcoder", "gpt_3_5"],
                    "fallback_model": "gpt_3_5",
                    "model_specializations": {
                        "claude-3-opus": [
                            "batch_processing",
                            "long_context",
                            "complex_reasoning",
                            "code_generation",
                            "technical_analysis"
                        ]
                    }
                },
                "performance": {
                    "max_threads": 16,
                    "memory_limit_mb": 8192,
                    "cache_enabled": True,
                    "cache_dir": "cache"
                },
                "logging": {
                    "enabled": True,
                    "level": "INFO",
                    "file": "logs/app.log",
                    "max_size_mb": 100,
                    "backup_count": 5
                }
            }
            
            with open(self.config_file, "w", encoding='utf-8') as f:
                json.dump(config_data, f, indent=4)
            
            logger.info(f"{self.config_file} päivitetty onnistuneesti.")
            return True
        
        except Exception as e:
            logger.error(f"Virhe config.json päivityksessä: {str(e)}")
            return False
    
    def update_agent_settings(self) -> bool:
        """
        Päivitä agenttien asetukset
        
        Returns:
            bool: True jos onnistui, False jos virhe
        """
        try:
            agent_config = {
                "cascade": {
                    "enabled": True,
                    "max_threads": 16,
                    "memory_limit_mb": 8192,
                    "task_types": [
                        "code_generation",
                        "code_review",
                        "debugging",
                        "optimization"
                    ]
                },
                "autogpt": {
                    "enabled": True,
                    "task_types": [
                        "idea_generation",
                        "workflow_management",
                        "research",
                        "planning"
                    ],
                    "models": ["gpt_4", "claude"]
                },
                "theia": {
                    "enabled": True,
                    "api_models": ["gpt_4", "starcoder", "claude"],
                    "features": [
                        "code_completion",
                        "error_detection",
                        "refactoring",
                        "documentation"
                    ]
                }
            }
            
            with open(self.agent_config_file, "w", encoding='utf-8') as f:
                json.dump(agent_config, f, indent=4)
            
            logger.info(f"{self.agent_config_file} päivitetty onnistuneesti.")
            return True
        
        except Exception as e:
            logger.error(f"Virhe agent_config.json päivityksessä: {str(e)}")
            return False
    
    def test_api_keys(self) -> bool:
        """
        Testaa API-avainten oikeellisuus
        
        Returns:
            bool: True jos kaikki pakolliset avaimet ovat kunnossa
        """
        errors: List[str] = []
        
        # Tarkista OpenAI API avain
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key.startswith("sk-"):
            logger.info("✅ OpenAI API avain on kelvollinen.")
        else:
            error = "❌ OpenAI API avain puuttuu tai on virheellinen."
            errors.append(error)
            logger.error(error)
        
        # Tarkista Hugging Face API avain
        hf_key = os.getenv("HUGGINGFACE_API_KEY", "")
        if hf_key.startswith("hf-"):
            logger.info("✅ Hugging Face API avain on kelvollinen.")
        else:
            error = "❌ Hugging Face API avain puuttuu tai on virheellinen."
            errors.append(error)
            logger.error(error)
        
        # Tarkista Anthropic API avain (valinnainen)
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        if anthropic_key.startswith("sk-ant-"):
            logger.info("✅ Anthropic API avain on kelvollinen.")
        else:
            logger.warning("⚠️ Anthropic API avain puuttuu (valinnainen).")
        
        if errors:
            logger.error("\nVirheet:")
            for error in errors:
                logger.error(error)
            logger.error("\nPäivitä .env tiedosto oikeilla API-avaimilla.")
            return False
        
        return True
    
    def setup(self) -> bool:
        """
        Suorita koko ympäristön asennus
        
        Returns:
            bool: True jos kaikki vaiheet onnistuivat
        """
        logger.info("Aloitetaan ympäristön asennus...")
        
        # Vaihe 1: Tarkista ja luo .env
        logger.info("\nVaihe 1: Tarkistetaan .env tiedosto...")
        if not self.check_and_create_env():
            return False
        
        # Vaihe 2: Päivitä config.json
        logger.info("\nVaihe 2: Päivitetään config.json...")
        if not self.update_config():
            return False
        
        # Vaihe 3: Validoi API-avaimet
        logger.info("\nVaihe 3: Validoidaan API-avaimet...")
        if not self.test_api_keys():
            return False
        
        # Vaihe 4: Päivitä agenttien asetukset
        logger.info("\nVaihe 4: Päivitetään agenttien asetukset...")
        if not self.update_agent_settings():
            return False
        
        logger.info("\nYmpäristön asennus valmis!")
        return True

def main():
    """Pääfunktio ympäristön asennukseen"""
    setup = EnvironmentSetup()
    if setup.setup():
        print("\nYmpäristö on nyt valmis käyttöön!")
    else:
        print("\nYmpäristön asennuksessa ilmeni ongelmia. Tarkista virhelokit.")

if __name__ == "__main__":
    main()
