"""
Cascade konfiguraation hallinta.
Tämä moduuli vastaa Cascade-järjestelmän konfiguraation hallinnasta ja validoinnista.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Konfiguroi lokitus
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AnthropicConfig:
    """Anthropic API konfiguraatio"""
    enabled: bool
    api_key: str
    default_model: str
    batch_mode: bool
    batch_limit: int
    models: Dict[str, Any]
    rate_limits: Dict[str, Any]
    retry_settings: Dict[str, Any]

class CascadeConfig:
    """Cascade konfiguraation hallinta"""
    
    def __init__(self, config_file: str = "config.json"):
        """
        Alusta Cascade konfiguraatio
        
        Args:
            config_file: Polku konfiguraatiotiedostoon
        """
        self.config_file = config_file
        self.config = self._load_config()
        self.anthropic = self._init_anthropic_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Lataa konfiguraatio tiedostosta"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Virhe konfiguraation latauksessa: {str(e)}")
            raise
    
    def _init_anthropic_config(self) -> AnthropicConfig:
        """Alusta Anthropic konfiguraatio"""
        anthropic_config = self.config.get('anthropic', {})
        
        return AnthropicConfig(
            enabled=anthropic_config.get('enabled', False),
            api_key=os.getenv('ANTHROPIC_API_KEY', ''),
            default_model=anthropic_config.get('default_model', 'claude-3-opus-20240229'),
            batch_mode=anthropic_config.get('batch_mode', True),
            batch_limit=anthropic_config.get('batch_limit', 15),
            models=anthropic_config.get('models', {}),
            rate_limits=anthropic_config.get('rate_limits', {}),
            retry_settings=anthropic_config.get('retry_settings', {})
        )
    
    def validate_config(self) -> bool:
        """
        Validoi konfiguraatio
        
        Returns:
            bool: True jos konfiguraatio on validi
        """
        try:
            # Tarkista pakolliset kentät
            required_fields = ['api_keys', 'models', 'task_allocation']
            for field in required_fields:
                if field not in self.config:
                    raise ValueError(f"Pakollinen kenttä puuttuu: {field}")
            
            # Tarkista Anthropic konfiguraatio
            if self.anthropic.enabled:
                if not self.anthropic.api_key:
                    raise ValueError("Anthropic API avain puuttuu")
                
                if not self.anthropic.models:
                    raise ValueError("Anthropic mallit puuttuvat")
            
            # Tarkista mallikohtaiset asetukset
            if 'claude-3-opus' in self.anthropic.models:
                model_config = self.anthropic.models['claude-3-opus']
                required_model_fields = ['id', 'context_window', 'max_tokens']
                for field in required_model_fields:
                    if field not in model_config:
                        raise ValueError(f"Mallin pakollinen kenttä puuttuu: {field}")
            
            logger.info("Konfiguraatio validoitu onnistuneesti")
            return True
            
        except Exception as e:
            logger.error(f"Konfiguraation validointi epäonnistui: {str(e)}")
            return False
    
    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Hae mallin konfiguraatio
        
        Args:
            model_name: Mallin nimi
        
        Returns:
            Dict tai None: Mallin konfiguraatio tai None jos ei löydy
        """
        if model_name in self.anthropic.models:
            return self.anthropic.models[model_name]
        return None
    
    def update_anthropic_settings(
        self,
        enabled: Optional[bool] = None,
        batch_mode: Optional[bool] = None,
        batch_limit: Optional[int] = None
    ) -> bool:
        """
        Päivitä Anthropic asetuksia
        
        Args:
            enabled: Onko käytössä
            batch_mode: Onko eräajo käytössä
            batch_limit: Eräajon raja
        
        Returns:
            bool: True jos päivitys onnistui
        """
        try:
            if enabled is not None:
                self.config['anthropic']['enabled'] = enabled
            
            if batch_mode is not None:
                self.config['anthropic']['batch_mode'] = batch_mode
            
            if batch_limit is not None:
                self.config['anthropic']['batch_limit'] = batch_limit
            
            # Tallenna muutokset
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
            
            # Päivitä sisäinen tila
            self.anthropic = self._init_anthropic_config()
            
            logger.info("Anthropic asetukset päivitetty onnistuneesti")
            return True
            
        except Exception as e:
            logger.error(f"Virhe Anthropic asetusten päivityksessä: {str(e)}")
            return False

def main():
    """Testaa konfiguraation toiminta"""
    config = CascadeConfig()
    
    print("\nTarkistetaan konfiguraatio...")
    if config.validate_config():
        print("[OK] Konfiguraatio on validi")
        
        print("\nAnthropic asetukset:")
        print(f"Enabled: {config.anthropic.enabled}")
        print(f"Default Model: {config.anthropic.default_model}")
        print(f"Batch Mode: {config.anthropic.batch_mode}")
        print(f"Batch Limit: {config.anthropic.batch_limit}")
        
        if 'claude-3-opus' in config.anthropic.models:
            model = config.anthropic.models['claude-3-opus']
            print(f"\nClaude-3 Opus asetukset:")
            print(f"Model ID: {model['id']}")
            print(f"Context Window: {model['context_window']}")
            print(f"Max Tokens: {model['max_tokens']}")
    else:
        print("[ERROR] Konfiguraatiossa on virheita")

if __name__ == "__main__":
    main()
