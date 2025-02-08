"""
API-avainten lataus config.json tiedostosta ja ympäristömuuttujista.

Käyttö:
1. Luo .env tiedosto projektin juurihakemistoon
2. Lisää seuraavat rivit .env tiedostoon:
   ```
   OPENAI_API_KEY=sk-your-openai-api-key-here
   HUGGINGFACE_API_KEY=hf_your-huggingface-api-key-here
   ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here
   ```
3. Korvaa API-avaimet omillasi

HUOM: Älä koskaan jaa .env tiedostoa tai lisää sitä versionhallintaan!
"""

import os
import json
from typing import Dict, Optional
from dotenv import load_dotenv
from user_messages import motivate_user

def create_env_template():
    """Luo .env.example tiedoston malliksi"""
    template = """# OpenAI API-avain (pakollinen)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Hugging Face API-avain (pakollinen)
HUGGINGFACE_API_KEY=hf_your-huggingface-api-key-here

# Anthropic API-avain (valinnainen)
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here
"""
    try:
        with open('.env.example', 'w') as f:
            f.write(template)
        print(".env.example tiedosto luotu onnistuneesti")
    except Exception as e:
        print(f"Virhe .env.example tiedoston luonnissa: {str(e)}")

def validate_api_key(key: Optional[str], prefix: str, name: str) -> bool:
    """Tarkistaa API-avaimen muodon"""
    if not key:
        print(f"VAROITUS: {name} API-avain puuttuu")
        return False
    if not key.startswith(prefix):
        print(f"VAROITUS: {name} API-avaimen tulee alkaa '{prefix}'")
        return False
    return True

def load_api_keys() -> Dict[str, str]:
    """Lataa API-avaimet config.json tiedostosta ja ympäristömuuttujista"""
    
    # Lataa .env tiedosto
    load_dotenv()
    
    # Lue config.json
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Kerää API-avaimet
    api_keys = {}
    
    # OpenAI
    if config["api"]["services"]["openai"]["enabled"]:
        openai_key = os.getenv("OPENAI_API_KEY")
        if validate_api_key(openai_key, "sk-", "OpenAI"):
            api_keys["gpt_4"] = openai_key
            api_keys["gpt_3_5"] = openai_key
    
    # Hugging Face
    if config["api"]["services"]["huggingface"]["enabled"]:
        hf_key = os.getenv("HUGGINGFACE_API_KEY")
        if validate_api_key(hf_key, "hf_", "Hugging Face"):
            api_keys["starcoder"] = hf_key
    
    # Anthropic (valinnainen)
    if config["api"]["services"]["anthropic"]["enabled"]:
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if validate_api_key(anthropic_key, "sk-ant-", "Anthropic"):
            api_keys["claude"] = anthropic_key
    
    # Tarkista että pakolliset avaimet löytyvät
    required_services = ["openai", "huggingface"]
    missing_keys = []
    
    for service in required_services:
        if service == "openai" and not (api_keys.get("gpt_4") and api_keys.get("gpt_3_5")):
            missing_keys.append("OpenAI")
        elif service == "huggingface" and not api_keys.get("starcoder"):
            missing_keys.append("Hugging Face")
    
    if missing_keys:
        raise ValueError(
            f"Puuttuvat pakolliset API-avaimet: {', '.join(missing_keys)}\n"
            "Varmista että .env tiedosto on olemassa ja sisältää oikeat avaimet."
        )
    
    print("API-avaimet ladattu onnistuneesti")
    motivate_user()
    return api_keys

def get_model_config(model_name: str) -> Dict:
    """Hakee mallin konfiguraation config.json tiedostosta"""
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    for service in config["api"]["services"].values():
        if model_name in service.get("models", {}):
            return service["models"][model_name]
    
    raise ValueError(f"Mallia {model_name} ei löydy konfiguraatiosta")

if __name__ == "__main__":
    try:
        # Luo .env.example tiedosto jos sitä ei ole
        if not os.path.exists('.env.example'):
            create_env_template()
        
        # Tarkista onko .env tiedosto olemassa
        if not os.path.exists('.env'):
            print("\nVAROITUS: .env tiedostoa ei löydy!")
            print("1. Kopioi .env.example tiedosto nimellä .env")
            print("2. Lisää omat API-avaimesi .env tiedostoon")
            print("3. Älä koskaan jaa .env tiedostoa tai lisää sitä versionhallintaan!\n")
        
        # Lataa ja näytä API-avaimet
        keys = load_api_keys()
        print("\nLadatut API-avaimet:")
        for service, key in keys.items():
            # Näytä vain avainten ensimmäiset ja viimeiset 4 merkkiä turvallisuuden vuoksi
            masked_key = f"{key[:4]}...{key[-4:]}" if key else "Puuttuu"
            print(f"{service}: {masked_key}")
            
            # Näytä mallin konfiguraatio
            try:
                model_config = get_model_config(service)
                print(f"  - Model ID: {model_config['model_id']}")
                print(f"  - Max Tokens: {model_config.get('max_tokens', model_config.get('max_length', 'N/A'))}")
                print(f"  - Temperature: {model_config.get('temperature', 'N/A')}")
            except ValueError:
                print("  - Konfiguraatiota ei löydy")
            
    except Exception as e:
        print(f"Virhe API-avainten lataamisessa: {str(e)}")
