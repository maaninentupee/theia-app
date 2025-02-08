import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from dotenv import load_dotenv

# Konfiguroi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KeyVaultManager:
    """Azure Key Vault -pohjainen avainten hallinta"""
    
    def __init__(self, vault_url: str):
        """
        Alusta Key Vault -yhteys
        
        Args:
            vault_url: Key Vault URL
        """
        self.vault_url = vault_url
        self.credential = DefaultAzureCredential()
        self.client = SecretClient(
            vault_url=vault_url,
            credential=self.credential
        )
        
        # Välimuisti avaimille
        self._cache: Dict[str, Dict] = {}
        self._last_refresh = datetime.now()
        self.refresh_interval = timedelta(minutes=5)
    
    async def get_api_key(
        self,
        service_name: str,
        force_refresh: bool = False
    ) -> Optional[str]:
        """
        Hae API-avain
        
        Args:
            service_name: Palvelun nimi
            force_refresh: Pakota päivitys
        
        Returns:
            str: API-avain tai None
        """
        cache_key = f"{service_name}_API_KEY"
        
        # Tarkista välimuisti
        if not force_refresh:
            cached = self._get_from_cache(cache_key)
            if cached:
                return cached
        
        try:
            # Hae avain Key Vaultista
            secret = await asyncio.to_thread(
                self.client.get_secret,
                cache_key
            )
            
            # Päivitä välimuisti
            self._update_cache(
                cache_key,
                secret.value,
                secret.properties.expires_on
            )
            
            return secret.value
        
        except Exception as e:
            logger.error(
                f"Error fetching API key for {service_name}: {str(e)}"
            )
            return None
    
    async def rotate_api_key(
        self,
        service_name: str,
        new_key: str
    ) -> bool:
        """
        Kierrätä API-avain
        
        Args:
            service_name: Palvelun nimi
            new_key: Uusi avain
        
        Returns:
            bool: True jos onnistui
        """
        secret_name = f"{service_name}_API_KEY"
        
        try:
            # Tallenna vanha avain varmuuskopioksi
            old_secret = await self.get_api_key(service_name)
            if old_secret:
                backup_name = f"{secret_name}_BACKUP"
                await asyncio.to_thread(
                    self.client.set_secret,
                    backup_name,
                    old_secret
                )
            
            # Aseta uusi avain
            await asyncio.to_thread(
                self.client.set_secret,
                secret_name,
                new_key
            )
            
            # Pakota välimuistin päivitys
            await self.get_api_key(service_name, force_refresh=True)
            
            return True
        
        except Exception as e:
            logger.error(
                f"Error rotating API key for {service_name}: {str(e)}"
            )
            return False
    
    def _get_from_cache(self, key: str) -> Optional[str]:
        """
        Hae avain välimuistista
        
        Args:
            key: Avaimen nimi
        
        Returns:
            str: Avain tai None
        """
        # Tarkista päivitystarve
        if (
            datetime.now() - self._last_refresh >
            self.refresh_interval
        ):
            return None
        
        cached = self._cache.get(key)
        if not cached:
            return None
        
        # Tarkista vanhentuminen
        if cached["expires_on"]:
            if datetime.now() > cached["expires_on"]:
                return None
        
        return cached["value"]
    
    def _update_cache(
        self,
        key: str,
        value: str,
        expires_on: datetime = None
    ):
        """
        Päivitä välimuisti
        
        Args:
            key: Avaimen nimi
            value: Arvo
            expires_on: Vanhentumisaika
        """
        self._cache[key] = {
            "value": value,
            "expires_on": expires_on,
            "updated_at": datetime.now()
        }
        self._last_refresh = datetime.now()

class SecretRotator:
    """Automaattinen avainten kierrätys"""
    
    def __init__(
        self,
        vault_manager: KeyVaultManager,
        config_path: str = "config.json"
    ):
        """
        Alusta kierrättäjä
        
        Args:
            vault_manager: Key Vault -manageri
            config_path: Konfiguraatiotiedoston polku
        """
        self.vault_manager = vault_manager
        self.config = self._load_config(config_path)
        
        # Kierrätysasetukset
        self.rotation_interval = timedelta(
            days=self.config.get("key_rotation_days", 30)
        )
        self.rotation_history: Dict[str, datetime] = {}
    
    def _load_config(self, path: str) -> Dict:
        """Lataa konfiguraatio"""
        with open(path) as f:
            return json.load(f)
    
    async def check_rotation_needs(self) -> Dict[str, bool]:
        """
        Tarkista kierrätystarve
        
        Returns:
            Dict: Palvelut ja kierrätystarve
        """
        needs = {}
        
        for service in self.config["api_keys"]:
            # Tarkista viimeisin kierrätys
            last_rotation = self.rotation_history.get(
                service,
                datetime.min
            )
            
            # Tarkista tarve
            needs[service] = (
                datetime.now() - last_rotation >
                self.rotation_interval
            )
        
        return needs
    
    async def rotate_keys(self) -> Dict[str, bool]:
        """
        Kierrätä avaimet
        
        Returns:
            Dict: Palvelut ja kierrätyksen tila
        """
        results = {}
        rotation_needs = await self.check_rotation_needs()
        
        for service, needs_rotation in rotation_needs.items():
            if not needs_rotation:
                continue
            
            # Generoi uusi avain
            new_key = self._generate_key()
            
            # Kierrätä avain
            success = await self.vault_manager.rotate_api_key(
                service,
                new_key
            )
            
            if success:
                self.rotation_history[service] = datetime.now()
            
            results[service] = success
        
        return results
    
    def _generate_key(self) -> str:
        """
        Generoi uusi API-avain
        
        Returns:
            str: Uusi avain
        """
        # TODO: Implementoi turvallinen avaingeneraattori
        import secrets
        return secrets.token_urlsafe(32)

async def main():
    """Testaa avainten hallintaa"""
    # Alusta Key Vault
    vault_url = "https://<key-vault-name>.vault.azure.net/"
    vault_manager = KeyVaultManager(vault_url)
    
    # Testaa avainten hakua
    openai_key = await vault_manager.get_api_key("OPENAI")
    logger.info(
        f"OpenAI API key found: {bool(openai_key)}"
    )
    
    # Alusta kierrättäjä
    rotator = SecretRotator(vault_manager)
    
    # Tarkista kierrätystarpeet
    needs = await rotator.check_rotation_needs()
    logger.info(f"Rotation needs: {needs}")
    
    # Kierrätä avaimet
    results = await rotator.rotate_keys()
    logger.info(f"Rotation results: {results}")

if __name__ == "__main__":
    asyncio.run(main())
