"""
Käyttöjärjestelmäriippumaton polkujen käsittely.
"""

import os
import platform
import json
from pathlib import Path
from typing import Union, Dict

class PathManager:
    """Polkujen hallinta eri käyttöjärjestelmille"""
    
    def __init__(self, config_file: Union[str, Path] = "config.json"):
        """
        Alusta polkujen hallinta
        
        Args:
            config_file: Konfiguraatiotiedoston polku
        """
        self.config_file = Path(config_file)
        self.system = platform.system().lower()
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Lataa konfiguraatio
        
        Returns:
            Dict: Konfiguraatio
        """
        try:
            return json.loads(self.config_file.read_text(encoding='utf-8'))
        except Exception as e:
            raise RuntimeError(f"Virhe ladattaessa konfiguraatiota: {e}")
            
    def get_path(self, path_type: str) -> Path:
        """
        Hae polku tietylle tyypille
        
        Args:
            path_type: Polun tyyppi (esim. 'cache', 'temp', 'data')
            
        Returns:
            Path: Käyttöjärjestelmäkohtainen polku
        """
        try:
            path = self.config['paths'][path_type][self.system]
            return Path(path)
        except KeyError:
            raise ValueError(f"Virheellinen polun tyyppi: {path_type}")
            
    def get_log_file(self, log_type: str = "app") -> Path:
        """
        Hae lokitiedoston polku
        
        Args:
            log_type: Lokitiedoston tyyppi ('app' tai 'metrics')
            
        Returns:
            Path: Lokitiedoston polku
        """
        try:
            if log_type == "app":
                path = self.config['logging']['file'][self.system]
            elif log_type == "metrics":
                path = self.config['logging']['metrics']['file'][self.system]
            else:
                raise ValueError(f"Virheellinen lokitiedoston tyyppi: {log_type}")
                
            return Path(path)
        except KeyError:
            raise ValueError(f"Virheellinen lokitiedoston tyyppi: {log_type}")
            
    def ensure_directory(self, path: Union[str, Path]):
        """
        Varmista että hakemisto on olemassa
        
        Args:
            path: Hakemiston polku
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        
    def join_paths(self, *paths: Union[str, Path]) -> Path:
        """
        Yhdistä polut käyttöjärjestelmäriippumattomasti
        
        Args:
            *paths: Polut
            
        Returns:
            Path: Yhdistetty polku
        """
        return Path(*paths)
