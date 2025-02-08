"""
Windsurf virheenkäsittely.
Tämä moduuli vastaa Windsurf IDE:n virheenkäsittelystä.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import anthropic
from anthropic import Client, RateLimitError, APIError, APITimeoutError, APIConnectionError

# Konfiguroi lokitus
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ErrorStats:
    """Virhetilastot"""
    total_errors: int = 0
    rate_limit_errors: int = 0
    timeout_errors: int = 0
    connection_errors: int = 0
    validation_errors: int = 0
    other_errors: int = 0
    last_error_time: Optional[datetime] = None
    error_window: List[Tuple[datetime, str]] = None
    
    def __post_init__(self):
        """Alusta virheikkuna"""
        self.error_window = []
    
    def add_error(self, error_type: str):
        """
        Lisää virhe tilastoihin
        
        Args:
            error_type: Virheen tyyppi
        """
        self.total_errors += 1
        self.last_error_time = datetime.now()
        
        # Lisää virhe ikkunaan
        self.error_window.append((self.last_error_time, error_type))
        
        # Poista vanhat virheet (yli tunnin vanhat)
        cutoff = datetime.now() - timedelta(hours=1)
        self.error_window = [
            (time, type) for time, type in self.error_window
            if time > cutoff
        ]
        
        # Päivitä virhetyypit
        if error_type == "rate_limit":
            self.rate_limit_errors += 1
        elif error_type == "timeout":
            self.timeout_errors += 1
        elif error_type == "connection":
            self.connection_errors += 1
        elif error_type == "validation":
            self.validation_errors += 1
        else:
            self.other_errors += 1
    
    def get_error_rate(self) -> float:
        """
        Laske virhetaajuus viimeisen tunnin ajalta
        
        Returns:
            float: Virheitä tunnissa
        """
        if not self.error_window:
            return 0.0
        
        # Laske virheet viimeisen tunnin ajalta
        cutoff = datetime.now() - timedelta(hours=1)
        recent_errors = [
            time for time, _ in self.error_window
            if time > cutoff
        ]
        
        return len(recent_errors)
    
    def should_backoff(self) -> bool:
        """
        Tarkista pitäisikö hidastaa kutsuja
        
        Returns:
            bool: True jos pitäisi hidastaa
        """
        error_rate = self.get_error_rate()
        return error_rate > 10  # Yli 10 virhettä tunnissa

class WindsurfErrorHandler:
    """Windsurf virheenkäsittelijä"""
    
    def __init__(self):
        """Alusta virheenkäsittelijä"""
        self.stats = ErrorStats()
        self.backoff_time = 1.0  # Sekunteina
    
    async def handle_api_error(
        self,
        error: Exception,
        context: str = ""
    ) -> Tuple[bool, str]:
        """
        Käsittele API-virhe
        
        Args:
            error: Virhe
            context: Virheen konteksti
        
        Returns:
            Tuple[bool, str]: (Pitäisikö yrittää uudelleen, Virheviesti)
        """
        error_type = "other"
        retry = False
        message = str(error)
        
        if isinstance(error, RateLimitError):
            error_type = "rate_limit"
            message = "API Rate Limit saavutettu"
            retry = True
            
            # Exponentiaalinen backoff
            await asyncio.sleep(self.backoff_time)
            self.backoff_time *= 2
        
        elif isinstance(error, APITimeoutError):
            error_type = "timeout"
            message = "API Timeout"
            retry = True
            
            # Lyhyt odotus
            await asyncio.sleep(1.0)
        
        elif isinstance(error, APIConnectionError):
            error_type = "connection"
            message = "API-yhteysongelma"
            retry = True
            
            # Keskipitkä odotus
            await asyncio.sleep(5.0)
        
        elif isinstance(error, APIError):
            if error.status_code == 400:
                error_type = "validation"
                message = "Virheellinen pyyntö"
                retry = False
            elif error.status_code == 401:
                error_type = "auth"
                message = "Autentikaatiovirhe"
                retry = False
            elif error.status_code == 404:
                error_type = "not_found"
                message = "Resurssia ei löydy"
                retry = False
            elif error.status_code >= 500:
                error_type = "server"
                message = "Palvelinvirhe"
                retry = True
                
                # Pitkä odotus
                await asyncio.sleep(10.0)
        
        # Päivitä tilastot
        self.stats.add_error(error_type)
        
        # Loki
        logger.error(
            f"API-virhe ({error_type}): {message} "
            f"[{context}] [Retry: {retry}]"
        )
        
        # Jos virhetaajuus on korkea, hidasta kutsuja
        if self.stats.should_backoff():
            logger.warning(
                f"Korkea virhetaajuus ({self.stats.get_error_rate():.1f} "
                f"virhettä/tunti). Hidastetaan kutsuja."
            )
            await asyncio.sleep(5.0)
        
        return retry, message

class WindsurfRetryHandler:
    """Windsurf uudelleenyrityslogiikka"""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0
    ):
        """
        Alusta uudelleenyrityslogiikka
        
        Args:
            max_retries: Maksimi uudelleenyritykset
            initial_delay: Alkuviive sekunteina
            max_delay: Maksimi viive sekunteina
            exponential_base: Exponentiaalisen kasvun kerroin
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    def calculate_delay(self, retry_count: int) -> float:
        """
        Laske odotusaika
        
        Args:
            retry_count: Uudelleenyritysten määrä
        
        Returns:
            float: Odotusaika sekunteina
        """
        delay = self.initial_delay * (self.exponential_base ** retry_count)
        return min(delay, self.max_delay)
    
    async def execute_with_retry(
        self,
        func: callable,
        *args,
        error_handler: Optional[WindsurfErrorHandler] = None,
        context: str = "",
        **kwargs
    ) -> Any:
        """
        Suorita funktio uudelleenyrityksillä
        
        Args:
            func: Suoritettava funktio
            error_handler: Virheenkäsittelijä
            context: Virheen konteksti
            *args: Funktion argumentit
            **kwargs: Funktion avainsana-argumentit
        
        Returns:
            Any: Funktion tulos
        
        Raises:
            Exception: Jos kaikki uudelleenyritykset epäonnistuvat
        """
        if error_handler is None:
            error_handler = WindsurfErrorHandler()
        
        last_error = None
        
        for retry in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            
            except Exception as e:
                last_error = e
                
                # Käsittele virhe
                should_retry, message = await error_handler.handle_api_error(
                    e,
                    context=f"{context} (Retry {retry+1}/{self.max_retries})"
                )
                
                if not should_retry or retry >= self.max_retries:
                    raise last_error
                
                # Odota ennen uutta yritystä
                delay = self.calculate_delay(retry)
                logger.info(
                    f"Odotetaan {delay:.1f}s ennen uutta yritystä "
                    f"({retry+1}/{self.max_retries})"
                )
                await asyncio.sleep(delay)
        
        raise last_error  # Ei pitäisi koskaan tapahtua

async def main():
    """Testaa virheenkäsittelyä"""
    error_handler = WindsurfErrorHandler()
    retry_handler = WindsurfRetryHandler()
    
    # Testifunktio joka heittää virheen
    async def test_func():
        raise RateLimitError("Test rate limit error")
    
    try:
        await retry_handler.execute_with_retry(
            test_func,
            error_handler=error_handler,
            context="Testitapaus"
        )
    except Exception as e:
        logger.error(f"Lopullinen virhe: {str(e)}")
        logger.info(f"Virhetilastot: {error_handler.stats.__dict__}")

if __name__ == "__main__":
    asyncio.run(main())
