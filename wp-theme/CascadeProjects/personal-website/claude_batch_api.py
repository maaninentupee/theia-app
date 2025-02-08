"""
Claude-3 Batch API integraatio.
Tämä moduuli tarjoaa rajapinnan Claude-3-opus-20240229 mallin käyttöön eräajoina.
"""

import os
import json
import time
import asyncio
import logging
import random
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import aiohttp
from dotenv import load_dotenv

# Konfiguroi lokitus
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BatchRequest:
    """Batch API -pyyntö"""
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    request_id: str
    system_prompt: Optional[str] = None

@dataclass
class BatchResponse:
    """Batch API -vastaus"""
    request_id: str
    completion: str
    finish_reason: str
    model: str
    usage: Dict[str, int]

class ClaudeBatchAPI:
    """Claude-3 Batch API -käsittelijä"""
    
    def __init__(self, config_file: str = "config.json", test_mode: bool = False):
        """
        Alusta Claude Batch API
        
        Args:
            config_file: Polku konfiguraatiotiedostoon
            test_mode: Jos True, käytetään mock-vastauksia ilman oikeaa API-kutsua
        """
        # Lataa ympäristömuuttujat ja konfiguraatio
        load_dotenv()
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Hae Claude Batch asetukset
        self.batch_config = self.config['claude_batch']
        self.test_mode = test_mode
        
        if not test_mode:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("Anthropic API avain puuttuu!")
        
        # Alusta asynkroninen sessio
        self.session = None
        self.rate_limiter = self._create_rate_limiter()
    
    async def __aenter__(self):
        """Kontekstimanagerin alustus"""
        if not self.test_mode:
            self.session = aiohttp.ClientSession(
                headers={
                    "X-API-Key": self.api_key,
                    "Content-Type": "application/json"
                }
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Kontekstimanagerin sulkeminen"""
        if self.session:
            await self.session.close()
    
    def _create_rate_limiter(self) -> asyncio.Semaphore:
        """Luo rate limiter"""
        requests_per_minute = self.batch_config['rate_limit']['requests_per_minute']
        return asyncio.Semaphore(requests_per_minute)
    
    async def _make_request(self, request: BatchRequest, max_retries: int = 3) -> BatchResponse:
        """
        Tee yksittäinen API-pyyntö
        
        Args:
            request: BatchRequest-olio
            max_retries: Maksimi uudelleenyritysten määrä
        
        Returns:
            BatchResponse: API-vastaus
        """
        last_error = None
        for retry in range(max_retries):
            try:
                async with self.rate_limiter:
                    # Test mode - palauta mock-vastaus
                    if self.test_mode:
                        # Simuloi satunnaista virhettä testausta varten
                        if random.random() < 0.4 and retry == 0:  # Virhe vain ensimmäisellä yrityksellä
                            raise Exception("Simuloitu API virhe testausta varten")
                            
                        return BatchResponse(
                            request_id=request.request_id,
                            completion="Tämä on testivastaus",
                            finish_reason="stop",
                            model="claude-3-opus-20240229",
                            usage={"prompt_tokens": 10, "completion_tokens": 20}
                        )

                    payload = {
                        "model": self.config['models']['claude'],
                        "prompt": request.prompt,
                        "max_tokens_to_sample": request.max_tokens,
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                        "request_id": request.request_id
                    }
                    
                    if request.system_prompt:
                        payload["system"] = request.system_prompt
                    
                    async with self.session.post(
                        self.batch_config['api_url'],
                        json=payload,
                        timeout=self.batch_config['timeout']
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(f"API virhe {response.status}: {error_text}")
                        
                        data = await response.json()
                        
                        return BatchResponse(
                            request_id=data['request_id'],
                            completion=data['completion'],
                            finish_reason=data['finish_reason'],
                            model=data['model'],
                            usage=data['usage']
                        )
            
            except Exception as e:
                last_error = e
                logger.error(f"Virhe API-pyynnössä (yritys {retry + 1}/{max_retries}): {str(e)}")
                if retry < max_retries - 1:
                    await asyncio.sleep(2 ** retry)  # Exponential backoff
                    continue
                raise last_error
    
    async def process_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> List[BatchResponse]:
        """
        Prosessoi lista prompteja eräajona
        
        Args:
            prompts: Lista prompteja
            system_prompt: Valinnainen system prompt
            max_tokens: Maksimi tokenien määrä
            temperature: Lämpötila (0-1)
            top_p: Top-p näytteistys (0-1)
        
        Returns:
            List[BatchResponse]: Lista vastauksia
        """
        # Käytä oletusarvoja jos parametreja ei annettu
        max_tokens = max_tokens or self.batch_config['max_tokens']
        temperature = temperature or self.batch_config['temperature']
        top_p = top_p or self.batch_config['top_p']
        
        # Jaa promptit eriin
        batch_size = self.batch_config['batch_size']
        batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
        
        all_responses = []
        
        # Prosessoi erät rinnakkain
        for batch in batches:
            requests = [
                BatchRequest(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    request_id=f"req_{int(time.time())}_{i}",
                    system_prompt=system_prompt
                )
                for i, prompt in enumerate(batch)
            ]
            
            # Suorita pyynnöt rinnakkain
            tasks = [
                self._make_request(request)
                for request in requests
            ]
            
            # Odota kaikki vastaukset
            batch_responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Käsittele virheet
            for i, response in enumerate(batch_responses):
                if isinstance(response, Exception):
                    logger.error(f"Virhe erässä {i}: {str(response)}")
                    continue
                all_responses.append(response)
        
        return all_responses

async def main():
    """Esimerkki käytöstä"""
    # Testipromptit
    prompts = [
        "Mikä on tekoälyn rooli ohjelmistokehityksessä?",
        "Miten voin optimoida Python-koodin suorituskykyä?",
        "Selitä mikropalveluarkkitehtuurin edut ja haitat."
    ]
    
    async with ClaudeBatchAPI(test_mode=True) as api:
        responses = await api.process_batch(
            prompts=prompts,
            system_prompt="Olet asiantunteva ja ytimekäs tekoälyassistentti."
        )
        
        for i, response in enumerate(responses):
            print(f"\nVastaus {i+1}:")
            print(f"Prompt: {prompts[i]}")
            print(f"Vastaus: {response.completion}")
            print(f"Tokenien käyttö: {response.usage}")

if __name__ == "__main__":
    asyncio.run(main())
