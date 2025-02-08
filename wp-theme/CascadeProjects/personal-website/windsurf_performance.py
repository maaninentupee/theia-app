"""
Windsurf suorituskykyoptimoinnit.
Tämä moduuli vastaa välimuistista ja kuorman skaalauksesta.
"""

import os
import json
import logging
import asyncio
import hashlib
import pickle
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import anthropic
from anthropic import Client, RateLimitError

# Konfiguroi lokitus
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    """Välimuistitilastot"""
    hits: int = 0
    misses: int = 0
    size: int = 0
    saved_tokens: int = 0
    saved_cost: float = 0.0

class LRUCache:
    """Least Recently Used -välimuisti"""
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl: int = 3600,  # 1 tunti
        persist_path: Optional[str] = None
    ):
        """
        Alusta välimuisti
        
        Args:
            max_size: Maksimi välimuistin koko
            ttl: Time-to-live sekunteina
            persist_path: Polku välimuistin tallennukseen
        """
        self.max_size = max_size
        self.ttl = ttl
        self.persist_path = persist_path
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.stats = CacheStats()
        
        # Lataa välimuisti levyltä
        if persist_path and os.path.exists(persist_path):
            try:
                with open(persist_path, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"Ladattiin {len(self.cache)} kohdetta välimuistista")
            except Exception as e:
                logger.error(f"Virhe välimuistin latauksessa: {str(e)}")
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generoi avain argumenteista"""
        key_data = f"{args}_{kwargs}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Hae arvo välimuistista"""
        if key not in self.cache:
            self.stats.misses += 1
            return None
        
        value, timestamp = self.cache[key]
        
        # Tarkista TTL
        if (datetime.now() - timestamp).total_seconds() > self.ttl:
            del self.cache[key]
            self.stats.misses += 1
            return None
        
        self.stats.hits += 1
        return value
    
    def set(self, key: str, value: Any):
        """Aseta arvo välimuistiin"""
        # Poista vanhin jos täynnä
        if len(self.cache) >= self.max_size:
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k][1]
            )
            del self.cache[oldest_key]
        
        self.cache[key] = (value, datetime.now())
        self.stats.size = len(self.cache)
        
        # Tallenna levylle
        if self.persist_path:
            try:
                with open(self.persist_path, 'wb') as f:
                    pickle.dump(self.cache, f)
            except Exception as e:
                logger.error(f"Virhe välimuistin tallennuksessa: {str(e)}")
    
    def clear(self):
        """Tyhjennä välimuisti"""
        self.cache.clear()
        self.stats = CacheStats()
        
        if self.persist_path and os.path.exists(self.persist_path):
            os.remove(self.persist_path)

class BatchProcessor:
    """Eräajojen käsittelijä"""
    
    def __init__(
        self,
        batch_size: int = 15,
        max_concurrent: int = 3,
        rate_limit: int = 50,  # pyyntöä/min
        cache: Optional[LRUCache] = None
    ):
        """
        Alusta eräajoprosessori
        
        Args:
            batch_size: Eräkoko
            max_concurrent: Maksimi rinnakkaiset erät
            rate_limit: Maksimi pyyntöä minuutissa
            cache: Välimuisti
        """
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.rate_limit = rate_limit
        self.cache = cache or LRUCache()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.last_request_times: List[datetime] = []
    
    async def _wait_for_rate_limit(self):
        """Odota rate limitiä"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Poista vanhat pyynnöt
        self.last_request_times = [
            t for t in self.last_request_times
            if t > minute_ago
        ]
        
        # Odota jos rate limit ylittyy
        if len(self.last_request_times) >= self.rate_limit:
            wait_time = (
                self.last_request_times[0] + timedelta(minutes=1) - now
            ).total_seconds()
            if wait_time > 0:
                logger.info(f"Odotetaan rate limitiä {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        self.last_request_times.append(now)
    
    async def process_batch(
        self,
        tasks: List[Any],
        process_func: Callable,
        *args,
        **kwargs
    ) -> List[Any]:
        """
        Käsittele tehtäväerä
        
        Args:
            tasks: Tehtävälista
            process_func: Käsittelyfunktio
            *args: Funktion argumentit
            **kwargs: Funktion avainsana-argumentit
        
        Returns:
            List[Any]: Tulokset
        """
        async with self.semaphore:
            await self._wait_for_rate_limit()
            
            results = []
            for task in tasks:
                # Tarkista välimuisti
                cache_key = self.cache._generate_key(task, *args, **kwargs)
                cached_result = self.cache.get(cache_key)
                
                if cached_result is not None:
                    results.append(cached_result)
                    continue
                
                # Suorita tehtävä
                try:
                    result = await process_func(task, *args, **kwargs)
                    results.append(result)
                    
                    # Tallenna välimuistiin
                    self.cache.set(cache_key, result)
                
                except Exception as e:
                    logger.error(
                        f"Virhe tehtävän suorituksessa: {str(e)}"
                    )
                    results.append(None)
            
            return results
    
    async def process_all(
        self,
        tasks: List[Any],
        process_func: Callable,
        *args,
        **kwargs
    ) -> List[Any]:
        """
        Käsittele kaikki tehtävät
        
        Args:
            tasks: Tehtävälista
            process_func: Käsittelyfunktio
            *args: Funktion argumentit
            **kwargs: Funktion avainsana-argumentit
        
        Returns:
            List[Any]: Tulokset
        """
        # Jaa tehtävät eriin
        batches = [
            tasks[i:i + self.batch_size]
            for i in range(0, len(tasks), self.batch_size)
        ]
        
        # Suorita erät rinnakkain
        tasks = [
            self.process_batch(batch, process_func, *args, **kwargs)
            for batch in batches
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Litistä tulokset
        return [
            result
            for batch_results in results
            for result in batch_results
        ]

class PerformanceMonitor:
    """Suorituskyvyn monitorointi"""
    
    def __init__(self):
        """Alusta monitorointi"""
        self.start_time = datetime.now()
        self.request_times: List[float] = []
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.token_usage: Dict[str, int] = defaultdict(int)
    
    def add_request(self, duration: float):
        """Lisää pyyntö"""
        self.request_times.append(duration)
    
    def add_error(self, error_type: str):
        """Lisää virhe"""
        self.error_counts[error_type] += 1
    
    def add_tokens(self, model: str, tokens: int):
        """Lisää tokenien käyttö"""
        self.token_usage[model] += tokens
    
    def get_stats(self) -> Dict[str, Any]:
        """Hae tilastot"""
        if not self.request_times:
            return {}
        
        return {
            "uptime": (datetime.now() - self.start_time).total_seconds(),
            "total_requests": len(self.request_times),
            "avg_request_time": sum(self.request_times) / len(self.request_times),
            "min_request_time": min(self.request_times),
            "max_request_time": max(self.request_times),
            "error_counts": dict(self.error_counts),
            "token_usage": dict(self.token_usage),
            "requests_per_minute": len(self.request_times) / (
                (datetime.now() - self.start_time).total_seconds() / 60
            )
        }

async def main():
    """Testaa suorituskykyä"""
    # Alusta komponentit
    cache = LRUCache(
        max_size=1000,
        ttl=3600,
        persist_path="windsurf_cache.pkl"
    )
    
    batch_processor = BatchProcessor(
        batch_size=15,
        max_concurrent=3,
        rate_limit=50,
        cache=cache
    )
    
    monitor = PerformanceMonitor()
    
    # Testitehtävät
    tasks = [f"Tehtävä {i}" for i in range(100)]
    
    # Testifunktio
    async def process_task(task: str) -> str:
        await asyncio.sleep(0.1)  # Simuloi API-kutsua
        return f"Tulos: {task}"
    
    # Suorita tehtävät
    start_time = datetime.now()
    
    results = await batch_processor.process_all(
        tasks,
        process_task
    )
    
    duration = (datetime.now() - start_time).total_seconds()
    monitor.add_request(duration)
    
    # Tulosta tilastot
    logger.info(f"Suoritettu {len(results)} tehtävää")
    logger.info(f"Välimuistitilastot: {cache.stats}")
    logger.info(f"Suorituskykytilastot: {monitor.get_stats()}")

if __name__ == "__main__":
    asyncio.run(main())
