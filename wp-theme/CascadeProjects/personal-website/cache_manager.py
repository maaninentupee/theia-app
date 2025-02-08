import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import aioredis
from cachetools import TTLCache, cached
from prometheus_client import Counter, Gauge, Histogram

# Konfiguroi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    """Välimuistitasot"""
    MEMORY = "memory"  # Nopein, pienin
    REDIS = "redis"    # Keskitaso
    DISK = "disk"     # Hitain, suurin

class CacheStrategy(Enum):
    """Välimuististrategiat"""
    LRU = "lru"       # Least Recently Used
    LFU = "lfu"       # Least Frequently Used
    FIFO = "fifo"     # First In First Out

class CacheConfig:
    """Välimuistikonfiguraatio"""
    
    def __init__(
        self,
        level: CacheLevel,
        strategy: CacheStrategy,
        ttl: int,
        max_size: int
    ):
        self.level = level
        self.strategy = strategy
        self.ttl = ttl
        self.max_size = max_size

class CacheMetrics:
    """Välimuistimetriikat"""
    
    def __init__(self):
        # Prometheus-mittarit
        self.hits = Counter(
            'cache_hits_total',
            'Cache hits',
            ['level']
        )
        
        self.misses = Counter(
            'cache_misses_total',
            'Cache misses',
            ['level']
        )
        
        self.size = Gauge(
            'cache_size_bytes',
            'Cache size in bytes',
            ['level']
        )
        
        self.latency = Histogram(
            'cache_operation_duration_seconds',
            'Cache operation duration',
            ['level', 'operation']
        )

class CacheManager:
    """Välimuistin hallinta"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost",
        disk_path: str = "cache"
    ):
        """
        Alusta välimuisti
        
        Args:
            redis_url: Redis-palvelimen osoite
            disk_path: Polku välimuistin tiedostoihin
        """
        # Välimuistit eri tasoille
        self.memory_cache = TTLCache(
            maxsize=1000,
            ttl=3600
        )
        
        self.redis = aioredis.from_url(redis_url)
        self.disk_path = Path(disk_path)
        self.disk_path.mkdir(exist_ok=True)
        
        # Oletuskonfiguraatiot
        self.default_configs = {
            CacheLevel.MEMORY: CacheConfig(
                CacheLevel.MEMORY,
                CacheStrategy.LRU,
                ttl=3600,
                max_size=1000
            ),
            CacheLevel.REDIS: CacheConfig(
                CacheLevel.REDIS,
                CacheStrategy.LFU,
                ttl=86400,
                max_size=10000
            ),
            CacheLevel.DISK: CacheConfig(
                CacheLevel.DISK,
                CacheStrategy.FIFO,
                ttl=604800,
                max_size=100000
            )
        }
        
        # Metriikat
        self.metrics = CacheMetrics()
    
    async def get(
        self,
        key: str,
        level: CacheLevel = CacheLevel.MEMORY
    ) -> Optional[Any]:
        """
        Hae arvo välimuistista
        
        Args:
            key: Avain
            level: Välimuistitaso
        
        Returns:
            Any: Arvo tai None
        """
        start_time = datetime.now()
        
        try:
            # Generoi hash-avain
            cache_key = self._generate_key(key)
            
            # Yritä hakea eri tasoilta
            value = None
            
            if level == CacheLevel.MEMORY:
                value = self.memory_cache.get(cache_key)
            
            elif level == CacheLevel.REDIS:
                value = await self.redis.get(cache_key)
                if value:
                    value = json.loads(value)
            
            elif level == CacheLevel.DISK:
                value = self._read_from_disk(cache_key)
            
            # Päivitä metriikat
            if value is not None:
                self.metrics.hits.labels(level.value).inc()
            else:
                self.metrics.misses.labels(level.value).inc()
            
            return value
        
        finally:
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics.latency.labels(
                level.value,
                'get'
            ).observe(duration)
    
    async def set(
        self,
        key: str,
        value: Any,
        level: CacheLevel = CacheLevel.MEMORY,
        ttl: Optional[int] = None
    ):
        """
        Aseta arvo välimuistiin
        
        Args:
            key: Avain
            value: Arvo
            level: Välimuistitaso
            ttl: Time-to-live (sekunteina)
        """
        start_time = datetime.now()
        
        try:
            # Generoi hash-avain
            cache_key = self._generate_key(key)
            
            # Käytä oletuskonfiguraatiota
            config = self.default_configs[level]
            if ttl is None:
                ttl = config.ttl
            
            # Tallenna arvot eri tasoille
            if level == CacheLevel.MEMORY:
                self.memory_cache[cache_key] = value
            
            elif level == CacheLevel.REDIS:
                await self.redis.setex(
                    cache_key,
                    ttl,
                    json.dumps(value)
                )
            
            elif level == CacheLevel.DISK:
                self._write_to_disk(cache_key, value)
            
            # Päivitä koko
            size = len(str(value).encode())
            self.metrics.size.labels(level.value).set(size)
        
        finally:
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics.latency.labels(
                level.value,
                'set'
            ).observe(duration)
    
    async def invalidate(
        self,
        key: str,
        level: Optional[CacheLevel] = None
    ):
        """
        Poista arvo välimuistista
        
        Args:
            key: Avain
            level: Välimuistitaso (jos None, poista kaikista)
        """
        cache_key = self._generate_key(key)
        
        if level is None:
            # Poista kaikista tasoista
            levels = list(CacheLevel)
        else:
            levels = [level]
        
        for l in levels:
            if l == CacheLevel.MEMORY:
                self.memory_cache.pop(cache_key, None)
            
            elif l == CacheLevel.REDIS:
                await self.redis.delete(cache_key)
            
            elif l == CacheLevel.DISK:
                self._delete_from_disk(cache_key)
    
    async def clear(self, level: Optional[CacheLevel] = None):
        """
        Tyhjennä välimuisti
        
        Args:
            level: Välimuistitaso (jos None, tyhjennä kaikki)
        """
        if level is None:
            # Tyhjennä kaikki tasot
            levels = list(CacheLevel)
        else:
            levels = [level]
        
        for l in levels:
            if l == CacheLevel.MEMORY:
                self.memory_cache.clear()
            
            elif l == CacheLevel.REDIS:
                await self.redis.flushdb()
            
            elif l == CacheLevel.DISK:
                import shutil
                shutil.rmtree(self.disk_path)
    
    def _generate_key(self, key: str) -> str:
        """
        Generoi hash-avain
        
        Args:
            key: Alkuperäinen avain
        
        Returns:
            str: Hash-avain
        """
        return hashlib.sha256(
            key.encode()
        ).hexdigest()
    
    def _read_from_disk(self, key: str) -> Optional[Any]:
        """
        Lue arvo levyltä
        
        Args:
            key: Avain
        
        Returns:
            Any: Arvo tai None
        """
        path = self.disk_path / key
        
        if not path.exists():
            return None
        
        try:
            with path.open('r') as f:
                data = json.load(f)
                
                # Tarkista TTL
                if datetime.fromisoformat(data['expires']) < datetime.now():
                    path.unlink()
                    return None
                
                return data['value']
        except Exception as e:
            logger.error(f"Virhe luettaessa välimuistia: {e}")
            return None
    
    def _write_to_disk(self, key: str, value: Any):
        """
        Kirjoita arvo levylle
        
        Args:
            key: Avain
            value: Arvo
        """
        try:
            path = self.disk_path / key
            config = self.default_configs[CacheLevel.DISK]
            expires = datetime.now() + timedelta(seconds=config.ttl)
            
            data = {
                'value': value,
                'expires': expires.isoformat()
            }
            
            with path.open('w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Virhe tallennettaessa välimuistiin: {e}")
    
    def _delete_from_disk(self, key: str):
        """
        Poista arvo levyltä
        
        Args:
            key: Avain
        """
        path = self.disk_path / key
        
        if path.exists():
            try:
                path.unlink()
            except Exception as e:
                logger.error(f"Virhe poistettaessa välimuistista: {e}")
    
    def get_stats(self) -> Dict:
        """
        Hae välimuistitilastot
        
        Returns:
            Dict: Tilastot
        """
        stats = {}
        
        for level in CacheLevel:
            hits = self.metrics.hits.labels(level.value)._value.get()
            misses = self.metrics.misses.labels(level.value)._value.get()
            total = hits + misses
            
            stats[level.value] = {
                'hits': hits,
                'misses': misses,
                'hit_ratio': hits / total if total > 0 else 0,
                'size': self.metrics.size.labels(
                    level.value
                )._value.get()
            }
        
        return stats

class CacheDecorator:
    """Välimuistikoristin"""
    
    def __init__(
        self,
        cache_manager: CacheManager,
        level: CacheLevel = CacheLevel.MEMORY,
        ttl: Optional[int] = None
    ):
        self.cache_manager = cache_manager
        self.level = level
        self.ttl = ttl
    
    def __call__(self, func):
        async def wrapper(*args, **kwargs):
            # Generoi avain funktiokutsusta
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Yritä hakea välimuistista
            cached_value = await self.cache_manager.get(
                key,
                self.level
            )
            
            if cached_value is not None:
                return cached_value
            
            # Suorita funktio
            result = await func(*args, **kwargs)
            
            # Tallenna välimuistiin
            await self.cache_manager.set(
                key,
                result,
                self.level,
                self.ttl
            )
            
            return result
        
        return wrapper

async def main():
    """Testaa välimuistia"""
    cache_manager = CacheManager()
    
    # Testaa eri tasoja
    for level in CacheLevel:
        # Aseta arvo
        await cache_manager.set(
            "test_key",
            {"data": "test_value"},
            level
        )
        
        # Hae arvo
        value = await cache_manager.get("test_key", level)
        print(f"\n{level.value} cache test:")
        print(f"Value: {value}")
    
    # Testaa koristinta
    @CacheDecorator(cache_manager)
    async def expensive_operation(x: int) -> int:
        await asyncio.sleep(1)  # Simuloi raskasta laskentaa
        return x * x
    
    # Suorita operaatio kahdesti
    start = datetime.now()
    result1 = await expensive_operation(5)
    duration1 = (datetime.now() - start).total_seconds()
    
    start = datetime.now()
    result2 = await expensive_operation(5)
    duration2 = (datetime.now() - start).total_seconds()
    
    print("\nCache decorator test:")
    print(f"First call: {result1} ({duration1:.2f}s)")
    print(f"Second call: {result2} ({duration2:.2f}s)")
    
    # Näytä tilastot
    stats = cache_manager.get_stats()
    print("\nCache stats:")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
