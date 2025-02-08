import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from cachetools import LRUCache, TTLCache
from prometheus_client import Counter, Gauge, Histogram
import redis.asyncio as redis

# Konfiguroi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    """Välimuistitasot"""
    MEMORY = "memory"    # Nopein, pienin
    REDIS = "redis"      # Keskitaso
    DISK = "disk"       # Hitain, suurin

class CacheStrategy(Enum):
    """Välimuististrategiat"""
    LRU = "lru"     # Least Recently Used
    LFU = "lfu"     # Least Frequently Used
    TTL = "ttl"     # Time To Live

@dataclass
class CacheConfig:
    """Välimuistikonfiguraatio"""
    strategy: CacheStrategy
    max_size: int
    ttl: Optional[int] = None
    compression: bool = True
    validation: bool = True

@dataclass
class CacheStats:
    """Välimuistitilastot"""
    hits: int = 0
    misses: int = 0
    size: int = 0
    latency: float = 0.0

class CacheOptimizer:
    """Välimuistin optimoija"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost",
        disk_path: str = "./cache"
    ):
        """
        Alusta optimoija
        
        Args:
            redis_url: Redis-palvelimen URL
            disk_path: Levyvälimuistin polku
        """
        # Välimuistikonfiguraatiot
        self.configs = {
            CacheLevel.MEMORY: CacheConfig(
                strategy=CacheStrategy.LRU,
                max_size=10000,
                ttl=300,        # 5min
                compression=False
            ),
            CacheLevel.REDIS: CacheConfig(
                strategy=CacheStrategy.LFU,
                max_size=100000,
                ttl=3600,       # 1h
                compression=True
            ),
            CacheLevel.DISK: CacheConfig(
                strategy=CacheStrategy.TTL,
                max_size=1000000,
                ttl=86400,      # 24h
                compression=True
            )
        }
        
        # Välimuistit
        self.caches = {
            CacheLevel.MEMORY: self._create_cache(
                self.configs[CacheLevel.MEMORY]
            ),
            CacheLevel.REDIS: redis.from_url(redis_url),
            CacheLevel.DISK: disk_path
        }
        
        # Tilastot
        self.stats = {
            level: CacheStats()
            for level in CacheLevel
        }
        
        # Metriikat
        self.cache_hits = Counter(
            'cache_hits_total',
            'Cache hits',
            ['level']
        )
        self.cache_misses = Counter(
            'cache_misses_total',
            'Cache misses',
            ['level']
        )
        self.cache_size = Gauge(
            'cache_size_bytes',
            'Cache size in bytes',
            ['level']
        )
        self.cache_latency = Histogram(
            'cache_latency_seconds',
            'Cache operation latency',
            ['level', 'operation']
        )
    
    def _create_cache(self, config: CacheConfig) -> Union[LRUCache, TTLCache]:
        """
        Luo välimuisti
        
        Args:
            config: Välimuistikonfiguraatio
        
        Returns:
            Union[LRUCache, TTLCache]: Välimuisti
        """
        if config.strategy == CacheStrategy.LRU:
            return LRUCache(maxsize=config.max_size)
        elif config.strategy == CacheStrategy.TTL:
            return TTLCache(
                maxsize=config.max_size,
                ttl=config.ttl
            )
        else:
            raise ValueError(f"Unknown strategy: {config.strategy}")
    
    def _generate_key(self, task: Any) -> str:
        """
        Generoi välimuistiavain
        
        Args:
            task: Tehtävä
        
        Returns:
            str: Välimuistiavain
        """
        # Muunna tehtävä merkkijonoksi
        if hasattr(task, 'to_cache_key'):
            key_str = task.to_cache_key()
        else:
            key_str = json.dumps(
                task,
                sort_keys=True,
                default=str
            )
        
        # Generoi hash
        return hashlib.sha256(
            key_str.encode()
        ).hexdigest()
    
    async def get(
        self,
        task: Any,
        level: CacheLevel = CacheLevel.MEMORY
    ) -> Optional[Any]:
        """
        Hae välimuistista
        
        Args:
            task: Tehtävä
            level: Välimuistitaso
        
        Returns:
            Optional[Any]: Tulos tai None
        """
        key = self._generate_key(task)
        start_time = datetime.now()
        
        try:
            # Hae välimuistista
            if level == CacheLevel.MEMORY:
                result = self.caches[level].get(key)
            
            elif level == CacheLevel.REDIS:
                result = await self.caches[level].get(key)
                if result:
                    result = json.loads(result)
            
            else:  # DISK
                # TODO: Implementoi levyvälimuisti
                result = None
            
            # Päivitä metriikat
            duration = (
                datetime.now() - start_time
            ).total_seconds()
            
            if result is not None:
                self.stats[level].hits += 1
                self.cache_hits.labels(level=level.value).inc()
            else:
                self.stats[level].misses += 1
                self.cache_misses.labels(level=level.value).inc()
            
            self.cache_latency.labels(
                level=level.value,
                operation="get"
            ).observe(duration)
            
            return result
        
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return None
    
    async def set(
        self,
        task: Any,
        result: Any,
        level: CacheLevel = CacheLevel.MEMORY
    ):
        """
        Tallenna välimuistiin
        
        Args:
            task: Tehtävä
            result: Tulos
            level: Välimuistitaso
        """
        key = self._generate_key(task)
        start_time = datetime.now()
        
        try:
            # Tallenna välimuistiin
            if level == CacheLevel.MEMORY:
                self.caches[level][key] = result
            
            elif level == CacheLevel.REDIS:
                await self.caches[level].set(
                    key,
                    json.dumps(result),
                    ex=self.configs[level].ttl
                )
            
            else:  # DISK
                # TODO: Implementoi levyvälimuisti
                pass
            
            # Päivitä metriikat
            duration = (
                datetime.now() - start_time
            ).total_seconds()
            
            self.cache_latency.labels(
                level=level.value,
                operation="set"
            ).observe(duration)
            
            # Päivitä koko
            if level == CacheLevel.MEMORY:
                size = len(self.caches[level])
            elif level == CacheLevel.REDIS:
                size = await self.caches[level].dbsize()
            else:
                size = 0
            
            self.cache_size.labels(
                level=level.value
            ).set(size)
        
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")
    
    async def invalidate(
        self,
        task: Any,
        level: CacheLevel = CacheLevel.MEMORY
    ):
        """
        Invalidoi välimuisti
        
        Args:
            task: Tehtävä
            level: Välimuistitaso
        """
        key = self._generate_key(task)
        
        try:
            # Poista välimuistista
            if level == CacheLevel.MEMORY:
                self.caches[level].pop(key, None)
            
            elif level == CacheLevel.REDIS:
                await self.caches[level].delete(key)
            
            else:  # DISK
                # TODO: Implementoi levyvälimuisti
                pass
        
        except Exception as e:
            logger.error(f"Cache invalidate error: {str(e)}")
    
    def get_stats(self) -> Dict:
        """
        Hae tilastot
        
        Returns:
            Dict: Välimuistitilastot
        """
        stats = {}
        
        for level in CacheLevel:
            level_stats = self.stats[level]
            total = level_stats.hits + level_stats.misses
            
            stats[level.value] = {
                "hits": level_stats.hits,
                "misses": level_stats.misses,
                "hit_ratio": (
                    level_stats.hits / total
                    if total > 0 else 0
                ),
                "size": level_stats.size,
                "latency": level_stats.latency
            }
        
        return stats
    
    def generate_report(self) -> str:
        """
        Generoi raportti
        
        Returns:
            str: Markdown-muotoinen raportti
        """
        stats = self.get_stats()
        
        report = """# Välimuistiraportti

## Tilastot

"""
        
        for level, level_stats in stats.items():
            report += f"### {level}\n"
            report += f"- Osumat: {level_stats['hits']}\n"
            report += f"- Ohitukset: {level_stats['misses']}\n"
            report += f"- Osumataajuus: {level_stats['hit_ratio']:.1%}\n"
            report += f"- Koko: {level_stats['size']} alkiota\n"
            report += f"- Latenssi: {level_stats['latency']*1000:.2f}ms\n\n"
        
        return report

async def main():
    """Testaa välimuistin optimointia"""
    optimizer = CacheOptimizer()
    
    # Simuloi tehtäviä
    for i in range(100):
        task = {
            "id": i % 10,  # 10 uniikkia tehtävää
            "type": "test",
            "data": f"data_{i}"
        }
        
        # Yritä hakea välimuistista
        result = await optimizer.get(task)
        
        if result is None:
            # Simuloi laskentaa
            await asyncio.sleep(0.1)
            result = f"result_{i}"
            
            # Tallenna välimuistiin
            await optimizer.set(task, result)
    
    # Tulosta raportti
    print("\nCache Report:")
    print(optimizer.generate_report())

if __name__ == "__main__":
    asyncio.run(main())
