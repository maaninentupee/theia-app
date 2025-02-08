import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
import json
import hashlib
import pickle
from collections import OrderedDict

from prometheus_client import Counter, Gauge, Histogram
import aioredis
from cachetools import LRUCache, TTLCache, cached
import msgpack

# Konfiguroi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Välimuististrategiat"""
    LRU = "lru"       # Least Recently Used
    TTL = "ttl"       # Time To Live
    HYBRID = "hybrid" # LRU + TTL
    ADAPTIVE = "adaptive"  # Adaptiivinen

class CacheLevel(Enum):
    """Välimuistitasot"""
    MEMORY = "memory"   # Muistivälimuisti
    REDIS = "redis"     # Redis-välimuisti
    DISK = "disk"       # Levyvälimuisti

@dataclass
class CacheConfig:
    """Välimuistikonfiguraatio"""
    strategy: CacheStrategy
    max_size: int
    ttl: int  # sekunteina
    levels: List[CacheLevel]
    compress: bool = True
    validate: bool = True

@dataclass
class CacheItem:
    """Välimuistiobjekti"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_access: Optional[datetime] = None
    size: int = 0

class CacheManager:
    """Välimuistin hallinta"""
    
    def __init__(
        self,
        config: CacheConfig,
        redis_url: Optional[str] = None
    ):
        """
        Alusta hallinta
        
        Args:
            config: Konfiguraatio
            redis_url: Redis-URL
        """
        self.config = config
        self.redis_url = redis_url
        
        # Muistivälimuisti
        if CacheLevel.MEMORY in config.levels:
            if config.strategy == CacheStrategy.LRU:
                self.memory_cache = LRUCache(
                    maxsize=config.max_size
                )
            elif config.strategy == CacheStrategy.TTL:
                self.memory_cache = TTLCache(
                    maxsize=config.max_size,
                    ttl=config.ttl
                )
            else:  # HYBRID tai ADAPTIVE
                self.memory_cache = OrderedDict()
        
        # Redis-välimuisti
        self.redis: Optional[aioredis.Redis] = None
        if CacheLevel.REDIS in config.levels:
            self.redis = aioredis.from_url(redis_url)
        
        # Metriikat
        self.hit_counter = Counter(
            'cache_hits_total',
            'Cache hits',
            ['level']
        )
        self.miss_counter = Counter(
            'cache_misses_total',
            'Cache misses',
            ['level']
        )
        self.size_gauge = Gauge(
            'cache_size_bytes',
            'Cache size',
            ['level']
        )
        self.item_gauge = Gauge(
            'cache_items',
            'Cache items',
            ['level']
        )
        self.latency = Histogram(
            'cache_latency_seconds',
            'Cache latency',
            ['level', 'operation']
        )
    
    async def get(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Hae arvo
        
        Args:
            key: Avain
            default: Oletusarvo
        
        Returns:
            Any: Arvo
        """
        # Generoi hash
        cache_key = self._hash_key(key)
        
        # Kokeile muistia
        if hasattr(self, 'memory_cache'):
            value = await self._get_from_memory(cache_key)
            if value is not None:
                return value
        
        # Kokeile Redisiä
        if self.redis:
            value = await self._get_from_redis(cache_key)
            if value is not None:
                return value
        
        # Kokeile levyä
        if CacheLevel.DISK in self.config.levels:
            value = await self._get_from_disk(cache_key)
            if value is not None:
                return value
        
        return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """
        Aseta arvo
        
        Args:
            key: Avain
            value: Arvo
            ttl: TTL sekunteina
        """
        # Generoi hash
        cache_key = self._hash_key(key)
        
        # Validoi arvo
        if self.config.validate:
            self._validate_value(value)
        
        # Pakkaa arvo
        packed_value = self._pack_value(value)
        
        # Tallenna muistiin
        if hasattr(self, 'memory_cache'):
            await self._set_to_memory(
                cache_key,
                packed_value,
                ttl
            )
        
        # Tallenna Redisiin
        if self.redis:
            await self._set_to_redis(
                cache_key,
                packed_value,
                ttl
            )
        
        # Tallenna levylle
        if CacheLevel.DISK in self.config.levels:
            await self._set_to_disk(
                cache_key,
                packed_value,
                ttl
            )
    
    async def delete(self, key: str):
        """
        Poista arvo
        
        Args:
            key: Avain
        """
        cache_key = self._hash_key(key)
        
        # Poista muistista
        if hasattr(self, 'memory_cache'):
            self.memory_cache.pop(cache_key, None)
        
        # Poista Redisistä
        if self.redis:
            await self.redis.delete(cache_key)
        
        # Poista levyltä
        if CacheLevel.DISK in self.config.levels:
            await self._delete_from_disk(cache_key)
    
    async def clear(self):
        """Tyhjennä välimuisti"""
        # Tyhjennä muisti
        if hasattr(self, 'memory_cache'):
            self.memory_cache.clear()
        
        # Tyhjennä Redis
        if self.redis:
            await self.redis.flushdb()
        
        # Tyhjennä levy
        if CacheLevel.DISK in self.config.levels:
            await self._clear_disk()
    
    def _hash_key(self, key: str) -> str:
        """
        Generoi hash
        
        Args:
            key: Avain
        
        Returns:
            str: Hash
        """
        if isinstance(key, str):
            key = key.encode()
        return hashlib.sha256(key).hexdigest()
    
    def _validate_value(self, value: Any):
        """
        Validoi arvo
        
        Args:
            value: Arvo
        """
        try:
            # Tarkista serialisoitavuus
            pickle.dumps(value)
        except Exception as e:
            raise ValueError(
                f"Value not serializable: {str(e)}"
            )
    
    def _pack_value(self, value: Any) -> bytes:
        """
        Pakkaa arvo
        
        Args:
            value: Arvo
        
        Returns:
            bytes: Pakattu arvo
        """
        if self.config.compress:
            return msgpack.packb(
                value,
                use_bin_type=True
            )
        return pickle.dumps(value)
    
    def _unpack_value(self, value: bytes) -> Any:
        """
        Pura arvo
        
        Args:
            value: Pakattu arvo
        
        Returns:
            Any: Arvo
        """
        if self.config.compress:
            return msgpack.unpackb(
                value,
                raw=False
            )
        return pickle.loads(value)
    
    async def _get_from_memory(
        self,
        key: str
    ) -> Optional[Any]:
        """
        Hae muistista
        
        Args:
            key: Avain
        
        Returns:
            Optional[Any]: Arvo
        """
        start_time = datetime.now()
        
        try:
            value = self.memory_cache.get(key)
            
            if value is not None:
                # Päivitä metriikat
                self.hit_counter.labels(
                    level='memory'
                ).inc()
                
                # Pura arvo
                return self._unpack_value(value)
            
            self.miss_counter.labels(
                level='memory'
            ).inc()
            
            return None
        
        finally:
            duration = (
                datetime.now() - start_time
            ).total_seconds()
            
            self.latency.labels(
                level='memory',
                operation='get'
            ).observe(duration)
    
    async def _get_from_redis(
        self,
        key: str
    ) -> Optional[Any]:
        """
        Hae Redisistä
        
        Args:
            key: Avain
        
        Returns:
            Optional[Any]: Arvo
        """
        start_time = datetime.now()
        
        try:
            value = await self.redis.get(key)
            
            if value is not None:
                # Päivitä metriikat
                self.hit_counter.labels(
                    level='redis'
                ).inc()
                
                # Pura arvo
                return self._unpack_value(value)
            
            self.miss_counter.labels(
                level='redis'
            ).inc()
            
            return None
        
        finally:
            duration = (
                datetime.now() - start_time
            ).total_seconds()
            
            self.latency.labels(
                level='redis',
                operation='get'
            ).observe(duration)
    
    async def _get_from_disk(
        self,
        key: str
    ) -> Optional[Any]:
        """
        Hae levyltä
        
        Args:
            key: Avain
        
        Returns:
            Optional[Any]: Arvo
        """
        start_time = datetime.now()
        
        try:
            # Tarkista tiedosto
            path = f"cache/{key}"
            
            try:
                with open(path, 'rb') as f:
                    value = f.read()
            except FileNotFoundError:
                self.miss_counter.labels(
                    level='disk'
                ).inc()
                return None
            
            # Päivitä metriikat
            self.hit_counter.labels(
                level='disk'
            ).inc()
            
            # Pura arvo
            return self._unpack_value(value)
        
        finally:
            duration = (
                datetime.now() - start_time
            ).total_seconds()
            
            self.latency.labels(
                level='disk',
                operation='get'
            ).observe(duration)
    
    async def _set_to_memory(
        self,
        key: str,
        value: bytes,
        ttl: Optional[int]
    ):
        """
        Tallenna muistiin
        
        Args:
            key: Avain
            value: Arvo
            ttl: TTL sekunteina
        """
        start_time = datetime.now()
        
        try:
            if self.config.strategy == CacheStrategy.ADAPTIVE:
                # Tarkista tila
                if len(self.memory_cache) >= self.config.max_size:
                    # Poista vähiten käytetty
                    self._evict_lru()
            
            # Tallenna
            self.memory_cache[key] = value
            
            # Päivitä metriikat
            self.size_gauge.labels(
                level='memory'
            ).set(len(value))
            
            self.item_gauge.labels(
                level='memory'
            ).set(len(self.memory_cache))
        
        finally:
            duration = (
                datetime.now() - start_time
            ).total_seconds()
            
            self.latency.labels(
                level='memory',
                operation='set'
            ).observe(duration)
    
    async def _set_to_redis(
        self,
        key: str,
        value: bytes,
        ttl: Optional[int]
    ):
        """
        Tallenna Redisiin
        
        Args:
            key: Avain
            value: Arvo
            ttl: TTL sekunteina
        """
        start_time = datetime.now()
        
        try:
            if ttl:
                await self.redis.setex(
                    key,
                    ttl,
                    value
                )
            else:
                await self.redis.set(key, value)
            
            # Päivitä metriikat
            self.size_gauge.labels(
                level='redis'
            ).set(len(value))
            
            self.item_gauge.labels(
                level='redis'
            ).set(
                await self.redis.dbsize()
            )
        
        finally:
            duration = (
                datetime.now() - start_time
            ).total_seconds()
            
            self.latency.labels(
                level='redis',
                operation='set'
            ).observe(duration)
    
    async def _set_to_disk(
        self,
        key: str,
        value: bytes,
        ttl: Optional[int]
    ):
        """
        Tallenna levylle
        
        Args:
            key: Avain
            value: Arvo
            ttl: TTL sekunteina
        """
        start_time = datetime.now()
        
        try:
            # Luo hakemisto
            import os
            os.makedirs('cache', exist_ok=True)
            
            # Tallenna tiedosto
            path = f"cache/{key}"
            with open(path, 'wb') as f:
                f.write(value)
            
            # Päivitä metriikat
            self.size_gauge.labels(
                level='disk'
            ).set(len(value))
            
            self.item_gauge.labels(
                level='disk'
            ).set(
                len(os.listdir('cache'))
            )
        
        finally:
            duration = (
                datetime.now() - start_time
            ).total_seconds()
            
            self.latency.labels(
                level='disk',
                operation='set'
            ).observe(duration)
    
    def _evict_lru(self):
        """Poista vähiten käytetty"""
        if not self.memory_cache:
            return
        
        # Poista vanhin
        self.memory_cache.popitem(last=False)
    
    async def _delete_from_disk(self, key: str):
        """
        Poista levyltä
        
        Args:
            key: Avain
        """
        import os
        path = f"cache/{key}"
        
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    
    async def _clear_disk(self):
        """Tyhjennä levy"""
        import os
        import shutil
        
        try:
            shutil.rmtree('cache')
        except FileNotFoundError:
            pass
    
    def get_stats(self) -> Dict:
        """
        Hae tilastot
        
        Returns:
            Dict: Tilastot
        """
        stats = {
            "strategy": self.config.strategy.value,
            "levels": [l.value for l in self.config.levels],
            "memory": {
                "size": 0,
                "items": 0
            },
            "redis": {
                "size": 0,
                "items": 0
            },
            "disk": {
                "size": 0,
                "items": 0
            }
        }
        
        # Muisti
        if hasattr(self, 'memory_cache'):
            stats["memory"] = {
                "size": sum(
                    len(v) for v in
                    self.memory_cache.values()
                ),
                "items": len(self.memory_cache)
            }
        
        # Redis
        if self.redis:
            stats["redis"] = {
                "size": self.size_gauge.labels(
                    level='redis'
                )._value.get(),
                "items": self.item_gauge.labels(
                    level='redis'
                )._value.get()
            }
        
        # Levy
        if CacheLevel.DISK in self.config.levels:
            import os
            
            try:
                files = os.listdir('cache')
                stats["disk"] = {
                    "size": sum(
                        os.path.getsize(f"cache/{f}")
                        for f in files
                    ),
                    "items": len(files)
                }
            except FileNotFoundError:
                pass
        
        return stats
    
    def generate_report(self) -> str:
        """
        Generoi raportti
        
        Returns:
            str: Markdown-muotoinen raportti
        """
        stats = self.get_stats()
        
        report = """# Välimuistiraportti

## Konfiguraatio

"""
        report += f"- Strategia: {stats['strategy']}\n"
        report += f"- Tasot: {', '.join(stats['levels'])}\n\n"
        
        report += "## Muisti\n\n"
        report += f"- Koko: {stats['memory']['size']} tavua\n"
        report += f"- Objekteja: {stats['memory']['items']}\n\n"
        
        report += "## Redis\n\n"
        report += f"- Koko: {stats['redis']['size']} tavua\n"
        report += f"- Objekteja: {stats['redis']['items']}\n\n"
        
        report += "## Levy\n\n"
        report += f"- Koko: {stats['disk']['size']} tavua\n"
        report += f"- Objekteja: {stats['disk']['items']}\n"
        
        return report

async def main():
    """Testaa välimuistia"""
    # Alusta konfiguraatio
    config = CacheConfig(
        strategy=CacheStrategy.ADAPTIVE,
        max_size=1000,
        ttl=3600,
        levels=[
            CacheLevel.MEMORY,
            CacheLevel.REDIS,
            CacheLevel.DISK
        ],
        compress=True
    )
    
    # Alusta hallinta
    cache = CacheManager(
        config,
        redis_url="redis://localhost"
    )
    
    # Testaa toimintoja
    await cache.set("key1", "value1")
    await cache.set("key2", {"data": "value2"})
    await cache.set("key3", [1, 2, 3])
    
    print("Value 1:", await cache.get("key1"))
    print("Value 2:", await cache.get("key2"))
    print("Value 3:", await cache.get("key3"))
    
    # Tulosta raportti
    print("\nCache Report:")
    print(cache.generate_report())

if __name__ == "__main__":
    asyncio.run(main())
