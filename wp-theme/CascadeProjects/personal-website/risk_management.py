import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

# Konfiguroi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Riskitasot"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskCategory(Enum):
    """Riskikategoriat"""
    API = "api"
    TOKEN = "token"
    PERFORMANCE = "performance"
    SECURITY = "security"

@dataclass
class RiskThreshold:
    """Riskikynnykset"""
    warning: float
    critical: float
    cooldown: int  # sekunteina

@dataclass
class Risk:
    """Riski"""
    category: RiskCategory
    level: RiskLevel
    description: str
    mitigation: str
    threshold: RiskThreshold
    current_value: float
    last_triggered: Optional[datetime] = None

class LoadBalancer:
    """Kuormantasaaja"""
    
    def __init__(self, max_concurrent: int = 10):
        """
        Alusta kuormantasaaja
        
        Args:
            max_concurrent: Maksimi rinnakkaisten tehtävien määrä
        """
        self.max_concurrent = max_concurrent
        self.current_load = 0
        self.queue: asyncio.Queue = asyncio.Queue()
        self.active_tasks: Dict[str, asyncio.Task] = {}
        
        # Metriikat
        self.queue_size = Gauge(
            'task_queue_size',
            'Number of tasks in queue'
        )
        self.active_tasks_gauge = Gauge(
            'active_tasks',
            'Number of active tasks'
        )
        self.task_latency = Histogram(
            'task_latency_seconds',
            'Task execution latency'
        )
    
    async def submit_task(
        self,
        task_id: str,
        coroutine,
        priority: int = 0
    ):
        """
        Lähetä tehtävä
        
        Args:
            task_id: Tehtävän ID
            coroutine: Suoritettava coroutine
            priority: Prioriteetti (korkeampi = tärkeämpi)
        """
        await self.queue.put((priority, task_id, coroutine))
        self.queue_size.inc()
        
        if self.current_load < self.max_concurrent:
            asyncio.create_task(self._process_queue())
    
    async def _process_queue(self):
        """Käsittele jonoa"""
        try:
            self.current_load += 1
            self.active_tasks_gauge.inc()
            
            while not self.queue.empty():
                priority, task_id, coroutine = await self.queue.get()
                self.queue_size.dec()
                
                start_time = time.time()
                try:
                    task = asyncio.create_task(coroutine)
                    self.active_tasks[task_id] = task
                    await task
                except Exception as e:
                    logger.error(
                        f"Task {task_id} failed: {str(e)}"
                    )
                finally:
                    self.active_tasks.pop(task_id, None)
                    self.task_latency.observe(
                        time.time() - start_time
                    )
        
        finally:
            self.current_load -= 1
            self.active_tasks_gauge.dec()

class TokenManager:
    """Token-hallinta"""
    
    def __init__(
        self,
        max_tokens_per_minute: int = 100000,
        max_tokens_per_request: int = 4000
    ):
        """
        Alusta token-hallinta
        
        Args:
            max_tokens_per_minute: Maksimi tokenit per minuutti
            max_tokens_per_request: Maksimi tokenit per pyyntö
        """
        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_tokens_per_request = max_tokens_per_request
        self.token_usage = []
        self.last_reset = datetime.now()
        
        # Metriikat
        self.token_usage_gauge = Gauge(
            'token_usage',
            'Token usage per minute'
        )
        self.token_limit_hits = Counter(
            'token_limit_hits',
            'Number of token limit hits'
        )
    
    def can_use_tokens(self, tokens: int) -> bool:
        """
        Tarkista voiko tokeneita käyttää
        
        Args:
            tokens: Tokenien määrä
        
        Returns:
            bool: True jos voi käyttää
        """
        # Päivitä käyttö
        self._update_usage()
        
        # Tarkista rajat
        if tokens > self.max_tokens_per_request:
            self.token_limit_hits.inc()
            return False
        
        current_usage = sum(self.token_usage)
        if current_usage + tokens > self.max_tokens_per_minute:
            self.token_limit_hits.inc()
            return False
        
        return True
    
    def use_tokens(self, tokens: int):
        """
        Käytä tokeneita
        
        Args:
            tokens: Tokenien määrä
        """
        self._update_usage()
        self.token_usage.append(tokens)
        self.token_usage_gauge.set(sum(self.token_usage))
    
    def _update_usage(self):
        """Päivitä token-käyttö"""
        now = datetime.now()
        if now - self.last_reset >= timedelta(minutes=1):
            self.token_usage = []
            self.last_reset = now

class RiskManager:
    """Riskienhallinta"""
    
    def __init__(
        self,
        load_balancer: LoadBalancer,
        token_manager: TokenManager
    ):
        """
        Alusta riskienhallinta
        
        Args:
            load_balancer: Kuormantasaaja
            token_manager: Token-hallinta
        """
        self.load_balancer = load_balancer
        self.token_manager = token_manager
        
        # Riskit ja kynnykset
        self.risks: Dict[str, Risk] = {
            "api_rate_limit": Risk(
                category=RiskCategory.API,
                level=RiskLevel.LOW,
                description="API-pyyntöjen määrä lähestyy rajaa",
                mitigation="Käytetään kuormantasausta",
                threshold=RiskThreshold(
                    warning=0.7,
                    critical=0.9,
                    cooldown=60
                ),
                current_value=0.0
            ),
            "token_usage": Risk(
                category=RiskCategory.TOKEN,
                level=RiskLevel.LOW,
                description="Token-käyttö lähestyy rajaa",
                mitigation="Käytetään adaptiivista hallintaa",
                threshold=RiskThreshold(
                    warning=0.7,
                    critical=0.9,
                    cooldown=60
                ),
                current_value=0.0
            )
        }
        
        # Metriikat
        self.risk_levels = Gauge(
            'risk_levels',
            'Current risk levels',
            ['category']
        )
    
    async def monitor_risks(self):
        """Monitoroi riskejä"""
        while True:
            try:
                # Päivitä API-riski
                api_load = (
                    self.load_balancer.current_load /
                    self.load_balancer.max_concurrent
                )
                self._update_risk(
                    "api_rate_limit",
                    api_load
                )
                
                # Päivitä token-riski
                token_usage = sum(
                    self.token_manager.token_usage
                ) / self.token_manager.max_tokens_per_minute
                self._update_risk(
                    "token_usage",
                    token_usage
                )
                
                # Odota sekunti
                await asyncio.sleep(1)
            
            except Exception as e:
                logger.error(f"Risk monitoring failed: {str(e)}")
                await asyncio.sleep(5)
    
    def _update_risk(self, risk_id: str, current_value: float):
        """
        Päivitä riski
        
        Args:
            risk_id: Riskin ID
            current_value: Nykyinen arvo
        """
        risk = self.risks[risk_id]
        risk.current_value = current_value
        
        # Tarkista kynnykset
        if current_value >= risk.threshold.critical:
            new_level = RiskLevel.CRITICAL
        elif current_value >= risk.threshold.warning:
            new_level = RiskLevel.HIGH
        elif current_value >= risk.threshold.warning * 0.5:
            new_level = RiskLevel.MEDIUM
        else:
            new_level = RiskLevel.LOW
        
        # Päivitä taso jos cooldown ohi
        if (
            risk.last_triggered is None or
            datetime.now() - risk.last_triggered >=
            timedelta(seconds=risk.threshold.cooldown)
        ):
            if new_level != risk.level:
                risk.level = new_level
                risk.last_triggered = datetime.now()
                self._handle_risk_change(risk_id, risk)
        
        # Päivitä metriikka
        self.risk_levels.labels(
            category=risk.category.value
        ).set(len(RiskLevel) - list(RiskLevel).index(risk.level))
    
    def _handle_risk_change(self, risk_id: str, risk: Risk):
        """
        Käsittele riskin muutos
        
        Args:
            risk_id: Riskin ID
            risk: Riski
        """
        logger.warning(
            f"Risk {risk_id} changed to {risk.level.value}: "
            f"{risk.description}"
        )
        
        if risk.level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            logger.error(
                f"Critical risk {risk_id}: {risk.mitigation}"
            )
            
            # Implementoi automaattiset toimenpiteet
            if risk.category == RiskCategory.API:
                self.load_balancer.max_concurrent = max(
                    1,
                    self.load_balancer.max_concurrent // 2
                )
            
            elif risk.category == RiskCategory.TOKEN:
                self.token_manager.max_tokens_per_request = max(
                    1000,
                    self.token_manager.max_tokens_per_request // 2
                )

async def main():
    """Testaa riskienhallintaa"""
    # Alusta komponentit
    load_balancer = LoadBalancer(max_concurrent=5)
    token_manager = TokenManager(
        max_tokens_per_minute=50000,
        max_tokens_per_request=2000
    )
    risk_manager = RiskManager(
        load_balancer,
        token_manager
    )
    
    # Käynnistä monitorointi
    monitor_task = asyncio.create_task(
        risk_manager.monitor_risks()
    )
    
    # Simuloi kuormaa
    async def test_task(i: int):
        logger.info(f"Task {i} started")
        await asyncio.sleep(1)
        logger.info(f"Task {i} completed")
    
    # Lähetä tehtäviä
    for i in range(10):
        await load_balancer.submit_task(
            f"task_{i}",
            test_task(i)
        )
        token_manager.use_tokens(1000)
    
    # Odota tehtävien valmistumista
    await asyncio.sleep(5)
    
    # Pysäytä monitorointi
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    asyncio.run(main())
