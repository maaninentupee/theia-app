import asyncio
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import json

import numpy as np
from prometheus_client import Counter, Gauge, Histogram
import aiohttp
import psutil

# Konfiguroi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Tehtävätyypit"""
    LIGHT = "light"     # Kevyt tehtävä
    MEDIUM = "medium"   # Keskitason tehtävä
    HEAVY = "heavy"     # Raskas tehtävä
    BURST = "burst"     # Pursketehtävä

class LoadPattern(Enum):
    """Kuormituskuviot"""
    CONSTANT = "constant"       # Tasainen kuorma
    RAMP_UP = "ramp_up"        # Nouseva kuorma
    SPIKE = "spike"            # Piikkikuorma
    RANDOM = "random"          # Satunnainen kuorma

@dataclass
class Task:
    """Tehtävä"""
    id: str
    type: TaskType
    payload: Any
    created_at: datetime
    timeout: Optional[float] = None
    retries: int = 0
    max_retries: int = 3

@dataclass
class TestConfig:
    """Testikonfiguraatio"""
    duration: int           # Testin kesto sekunteina
    pattern: LoadPattern    # Kuormituskuvio
    max_rps: float         # Maksimi RPS
    task_distribution: Dict[TaskType, float]  # Tehtävätyyppien jakauma

class LoadTester:
    """Kuormitustestaaja"""
    
    def __init__(
        self,
        config: TestConfig,
        worker_count: int = 8
    ):
        """
        Alusta testaaja
        
        Args:
            config: Testikonfiguraatio
            worker_count: Työntekijöiden määrä
        """
        self.config = config
        self.worker_count = worker_count
        
        # Tehtäväjono
        self.task_queue: asyncio.Queue = asyncio.Queue()
        
        # Työntekijät
        self.workers: List[asyncio.Task] = []
        
        # Tilastot
        self.task_counter = Counter(
            'test_tasks_total',
            'Total test tasks',
            ['type', 'status']
        )
        self.task_latency = Histogram(
            'test_latency_seconds',
            'Test task latency',
            ['type']
        )
        self.worker_gauge = Gauge(
            'test_workers',
            'Test workers'
        )
        self.queue_gauge = Gauge(
            'test_queue_size',
            'Test queue size'
        )
    
    async def run_test(self):
        """Suorita testi"""
        logger.info(
            f"Starting load test with pattern: {self.config.pattern.value}"
        )
        
        # Käynnistä työntekijät
        self.workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self.worker_count)
        ]
        
        # Käynnistä kuormageneraattori
        generator = asyncio.create_task(
            self._generate_load()
        )
        
        # Odota testin valmistumista
        try:
            await asyncio.sleep(self.config.duration)
        
        finally:
            # Pysäytä generaattori
            generator.cancel()
            
            # Odota jonon tyhjenemistä
            await self.task_queue.join()
            
            # Pysäytä työntekijät
            for worker in self.workers:
                worker.cancel()
            
            await asyncio.gather(
                *self.workers,
                return_exceptions=True
            )
    
    async def _generate_load(self):
        """Generoi kuormaa"""
        start_time = datetime.now()
        
        while True:
            try:
                # Laske nykyinen RPS
                elapsed = (
                    datetime.now() - start_time
                ).total_seconds()
                
                target_rps = self._calculate_target_rps(
                    elapsed
                )
                
                if target_rps > 0:
                    # Generoi tehtävä
                    task = self._generate_task()
                    await self.task_queue.put(task)
                    
                    # Päivitä metriikat
                    self.queue_gauge.set(
                        self.task_queue.qsize()
                    )
                    
                    # Odota seuraavaan tehtävään
                    await asyncio.sleep(1 / target_rps)
                
                else:
                    await asyncio.sleep(0.1)
            
            except asyncio.CancelledError:
                break
            
            except Exception as e:
                logger.error(
                    f"Load generation error: {str(e)}"
                )
                await asyncio.sleep(1)
    
    def _calculate_target_rps(
        self,
        elapsed: float
    ) -> float:
        """
        Laske kohde-RPS
        
        Args:
            elapsed: Kulunut aika sekunteina
        
        Returns:
            float: Kohde-RPS
        """
        if self.config.pattern == LoadPattern.CONSTANT:
            return self.config.max_rps
        
        elif self.config.pattern == LoadPattern.RAMP_UP:
            progress = elapsed / self.config.duration
            return self.config.max_rps * min(1.0, progress)
        
        elif self.config.pattern == LoadPattern.SPIKE:
            period = self.config.duration / 4
            phase = (elapsed % period) / period
            
            if phase < 0.4:  # Nouseva
                return self.config.max_rps * (phase / 0.4)
            elif phase < 0.6:  # Huippu
                return self.config.max_rps
            else:  # Laskeva
                return self.config.max_rps * (1 - (phase - 0.6) / 0.4)
        
        else:  # RANDOM
            return random.uniform(
                0,
                self.config.max_rps
            )
    
    def _generate_task(self) -> Task:
        """
        Generoi tehtävä
        
        Returns:
            Task: Generoitu tehtävä
        """
        # Valitse tyyppi
        task_type = random.choices(
            list(TaskType),
            weights=[
                self.config.task_distribution[t]
                for t in TaskType
            ]
        )[0]
        
        # Generoi payload
        if task_type == TaskType.LIGHT:
            payload = {
                "data": "x" * random.randint(100, 1000)
            }
            timeout = 1.0
        
        elif task_type == TaskType.MEDIUM:
            payload = {
                "data": "x" * random.randint(1000, 10000)
            }
            timeout = 2.0
        
        elif task_type == TaskType.HEAVY:
            payload = {
                "data": "x" * random.randint(10000, 100000)
            }
            timeout = 5.0
        
        else:  # BURST
            payload = {
                "data": "x" * random.randint(1000, 5000),
                "burst_size": random.randint(10, 50)
            }
            timeout = 3.0
        
        return Task(
            id=f"task_{datetime.now().timestamp()}",
            type=task_type,
            payload=payload,
            created_at=datetime.now(),
            timeout=timeout
        )
    
    async def _worker(self, worker_id: int):
        """
        Työntekijä
        
        Args:
            worker_id: Työntekijän ID
        """
        logger.info(f"Worker {worker_id} started")
        self.worker_gauge.inc()
        
        try:
            while True:
                # Hae tehtävä
                task = await self.task_queue.get()
                start_time = datetime.now()
                
                try:
                    # Suorita tehtävä
                    await self._process_task(task)
                    
                    # Päivitä metriikat
                    duration = (
                        datetime.now() - start_time
                    ).total_seconds()
                    
                    self.task_counter.labels(
                        type=task.type.value,
                        status="success"
                    ).inc()
                    
                    self.task_latency.labels(
                        type=task.type.value
                    ).observe(duration)
                
                except Exception as e:
                    logger.error(
                        f"Task {task.id} failed: {str(e)}"
                    )
                    
                    self.task_counter.labels(
                        type=task.type.value,
                        status="error"
                    ).inc()
                
                finally:
                    self.task_queue.task_done()
                    self.queue_gauge.set(
                        self.task_queue.qsize()
                    )
        
        except asyncio.CancelledError:
            logger.info(f"Worker {worker_id} stopped")
            self.worker_gauge.dec()
    
    async def _process_task(self, task: Task):
        """
        Suorita tehtävä
        
        Args:
            task: Tehtävä
        """
        # Simuloi prosessointia
        if task.type == TaskType.BURST:
            # Suorita purske
            for _ in range(task.payload["burst_size"]):
                await asyncio.sleep(0.1)
        else:
            # Simuloi työkuormaa
            await asyncio.sleep(
                random.uniform(0.1, task.timeout or 1.0)
            )
        
        # Simuloi virheitä
        if random.random() < 0.05:  # 5% virhetodennäköisyys
            raise Exception("Simulated error")
    
    def get_stats(self) -> Dict:
        """
        Hae tilastot
        
        Returns:
            Dict: Tilastot
        """
        stats = {
            "workers": len(self.workers),
            "queue_size": self.task_queue.qsize(),
            "tasks": {
                task_type.value: {
                    "success": self.task_counter.labels(
                        type=task_type.value,
                        status="success"
                    )._value.get(),
                    "error": self.task_counter.labels(
                        type=task_type.value,
                        status="error"
                    )._value.get()
                }
                for task_type in TaskType
            }
        }
        
        return stats
    
    def generate_report(self) -> str:
        """
        Generoi raportti
        
        Returns:
            str: Markdown-muotoinen raportti
        """
        stats = self.get_stats()
        
        report = f"""# Kuormitustestiraportti

## Konfiguraatio

- Kuvio: {self.config.pattern.value}
- Kesto: {self.config.duration}s
- Maksimi RPS: {self.config.max_rps}
- Työntekijät: {self.worker_count}

## Suorituskyky

- Aktiiviset työntekijät: {stats['workers']}
- Jonon pituus: {stats['queue_size']}

## Tehtävät tyypeittäin

"""
        
        for task_type, counts in stats["tasks"].items():
            total = counts["success"] + counts["error"]
            if total > 0:
                error_rate = counts["error"] / total * 100
                report += f"### {task_type}\n"
                report += f"- Onnistuneet: {counts['success']}\n"
                report += f"- Virheet: {counts['error']}\n"
                report += f"- Virheprosentti: {error_rate:.1f}%\n\n"
        
        return report

async def main():
    """Testaa kuormitustestaajaa"""
    # Alusta konfiguraatio
    config = TestConfig(
        duration=60,  # 1min testi
        pattern=LoadPattern.RAMP_UP,
        max_rps=100.0,
        task_distribution={
            TaskType.LIGHT: 0.4,    # 40%
            TaskType.MEDIUM: 0.3,   # 30%
            TaskType.HEAVY: 0.2,    # 20%
            TaskType.BURST: 0.1     # 10%
        }
    )
    
    # Alusta testaaja
    tester = LoadTester(config)
    
    # Suorita testi
    await tester.run_test()
    
    # Tulosta raportti
    print("\nTest Report:")
    print(tester.generate_report())

if __name__ == "__main__":
    asyncio.run(main())
