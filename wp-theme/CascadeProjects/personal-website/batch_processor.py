import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Iterator
import json
import math

from prometheus_client import Counter, Gauge, Histogram
import numpy as np

# Konfiguroi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchStrategy(Enum):
    """Erästrategiat"""
    FIXED = "fixed"       # Kiinteä koko
    DYNAMIC = "dynamic"   # Dynaaminen koko
    ADAPTIVE = "adaptive" # Adaptiivinen
    SMART = "smart"       # Älykäs

class BatchPriority(Enum):
    """Eräprioriteetit"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class BatchConfig:
    """Eräkonfiguraatio"""
    strategy: BatchStrategy
    min_size: int
    max_size: int
    timeout: int  # sekunteina
    max_retries: int = 3
    parallel_batches: int = 4

@dataclass
class Batch:
    """Erä"""
    id: str
    tasks: List[Any]
    priority: BatchPriority
    created_at: datetime
    timeout: int
    retries: int = 0
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None

class BatchProcessor:
    """Eräkäsittelijä"""
    
    def __init__(
        self,
        config: BatchConfig,
        process_func: Callable
    ):
        """
        Alusta käsittelijä
        
        Args:
            config: Konfiguraatio
            process_func: Käsittelyfunktio
        """
        self.config = config
        self.process_func = process_func
        
        # Erät
        self.batches: Dict[str, Batch] = {}
        self.pending: asyncio.Queue = asyncio.Queue()
        
        # Työntekijät
        self.workers: List[asyncio.Task] = []
        
        # Metriikat
        self.batch_counter = Counter(
            'batches_total',
            'Total batches',
            ['priority', 'status']
        )
        self.task_counter = Counter(
            'batch_tasks_total',
            'Total batch tasks',
            ['status']
        )
        self.batch_size = Histogram(
            'batch_size',
            'Batch size'
        )
        self.batch_latency = Histogram(
            'batch_latency_seconds',
            'Batch latency',
            ['priority']
        )
        self.worker_gauge = Gauge(
            'batch_workers',
            'Batch workers'
        )
    
    async def start(self):
        """Käynnistä käsittelijä"""
        logger.info("Starting batch processor")
        
        # Käynnistä työntekijät
        self.workers = [
            asyncio.create_task(self._worker())
            for _ in range(self.config.parallel_batches)
        ]
        
        # Päivitä metriikat
        self.worker_gauge.set(len(self.workers))
    
    async def stop(self):
        """Pysäytä käsittelijä"""
        logger.info("Stopping batch processor")
        
        # Pysäytä työntekijät
        for worker in self.workers:
            worker.cancel()
        
        await asyncio.gather(
            *self.workers,
            return_exceptions=True
        )
    
    def split_tasks(
        self,
        tasks: List[Any],
        priority: BatchPriority = BatchPriority.NORMAL
    ) -> Iterator[List[Any]]:
        """
        Jaa tehtävät eriin
        
        Args:
            tasks: Tehtävät
            priority: Prioriteetti
        
        Returns:
            Iterator[List[Any]]: Erät
        """
        if self.config.strategy == BatchStrategy.FIXED:
            # Kiinteä koko
            batch_size = self.config.max_size
        
        elif self.config.strategy == BatchStrategy.DYNAMIC:
            # Dynaaminen koko perustuen jonoon
            queue_size = self.pending.qsize()
            if queue_size > 100:
                batch_size = self.config.max_size
            elif queue_size > 50:
                batch_size = self.config.max_size // 2
            else:
                batch_size = self.config.min_size
        
        elif self.config.strategy == BatchStrategy.ADAPTIVE:
            # Adaptiivinen koko perustuen suorituskykyyn
            performance = self._analyze_performance()
            batch_size = int(
                self.config.min_size +
                (self.config.max_size - self.config.min_size) *
                performance
            )
        
        else:  # SMART
            # Älykäs koko perustuen moneen tekijään
            batch_size = self._calculate_smart_size(
                len(tasks),
                priority
            )
        
        # Jaa eriin
        for i in range(0, len(tasks), batch_size):
            yield tasks[i:i + batch_size]
    
    async def submit(
        self,
        tasks: List[Any],
        priority: BatchPriority = BatchPriority.NORMAL
    ) -> List[str]:
        """
        Lähetä tehtävät
        
        Args:
            tasks: Tehtävät
            priority: Prioriteetti
        
        Returns:
            List[str]: Erien ID:t
        """
        batch_ids = []
        
        # Jaa eriin
        for batch_tasks in self.split_tasks(tasks, priority):
            # Luo erä
            batch = Batch(
                id=f"batch_{len(self.batches)}",
                tasks=batch_tasks,
                priority=priority,
                created_at=datetime.now(),
                timeout=self.config.timeout
            )
            
            # Tallenna erä
            self.batches[batch.id] = batch
            await self.pending.put(batch)
            batch_ids.append(batch.id)
            
            # Päivitä metriikat
            self.batch_counter.labels(
                priority=priority.name,
                status="pending"
            ).inc()
            
            self.task_counter.labels(
                status="pending"
            ).inc(len(batch_tasks))
            
            self.batch_size.observe(len(batch_tasks))
        
        return batch_ids
    
    async def get_results(
        self,
        batch_ids: List[str],
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Hae tulokset
        
        Args:
            batch_ids: Erien ID:t
            timeout: Aikakatkaisu
        
        Returns:
            Dict[str, Any]: Tulokset
        """
        results = {}
        start_time = datetime.now()
        
        for batch_id in batch_ids:
            batch = self.batches.get(batch_id)
            if not batch:
                continue
            
            # Odota valmistumista
            while batch.status == "pending":
                if timeout and (
                    datetime.now() - start_time
                ).total_seconds() > timeout:
                    raise TimeoutError(
                        f"Batch {batch_id} timed out"
                    )
                
                await asyncio.sleep(0.1)
            
            if batch.status == "completed":
                results[batch_id] = batch.result
            else:
                results[batch_id] = {
                    "error": batch.error
                }
        
        return results
    
    async def _worker(self):
        """Työntekijä"""
        while True:
            try:
                # Hae erä
                batch = await self.pending.get()
                start_time = datetime.now()
                
                try:
                    # Suorita erä
                    batch.status = "running"
                    batch.result = await self.process_func(
                        batch.tasks
                    )
                    batch.status = "completed"
                    
                    # Päivitä metriikat
                    self.batch_counter.labels(
                        priority=batch.priority.name,
                        status="completed"
                    ).inc()
                    
                    self.task_counter.labels(
                        status="completed"
                    ).inc(len(batch.tasks))
                
                except Exception as e:
                    # Yritä uudelleen
                    if batch.retries < self.config.max_retries:
                        batch.retries += 1
                        batch.status = "pending"
                        await self.pending.put(batch)
                        
                        self.batch_counter.labels(
                            priority=batch.priority.name,
                            status="retrying"
                        ).inc()
                    
                    else:
                        batch.status = "failed"
                        batch.error = str(e)
                        
                        self.batch_counter.labels(
                            priority=batch.priority.name,
                            status="failed"
                        ).inc()
                        
                        self.task_counter.labels(
                            status="failed"
                        ).inc(len(batch.tasks))
                
                finally:
                    # Päivitä latenssi
                    duration = (
                        datetime.now() - start_time
                    ).total_seconds()
                    
                    self.batch_latency.labels(
                        priority=batch.priority.name
                    ).observe(duration)
                    
                    # Merkitse jono käsitellyksi
                    self.pending.task_done()
            
            except asyncio.CancelledError:
                break
            
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")
                await asyncio.sleep(1)
    
    def _analyze_performance(self) -> float:
        """
        Analysoi suorituskyky
        
        Returns:
            float: Suorituskyky (0-1)
        """
        # Hae latenssihistoria
        latencies = []
        for batch in self.batches.values():
            if batch.status == "completed":
                duration = (
                    batch.completed_at -
                    batch.created_at
                ).total_seconds()
                latencies.append(duration)
        
        if not latencies:
            return 0.5  # Oletusarvo
        
        # Laske suorituskyky
        avg_latency = np.mean(latencies)
        target_latency = 1.0  # Tavoitelatenssi
        
        return max(0, min(
            1,
            1 - (avg_latency / target_latency)
        ))
    
    def _calculate_smart_size(
        self,
        total_tasks: int,
        priority: BatchPriority
    ) -> int:
        """
        Laske älykäs eräkoko
        
        Args:
            total_tasks: Tehtävien määrä
            priority: Prioriteetti
        
        Returns:
            int: Eräkoko
        """
        # Huomioi prioriteetti
        priority_factor = {
            BatchPriority.LOW: 0.5,
            BatchPriority.NORMAL: 1.0,
            BatchPriority.HIGH: 1.5,
            BatchPriority.CRITICAL: 2.0
        }[priority]
        
        # Huomioi jonon koko
        queue_factor = max(0.5, min(
            2.0,
            1 / (1 + self.pending.qsize() / 100)
        ))
        
        # Huomioi suorituskyky
        performance = self._analyze_performance()
        
        # Laske koko
        base_size = self.config.min_size
        size_range = (
            self.config.max_size -
            self.config.min_size
        )
        
        smart_size = int(
            base_size +
            size_range *
            priority_factor *
            queue_factor *
            performance
        )
        
        return max(
            self.config.min_size,
            min(
                smart_size,
                self.config.max_size,
                total_tasks
            )
        )
    
    def get_stats(self) -> Dict:
        """
        Hae tilastot
        
        Returns:
            Dict: Tilastot
        """
        stats = {
            "strategy": self.config.strategy.value,
            "workers": len(self.workers),
            "queue_size": self.pending.qsize(),
            "batches": {
                "total": len(self.batches),
                "by_status": defaultdict(int),
                "by_priority": defaultdict(int)
            },
            "tasks": {
                "total": sum(
                    len(b.tasks)
                    for b in self.batches.values()
                ),
                "by_status": defaultdict(int)
            }
        }
        
        # Laske erät
        for batch in self.batches.values():
            stats["batches"]["by_status"][
                batch.status
            ] += 1
            stats["batches"]["by_priority"][
                batch.priority.name
            ] += 1
            
            stats["tasks"]["by_status"][
                batch.status
            ] += len(batch.tasks)
        
        return stats
    
    def generate_report(self) -> str:
        """
        Generoi raportti
        
        Returns:
            str: Markdown-muotoinen raportti
        """
        stats = self.get_stats()
        
        report = """# Eräkäsittelyraportti

## Konfiguraatio

"""
        report += f"- Strategia: {stats['strategy']}\n"
        report += f"- Työntekijät: {stats['workers']}\n"
        report += f"- Jonossa: {stats['queue_size']}\n\n"
        
        report += "## Erät\n\n"
        report += f"Yhteensä: {stats['batches']['total']}\n\n"
        
        report += "### Tilan mukaan\n\n"
        for status, count in stats["batches"]["by_status"].items():
            report += f"- {status}: {count}\n"
        
        report += "\n### Prioriteetin mukaan\n\n"
        for priority, count in stats["batches"]["by_priority"].items():
            report += f"- {priority}: {count}\n"
        
        report += "\n## Tehtävät\n\n"
        report += f"Yhteensä: {stats['tasks']['total']}\n\n"
        
        report += "### Tilan mukaan\n\n"
        for status, count in stats["tasks"]["by_status"].items():
            report += f"- {status}: {count}\n"
        
        return report

async def main():
    """Testaa eräkäsittelijää"""
    # Testifunktio
    async def process_batch(tasks: List[Any]) -> List[Any]:
        """Käsittele erä"""
        await asyncio.sleep(0.1)  # Simuloi työtä
        return [f"processed_{t}" for t in tasks]
    
    # Alusta konfiguraatio
    config = BatchConfig(
        strategy=BatchStrategy.SMART,
        min_size=10,
        max_size=100,
        timeout=60,
        parallel_batches=4
    )
    
    # Alusta käsittelijä
    processor = BatchProcessor(config, process_batch)
    await processor.start()
    
    try:
        # Testitehtävät
        tasks = [f"task_{i}" for i in range(1000)]
        
        # Lähetä tehtävät
        batch_ids = await processor.submit(
            tasks,
            priority=BatchPriority.HIGH
        )
        
        # Odota tuloksia
        results = await processor.get_results(
            batch_ids,
            timeout=30
        )
        
        # Tulosta tulokset
        for batch_id, result in results.items():
            print(f"Batch {batch_id}:", result)
        
        # Tulosta raportti
        print("\nBatch Report:")
        print(processor.generate_report())
    
    finally:
        # Pysäytä käsittelijä
        await processor.stop()

if __name__ == "__main__":
    asyncio.run(main())
