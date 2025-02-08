import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from prometheus_client import Counter, Gauge, Histogram

# Konfiguroi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchStrategy(Enum):
    """Erästrategiat"""
    SIZE = "size"           # Kiinteä koko
    TIME = "time"           # Aikaikkuna
    DYNAMIC = "dynamic"     # Dynaaminen koko
    PRIORITY = "priority"   # Prioriteettipohjainen

class TaskPriority(Enum):
    """Tehtävien prioriteetit"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class BatchConfig:
    """Eräkonfiguraatio"""
    strategy: BatchStrategy
    max_size: int
    max_delay: float
    target_qps: float
    min_batch_size: int = 1

@dataclass
class Task:
    """Tehtävä"""
    id: str
    priority: TaskPriority
    payload: Any
    created_at: datetime
    timeout: Optional[float] = None
    dependencies: Set[str] = None

class TaskBatcher:
    """Tehtävien jakaja"""
    
    def __init__(
        self,
        config: BatchConfig,
        window_size: int = 60  # 1min
    ):
        """
        Alusta jakaja
        
        Args:
            config: Eräkonfiguraatio
            window_size: Tilastoikkuna sekunteina
        """
        self.config = config
        self.window_size = window_size
        
        # Aktiiviset erät
        self.current_batch: List[Task] = []
        self.last_flush = datetime.now()
        
        # Tehtäväjonot prioriteeteittain
        self.task_queues: Dict[TaskPriority, List[Task]] = {
            priority: []
            for priority in TaskPriority
        }
        
        # Tilastot
        self.task_history: List[Tuple[datetime, int]] = []
        self.batch_sizes: List[int] = []
        
        # Metriikat
        self.batch_counter = Counter(
            'batches_total',
            'Total number of batches',
            ['strategy']
        )
        self.batch_size = Histogram(
            'batch_size',
            'Batch size distribution',
            ['strategy']
        )
        self.task_latency = Histogram(
            'task_latency_seconds',
            'Task latency in batch',
            ['priority']
        )
        self.queue_size = Gauge(
            'queue_size',
            'Queue size',
            ['priority']
        )
    
    async def add_task(self, task: Task) -> Optional[List[Task]]:
        """
        Lisää tehtävä
        
        Args:
            task: Tehtävä
        
        Returns:
            Optional[List[Task]]: Valmis erä tai None
        """
        # Lisää jonoon
        self.task_queues[task.priority].append(task)
        
        # Päivitä metriikat
        self.queue_size.labels(
            priority=task.priority.name
        ).inc()
        
        # Tarkista erän valmius
        return await self._check_batch_ready()
    
    async def _check_batch_ready(self) -> Optional[List[Task]]:
        """
        Tarkista erän valmius
        
        Returns:
            Optional[List[Task]]: Valmis erä tai None
        """
        now = datetime.now()
        
        # Tarkista strategian mukaan
        if self.config.strategy == BatchStrategy.SIZE:
            return await self._check_size_based()
        
        elif self.config.strategy == BatchStrategy.TIME:
            return await self._check_time_based(now)
        
        elif self.config.strategy == BatchStrategy.DYNAMIC:
            return await self._check_dynamic(now)
        
        else:  # PRIORITY
            return await self._check_priority_based()
    
    async def _check_size_based(self) -> Optional[List[Task]]:
        """
        Tarkista kokopohjainen strategia
        
        Returns:
            Optional[List[Task]]: Valmis erä tai None
        """
        total_tasks = sum(
            len(queue)
            for queue in self.task_queues.values()
        )
        
        if total_tasks >= self.config.max_size:
            return await self._create_batch()
        
        return None
    
    async def _check_time_based(
        self,
        now: datetime
    ) -> Optional[List[Task]]:
        """
        Tarkista aikapohjainen strategia
        
        Args:
            now: Nykyhetki
        
        Returns:
            Optional[List[Task]]: Valmis erä tai None
        """
        if (
            now - self.last_flush
        ).total_seconds() >= self.config.max_delay:
            return await self._create_batch()
        
        return None
    
    async def _check_dynamic(
        self,
        now: datetime
    ) -> Optional[List[Task]]:
        """
        Tarkista dynaaminen strategia
        
        Args:
            now: Nykyhetki
        
        Returns:
            Optional[List[Task]]: Valmis erä tai None
        """
        # Laske nykyinen QPS
        cutoff = now - timedelta(seconds=self.window_size)
        recent_tasks = [
            (dt, count)
            for dt, count in self.task_history
            if dt >= cutoff
        ]
        
        if recent_tasks:
            total_tasks = sum(count for _, count in recent_tasks)
            current_qps = total_tasks / self.window_size
            
            # Säädä eräkokoa
            if current_qps > self.config.target_qps:
                # Kasvata eräkokoa
                self.config.max_size = min(
                    self.config.max_size * 1.5,
                    100  # Maksimi eräkoko
                )
            else:
                # Pienennä eräkokoa
                self.config.max_size = max(
                    self.config.max_size / 1.5,
                    self.config.min_batch_size
                )
        
        # Tarkista eräkoko
        return await self._check_size_based()
    
    async def _check_priority_based(self) -> Optional[List[Task]]:
        """
        Tarkista prioriteettipohjainen strategia
        
        Returns:
            Optional[List[Task]]: Valmis erä tai None
        """
        # Tarkista kriittiset tehtävät
        if self.task_queues[TaskPriority.CRITICAL]:
            return await self._create_batch(
                priority=TaskPriority.CRITICAL
            )
        
        # Tarkista muut prioriteetit
        for priority in reversed(list(TaskPriority)):
            queue = self.task_queues[priority]
            if len(queue) >= self.config.max_size:
                return await self._create_batch(priority=priority)
        
        return None
    
    async def _create_batch(
        self,
        priority: Optional[TaskPriority] = None
    ) -> List[Task]:
        """
        Luo erä
        
        Args:
            priority: Tehtävien prioriteetti
        
        Returns:
            List[Task]: Tehtäväerä
        """
        batch = []
        now = datetime.now()
        
        if priority:
            # Kerää yhden prioriteetin tehtävät
            queue = self.task_queues[priority]
            batch = queue[:self.config.max_size]
            self.task_queues[priority] = queue[self.config.max_size:]
        
        else:
            # Kerää kaikki tehtävät prioriteettijärjestyksessä
            remaining = self.config.max_size
            
            for priority in reversed(list(TaskPriority)):
                if remaining <= 0:
                    break
                
                queue = self.task_queues[priority]
                take = min(len(queue), remaining)
                
                batch.extend(queue[:take])
                self.task_queues[priority] = queue[take:]
                remaining -= take
        
        # Päivitä tilastot
        self.last_flush = now
        self.batch_sizes.append(len(batch))
        self.task_history.append((now, len(batch)))
        
        # Siivoa vanha historia
        cutoff = now - timedelta(seconds=self.window_size)
        self.task_history = [
            (dt, count)
            for dt, count in self.task_history
            if dt >= cutoff
        ]
        
        # Päivitä metriikat
        self.batch_counter.labels(
            strategy=self.config.strategy.value
        ).inc()
        
        self.batch_size.labels(
            strategy=self.config.strategy.value
        ).observe(len(batch))
        
        for task in batch:
            latency = (
                now - task.created_at
            ).total_seconds()
            
            self.task_latency.labels(
                priority=task.priority.name
            ).observe(latency)
            
            self.queue_size.labels(
                priority=task.priority.name
            ).dec()
        
        return batch
    
    def get_stats(self) -> Dict:
        """
        Hae tilastot
        
        Returns:
            Dict: Tilastot
        """
        stats = {
            "current": {
                "queue_sizes": {
                    priority.name: len(queue)
                    for priority, queue in self.task_queues.items()
                },
                "batch_size": self.config.max_size,
                "target_qps": self.config.target_qps
            },
            "history": {
                "avg_batch_size": (
                    np.mean(self.batch_sizes)
                    if self.batch_sizes else 0
                ),
                "max_batch_size": (
                    max(self.batch_sizes)
                    if self.batch_sizes else 0
                ),
                "total_batches": len(self.batch_sizes)
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
        
        report = f"""# Tehtävien jakamisraportti

## Nykyinen tila

- Strategia: {self.config.strategy.value}
- Eräkoko: {stats['current']['batch_size']}
- Kohde QPS: {stats['current']['target_qps']}

### Jononpituudet
"""
        
        for priority, size in stats['current']['queue_sizes'].items():
            report += f"- {priority}: {size}\n"
        
        report += f"""
## Historia

- Keskimääräinen eräkoko: {stats['history']['avg_batch_size']:.1f}
- Suurin eräkoko: {stats['history']['max_batch_size']}
- Eriä yhteensä: {stats['history']['total_batches']}
"""
        
        return report

async def main():
    """Testaa tehtävien jakajaa"""
    # Alusta konfiguraatio
    config = BatchConfig(
        strategy=BatchStrategy.DYNAMIC,
        max_size=10,
        max_delay=1.0,
        target_qps=100.0
    )
    
    # Alusta jakaja
    batcher = TaskBatcher(config)
    
    # Simuloi tehtäviä
    for i in range(100):
        # Valitse prioriteetti
        if i % 10 == 0:
            priority = TaskPriority.CRITICAL
        elif i % 5 == 0:
            priority = TaskPriority.HIGH
        elif i % 3 == 0:
            priority = TaskPriority.MEDIUM
        else:
            priority = TaskPriority.LOW
        
        # Luo tehtävä
        task = Task(
            id=f"task_{i}",
            priority=priority,
            payload=f"data_{i}",
            created_at=datetime.now()
        )
        
        # Lisää tehtävä
        batch = await batcher.add_task(task)
        
        if batch:
            logger.info(
                f"Created batch with {len(batch)} tasks"
            )
        
        # Odota vähän
        await asyncio.sleep(0.1)
    
    # Tulosta raportti
    print("\nBatch Report:")
    print(batcher.generate_report())

if __name__ == "__main__":
    asyncio.run(main())
