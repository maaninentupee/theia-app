import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
import json
import random

from prometheus_client import Counter, Gauge, Histogram
import aiohttp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Konfiguroi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Tehtävien prioriteetit"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

class TaskStatus(Enum):
    """Tehtävien tilat"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class Task:
    """Tehtävä"""
    id: str
    func: Callable
    args: tuple
    kwargs: dict
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3

class TaskProcessor:
    """Tehtävienkäsittelijä"""
    
    def __init__(
        self,
        thread_workers: int = 8,
        process_workers: int = 4,
        queue_size: int = 1000,
        batch_size: int = 10
    ):
        """
        Alusta käsittelijä
        
        Args:
            thread_workers: Säikeiden määrä
            process_workers: Prosessien määrä
            queue_size: Jonon koko
            batch_size: Eräkoko
        """
        self.thread_workers = thread_workers
        self.process_workers = process_workers
        self.queue_size = queue_size
        self.batch_size = batch_size
        
        # Jonot prioriteeteittain
        self.queues: Dict[
            TaskPriority,
            asyncio.Queue
        ] = {
            p: asyncio.Queue(maxsize=queue_size)
            for p in TaskPriority
        }
        
        # Thread pool kevyille tehtäville
        self.thread_pool = ThreadPoolExecutor(
            max_workers=thread_workers
        )
        
        # Process pool raskaille tehtäville
        self.process_pool = ProcessPoolExecutor(
            max_workers=process_workers
        )
        
        # Tehtävät
        self.tasks: Dict[str, Task] = {}
        
        # Työntekijät
        self.workers: List[asyncio.Task] = []
        
        # Metriikat
        self.task_counter = Counter(
            'tasks_total',
            'Total tasks',
            ['priority', 'status']
        )
        self.queue_gauge = Gauge(
            'queue_size',
            'Queue size',
            ['priority']
        )
        self.worker_gauge = Gauge(
            'workers',
            'Active workers',
            ['type']
        )
        self.task_latency = Histogram(
            'task_latency_seconds',
            'Task latency',
            ['priority']
        )
    
    async def start(self):
        """Käynnistä käsittelijä"""
        logger.info("Starting task processor")
        
        # Käynnistä työntekijät
        self.workers = [
            asyncio.create_task(self._worker())
            for _ in range(self.thread_workers)
        ]
        
        # Päivitä metriikat
        self.worker_gauge.labels(
            type='thread'
        ).set(self.thread_workers)
        
        self.worker_gauge.labels(
            type='process'
        ).set(self.process_workers)
    
    async def stop(self):
        """Pysäytä käsittelijä"""
        logger.info("Stopping task processor")
        
        # Pysäytä työntekijät
        for worker in self.workers:
            worker.cancel()
        
        await asyncio.gather(
            *self.workers,
            return_exceptions=True
        )
        
        # Sulje poolit
        self.thread_pool.shutdown()
        self.process_pool.shutdown()
    
    async def submit(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Task:
        """
        Lähetä tehtävä
        
        Args:
            func: Tehtäväfunktio
            *args: Argumentit
            **kwargs: Avainsana-argumentit
        
        Returns:
            Task: Tehtävä
        """
        # Luo tehtävä
        task = Task(
            id=f"task_{len(self.tasks)}",
            func=func,
            args=args,
            kwargs=kwargs,
            priority=kwargs.pop(
                'priority',
                TaskPriority.NORMAL
            ),
            created_at=datetime.now()
        )
        
        # Tallenna tehtävä
        self.tasks[task.id] = task
        
        # Lisää jonoon
        await self.queues[task.priority].put(task)
        
        # Päivitä metriikat
        self.task_counter.labels(
            priority=task.priority.name,
            status=task.status.value
        ).inc()
        
        self.queue_gauge.labels(
            priority=task.priority.name
        ).set(
            self.queues[task.priority].qsize()
        )
        
        return task
    
    async def submit_batch(
        self,
        tasks: List[Tuple[Callable, tuple, dict]]
    ) -> List[Task]:
        """
        Lähetä tehtäväerä
        
        Args:
            tasks: Tehtävät (func, args, kwargs)
        
        Returns:
            List[Task]: Tehtävät
        """
        return [
            await self.submit(func, *args, **kwargs)
            for func, args, kwargs in tasks
        ]
    
    async def get_result(
        self,
        task_id: str,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Hae tulos
        
        Args:
            task_id: Tehtävän ID
            timeout: Aikakatkaisu
        
        Returns:
            Any: Tulos
        """
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        # Odota valmistumista
        start_time = datetime.now()
        
        while task.status not in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED
        }:
            if timeout and (
                datetime.now() - start_time
            ).total_seconds() > timeout:
                raise TimeoutError(
                    f"Task {task_id} timed out"
                )
            
            await asyncio.sleep(0.1)
        
        if task.status == TaskStatus.FAILED:
            raise Exception(
                f"Task {task_id} failed: {task.error}"
            )
        
        return task.result
    
    async def _worker(self):
        """Työntekijä"""
        while True:
            try:
                # Hae tehtävä korkeimmalla prioriteetilla
                task = None
                
                for priority in reversed(TaskPriority):
                    if not self.queues[priority].empty():
                        task = await self.queues[priority].get()
                        break
                
                if not task:
                    await asyncio.sleep(0.1)
                    continue
                
                # Suorita tehtävä
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
                
                try:
                    # Valitse suoritustapa
                    if self._is_cpu_bound(task):
                        # Suorita process poolissa
                        loop = asyncio.get_event_loop()
                        task.result = await loop.run_in_executor(
                            self.process_pool,
                            task.func,
                            *task.args,
                            **task.kwargs
                        )
                    
                    else:
                        # Suorita thread poolissa
                        loop = asyncio.get_event_loop()
                        task.result = await loop.run_in_executor(
                            self.thread_pool,
                            task.func,
                            *task.args,
                            **task.kwargs
                        )
                    
                    task.status = TaskStatus.COMPLETED
                
                except Exception as e:
                    # Yritä uudelleen
                    if task.retries < task.max_retries:
                        task.retries += 1
                        task.status = TaskStatus.RETRYING
                        await self.queues[task.priority].put(task)
                    
                    else:
                        task.status = TaskStatus.FAILED
                        task.error = str(e)
                
                finally:
                    # Merkitse valmistuminen
                    task.completed_at = datetime.now()
                    
                    # Päivitä metriikat
                    duration = (
                        task.completed_at - task.started_at
                    ).total_seconds()
                    
                    self.task_counter.labels(
                        priority=task.priority.name,
                        status=task.status.value
                    ).inc()
                    
                    self.task_latency.labels(
                        priority=task.priority.name
                    ).observe(duration)
                    
                    self.queue_gauge.labels(
                        priority=task.priority.name
                    ).set(
                        self.queues[task.priority].qsize()
                    )
                    
                    # Merkitse jono käsitellyksi
                    self.queues[task.priority].task_done()
            
            except asyncio.CancelledError:
                break
            
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")
                await asyncio.sleep(1)
    
    def _is_cpu_bound(self, task: Task) -> bool:
        """
        Tarkista onko tehtävä CPU-intensiivinen
        
        Args:
            task: Tehtävä
        
        Returns:
            bool: True jos CPU-intensiivinen
        """
        # Tarkista funktion attribuutit
        if hasattr(task.func, 'cpu_bound'):
            return task.func.cpu_bound
        
        # Tarkista funktion nimi
        name = task.func.__name__.lower()
        return any(
            kw in name
            for kw in ['calculate', 'process', 'compute']
        )
    
    def get_stats(self) -> Dict:
        """
        Hae tilastot
        
        Returns:
            Dict: Tilastot
        """
        stats = {
            "queues": {
                p.name: self.queues[p].qsize()
                for p in TaskPriority
            },
            "tasks": {
                "total": len(self.tasks),
                "by_status": defaultdict(int),
                "by_priority": defaultdict(int)
            },
            "workers": {
                "thread": self.thread_workers,
                "process": self.process_workers
            }
        }
        
        # Laske tehtävät
        for task in self.tasks.values():
            stats["tasks"]["by_status"][
                task.status.value
            ] += 1
            stats["tasks"]["by_priority"][
                task.priority.name
            ] += 1
        
        return stats
    
    def generate_report(self) -> str:
        """
        Generoi raportti
        
        Returns:
            str: Markdown-muotoinen raportti
        """
        stats = self.get_stats()
        
        report = """# Tehtäväraportti

## Jonot

"""
        
        for priority, size in stats["queues"].items():
            report += f"- {priority}: {size}\n"
        
        report += "\n## Tehtävät\n\n"
        report += f"Yhteensä: {stats['tasks']['total']}\n\n"
        
        report += "### Tilan mukaan\n\n"
        for status, count in stats["tasks"]["by_status"].items():
            report += f"- {status}: {count}\n"
        
        report += "\n### Prioriteetin mukaan\n\n"
        for priority, count in stats["tasks"]["by_priority"].items():
            report += f"- {priority}: {count}\n"
        
        report += "\n## Työntekijät\n\n"
        report += f"- Thread pool: {stats['workers']['thread']}\n"
        report += f"- Process pool: {stats['workers']['process']}\n"
        
        return report

async def main():
    """Testaa tehtävienkäsittelijää"""
    # Alusta käsittelijä
    processor = TaskProcessor()
    await processor.start()
    
    try:
        # Testitehtävät
        def cpu_task(n: int) -> int:
            """CPU-intensiivinen tehtävä"""
            return sum(i * i for i in range(n))
        
        def io_task(delay: float) -> str:
            """I/O-tehtävä"""
            time.sleep(delay)
            return f"Slept for {delay}s"
        
        # Lähetä tehtäviä
        tasks = []
        
        # CPU-tehtävät
        for i in range(5):
            task = await processor.submit(
                cpu_task,
                1000000,
                priority=TaskPriority.HIGH
            )
            tasks.append(task)
        
        # I/O-tehtävät
        for i in range(10):
            task = await processor.submit(
                io_task,
                0.5,
                priority=TaskPriority.NORMAL
            )
            tasks.append(task)
        
        # Odota tuloksia
        for task in tasks:
            try:
                result = await processor.get_result(
                    task.id,
                    timeout=10
                )
                logger.info(
                    f"Task {task.id} result: {result}"
                )
            
            except Exception as e:
                logger.error(
                    f"Task {task.id} failed: {str(e)}"
                )
        
        # Tulosta raportti
        print("\nTask Report:")
        print(processor.generate_report())
    
    finally:
        # Pysäytä käsittelijä
        await processor.stop()

if __name__ == "__main__":
    asyncio.run(main())
