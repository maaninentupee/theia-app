import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

# Konfiguroi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Tehtävien prioriteetit"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class Task:
    """Tehtävä"""
    id: str
    priority: TaskPriority
    payload: Any
    created_at: datetime
    timeout: Optional[float] = None
    retries: int = 0
    max_retries: int = 3

class AdaptiveExecutor:
    """Adaptiivinen tehtävien suorittaja"""
    
    def __init__(
        self,
        min_workers: int = 4,
        max_workers: int = 32,
        scale_factor: float = 1.5,
        scale_down_delay: int = 300,  # 5min
        target_utilization: float = 0.7
    ):
        """
        Alusta suorittaja
        
        Args:
            min_workers: Minimi työntekijämäärä
            max_workers: Maksimi työntekijämäärä
            scale_factor: Skaalauskerroin
            scale_down_delay: Viive ennen skaalaamista alas (s)
            target_utilization: Tavoitekäyttöaste
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_factor = scale_factor
        self.scale_down_delay = scale_down_delay
        self.target_utilization = target_utilization
        
        # Nykyinen tila
        self.current_workers = min_workers
        self.last_scale_time = datetime.now()
        self.active_tasks: Set[str] = set()
        
        # Tehtäväjonot prioriteeteittain
        self.task_queues: Dict[TaskPriority, asyncio.Queue] = {
            priority: asyncio.Queue()
            for priority in TaskPriority
        }
        
        # ThreadPoolExecutor tehtävien suoritukseen
        self.executor = ThreadPoolExecutor(
            max_workers=self.current_workers
        )
        
        # Metriikat
        self.task_counter = Counter(
            'tasks_total',
            'Total number of tasks',
            ['priority', 'status']
        )
        self.worker_gauge = Gauge(
            'workers_current',
            'Current number of workers'
        )
        self.queue_size = Gauge(
            'queue_size',
            'Queue size',
            ['priority']
        )
        self.task_latency = Histogram(
            'task_latency_seconds',
            'Task latency in seconds',
            ['priority']
        )
        
        # Käynnistä työntekijät
        self.worker_tasks = set()
        for _ in range(self.current_workers):
            self._start_worker()
    
    def _start_worker(self):
        """Käynnistä uusi työntekijä"""
        task = asyncio.create_task(self._worker_loop())
        self.worker_tasks.add(task)
        task.add_done_callback(self.worker_tasks.discard)
    
    async def _worker_loop(self):
        """Työntekijän pääsilmukka"""
        while True:
            try:
                # Tarkista jonot prioriteettijärjestyksessä
                for priority in reversed(list(TaskPriority)):
                    queue = self.task_queues[priority]
                    
                    try:
                        # Yritä hakea tehtävä
                        task = await asyncio.wait_for(
                            queue.get(),
                            timeout=1.0
                        )
                        
                        # Suorita tehtävä
                        start_time = datetime.now()
                        try:
                            self.active_tasks.add(task.id)
                            result = await self._execute_task(task)
                            
                            # Päivitä metriikat
                            self.task_counter.labels(
                                priority=priority.name,
                                status="success"
                            ).inc()
                            
                            duration = (
                                datetime.now() - start_time
                            ).total_seconds()
                            self.task_latency.labels(
                                priority=priority.name
                            ).observe(duration)
                            
                            logger.info(
                                f"Task {task.id} completed in {duration:.2f}s"
                            )
                        
                        except Exception as e:
                            logger.error(
                                f"Task {task.id} failed: {str(e)}"
                            )
                            self.task_counter.labels(
                                priority=priority.name,
                                status="error"
                            ).inc()
                            
                            # Yritä uudelleen jos mahdollista
                            if task.retries < task.max_retries:
                                task.retries += 1
                                await self.submit_task(task)
                        
                        finally:
                            self.active_tasks.discard(task.id)
                            queue.task_done()
                    
                    except asyncio.TimeoutError:
                        continue
                
                # Tarkista skaalaustarve
                await self._check_scaling()
            
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")
                await asyncio.sleep(1)
    
    async def _execute_task(self, task: Task) -> Any:
        """
        Suorita tehtävä
        
        Args:
            task: Suoritettava tehtävä
        
        Returns:
            Any: Tehtävän tulos
        """
        # Suorita ThreadPoolExecutorissa
        loop = asyncio.get_event_loop()
        if task.timeout:
            # Aikakatkaisu
            try:
                return await asyncio.wait_for(
                    loop.run_in_executor(
                        self.executor,
                        task.payload
                    ),
                    timeout=task.timeout
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Task {task.id} timed out after {task.timeout}s"
                )
        else:
            # Ei aikakatkaisua
            return await loop.run_in_executor(
                self.executor,
                task.payload
            )
    
    async def submit_task(self, task: Task):
        """
        Lähetä tehtävä suoritettavaksi
        
        Args:
            task: Suoritettava tehtävä
        """
        # Lisää jonoon
        await self.task_queues[task.priority].put(task)
        
        # Päivitä metriikat
        self.queue_size.labels(
            priority=task.priority.name
        ).inc()
    
    async def _check_scaling(self):
        """Tarkista skaalaustarve"""
        now = datetime.now()
        
        # Laske käyttöaste
        total_queue_size = sum(
            queue.qsize()
            for queue in self.task_queues.values()
        )
        active_tasks = len(self.active_tasks)
        utilization = (
            (total_queue_size + active_tasks) /
            (self.current_workers * 2)  # 2 tehtävää per työntekijä
        )
        
        # Tarkista skaalaus ylös
        if utilization > self.target_utilization:
            # Laske tarvittavat työntekijät
            target_workers = min(
                self.max_workers,
                int(
                    self.current_workers *
                    self.scale_factor
                )
            )
            
            if target_workers > self.current_workers:
                logger.info(
                    f"Scaling up from {self.current_workers} to "
                    f"{target_workers} workers"
                )
                
                # Lisää työntekijöitä
                for _ in range(target_workers - self.current_workers):
                    self._start_worker()
                
                self.current_workers = target_workers
                self.last_scale_time = now
                
                # Päivitä executor
                self.executor._max_workers = target_workers
                
                # Päivitä metriikat
                self.worker_gauge.set(self.current_workers)
        
        # Tarkista skaalaus alas
        elif (
            utilization < self.target_utilization * 0.5 and
            now - self.last_scale_time >
            timedelta(seconds=self.scale_down_delay)
        ):
            # Laske tarvittavat työntekijät
            target_workers = max(
                self.min_workers,
                int(
                    self.current_workers /
                    self.scale_factor
                )
            )
            
            if target_workers < self.current_workers:
                logger.info(
                    f"Scaling down from {self.current_workers} to "
                    f"{target_workers} workers"
                )
                
                # Vähennä työntekijöitä
                workers_to_remove = (
                    self.current_workers - target_workers
                )
                for _ in range(workers_to_remove):
                    if self.worker_tasks:
                        task = self.worker_tasks.pop()
                        task.cancel()
                
                self.current_workers = target_workers
                self.last_scale_time = now
                
                # Päivitä executor
                self.executor._max_workers = target_workers
                
                # Päivitä metriikat
                self.worker_gauge.set(self.current_workers)
    
    async def shutdown(self):
        """Sammuta suorittaja"""
        # Peruuta työntekijät
        for task in self.worker_tasks:
            task.cancel()
        
        # Odota peruutukset
        if self.worker_tasks:
            await asyncio.wait(self.worker_tasks)
        
        # Sulje executor
        self.executor.shutdown()

async def main():
    """Testaa adaptiivista suorittajaa"""
    executor = AdaptiveExecutor()
    
    # Simuloi tehtäviä
    async def test_task(i: int):
        logger.info(f"Processing task {i}")
        await asyncio.sleep(np.random.uniform(0.1, 2.0))
        return i * 2
    
    # Luo tehtäviä eri prioriteeteilla
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
            payload=lambda x=i: test_task(x),
            created_at=datetime.now(),
            timeout=5.0
        )
        
        # Lähetä tehtävä
        await executor.submit_task(task)
        
        # Odota vähän
        await asyncio.sleep(0.1)
    
    # Odota tehtävien valmistumista
    for queue in executor.task_queues.values():
        await queue.join()
    
    # Sammuta
    await executor.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
