import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

import aiohttp
from prometheus_client import Counter, Gauge, Histogram

# Konfiguroi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Tehtävän prioriteetti"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class Task:
    """Tehtävä"""
    id: str
    type: str
    content: str
    model: str
    priority: TaskPriority
    max_retries: int = 3
    timeout: int = 30
    batch_size: Optional[int] = None
    dependencies: List[str] = None

class TaskStatus(Enum):
    """Tehtävän tila"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

class TaskResult:
    """Tehtävän tulos"""
    
    def __init__(
        self,
        task_id: str,
        status: TaskStatus,
        result: Optional[str] = None,
        error: Optional[str] = None
    ):
        self.task_id = task_id
        self.status = status
        self.result = result
        self.error = error
        self.completion_time = datetime.now()

class TaskExecutor:
    """Tehtävien suorittaja"""
    
    def __init__(
        self,
        max_workers: int = 4,
        max_batch_size: int = 10
    ):
        """
        Alusta suorittaja
        
        Args:
            max_workers: Maksimi työntekijämäärä
            max_batch_size: Maksimi eräkoko
        """
        self.max_workers = max_workers
        self.max_batch_size = max_batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue: List[Task] = []
        self.running_tasks: Dict[str, Task] = {}
        self.results: Dict[str, TaskResult] = {}
        self.retry_counts: Dict[str, int] = {}
        
        # Metriikat
        self.task_counter = Counter(
            'task_executor_tasks_total',
            'Total number of tasks',
            ['status', 'model']
        )
        
        self.active_tasks_gauge = Gauge(
            'task_executor_active_tasks',
            'Number of active tasks'
        )
        
        self.task_duration = Histogram(
            'task_executor_duration_seconds',
            'Task duration in seconds',
            ['model']
        )
    
    async def submit_task(self, task: Task) -> str:
        """
        Lähetä tehtävä suoritettavaksi
        
        Args:
            task: Tehtävä
        
        Returns:
            str: Tehtävän ID
        """
        self.task_queue.append(task)
        self.task_counter.labels(
            status='submitted',
            model=task.model
        ).inc()
        
        logger.info(
            f"Task {task.id} submitted | "
            f"Type: {task.type} | "
            f"Priority: {task.priority.name}"
        )
        
        return task.id
    
    async def process_tasks(self):
        """Prosessoi tehtäväjonoa"""
        while True:
            # Järjestä tehtävät prioriteetin mukaan
            self.task_queue.sort(
                key=lambda x: x.priority.value,
                reverse=True
            )
            
            # Kerää erä
            batch = self._collect_batch()
            if not batch:
                await asyncio.sleep(0.1)
                continue
            
            # Suorita erä rinnakkain
            async with aiohttp.ClientSession() as session:
                tasks = []
                for task in batch:
                    self.running_tasks[task.id] = task
                    self.active_tasks_gauge.inc()
                    
                    tasks.append(
                        asyncio.create_task(
                            self._execute_task(task, session)
                        )
                    )
                
                # Odota tulokset
                results = await asyncio.gather(
                    *tasks,
                    return_exceptions=True
                )
                
                # Käsittele tulokset
                for task, result in zip(batch, results):
                    await self._handle_result(task, result)
    
    def _collect_batch(self) -> List[Task]:
        """
        Kerää suoritettava erä
        
        Returns:
            List[Task]: Tehtäväerä
        """
        batch = []
        batch_size = 0
        
        while (
            self.task_queue and
            len(batch) < self.max_batch_size and
            batch_size < self.max_workers
        ):
            task = self.task_queue[0]
            
            # Tarkista riippuvuudet
            if task.dependencies:
                if not all(
                    dep in self.results and
                    self.results[dep].status == TaskStatus.COMPLETED
                    for dep in task.dependencies
                ):
                    # Siirrä tehtävä jonon loppuun
                    self.task_queue.append(
                        self.task_queue.pop(0)
                    )
                    continue
            
            # Lisää tehtävä erään
            batch.append(self.task_queue.pop(0))
            batch_size += task.batch_size or 1
        
        return batch
    
    async def _execute_task(
        self,
        task: Task,
        session: aiohttp.ClientSession
    ) -> TaskResult:
        """
        Suorita tehtävä
        
        Args:
            task: Tehtävä
            session: HTTP-sessio
        
        Returns:
            TaskResult: Tehtävän tulos
        """
        start_time = datetime.now()
        
        try:
            # Suorita tehtävä asynkronisesti
            async with session.post(
                f"http://api/{task.model}/execute",
                json={
                    "type": task.type,
                    "content": task.content
                },
                timeout=task.timeout
            ) as response:
                if response.status == 200:
                    result = await response.text()
                    status = TaskStatus.COMPLETED
                    error = None
                else:
                    result = None
                    status = TaskStatus.FAILED
                    error = f"API error: {response.status}"
            
            # Mittaa kesto
            duration = (datetime.now() - start_time).total_seconds()
            self.task_duration.labels(task.model).observe(duration)
            
            return TaskResult(task.id, status, result, error)
        
        except Exception as e:
            logger.error(
                f"Task {task.id} failed: {str(e)}"
            )
            return TaskResult(
                task.id,
                TaskStatus.FAILED,
                error=str(e)
            )
        
        finally:
            self.active_tasks_gauge.dec()
            del self.running_tasks[task.id]
    
    async def _handle_result(
        self,
        task: Task,
        result: TaskResult
    ):
        """
        Käsittele tehtävän tulos
        
        Args:
            task: Tehtävä
            result: Tulos
        """
        if result.status == TaskStatus.FAILED:
            # Yritä uudelleen
            retry_count = self.retry_counts.get(task.id, 0)
            if retry_count < task.max_retries:
                self.retry_counts[task.id] = retry_count + 1
                self.task_queue.append(task)
                result.status = TaskStatus.RETRYING
                
                logger.warning(
                    f"Retrying task {task.id} | "
                    f"Attempt {retry_count + 1}/{task.max_retries}"
                )
            else:
                logger.error(
                    f"Task {task.id} failed after "
                    f"{task.max_retries} retries"
                )
        
        # Tallenna tulos
        self.results[task.id] = result
        
        # Päivitä metriikat
        self.task_counter.labels(
            status=result.status.value,
            model=task.model
        ).inc()
        
        logger.info(
            f"Task {task.id} {result.status.value} | "
            f"Model: {task.model}"
        )
    
    async def get_task_status(self, task_id: str) -> TaskStatus:
        """
        Hae tehtävän tila
        
        Args:
            task_id: Tehtävän ID
        
        Returns:
            TaskStatus: Tehtävän tila
        """
        if task_id in self.results:
            return self.results[task_id].status
        elif task_id in self.running_tasks:
            return TaskStatus.RUNNING
        elif any(t.id == task_id for t in self.task_queue):
            return TaskStatus.PENDING
        else:
            return None
    
    async def get_task_result(
        self,
        task_id: str
    ) -> Optional[TaskResult]:
        """
        Hae tehtävän tulos
        
        Args:
            task_id: Tehtävän ID
        
        Returns:
            Optional[TaskResult]: Tehtävän tulos
        """
        return self.results.get(task_id)
    
    def get_queue_stats(self) -> Dict:
        """
        Hae jonotilastot
        
        Returns:
            Dict: Tilastot
        """
        return {
            "queue_length": len(self.task_queue),
            "active_tasks": len(self.running_tasks),
            "completed_tasks": len([
                r for r in self.results.values()
                if r.status == TaskStatus.COMPLETED
            ]),
            "failed_tasks": len([
                r for r in self.results.values()
                if r.status == TaskStatus.FAILED
            ]),
            "retry_counts": self.retry_counts
        }

async def main():
    """Testaa tehtävien suoritusta"""
    executor = TaskExecutor(max_workers=4)
    
    # Luo testitehtäviä
    tasks = [
        Task(
            id=f"task_{i}",
            type="analysis",
            content=f"Test content {i}",
            model="gpt-4",
            priority=TaskPriority.MEDIUM
        )
        for i in range(10)
    ]
    
    # Lisää riippuvuuksia
    tasks[5].dependencies = ["task_0", "task_1"]
    tasks[8].dependencies = ["task_5"]
    
    # Lähetä tehtävät
    for task in tasks:
        await executor.submit_task(task)
    
    # Käynnistä prosessointi
    await executor.process_tasks()
    
    # Näytä tilastot
    stats = executor.get_queue_stats()
    print("\nQueue Stats:")
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    asyncio.run(main())
