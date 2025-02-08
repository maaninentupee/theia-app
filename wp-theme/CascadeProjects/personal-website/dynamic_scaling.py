import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import json
import numpy as np
from collections import defaultdict

from prometheus_client import Counter, Gauge, Histogram
import psutil
import aiohttp

# Konfiguroi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Tehtävätyypit"""
    LIGHT = "light"     # Kevyt tehtävä
    MEDIUM = "medium"   # Keskitason tehtävä
    HEAVY = "heavy"     # Raskas tehtävä
    BURST = "burst"     # Pursketehtävä

@dataclass
class Task:
    """Tehtävä"""
    id: str
    type: TaskType
    payload: Any
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[str] = None
    error: Optional[str] = None

class ScalingStrategy(Enum):
    """Skaalausstrategiat"""
    LINEAR = "linear"           # Lineaarinen skaalaus
    EXPONENTIAL = "exponential" # Eksponentiaalinen skaalaus
    ADAPTIVE = "adaptive"       # Adaptiivinen skaalaus

class DynamicScaler:
    """Dynaaminen skaalaaja"""
    
    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 10,
        target_utilization: float = 0.7,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        cooldown_period: int = 60,
        strategy: ScalingStrategy = ScalingStrategy.ADAPTIVE
    ):
        """
        Alusta skaalaaja
        
        Args:
            min_workers: Minimi työntekijät
            max_workers: Maksimi työntekijät
            target_utilization: Tavoitekäyttöaste
            scale_up_threshold: Skaalausraja ylös
            scale_down_threshold: Skaalausraja alas
            cooldown_period: Jäähdytysaika
            strategy: Skaalausstrategia
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_utilization = target_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        self.strategy = strategy
        
        # Työntekijät
        self.workers: Dict[str, Dict] = {}
        
        # Tehtävämetriikat
        self.task_metrics: Dict[TaskType, List[float]] = defaultdict(list)
        
        # Skaalaushistoria
        self.scaling_history: List[Dict] = []
        
        # Viimeisin skaalaus
        self.last_scale: Optional[datetime] = None
        
        # Metriikat
        self.worker_gauge = Gauge(
            'workers_total',
            'Total workers',
            ['type']
        )
        self.task_counter = Counter(
            'tasks_total',
            'Total tasks',
            ['type', 'status']
        )
        self.task_latency = Histogram(
            'task_latency_seconds',
            'Task latency',
            ['type']
        )
        self.utilization_gauge = Gauge(
            'worker_utilization',
            'Worker utilization'
        )
        self.queue_gauge = Gauge(
            'task_queue_size',
            'Task queue size',
            ['type']
        )
    
    async def track_task(
        self,
        task: Task,
        worker_id: str
    ):
        """
        Seuraa tehtävää
        
        Args:
            task: Tehtävä
            worker_id: Työntekijän ID
        """
        # Merkitse aloitus
        task.started_at = datetime.now()
        task.worker_id = worker_id
        
        try:
            # Simuloi työtä
            await self._simulate_work(task)
            
            # Merkitse valmistuminen
            task.completed_at = datetime.now()
            
            # Laske kesto
            duration = (
                task.completed_at - task.started_at
            ).total_seconds()
            
            # Tallenna metriikka
            self.task_metrics[task.type].append(duration)
            
            # Päivitä metriikat
            self.task_counter.labels(
                type=task.type.value,
                status="success"
            ).inc()
            
            self.task_latency.labels(
                type=task.type.value
            ).observe(duration)
            
            logger.info(
                f"Task {task.id} completed in {duration:.2f}s"
            )
        
        except Exception as e:
            task.error = str(e)
            self.task_counter.labels(
                type=task.type.value,
                status="error"
            ).inc()
            
            logger.error(
                f"Task {task.id} failed: {str(e)}"
            )
    
    async def _simulate_work(self, task: Task):
        """
        Simuloi työtä
        
        Args:
            task: Tehtävä
        """
        if task.type == TaskType.LIGHT:
            await asyncio.sleep(
                random.uniform(0.1, 0.5)
            )
        
        elif task.type == TaskType.MEDIUM:
            await asyncio.sleep(
                random.uniform(0.5, 2.0)
            )
        
        elif task.type == TaskType.HEAVY:
            await asyncio.sleep(
                random.uniform(2.0, 5.0)
            )
        
        else:  # BURST
            burst_size = task.payload.get("burst_size", 10)
            for _ in range(burst_size):
                await asyncio.sleep(0.1)
    
    async def scale_workers(self):
        """Skaalaa työntekijöitä"""
        # Tarkista jäähdytysaika
        if self.last_scale and (
            datetime.now() - self.last_scale
        ).total_seconds() < self.cooldown_period:
            return
        
        # Laske metriikat
        utilization = self._calculate_utilization()
        queue_size = self._calculate_queue_size()
        latency = self._calculate_average_latency()
        
        # Päivitä metriikat
        self.utilization_gauge.set(utilization)
        
        # Tarkista skaalaustarve
        if utilization > self.scale_up_threshold:
            await self._scale_up(
                utilization,
                queue_size,
                latency
            )
        
        elif utilization < self.scale_down_threshold:
            await self._scale_down(
                utilization,
                queue_size,
                latency
            )
    
    async def _scale_up(
        self,
        utilization: float,
        queue_size: int,
        latency: float
    ):
        """
        Skaalaa ylös
        
        Args:
            utilization: Käyttöaste
            queue_size: Jonon pituus
            latency: Keskimääräinen latenssi
        """
        current_workers = len(self.workers)
        if current_workers >= self.max_workers:
            return
        
        # Laske lisättävät työntekijät
        if self.strategy == ScalingStrategy.LINEAR:
            workers_to_add = 1
        
        elif self.strategy == ScalingStrategy.EXPONENTIAL:
            workers_to_add = max(
                1,
                current_workers // 2
            )
        
        else:  # ADAPTIVE
            # Huomioi käyttöaste ja jono
            pressure = (
                utilization +
                queue_size / 100 +
                latency / 10
            ) / 3
            
            workers_to_add = max(
                1,
                int(pressure * current_workers)
            )
        
        # Rajoita maksimiin
        workers_to_add = min(
            workers_to_add,
            self.max_workers - current_workers
        )
        
        # Lisää työntekijät
        for _ in range(workers_to_add):
            worker_id = f"worker_{len(self.workers) + 1}"
            self.workers[worker_id] = {
                "id": worker_id,
                "started_at": datetime.now(),
                "tasks_completed": 0,
                "total_duration": 0
            }
        
        # Päivitä metriikat
        self.worker_gauge.set(len(self.workers))
        
        # Tallenna historia
        self.scaling_history.append({
            "timestamp": datetime.now(),
            "action": "scale_up",
            "workers_added": workers_to_add,
            "total_workers": len(self.workers),
            "utilization": utilization,
            "queue_size": queue_size,
            "latency": latency
        })
        
        self.last_scale = datetime.now()
        
        logger.info(
            f"Scaled up by {workers_to_add} workers "
            f"(total: {len(self.workers)})"
        )
    
    async def _scale_down(
        self,
        utilization: float,
        queue_size: int,
        latency: float
    ):
        """
        Skaalaa alas
        
        Args:
            utilization: Käyttöaste
            queue_size: Jonon pituus
            latency: Keskimääräinen latenssi
        """
        current_workers = len(self.workers)
        if current_workers <= self.min_workers:
            return
        
        # Laske poistettavat työntekijät
        if self.strategy == ScalingStrategy.LINEAR:
            workers_to_remove = 1
        
        elif self.strategy == ScalingStrategy.EXPONENTIAL:
            workers_to_remove = max(
                1,
                current_workers // 4
            )
        
        else:  # ADAPTIVE
            # Huomioi käyttöaste ja jono
            pressure = 1 - (
                utilization +
                queue_size / 100 +
                latency / 10
            ) / 3
            
            workers_to_remove = max(
                1,
                int(pressure * current_workers)
            )
        
        # Rajoita minimiin
        workers_to_remove = min(
            workers_to_remove,
            current_workers - self.min_workers
        )
        
        # Poista työntekijät
        for _ in range(workers_to_remove):
            worker_id, worker = sorted(
                self.workers.items(),
                key=lambda x: x[1]["tasks_completed"]
            )[0]
            del self.workers[worker_id]
        
        # Päivitä metriikat
        self.worker_gauge.set(len(self.workers))
        
        # Tallenna historia
        self.scaling_history.append({
            "timestamp": datetime.now(),
            "action": "scale_down",
            "workers_removed": workers_to_remove,
            "total_workers": len(self.workers),
            "utilization": utilization,
            "queue_size": queue_size,
            "latency": latency
        })
        
        self.last_scale = datetime.now()
        
        logger.info(
            f"Scaled down by {workers_to_remove} workers "
            f"(total: {len(self.workers)})"
        )
    
    def _calculate_utilization(self) -> float:
        """
        Laske käyttöaste
        
        Returns:
            float: Käyttöaste
        """
        if not self.workers:
            return 0.0
        
        active_tasks = sum(
            1 for w in self.workers.values()
            if w.get("current_task")
        )
        
        return active_tasks / len(self.workers)
    
    def _calculate_queue_size(self) -> int:
        """
        Laske jonon pituus
        
        Returns:
            int: Jonon pituus
        """
        return sum(
            self.queue_gauge.labels(type=t.value)._value.get()
            for t in TaskType
        )
    
    def _calculate_average_latency(self) -> float:
        """
        Laske keskimääräinen latenssi
        
        Returns:
            float: Keskimääräinen latenssi
        """
        latencies = []
        
        for metrics in self.task_metrics.values():
            if metrics:
                latencies.extend(metrics[-100:])  # Viimeiset 100
        
        return (
            sum(latencies) / len(latencies)
            if latencies
            else 0.0
        )
    
    def generate_report(self) -> str:
        """
        Generoi raportti
        
        Returns:
            str: Markdown-muotoinen raportti
        """
        report = """# Skaalausraportti

## Nykyinen tila

"""
        
        report += f"- Työntekijöitä: {len(self.workers)}\n"
        report += f"- Käyttöaste: {self._calculate_utilization():.1%}\n"
        report += f"- Jonon pituus: {self._calculate_queue_size()}\n"
        report += f"- Keskimääräinen latenssi: {self._calculate_average_latency():.3f}s\n\n"
        
        report += "## Tehtävätyypit\n\n"
        
        for task_type in TaskType:
            metrics = self.task_metrics[task_type]
            if metrics:
                avg_duration = sum(metrics) / len(metrics)
                report += f"### {task_type.value}\n"
                report += f"- Tehtäviä: {len(metrics)}\n"
                report += f"- Keskimääräinen kesto: {avg_duration:.3f}s\n\n"
        
        report += "## Skaalaushistoria\n\n"
        
        for event in self.scaling_history[-10:]:  # Viimeiset 10
            action = event["action"]
            workers = (
                event["workers_added"]
                if action == "scale_up"
                else event["workers_removed"]
            )
            
            report += (
                f"- {event['timestamp']}: "
                f"{action}, {workers} työntekijää "
                f"(yhteensä: {event['total_workers']})\n"
            )
        
        return report

async def main():
    """Testaa dynaamista skaalausta"""
    # Alusta skaalaaja
    scaler = DynamicScaler(
        min_workers=2,
        max_workers=10,
        strategy=ScalingStrategy.ADAPTIVE
    )
    
    # Simuloi tehtäviä
    for i in range(100):
        # Luo tehtävä
        task = Task(
            id=f"task_{i}",
            type=random.choice(list(TaskType)),
            payload={"data": f"test_{i}"},
            created_at=datetime.now()
        )
        
        # Valitse työntekijä
        available_workers = [
            w for w in scaler.workers.keys()
            if not scaler.workers[w].get("current_task")
        ]
        
        if available_workers:
            worker_id = random.choice(available_workers)
            await scaler.track_task(task, worker_id)
        
        # Skaalaa tarvittaessa
        await scaler.scale_workers()
        
        await asyncio.sleep(0.1)
    
    # Tulosta raportti
    print("\nScaling Report:")
    print(scaler.generate_report())

if __name__ == "__main__":
    asyncio.run(main())
