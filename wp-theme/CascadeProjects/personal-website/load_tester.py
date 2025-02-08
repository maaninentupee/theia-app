import asyncio
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
import json

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

# Konfiguroi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoadPattern(Enum):
    """Kuormituskuviot"""
    CONSTANT = "constant"   # Tasainen kuorma
    SPIKE = "spike"         # Piikkikuorma
    RAMP = "ramp"          # Nouseva kuorma
    WAVE = "wave"          # Aaltoileva kuorma
    RANDOM = "random"      # Satunnainen kuorma

class TaskType(Enum):
    """Tehtävätyypit"""
    CPU_BOUND = "cpu_bound"     # CPU-intensiivinen
    IO_BOUND = "io_bound"       # I/O-intensiivinen
    MEMORY_BOUND = "memory_bound"  # Muisti-intensiivinen
    NETWORK_BOUND = "network_bound"  # Verkko-intensiivinen

@dataclass
class LoadTestConfig:
    """Kuormitustestin konfiguraatio"""
    pattern: LoadPattern
    duration: int  # sekunteina
    min_tasks: int
    max_tasks: int
    task_types: List[TaskType]
    task_timeout: int = 30
    parallel_tasks: int = 10
    metrics_interval: int = 1

class LoadGenerator:
    """Kuormitusgeneraattori"""
    
    def __init__(self, config: LoadTestConfig):
        """
        Alusta generaattori
        
        Args:
            config: Konfiguraatio
        """
        self.config = config
        
        # Metriikat
        self.task_counter = Counter(
            'load_tasks_total',
            'Total load test tasks',
            ['type', 'status']
        )
        self.active_tasks = Gauge(
            'load_active_tasks',
            'Active load test tasks',
            ['type']
        )
        self.task_latency = Histogram(
            'load_task_latency_seconds',
            'Load test task latency',
            ['type']
        )
        self.resource_usage = Gauge(
            'load_resource_usage',
            'Load test resource usage',
            ['resource']
        )
    
    def generate_task(
        self,
        task_type: TaskType
    ) -> Dict[str, Any]:
        """
        Generoi tehtävä
        
        Args:
            task_type: Tehtävätyyppi
        
        Returns:
            Dict[str, Any]: Tehtävä
        """
        task = {
            "id": f"task_{random.randint(0, 1000000)}",
            "type": task_type.value,
            "size": random.randint(1, 100),
            "created_at": datetime.now().isoformat()
        }
        
        if task_type == TaskType.CPU_BOUND:
            # CPU-intensiivinen tehtävä
            task["operations"] = random.randint(
                1000000,
                10000000
            )
        
        elif task_type == TaskType.IO_BOUND:
            # I/O-intensiivinen tehtävä
            task["file_size"] = random.randint(
                1024,
                1024*1024
            )
        
        elif task_type == TaskType.MEMORY_BOUND:
            # Muisti-intensiivinen tehtävä
            task["memory_size"] = random.randint(
                1024*1024,
                10*1024*1024
            )
        
        else:  # NETWORK_BOUND
            # Verkko-intensiivinen tehtävä
            task["request_size"] = random.randint(
                1024,
                1024*1024
            )
        
        return task
    
    async def simulate_task(
        self,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simuloi tehtävä
        
        Args:
            task: Tehtävä
        
        Returns:
            Dict[str, Any]: Tulos
        """
        start_time = datetime.now()
        task_type = TaskType(task["type"])
        
        try:
            # Päivitä metriikat
            self.active_tasks.labels(
                type=task_type.value
            ).inc()
            
            # Simuloi työtä
            if task_type == TaskType.CPU_BOUND:
                # CPU-intensiivinen työ
                result = await self._simulate_cpu_work(
                    task["operations"]
                )
            
            elif task_type == TaskType.IO_BOUND:
                # I/O-intensiivinen työ
                result = await self._simulate_io_work(
                    task["file_size"]
                )
            
            elif task_type == TaskType.MEMORY_BOUND:
                # Muisti-intensiivinen työ
                result = await self._simulate_memory_work(
                    task["memory_size"]
                )
            
            else:  # NETWORK_BOUND
                # Verkko-intensiivinen työ
                result = await self._simulate_network_work(
                    task["request_size"]
                )
            
            # Laske kesto
            duration = (
                datetime.now() - start_time
            ).total_seconds()
            
            # Päivitä metriikat
            self.task_counter.labels(
                type=task_type.value,
                status="success"
            ).inc()
            
            self.task_latency.labels(
                type=task_type.value
            ).observe(duration)
            
            return {
                "task_id": task["id"],
                "status": "success",
                "duration": duration,
                "result": result
            }
        
        except Exception as e:
            # Päivitä metriikat
            self.task_counter.labels(
                type=task_type.value,
                status="error"
            ).inc()
            
            return {
                "task_id": task["id"],
                "status": "error",
                "error": str(e)
            }
        
        finally:
            # Päivitä metriikat
            self.active_tasks.labels(
                type=task_type.value
            ).dec()
    
    async def _simulate_cpu_work(
        self,
        operations: int
    ) -> Dict[str, Any]:
        """
        Simuloi CPU-työtä
        
        Args:
            operations: Operaatiot
        
        Returns:
            Dict[str, Any]: Tulos
        """
        result = 0
        for _ in range(operations):
            result += random.random()
        
        return {
            "operations": operations,
            "result": result
        }
    
    async def _simulate_io_work(
        self,
        size: int
    ) -> Dict[str, Any]:
        """
        Simuloi I/O-työtä
        
        Args:
            size: Koko tavuina
        
        Returns:
            Dict[str, Any]: Tulos
        """
        # Simuloi levyoperaatioita
        data = bytearray(random.getrandbits(8) for _ in range(size))
        
        # Simuloi viivettä
        await asyncio.sleep(size / (1024*1024))  # 1MB/s
        
        return {
            "size": size,
            "checksum": sum(data) % 1000000007
        }
    
    async def _simulate_memory_work(
        self,
        size: int
    ) -> Dict[str, Any]:
        """
        Simuloi muistityötä
        
        Args:
            size: Koko tavuina
        
        Returns:
            Dict[str, Any]: Tulos
        """
        # Allokoi muistia
        data = [
            random.random()
            for _ in range(size // 8)  # 8 tavua per float
        ]
        
        # Simuloi käsittelyä
        result = sum(data)
        mean = result / len(data)
        
        return {
            "size": size,
            "mean": mean,
            "sum": result
        }
    
    async def _simulate_network_work(
        self,
        size: int
    ) -> Dict[str, Any]:
        """
        Simuloi verkkotyötä
        
        Args:
            size: Koko tavuina
        
        Returns:
            Dict[str, Any]: Tulos
        """
        # Simuloi verkkoviivettä
        latency = random.uniform(0.05, 0.5)  # 50-500ms
        await asyncio.sleep(latency)
        
        # Simuloi siirtoa
        transfer_time = size / (1024*1024)  # 1MB/s
        await asyncio.sleep(transfer_time)
        
        return {
            "size": size,
            "latency": latency,
            "transfer_time": transfer_time
        }
    
    def _calculate_task_count(
        self,
        elapsed: float
    ) -> int:
        """
        Laske tehtävien määrä
        
        Args:
            elapsed: Kulunut aika
        
        Returns:
            int: Tehtävien määrä
        """
        progress = elapsed / self.config.duration
        
        if self.config.pattern == LoadPattern.CONSTANT:
            # Tasainen kuorma
            return self.config.max_tasks
        
        elif self.config.pattern == LoadPattern.SPIKE:
            # Piikkikuorma
            if 0.4 <= progress <= 0.6:
                return self.config.max_tasks
            return self.config.min_tasks
        
        elif self.config.pattern == LoadPattern.RAMP:
            # Nouseva kuorma
            return int(
                self.config.min_tasks +
                (self.config.max_tasks - self.config.min_tasks) *
                progress
            )
        
        elif self.config.pattern == LoadPattern.WAVE:
            # Aaltoileva kuorma
            amplitude = (
                self.config.max_tasks -
                self.config.min_tasks
            ) / 2
            offset = (
                self.config.max_tasks +
                self.config.min_tasks
            ) / 2
            return int(
                offset +
                amplitude *
                np.sin(progress * 2 * np.pi)
            )
        
        else:  # RANDOM
            # Satunnainen kuorma
            return random.randint(
                self.config.min_tasks,
                self.config.max_tasks
            )
    
    async def run_load_test(self) -> Dict[str, Any]:
        """
        Suorita kuormitustesti
        
        Returns:
            Dict[str, Any]: Tulokset
        """
        logger.info("Starting load test")
        start_time = datetime.now()
        results = []
        
        # Suorita testi
        while (
            datetime.now() - start_time
        ).total_seconds() < self.config.duration:
            # Laske tehtävien määrä
            elapsed = (
                datetime.now() - start_time
            ).total_seconds()
            task_count = self._calculate_task_count(elapsed)
            
            # Generoi tehtävät
            tasks = [
                self.generate_task(
                    random.choice(self.config.task_types)
                )
                for _ in range(task_count)
            ]
            
            # Suorita tehtävät
            tasks_iter = iter(tasks)
            pending = set()
            
            while tasks_iter or pending:
                # Lisää tehtäviä
                while (
                    len(pending) < self.config.parallel_tasks and
                    tasks_iter
                ):
                    try:
                        task = next(tasks_iter)
                        pending.add(
                            asyncio.create_task(
                                self.simulate_task(task)
                            )
                        )
                    except StopIteration:
                        break
                
                if not pending:
                    break
                
                # Odota valmistumista
                done, pending = await asyncio.wait(
                    pending,
                    timeout=self.config.task_timeout,
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Kerää tulokset
                results.extend([t.result() for t in done])
            
            # Odota seuraavaan intervalliin
            await asyncio.sleep(
                self.config.metrics_interval
            )
        
        # Laske tilastot
        stats = self._calculate_stats(results)
        
        logger.info("Load test completed")
        return stats
    
    def _calculate_stats(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Laske tilastot
        
        Args:
            results: Tulokset
        
        Returns:
            Dict[str, Any]: Tilastot
        """
        stats = {
            "total_tasks": len(results),
            "success_rate": sum(
                1 for r in results
                if r["status"] == "success"
            ) / len(results),
            "error_rate": sum(
                1 for r in results
                if r["status"] == "error"
            ) / len(results),
            "latency": {
                "min": float('inf'),
                "max": float('-inf'),
                "mean": 0,
                "p50": 0,
                "p95": 0,
                "p99": 0
            },
            "by_type": {}
        }
        
        # Kerää latenssit
        latencies = [
            r["duration"]
            for r in results
            if r["status"] == "success"
        ]
        
        if latencies:
            stats["latency"].update({
                "min": min(latencies),
                "max": max(latencies),
                "mean": np.mean(latencies),
                "p50": np.percentile(latencies, 50),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99)
            })
        
        # Laske tyypit
        for task_type in TaskType:
            type_results = [
                r for r in results
                if r.get("type") == task_type.value
            ]
            
            if type_results:
                stats["by_type"][task_type.value] = {
                    "total": len(type_results),
                    "success_rate": sum(
                        1 for r in type_results
                        if r["status"] == "success"
                    ) / len(type_results),
                    "error_rate": sum(
                        1 for r in type_results
                        if r["status"] == "error"
                    ) / len(type_results),
                    "mean_latency": np.mean([
                        r["duration"]
                        for r in type_results
                        if r["status"] == "success"
                    ])
                }
        
        return stats

async def main():
    """Testaa kuormitusgeneraattoria"""
    # Alusta konfiguraatio
    config = LoadTestConfig(
        pattern=LoadPattern.WAVE,
        duration=300,  # 5min
        min_tasks=10,
        max_tasks=100,
        task_types=[
            TaskType.CPU_BOUND,
            TaskType.IO_BOUND,
            TaskType.MEMORY_BOUND,
            TaskType.NETWORK_BOUND
        ],
        parallel_tasks=10,
        metrics_interval=1
    )
    
    # Alusta generaattori
    generator = LoadGenerator(config)
    
    # Suorita testi
    results = await generator.run_load_test()
    
    # Tulosta tulokset
    print("\nLoad Test Results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
