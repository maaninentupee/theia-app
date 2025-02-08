"""
API kuormitustestaus eri säiemäärillä.
Suorittaa 100 tehtävää käyttäen 4, 8 ja 16 säiettä ja tallentaa metriikat CSV-tiedostoon.
"""

import time
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List
import psutil
from datetime import datetime
import csv

@dataclass
class TaskResult:
    task_id: int
    duration: float
    success: bool
    load_level: str
    cpu_usage: float
    memory_usage: float
    worker_count: int
    timestamp: datetime = None

class LoadTester:
    def __init__(self):
        self.results: List[TaskResult] = []
        self.process = psutil.Process()
        self.log_file = "task_metrics.log"
        
        # Initialize log file with headers
        with open(self.log_file, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp",
                "TaskID",
                "LoadLevel",
                "Duration",
                "Status",
                "CPUUsage",
                "MemoryMB",
                "WorkerCount"
            ])
    
    def log_result(self, result: TaskResult):
        """Log task result to CSV file"""
        with open(self.log_file, "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                result.task_id,
                result.load_level,
                f"{result.duration:.2f}",
                "success" if result.success else "failure",
                f"{result.cpu_usage:.1f}",
                f"{result.memory_usage:.1f}",
                result.worker_count
            ])
    
    def simulate_task(self, task_id: int, load_level: str, worker_count: int) -> TaskResult:
        """Simuloi yksittäisen tehtävän suorituksen"""
        start_time = time.time()
        
        if load_level == "kevyt":
            processing_time = random.uniform(0.1, 0.5)
        elif load_level == "keskitaso":
            processing_time = random.uniform(0.5, 1.5)
        else:  # raskas
            processing_time = random.uniform(1.5, 3.0)
            
        time.sleep(processing_time)
        success = random.random() > 0.05
        
        duration = time.time() - start_time
        cpu_usage = psutil.cpu_percent()
        memory_usage = self.process.memory_info().rss / 1024 / 1024
        
        result = TaskResult(
            task_id=task_id,
            duration=duration,
            success=success,
            load_level=load_level,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            worker_count=worker_count,
            timestamp=datetime.now()
        )
        
        self.log_result(result)
        return result
    
    def run_load_test_with_workers(self, num_tasks: int, worker_count: int):
        """Suorittaa kuormitustestin tietyllä säiemäärällä"""
        load_levels = ["kevyt", "keskitaso", "raskas"]
        tasks_per_level = num_tasks // len(load_levels)
        
        print(f"\nAloitetaan testi {worker_count} säikeellä:")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for load_level in load_levels:
                print(f"\n{load_level.capitalize()} kuormitustaso ({worker_count} säiettä):")
                futures = []
                
                for i in range(tasks_per_level):
                    task_id = len(self.results) + 1
                    future = executor.submit(self.simulate_task, task_id, load_level, worker_count)
                    futures.append(future)
                
                # Kerää tulokset
                for future in futures:
                    result = future.result()
                    self.results.append(result)
                    print(f"Tehtävä {result.task_id} valmis: kesto={result.duration:.2f}s, "
                          f"CPU={result.cpu_usage:.1f}%, Muisti={result.memory_usage:.1f}MB")
        
        total_time = time.time() - start_time
        print(f"\nKokonaisaika {worker_count} säikeellä: {total_time:.2f}s")
        return total_time
    
    def run_all_tests(self, num_tasks: int = 100):
        """Suorittaa testit eri säiemäärillä"""
        worker_counts = [4, 8, 16]
        total_times = {}
        
        print(f"Aloitetaan kuormitustestit {datetime.now()}")
        print(f"Suoritetaan {num_tasks} tehtävää kolmella eri kuormitustasolla")
        print(f"Metriikat tallennetaan tiedostoon: {self.log_file}")
        
        for worker_count in worker_counts:
            total_time = self.run_load_test_with_workers(num_tasks, worker_count)
            total_times[worker_count] = total_time
        
        self.print_summary(worker_counts, total_times)
    
    def print_summary(self, worker_counts: List[int], total_times: Dict[int, float]):
        """Tulostaa yhteenvedon kaikista testeistä"""
        print("\nYhteenveto kaikista testeistä:")
        
        for worker_count in worker_counts:
            print(f"\n{worker_count} säikeen tulokset:")
            results = [r for r in self.results if r.worker_count == worker_count]
            
            for load_level in ["kevyt", "keskitaso", "raskas"]:
                level_results = [r for r in results if r.load_level == load_level]
                
                if not level_results:
                    continue
                    
                avg_duration = sum(r.duration for r in level_results) / len(level_results)
                avg_cpu = sum(r.cpu_usage for r in level_results) / len(level_results)
                avg_memory = sum(r.memory_usage for r in level_results) / len(level_results)
                success_rate = sum(1 for r in level_results if r.success) / len(level_results) * 100
                
                print(f"\n  {load_level.capitalize()} kuormitus:")
                print(f"    Keskimääräinen kesto: {avg_duration:.2f}s")
                print(f"    Keskimääräinen CPU-käyttö: {avg_cpu:.1f}%")
                print(f"    Keskimääräinen muistinkäyttö: {avg_memory:.1f}MB")
                print(f"    Onnistumisprosentti: {success_rate:.1f}%")
            
            print(f"  Kokonaissuoritusaika: {total_times[worker_count]:.2f}s")
        
        print(f"\nKaikki metriikat on tallennettu tiedostoon: {self.log_file}")

if __name__ == "__main__":
    tester = LoadTester()
    tester.run_all_tests()
