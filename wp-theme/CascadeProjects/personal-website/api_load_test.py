"""
API kuormitustestaus eri tasoilla.
Suorittaa 100 tehtävää erilaisilla kuormitustasoilla.
"""

import time
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List
import psutil
from datetime import datetime

@dataclass
class TaskResult:
    task_id: int
    duration: float
    success: bool
    load_level: str
    cpu_usage: float
    memory_usage: float

class LoadTester:
    def __init__(self):
        self.results: List[TaskResult] = []
        self.process = psutil.Process()
    
    def simulate_task(self, task_id: int, load_level: str) -> TaskResult:
        """Simuloi yksittäisen tehtävän suorituksen"""
        start_time = time.time()
        
        # Simuloi eri kuormitustasoja
        if load_level == "kevyt":
            processing_time = random.uniform(0.1, 0.5)
        elif load_level == "keskitaso":
            processing_time = random.uniform(0.5, 1.5)
        else:  # raskas
            processing_time = random.uniform(1.5, 3.0)
            
        # Simuloi prosessointia
        time.sleep(processing_time)
        
        # Simuloi satunnainen onnistuminen (95% onnistumisprosentti)
        success = random.random() > 0.05
        
        duration = time.time() - start_time
        cpu_usage = psutil.cpu_percent()
        memory_usage = self.process.memory_info().rss / 1024 / 1024  # MB
        
        return TaskResult(
            task_id=task_id,
            duration=duration,
            success=success,
            load_level=load_level,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage
        )
    
    def run_load_test(self, num_tasks: int = 100):
        """Suorittaa kuormitustestin eri tasoilla"""
        load_levels = ["kevyt", "keskitaso", "raskas"]
        tasks_per_level = num_tasks // len(load_levels)
        
        print(f"Aloitetaan kuormitustesti {datetime.now()}")
        print(f"Suoritetaan {tasks_per_level} tehtävää per kuormitustaso")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            for load_level in load_levels:
                print(f"\nAloitetaan {load_level} kuormitustaso:")
                futures = []
                
                for i in range(tasks_per_level):
                    task_id = len(self.results) + 1
                    future = executor.submit(self.simulate_task, task_id, load_level)
                    futures.append(future)
                
                # Kerää tulokset
                for future in futures:
                    result = future.result()
                    self.results.append(result)
                    print(f"Tehtävä {result.task_id} valmis: kesto={result.duration:.2f}s, "
                          f"CPU={result.cpu_usage:.1f}%, Muisti={result.memory_usage:.1f}MB")
        
        self.print_summary()
    
    def print_summary(self):
        """Tulostaa yhteenvedon testituloksista"""
        print("\nYhteenveto kuormitustestistä:")
        
        for load_level in ["kevyt", "keskitaso", "raskas"]:
            level_results = [r for r in self.results if r.load_level == load_level]
            
            if not level_results:
                continue
                
            avg_duration = sum(r.duration for r in level_results) / len(level_results)
            avg_cpu = sum(r.cpu_usage for r in level_results) / len(level_results)
            avg_memory = sum(r.memory_usage for r in level_results) / len(level_results)
            success_rate = sum(1 for r in level_results if r.success) / len(level_results) * 100
            
            print(f"\n{load_level.capitalize()} kuormitus:")
            print(f"  Keskimääräinen kesto: {avg_duration:.2f}s")
            print(f"  Keskimääräinen CPU-käyttö: {avg_cpu:.1f}%")
            print(f"  Keskimääräinen muistinkäyttö: {avg_memory:.1f}MB")
            print(f"  Onnistumisprosentti: {success_rate:.1f}%")

if __name__ == "__main__":
    tester = LoadTester()
    tester.run_load_test()
