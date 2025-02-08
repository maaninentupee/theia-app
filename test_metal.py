#!/usr/bin/env python3
"""
Mac Metal -tuen testausskripti.
Testaa GPU-kiihdytyksen toimivuuden ja suorituskyvyn.
"""

import platform
import torch
import numpy as np
import time
from typing import Tuple, List
import json
from pathlib import Path

def check_system() -> dict:
    """Tarkista järjestelmän tiedot."""
    return {
        "os": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    }

def run_performance_test(size: Tuple[int, int] = (5000, 5000), 
                        device: str = "cpu") -> float:
    """Suorita suorituskykytesti matriisikertolaskulla."""
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    
    start_time = time.time()
    c = torch.matmul(a, b)
    torch.sync(device)  # Varmista että operaatio on valmis
    
    return time.time() - start_time

def test_memory_usage(device: str = "cpu") -> dict:
    """Testaa muistinkäyttöä eri kokoisilla tensoreilla."""
    results = {}
    sizes = [(1000, 1000), (2000, 2000), (5000, 5000)]
    
    for size in sizes:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        try:
            a = torch.randn(size, device=device)
            b = torch.randn(size, device=device)
            c = torch.matmul(a, b)
            memory = torch.cuda.memory_allocated() if device == "cuda" else "N/A"
            results[f"{size[0]}x{size[1]}"] = {
                "success": True,
                "memory_usage": memory if isinstance(memory, int) else "N/A"
            }
        except RuntimeError as e:
            results[f"{size[0]}x{size[1]}"] = {
                "success": False,
                "error": str(e)
            }
    
    return results

def main():
    """Pääfunktio testien suorittamiseen."""
    # Tarkista järjestelmä
    system_info = check_system()
    print("\n=== Järjestelmätiedot ===")
    print(json.dumps(system_info, indent=2))
    
    # Määritä testattavat laitteet
    devices = ["cpu"]
    if system_info["cuda_available"]:
        devices.append("cuda")
    if system_info["mps_available"]:
        devices.append("mps")
    
    # Suorita suorituskykytestit
    print("\n=== Suorituskykytestit ===")
    for device in devices:
        try:
            time_taken = run_performance_test(device=device)
            print(f"{device.upper()}: {time_taken:.4f} sekuntia")
        except Exception as e:
            print(f"{device.upper()}: Virhe - {str(e)}")
    
    # Testaa muistinkäyttö
    print("\n=== Muistinkäyttötestit ===")
    for device in devices:
        print(f"\nTestataan laitetta: {device.upper()}")
        memory_results = test_memory_usage(device)
        print(json.dumps(memory_results, indent=2))
    
    # Tallenna tulokset
    results = {
        "system_info": system_info,
        "devices_tested": devices,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    Path("test_results").mkdir(exist_ok=True)
    with open("test_results/metal_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n=== Suositukset ===")
    if system_info["mps_available"]:
        print("✅ Metal-tuki on käytettävissä - suositellaan MPS:n käyttöä")
    elif system_info["cuda_available"]:
        print("✅ CUDA on käytettävissä - suositellaan CUDA:n käyttöä")
    else:
        print("ℹ️ Käytetään CPU:ta - harkitse kvantisaation käyttöä")

if __name__ == "__main__":
    main()
