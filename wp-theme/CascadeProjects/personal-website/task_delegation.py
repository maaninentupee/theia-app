"""
Tehtävien delegointi ja optimointi.
Sisältää suorituskyvyn parannukset, tehtäväjonon hallinnan,
ja kattavan lokituksen ja raportoinnin.
"""

import os
import json
import logging
import asyncio
import time
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
from enum import Enum, auto
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from heapq import heappush, heappop
import csv
from pathlib import Path
from cryptography.fernet import Fernet
import base64
from dotenv import load_dotenv

# Luo logs-kansio
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Konfiguroi lokitus
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            log_dir / f"task_delegation_{datetime.now():%Y%m%d}.log"
        )
    ]
)
logger = logging.getLogger(__name__)

class APIKeyManager:
    """API-avainten turvallinen hallinta"""
    
    def __init__(self):
        """Alusta avainten hallinta"""
        # Lataa ympäristömuuttujat
        load_dotenv()
        
        # Määritä vaaditut avaimet
        self.required_keys = {
            "OPENAI": ["OPENAI_API_KEY", "OPENAI_ORG_ID"],
            "ANTHROPIC": ["ANTHROPIC_API_KEY"],
            "HUGGINGFACE": ["HUGGINGFACE_API_KEY"]
        }
        
        # Alusta avainten tila
        self.key_status = self._validate_keys()
        
        # Lokita puuttuvat avaimet
        self._log_missing_keys()
    
    def _validate_keys(self) -> Dict[str, Dict[str, bool]]:
        """
        Tarkista API-avainten tila
        
        Returns:
            Dict: Avainten tila palveluittain
        """
        status = {}
        for service, keys in self.required_keys.items():
            status[service] = {}
            for key in keys:
                value = os.getenv(key)
                status[service][key] = bool(value and len(value) > 10)
        return status
    
    def _log_missing_keys(self):
        """Lokita puuttuvat avaimet"""
        for service, keys in self.key_status.items():
            missing = [
                key for key, valid in keys.items()
                if not valid
            ]
            if missing:
                logger.warning(
                    f"Missing or invalid API keys for {service}: "
                    f"{', '.join(missing)}"
                )
    
    def get_service_keys(
        self,
        service: str
    ) -> Optional[Dict[str, str]]:
        """
        Hae palvelun API-avaimet
        
        Args:
            service: Palvelun nimi (OPENAI, ANTHROPIC, HUGGINGFACE)
        
        Returns:
            Dict: API-avaimet tai None jos puuttuu
        """
        if service not in self.required_keys:
            logger.error(f"Unknown service: {service}")
            return None
        
        # Tarkista avainten tila
        if not all(self.key_status[service].values()):
            logger.warning(
                f"Some API keys missing for {service}"
            )
            return None
        
        # Hae avaimet
        return {
            key: os.getenv(key)
            for key in self.required_keys[service]
        }
    
    def get_key(self, key_name: str) -> Optional[str]:
        """
        Hae yksittäinen API-avain
        
        Args:
            key_name: Avaimen nimi (esim. OPENAI_API_KEY)
        
        Returns:
            str: API-avain tai None jos puuttuu
        """
        value = os.getenv(key_name)
        if not value:
            logger.warning(f"API key not found: {key_name}")
            return None
        
        if len(value) < 10:
            logger.warning(f"Invalid API key: {key_name}")
            return None
        
        return value
    
    def validate_service(self, service: str) -> bool:
        """
        Tarkista palvelun avainten tila
        
        Args:
            service: Palvelun nimi
        
        Returns:
            bool: True jos kaikki avaimet OK
        """
        if service not in self.key_status:
            return False
        return all(self.key_status[service].values())
    
    def refresh_status(self):
        """Päivitä avainten tila"""
        self.key_status = self._validate_keys()
        self._log_missing_keys()

class TaskType(Enum):
    """Tehtävätyypit"""
    ANALYSIS = auto()      # Syväluotaava analyysi
    QUICK = auto()         # Nopeat tehtävät
    CODE = auto()          # Koodin generointi
    CONTEXTUAL = auto()    # Pitkät keskustelut

class ModelType(Enum):
    """Mallityypit"""
    GPT4 = "gpt-4"
    GPT35 = "gpt-3.5-turbo"
    STARCODER = "bigcode/starcoder"
    CLAUDE = "claude-3-opus-20240229"

@dataclass
class ModelConfig:
    """Mallin konfiguraatio"""
    model: ModelType
    max_tokens: int
    temperature: float
    timeout: float
    cost_per_token: float
    context_length: int
    concurrent_requests: int = 3  # Rinnakkaisten pyyntöjen määrä
    batch_size: int = 5          # Eräkoko
    retry_limit: int = 3         # Uudelleenyritysraja
    backoff_factor: float = 1.5  # Odotusajan kerroin

@dataclass
class Task:
    """Tehtävä"""
    type: TaskType
    content: str
    priority: int = 0
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    context: Optional[str] = None
    created_at: datetime = datetime.now()
    deadline: Optional[datetime] = None
    cost_limit: Optional[float] = None

class TaskMetrics:
    """Tehtävien metriikat"""
    
    def __init__(self):
        """Alusta metriikat"""
        self.start_times = {}
        self.metrics = defaultdict(list)
        self.error_counts = defaultdict(int)
        
        # CSV-tiedostot
        self.metrics_file = log_dir / "task_metrics.csv"
        self.errors_file = log_dir / "task_errors.csv"
        
        # Alusta CSV-tiedostot
        self._init_csv_files()
    
    def _init_csv_files(self):
        """Alusta CSV-tiedostot"""
        # Metriikat
        if not self.metrics_file.exists():
            with open(self.metrics_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "task_type",
                    "model",
                    "duration_ms",
                    "tokens",
                    "cost",
                    "success"
                ])
        
        # Virheet
        if not self.errors_file.exists():
            with open(self.errors_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "task_type",
                    "model",
                    "error",
                    "context"
                ])
    
    def start_task(self, task: Task):
        """Aloita tehtävän mittaus"""
        self.start_times[id(task)] = time.time()
    
    def complete_task(
        self,
        task: Task,
        model: str,
        tokens: int,
        cost: float,
        success: bool = True
    ):
        """
        Kirjaa valmis tehtävä
        
        Args:
            task: Tehtävä
            model: Käytetty malli
            tokens: Tokenien määrä
            cost: Kustannus
            success: Onnistuiko tehtävä
        """
        # Laske kesto
        start_time = self.start_times.pop(id(task))
        duration_ms = (time.time() - start_time) * 1000
        
        # Tallenna metriikka
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                task.type.name,
                model,
                f"{duration_ms:.2f}",
                tokens,
                f"{cost:.4f}",
                success
            ])
        
        # Päivitä keskiarvot
        self.metrics[task.type.name].append({
            "duration_ms": duration_ms,
            "tokens": tokens,
            "cost": cost
        })
        
        # Loki
        logger.info(
            f"Task completed: {task.type.name}, "
            f"model={model}, "
            f"duration={duration_ms:.2f}ms, "
            f"tokens={tokens}, "
            f"cost=${cost:.4f}"
        )
    
    def log_error(
        self,
        task: Task,
        model: str,
        error: str,
        context: Optional[Dict] = None
    ):
        """
        Kirjaa virhe
        
        Args:
            task: Tehtävä
            model: Käytetty malli
            error: Virheilmoitus
            context: Lisätiedot virheestä
        """
        # Kasvata virheiden määrää
        self.error_counts[task.type.name] += 1
        
        # Tallenna virhe
        with open(self.errors_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                task.type.name,
                model,
                error,
                json.dumps(context) if context else ""
            ])
        
        # Loki
        logger.error(
            f"Task failed: {task.type.name}, "
            f"model={model}, "
            f"error={error}"
        )
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Hae tilastot
        
        Returns:
            Dict: Tilastot tehtävätyypeittäin
        """
        stats = {}
        
        for task_type, measurements in self.metrics.items():
            if not measurements:
                continue
            
            # Laske keskiarvot
            durations = [m["duration_ms"] for m in measurements]
            tokens = [m["tokens"] for m in measurements]
            costs = [m["cost"] for m in measurements]
            
            stats[task_type] = {
                "avg_duration_ms": sum(durations) / len(durations),
                "avg_tokens": sum(tokens) / len(tokens),
                "avg_cost": sum(costs) / len(costs),
                "total_cost": sum(costs),
                "error_rate": (
                    self.error_counts[task_type] /
                    (len(measurements) + self.error_counts[task_type])
                )
            }
        
        return stats

class TaskQueue:
    """Tehtäväjono prioriteetilla"""
    
    def __init__(self, max_size: int = 1000):
        """Alusta tehtäväjono"""
        self.queue = []
        self.max_size = max_size
        self.processing = set()
        self.completed = deque(maxlen=100)
        self.failed = deque(maxlen=100)
        self._task_counter = 0
        self.metrics = TaskMetrics()
    
    def add_task(self, task: Task) -> bool:
        """Lisää tehtävä jonoon"""
        if len(self.queue) >= self.max_size:
            return False
        
        priority = self._calculate_priority(task)
        self._task_counter += 1
        heappush(self.queue, (priority, self._task_counter, task))
        
        logger.info(
            f"Task added: {task.type.name}, "
            f"priority={priority}, "
            f"queue_size={len(self.queue)}"
        )
        return True
    
    def get_next_task(self) -> Optional[Task]:
        """Hae seuraava tehtävä"""
        if not self.queue:
            return None
        
        priority, counter, task = heappop(self.queue)
        self.processing.add(id(task))
        
        # Aloita mittaus
        self.metrics.start_task(task)
        
        logger.info(
            f"Task retrieved: {task.type.name}, "
            f"priority={priority}, "
            f"queue_size={len(self.queue)}"
        )
        return task
    
    def complete_task(
        self,
        task: Task,
        result: str,
        model: str,
        tokens: int,
        cost: float
    ):
        """Merkitse tehtävä valmiiksi"""
        self.processing.remove(id(task))
        self.completed.append((task, result))
        
        # Kirjaa metriikka
        self.metrics.complete_task(
            task=task,
            model=model,
            tokens=tokens,
            cost=cost,
            success=True
        )
    
    def fail_task(
        self,
        task: Task,
        error: str,
        model: str,
        context: Optional[Dict] = None
    ):
        """Merkitse tehtävä epäonnistuneeksi"""
        self.processing.remove(id(task))
        self.failed.append((task, error))
        
        # Kirjaa virhe
        self.metrics.log_error(
            task=task,
            model=model,
            error=error,
            context=context
        )
    
    def _calculate_priority(self, task: Task) -> float:
        """
        Laske tehtävän prioriteetti
        
        Huomioi:
        1. Käyttäjän antama prioriteetti
        2. Tehtävän ikä
        3. Deadline jos määritetty
        4. Tehtävätyyppi
        
        Args:
            task: Tehtävä
        
        Returns:
            float: Prioriteetti (pienempi = tärkeämpi)
        """
        now = datetime.now()
        base_priority = float(task.priority)
        
        # Tehtävän ikä (vanhemmat tärkeämpiä)
        age_factor = (now - task.created_at).total_seconds() / 3600.0
        base_priority -= min(age_factor, 24.0)  # Max 24h vaikutus
        
        # Deadline jos määritetty
        if task.deadline:
            time_left = (task.deadline - now).total_seconds()
            if time_left <= 0:
                base_priority = float('-inf')  # Heti suoritukseen
            else:
                urgency = 100.0 / max(time_left, 1.0)
                base_priority -= urgency
        
        # Tehtävätyypin prioriteetti
        type_priority = {
            TaskType.QUICK: -5,      # Nopeat ensin
            TaskType.ANALYSIS: 0,    # Normaali
            TaskType.CODE: 0,        # Normaali
            TaskType.CONTEXTUAL: 5   # Hitaat viimeiseksi
        }
        base_priority += type_priority[task.type]
        
        return base_priority

class ModelSelector:
    """Mallien valinta"""
    
    def __init__(self):
        """Alusta mallien valinta"""
        self.configs = {
            ModelType.GPT4: ModelConfig(
                model=ModelType.GPT4,
                max_tokens=4000,
                temperature=0.7,
                timeout=30.0,
                cost_per_token=0.01,
                context_length=8000,
                concurrent_requests=2,  # Kallis malli
                batch_size=3
            ),
            ModelType.GPT35: ModelConfig(
                model=ModelType.GPT35,
                max_tokens=2000,
                temperature=0.9,
                timeout=10.0,
                cost_per_token=0.002,
                context_length=4000,
                concurrent_requests=5,  # Halpa ja nopea
                batch_size=10
            ),
            ModelType.STARCODER: ModelConfig(
                model=ModelType.STARCODER,
                max_tokens=1000,
                temperature=0.5,
                timeout=20.0,
                cost_per_token=0.0,
                context_length=8000,
                concurrent_requests=4,
                batch_size=8
            ),
            ModelType.CLAUDE: ModelConfig(
                model=ModelType.CLAUDE,
                max_tokens=4000,
                temperature=0.7,
                timeout=30.0,
                cost_per_token=0.008,
                context_length=100000,
                concurrent_requests=3,
                batch_size=5
            )
        }
        
        # Mallikohtaiset laskurit ja rajoittimet
        self.request_counters = {
            model: 0 for model in ModelType
        }
        self.last_request_time = {
            model: datetime.min for model in ModelType
        }
    
    def select_model(self, task: Task) -> Tuple[ModelConfig, float]:
        """
        Valitse tehtävälle sopiva malli
        
        Args:
            task: Tehtävä
        
        Returns:
            Tuple[ModelConfig, float]: Mallin konfiguraatio ja arvioitu hinta
        """
        if task.type == TaskType.ANALYSIS:
            config = self.configs[ModelType.GPT4]
        elif task.type == TaskType.QUICK:
            config = self.configs[ModelType.GPT35]
        elif task.type == TaskType.CODE:
            config = self.configs[ModelType.STARCODER]
        elif task.type == TaskType.CONTEXTUAL:
            config = self.configs[ModelType.CLAUDE]
        else:
            raise ValueError(f"Unknown task type: {task.type}")
        
        # Arvioi hinta
        tokens = len(task.content.split()) * 1.5  # Karkea arvio
        if task.context:
            tokens += len(task.context.split()) * 1.5
        
        estimated_cost = tokens * config.cost_per_token
        
        # Tarkista kustannusraja
        if task.cost_limit and estimated_cost > task.cost_limit:
            # Vaihda halvempaan malliin
            if config.model == ModelType.GPT4:
                config = self.configs[ModelType.GPT35]
                estimated_cost = tokens * config.cost_per_token
            elif config.model == ModelType.CLAUDE:
                config = self.configs[ModelType.GPT35]
                estimated_cost = tokens * config.cost_per_token
        
        return config, estimated_cost

class ModelLimits:
    """Mallikohtaiset rajoitukset"""
    
    # Mallikohtaiset maksimit
    GPT4_MAX = {
        "total_tokens": 8192,
        "input_tokens": 6144,
        "output_tokens": 2048,
        "max_batch": 20,
        "max_batch_tokens": 30000,
        "cost_per_1k_input": 0.03,
        "cost_per_1k_output": 0.06
    }
    
    GPT35_MAX = {
        "total_tokens": 4096,
        "input_tokens": 3072,
        "output_tokens": 1024,
        "max_batch": 30,
        "max_batch_tokens": 40000,
        "cost_per_1k_input": 0.0015,
        "cost_per_1k_output": 0.002
    }
    
    CLAUDE_MAX = {
        "total_tokens": 100000,
        "input_tokens": 75000,
        "output_tokens": 25000,
        "max_batch": 50,
        "max_batch_tokens": 200000,
        "cost_per_1k_input": 0.008,
        "cost_per_1k_output": 0.024
    }
    
    STARCODER_MAX = {
        "total_tokens": 2048,
        "input_tokens": 1536,
        "output_tokens": 512,
        "max_batch": 10,
        "max_batch_tokens": 15000,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0
    }
    
    @classmethod
    def get_limits(cls, model: ModelType) -> Dict[str, int]:
        """
        Hae mallin rajoitukset
        
        Args:
            model: Malli
        
        Returns:
            Dict: Mallin rajoitukset
        """
        return {
            ModelType.GPT4: cls.GPT4_MAX,
            ModelType.GPT35: cls.GPT35_MAX,
            ModelType.CLAUDE: cls.CLAUDE_MAX,
            ModelType.STARCODER: cls.STARCODER_MAX
        }.get(model, cls.GPT35_MAX)

class ContextManager:
    """Kontekstin hallinta"""
    
    def __init__(self):
        """Alusta kontekstin hallinta"""
        self.context_window = []
        self.token_counts = defaultdict(int)
    
    def estimate_tokens(self, text: str) -> int:
        """
        Arvioi tekstin token-määrä
        
        Args:
            text: Teksti
        
        Returns:
            int: Token-määrä
        """
        # Yksinkertainen arvio: sanat * 1.3
        words = len(text.split())
        return int(words * 1.3)
    
    def truncate_context(
        self,
        context: str,
        model: ModelType,
        reserve_tokens: int
    ) -> str:
        """
        Lyhennä konteksti mallin rajoihin
        
        Args:
            context: Konteksti
            model: Malli
            reserve_tokens: Varattavat tokenit
        
        Returns:
            str: Lyhennetty konteksti
        """
        limits = ModelLimits.get_limits(model)
        max_input = limits["input_tokens"] - reserve_tokens
        
        # Arvioi nykyinen koko
        current_tokens = self.estimate_tokens(context)
        if current_tokens <= max_input:
            return context
        
        # Jaa konteksti osiin
        parts = context.split("\n\n")
        result_parts = []
        total_tokens = 0
        
        # Lisää osia kunnes raja tulee vastaan
        for part in reversed(parts):
            part_tokens = self.estimate_tokens(part)
            if total_tokens + part_tokens > max_input:
                break
            result_parts.insert(0, part)
            total_tokens += part_tokens
        
        return "\n\n".join(result_parts)
    
    def optimize_prompt(
        self,
        task: Task,
        model: ModelType
    ) -> Tuple[str, str]:
        """
        Optimoi tehtävän syöte ja konteksti
        
        Args:
            task: Tehtävä
            model: Malli
        
        Returns:
            Tuple: (syöte, konteksti)
        """
        limits = ModelLimits.get_limits(model)
        
        # Varaa tokeneita vastaukselle
        reserve = min(
            task.max_tokens or limits["output_tokens"],
            limits["output_tokens"]
        )
        
        # Optimoi konteksti
        if task.context:
            context = self.truncate_context(
                task.context,
                model,
                reserve_tokens=100  # Varmuusvara
            )
        else:
            context = ""
        
        # Optimoi syöte
        content = task.content
        content_tokens = self.estimate_tokens(content)
        if content_tokens > limits["input_tokens"] - 100:
            # Lyhennä sisältöä jos liian pitkä
            words = content.split()
            max_words = int(
                (limits["input_tokens"] - 100) / 1.3
            )
            content = " ".join(words[:max_words])
            logger.warning(
                f"Content truncated from {len(words)} to {max_words} words"
            )
        
        return content, context
    
    def update_token_counts(
        self,
        model: ModelType,
        input_tokens: int,
        output_tokens: int
    ):
        """
        Päivitä token-laskurit
        
        Args:
            model: Malli
            input_tokens: Syötetokenit
            output_tokens: Vastauksen tokenit
        """
        self.token_counts[f"{model.value}_input"] += input_tokens
        self.token_counts[f"{model.value}_output"] += output_tokens
        
        # Laske kustannukset
        limits = ModelLimits.get_limits(model)
        cost = (
            input_tokens * limits["cost_per_1k_input"] / 1000 +
            output_tokens * limits["cost_per_1k_output"] / 1000
        )
        self.token_counts[f"{model.value}_cost"] += cost
    
    def get_token_usage(self) -> Dict[str, float]:
        """
        Hae token-käyttö
        
        Returns:
            Dict: Token-käyttö ja kustannukset
        """
        return dict(self.token_counts)

class BatchOptimizer:
    """Eräkäsittelyn optimointi"""
    
    def __init__(self):
        """Alusta optimoija"""
        self.token_estimator = TokenEstimator()
    
    def split_task(
        self,
        task: Task,
        max_tokens: int,
        overlap: int = 100
    ) -> List[Task]:
        """
        Jaa tehtävä pienempiin osiin
        
        Args:
            task: Tehtävä
            max_tokens: Maksimi token-määrä
            overlap: Päällekkäisyys osien välillä
        
        Returns:
            List[Task]: Osatehtävät
        """
        # Arvioi tokenien määrä
        content_tokens = self.token_estimator.estimate_tokens(
            task.content
        )
        
        # Jos mahtuu rajoihin, palauta sellaisenaan
        if content_tokens <= max_tokens:
            return [task]
        
        # Jaa sisältö osiin
        parts = []
        start = 0
        while start < len(task.content):
            # Etsi sopiva katkaisukohta
            end = self._find_split_point(
                task.content[start:],
                max_tokens - overlap
            )
            
            # Lisää päällekkäisyys
            if start > 0:
                part_start = max(0, start - overlap)
            else:
                part_start = 0
            
            # Luo osatehtävä
            part = Task(
                type=task.type,
                content=task.content[part_start:start + end],
                context=task.context,
                priority=task.priority,
                created_at=task.created_at,
                max_tokens=task.max_tokens,
                temperature=task.temperature,
                metadata={
                    **task.metadata,
                    "part": len(parts),
                    "is_split": True,
                    "original_task_id": id(task)
                }
            )
            parts.append(part)
            
            # Siirry eteenpäin
            start += end - (overlap if start > 0 else 0)
        
        return parts
    
    def _find_split_point(
        self,
        text: str,
        max_tokens: int
    ) -> int:
        """
        Etsi sopiva katkaisukohta
        
        Args:
            text: Teksti
            max_tokens: Maksimi token-määrä
        
        Returns:
            int: Katkaisukohta
        """
        # Arvioi paljonko tekstiä mahtuu
        chars_per_token = 4  # Keskimäärin
        target_length = max_tokens * chars_per_token
        
        if len(text) <= target_length:
            return len(text)
        
        # Etsi lähin virkkeen loppu
        for i in range(target_length, -1, -1):
            if i >= len(text):
                continue
            
            # Virkkeen loppu
            if text[i] in ".!?" and (
                i + 1 >= len(text) or
                text[i + 1].isspace()
            ):
                return i + 1
        
        # Jos ei löydy virkkeen loppua, etsi sanan loppu
        for i in range(target_length, -1, -1):
            if i >= len(text):
                continue
            
            if text[i].isspace():
                return i
        
        # Viimeinen vaihtoehto: katkaise keskeltä
        return target_length
    
    def merge_results(
        self,
        tasks: List[Task],
        results: List[str]
    ) -> str:
        """
        Yhdistä osatehtävien tulokset
        
        Args:
            tasks: Osatehtävät
            results: Tulokset
        
        Returns:
            str: Yhdistetty tulos
        """
        # Järjestä osat
        sorted_results = sorted(
            zip(tasks, results),
            key=lambda x: x[0].metadata.get("part", 0)
        )
        
        # Yhdistä tulokset
        merged = []
        for task, result in sorted_results:
            # Poista päällekkäisyys
            if (
                merged and
                task.metadata.get("is_split") and
                task.metadata.get("part") > 0
            ):
                # Etsi yhteinen osa
                overlap = self._find_overlap(merged[-1], result)
                if overlap > 0:
                    result = result[overlap:]
            
            merged.append(result)
        
        return "\n".join(merged)
    
    def _find_overlap(self, text1: str, text2: str) -> int:
        """
        Etsi tekstien päällekkäisyys
        
        Args:
            text1: Ensimmäinen teksti
            text2: Toinen teksti
        
        Returns:
            int: Päällekkäisyyden pituus
        """
        min_length = min(len(text1), len(text2))
        max_overlap = min(200, min_length)  # Max 200 merkkiä
        
        for length in range(max_overlap, 0, -1):
            if text1[-length:] == text2[:length]:
                return length
        
        return 0

class TokenEstimator:
    """Token-määrän arviointi"""
    
    def __init__(self):
        """Alusta estimaattori"""
        # Tilastot arviointitarkkuudesta
        self.stats = {
            "total_estimates": 0,
            "total_error": 0.0,
            "max_error": 0.0
        }
    
    def estimate_tokens(self, text: str) -> int:
        """
        Arvioi tekstin token-määrä
        
        Args:
            text: Teksti
        
        Returns:
            int: Arvioitu token-määrä
        """
        if not text:
            return 0
        
        # Laske sanat ja erikoismerkit
        words = len(text.split())
        special_chars = sum(
            1 for c in text
            if not c.isalnum() and not c.isspace()
        )
        
        # Arvioi token-määrä
        # - Sanat: keskimäärin 1.3 tokenia
        # - Erikoismerkit: 0.5 tokenia per merkki
        estimated = int(
            words * 1.3 +
            special_chars * 0.5
        )
        
        # Päivitä tilastot
        self.stats["total_estimates"] += 1
        
        return estimated
    
    def get_stats(self) -> Dict[str, float]:
        """
        Hae arviointitilastot
        
        Returns:
            Dict: Tilastot
        """
        if self.stats["total_estimates"] == 0:
            return {
                "avg_error": 0.0,
                "max_error": 0.0
            }
        
        return {
            "avg_error": (
                self.stats["total_error"] /
                self.stats["total_estimates"]
            ),
            "max_error": self.stats["max_error"]
        }

class BatchProcessor:
    """Tehtävien eräprosessointi"""
    
    def __init__(self, max_batch_size: int = 10):
        """Alusta eräprosessointi"""
        self.max_batch_size = max_batch_size
        self.batches: Dict[ModelType, List[Task]] = defaultdict(list)
        self.batch_metrics = defaultdict(lambda: {
            "success_rate": 0.0,
            "avg_tokens": 0,
            "total_cost": 0.0
        })
        self.optimizer = BatchOptimizer()
    
    async def process_batch(
        self,
        model: ModelType,
        executor: 'TaskExecutor'
    ) -> List[Tuple[Task, str]]:
        """Prosessoi erä"""
        batch = self.batches[model]
        if not batch:
            return []
        
        try:
            # Optimoi erä
            limits = ModelLimits.get_limits(model)
            optimized_batch = []
            
            for task in batch:
                # Jaa tehtävä tarvittaessa osiin
                subtasks = self.optimizer.split_task(
                    task,
                    limits["input_tokens"] - 100  # Varmuusvara
                )
                optimized_batch.extend(subtasks)
            
            # Claude-mallin erityiskäsittely
            if model == ModelType.CLAUDE:
                return await self._process_claude_batch(
                    optimized_batch,
                    executor
                )
            
            # Muut mallit
            results = []
            for task in optimized_batch:
                result = await executor.execute_task(task)
                
                # Jos tehtävä oli jaettu osiin, yhdistä tulokset
                if task.metadata.get("is_split"):
                    original_id = task.metadata["original_task_id"]
                    original_task = next(
                        t for t in batch
                        if id(t) == original_id
                    )
                    # Kerää kaikki osatulokset
                    part_results = [
                        r for t, r in results
                        if (
                            t.metadata.get("original_task_id") ==
                            original_id
                        )
                    ]
                    part_results.append(result)
                    
                    # Yhdistä tulokset
                    if len(part_results) == len(subtasks):
                        merged = self.optimizer.merge_results(
                            subtasks,
                            part_results
                        )
                        results.append((original_task, merged))
                else:
                    results.append((task, result))
            
            # Päivitä metriikat
            success_count = sum(
                1 for _, r in results
                if not r.startswith("Error")
            )
            self.batch_metrics[model]["success_rate"] = (
                success_count / len(results)
            )
            
            return results
        
        finally:
            # Tyhjennä erä
            self.batches[model].clear()

class TaskOptimizer:
    """Tehtävien optimointi"""
    
    def __init__(self):
        """Alusta optimoija"""
        self.selector = ModelSelector()
        self.queue = TaskQueue()
    
    def optimize_task(self, task: Task) -> Tuple[Task, ModelConfig, float]:
        """
        Optimoi tehtävä
        
        Args:
            task: Tehtävä
        
        Returns:
            Tuple[Task, ModelConfig, float]: 
                Optimoitu tehtävä, mallin konfiguraatio ja arvioitu hinta
        """
        # Valitse malli ja arvioi hinta
        config, cost = self.selector.select_model(task)
        
        # Aseta oletusarvot jos ei määritetty
        if task.max_tokens is None:
            task.max_tokens = config.max_tokens
        
        if task.temperature is None:
            task.temperature = config.temperature
        
        # Tarkista kontekstin pituus
        if task.context:
            tokens = len(task.context.split())
            if tokens > config.context_length:
                logger.warning(
                    f"Context too long ({tokens} tokens)"
                    f" for {config.model.value}"
                    f" (max {config.context_length})"
                )
                # Lyhennä kontekstia
                words = task.context.split()
                task.context = " ".join(
                    words[:config.context_length]
                )
        
        return task, config, cost

class TaskExecutor:
    """Tehtävien suoritus"""
    
    def __init__(self):
        """Alusta suorittaja"""
        self.optimizer = TaskOptimizer()
        self.queue = self.optimizer.queue
        self.semaphores = {
            model: asyncio.Semaphore(config.concurrent_requests)
            for model, config in self.optimizer.selector.configs.items()
        }
        self.batch_processor = BatchProcessor()
        self.key_manager = APIKeyManager()
        self.context_manager = ContextManager()
        
        # Tarkista avaimet
        key_status = self.key_manager.validate_service("OPENAI")
        if not key_status:
            logger.warning(
                "Some API keys are missing or invalid. "
                "Tasks requiring these APIs will fail."
            )

    async def execute_task(self, task: Task) -> str:
        """Suorita tehtävä"""
        task, config, cost = self.optimizer.optimize_task(task)
        
        try:
            # Tarkista API-avaimet
            service = self._get_service_for_model(config.model)
            keys = self.key_manager.get_service_keys(service)
            
            if not keys:
                raise ValueError(
                    f"API keys not found for {service}"
                )
            
            # Optimoi syöte ja konteksti
            content, context = self.context_manager.optimize_prompt(
                task,
                config.model
            )
            task.content = content
            task.context = context
            
            # Yritä lisätä erään
            if self.batch_processor.add_to_batch(task, config.model):
                logger.info(
                    f"Task added to batch for {config.model.value}"
                )
                # Prosessoi erä jos täynnä
                if len(
                    self.batch_processor.batches[config.model]
                ) >= self.batch_processor.max_batch_size:
                    results = await self.batch_processor.process_batch(
                        config.model,
                        self
                    )
                    # Etsi tämän tehtävän tulos
                    for t, r in results:
                        if t == task:
                            return r
            
            # Jos ei erässä, suorita normaalisti
            async with self.semaphores[config.model]:
                # Simuloi API-kutsua
                await asyncio.sleep(0.5)
                
                # Simuloi tokenien määrä
                input_tokens = self.context_manager.estimate_tokens(
                    task.content + (task.context or "")
                )
                output_tokens = len(task.content.split()) * 2
                
                # Päivitä token-laskurit
                self.context_manager.update_token_counts(
                    config.model,
                    input_tokens,
                    output_tokens
                )
                
                result = (
                    f"[TEST] {config.model.value} processed task: "
                    f"{task.content} "
                    f"(max_tokens={task.max_tokens}, "
                    f"temp={task.temperature}, "
                    f"cost={cost:.4f})"
                )
                
                # Merkitse onnistuneeksi
                self.queue.complete_task(
                    task=task,
                    result=result,
                    model=config.model.value,
                    tokens=input_tokens + output_tokens,
                    cost=cost
                )
                return result
        
        except Exception as e:
            # Merkitse epäonnistuneeksi
            error = f"Error: {str(e)}"
            self.queue.fail_task(
                task=task,
                error=error,
                model=config.model.value,
                context={
                    "task": asdict(task),
                    "config": asdict(config),
                    "service": service,
                    "key_status": self.key_manager.key_status.get(
                        service, {}
                    ),
                    "token_usage": self.context_manager.get_token_usage()
                }
            )
            return error
    
    def _get_service_for_model(self, model: ModelType) -> str:
        """
        Hae mallin palvelu
        
        Args:
            model: Malli
        
        Returns:
            str: Palvelun nimi
        """
        if model in [ModelType.GPT4, ModelType.GPT35]:
            return "OPENAI"
        elif model == ModelType.CLAUDE:
            return "ANTHROPIC"
        elif model == ModelType.STARCODER:
            return "HUGGINGFACE"
        else:
            raise ValueError(f"Unknown model: {model}")

class CascadeAgent:
    """Cascade-agentin integraatio"""
    
    def __init__(self):
        """Alusta Cascade-agentti"""
        self.executor = TaskExecutor()
        self.conversation_history = []
        self.agent_state = {
            "current_task": None,
            "context_window": [],
            "active_tools": set(),
            "last_model": None
        }
        self.performance_metrics = defaultdict(lambda: {
            "success_rate": 0.0,
            "avg_latency": 0.0,
            "token_usage": 0,
            "cost": 0.0
        })
        self.optimizer = TaskOptimizer()
        self.context_manager = ContextManager()
        self.metrics = AgentMetrics()
        
        # Mallien konfiguraatiot
        self.model_configs = {
            ModelType.GPT4: {
                "task_types": [
                    TaskType.ANALYSIS,
                    TaskType.CONTEXTUAL
                ],
                "min_priority": 0.7,
                "max_context": 6000,
                "cost_per_token": 0.03,
                "quality_score": 0.95
            },
            ModelType.GPT35: {
                "task_types": [
                    TaskType.QUICK,
                    TaskType.CODE
                ],
                "min_priority": 0.0,
                "max_context": 3000,
                "cost_per_token": 0.002,
                "quality_score": 0.85
            },
            ModelType.CLAUDE: {
                "task_types": [
                    TaskType.ANALYSIS,
                    TaskType.CONTEXTUAL
                ],
                "min_priority": 0.8,
                "max_context": 8000,
                "cost_per_token": 0.08,
                "quality_score": 0.98
            },
            ModelType.STARCODER: {
                "task_types": [
                    TaskType.CODE
                ],
                "min_priority": 0.5,
                "max_context": 2000,
                "cost_per_token": 0.001,
                "quality_score": 0.90
            }
        }
    
    async def delegate_task(self, task: Task) -> str:
        """Delegoi tehtävä sopivalle mallille"""
        start_time = time.time()
        
        try:
            # Valitse malli
            model = self._select_model(task)
            logger.info(
                f"Selected model {model.value} for task type "
                f"{task.type.value}"
            )
            
            # Optimoi konteksti
            task = self._optimize_context(task, model)
            
            # Suorita tehtävä
            result = await self.executor.execute_task(task)
            
            # Päivitä metriikat
            self.metrics.update_task_metrics(
                task=task,
                model=model,
                result=result,
                duration=time.time() - start_time
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Task delegation failed: {str(e)}")
            self.metrics.log_error(task, str(e))
            raise
    
    def _select_model(self, task: Task) -> ModelType:
        """
        Valitse sopivin malli tehtävälle
        
        Args:
            task: Tehtävä
        
        Returns:
            ModelType: Valittu malli
        """
        scores = {}
        
        for model, config in self.model_configs.items():
            # Tarkista tehtävätyyppi
            if task.type not in config["task_types"]:
                continue
            
            # Tarkista prioriteetti
            if task.priority < config["min_priority"]:
                continue
            
            # Arvioi kontekstin koko
            context_size = self.context_manager.estimate_tokens(
                task.content + (task.context or "")
            )
            if context_size > config["max_context"]:
                continue
            
            # Laske pisteet
            score = self._calculate_model_score(
                model=model,
                config=config,
                task=task,
                context_size=context_size
            )
            scores[model] = score
        
        if not scores:
            # Käytä GPT-3.5:ttä fallbackina
            return ModelType.GPT35
        
        # Valitse paras malli
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _calculate_model_score(
        self,
        model: ModelType,
        config: Dict,
        task: Task,
        context_size: int
    ) -> float:
        """
        Laske mallin sopivuuspisteet
        
        Args:
            model: Malli
            config: Mallin konfiguraatio
            task: Tehtävä
            context_size: Kontekstin koko
        
        Returns:
            float: Pisteet (0-1)
        """
        # Peruslaatupisteet
        score = config["quality_score"]
        
        # Kontekstin koko
        context_ratio = context_size / config["max_context"]
        score *= (1 - context_ratio)  # Pienempi konteksti = parempi
        
        # Kustannustehokkuus
        if task.priority < 0.8:  # Matala prioriteetti
            cost_factor = config["cost_per_token"]
            score *= (1 - cost_factor * 10)  # Halvempi = parempi
        
        # Tehtävätyyppikohtaiset säädöt
        type_weights = {
            TaskType.ANALYSIS: {"GPT4": 1.2, "CLAUDE": 1.3},
            TaskType.CODE: {"STARCODER": 1.4, "GPT35": 1.1},
            TaskType.QUICK: {"GPT35": 1.2},
            TaskType.CONTEXTUAL: {"CLAUDE": 1.2, "GPT4": 1.1}
        }
        
        weight = type_weights.get(task.type, {}).get(
            model.value,
            1.0
        )
        score *= weight
        
        # Aiempi suorituskyky
        history = self.metrics.get_model_metrics(model)
        if history:
            success_rate = history["success_rate"]
            avg_latency = history["avg_latency"]
            
            score *= (
                success_rate * 0.8 +  # Painota onnistumisia
                (1 - min(avg_latency / 10, 1)) * 0.2  # Nopeampi = parempi
            )
        
        return score
    
    def _optimize_context(
        self,
        task: Task,
        model: ModelType
    ) -> Task:
        """
        Optimoi tehtävän konteksti
        
        Args:
            task: Tehtävä
            model: Valittu malli
        
        Returns:
            Task: Optimoitu tehtävä
        """
        config = self.model_configs[model]
        max_tokens = config["max_context"]
        
        # Optimoi konteksti
        content, context = self.context_manager.optimize_prompt(
            task,
            model
        )
        
        # Päivitä tehtävä
        task.content = content
        task.context = context
        
        # Lisää mallikohtaiset metatiedot
        task.metadata.update({
            "selected_model": model.value,
            "context_tokens": self.context_manager.estimate_tokens(
                content + (context or "")
            ),
            "quality_threshold": config["quality_score"],
            "cost_estimate": (
                self.context_manager.estimate_tokens(content) *
                config["cost_per_token"]
            )
        })
        
        return task

class AgentMetrics:
    """Agentin metriikat"""
    
    def __init__(self):
        """Alusta metriikat"""
        self.model_metrics = defaultdict(lambda: {
            "total_tasks": 0,
            "success_count": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "total_latency": 0.0,
            "error_count": 0,
            "last_errors": deque(maxlen=10)
        })
    
    def update_task_metrics(
        self,
        task: Task,
        model: ModelType,
        result: str,
        duration: float
    ):
        """Päivitä tehtävän metriikat"""
        metrics = self.model_metrics[model]
        metrics["total_tasks"] += 1
        metrics["total_latency"] += duration
        
        # Onnistuminen/epäonnistuminen
        if not result.startswith("Error"):
            metrics["success_count"] += 1
        else:
            metrics["error_count"] += 1
        
        # Token-määrät ja kustannukset
        tokens = task.metadata.get("context_tokens", 0)
        metrics["total_tokens"] += tokens
        
        cost = task.metadata.get("cost_estimate", 0.0)
        metrics["total_cost"] += cost
    
    def log_error(self, task: Task, error: str):
        """Lokita virhe"""
        model = task.metadata.get("selected_model")
        if model:
            self.model_metrics[model]["last_errors"].append({
                "error": error,
                "task_type": task.type.value,
                "timestamp": time.time()
            })
    
    def get_model_metrics(
        self,
        model: ModelType
    ) -> Optional[Dict]:
        """
        Hae mallin metriikat
        
        Args:
            model: Malli
        
        Returns:
            Dict: Metriikat tai None
        """
        metrics = self.model_metrics[model]
        if not metrics["total_tasks"]:
            return None
        
        return {
            "success_rate": (
                metrics["success_count"] /
                metrics["total_tasks"]
            ),
            "avg_latency": (
                metrics["total_latency"] /
                metrics["total_tasks"]
            ),
            "avg_tokens": (
                metrics["total_tokens"] /
                metrics["total_tasks"]
            ),
            "total_cost": metrics["total_cost"],
            "error_rate": (
                metrics["error_count"] /
                metrics["total_tasks"]
            ),
            "recent_errors": list(metrics["last_errors"])
        }

async def main():
    """Testaa delegointia"""
    executor = TaskExecutor()
    cascade_agent = CascadeAgent()
    
    # Testitehtävät
    tasks = [
        Task(
            type=TaskType.ANALYSIS,
            content="Analyze the impact of AI on society",
            priority=1,
            deadline=datetime.now(),
            cost_limit=0.05
        ),
        Task(
            type=TaskType.QUICK,
            content="What's the weather like?",
            priority=2
        ),
        Task(
            type=TaskType.CODE,
            content="Generate a Python function to sort a list",
            priority=1
        ),
        Task(
            type=TaskType.CONTEXTUAL,
            content="Explain the history of AI",
            context="Long discussion about AI...",
            priority=3
        )
    ]
    
    # Lisää tehtävät jonoon
    for task in tasks:
        executor.queue.add_task(task)
    
    # Suorita tehtävät rinnakkain
    pending = set()
    while executor.queue.queue or pending:
        while len(pending) < 5:
            task = executor.queue.get_next_task()
            if not task:
                break
            coro = executor.execute_task(task)
            pending.add(asyncio.create_task(coro))
        
        if not pending:
            break
        
        done, pending = await asyncio.wait(
            pending,
            return_when=asyncio.FIRST_COMPLETED
        )
        
        for task in done:
            result = await task
            logger.info(f"Result: {result}")
    
    # Tulosta tilastot
    stats = executor.queue.metrics.get_stats()
    logger.info("\nTask Statistics:")
    for task_type, metrics in stats.items():
        logger.info(f"\n{task_type}:")
        for metric, value in metrics.items():
            if metric.startswith("avg"):
                logger.info(f"  {metric}: {value:.2f}")
            elif metric == "error_rate":
                logger.info(f"  {metric}: {value*100:.1f}%")
            else:
                logger.info(f"  {metric}: ${value:.4f}")

    # Testaa Cascade-agenttia
    message = "What is the definition of artificial intelligence?"
    result = await cascade_agent.delegate_task(Task(
        type=TaskType.QUICK,
        content=message,
        priority=1
    ))
    logger.info(f"Cascade Agent Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
