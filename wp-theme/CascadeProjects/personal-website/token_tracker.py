import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

# Konfiguroi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """API-mallityypit"""
    GPT4 = "gpt-4"
    GPT35 = "gpt-3.5-turbo"
    CLAUDE = "claude-v1"
    STARCODER = "starcoder"

@dataclass
class TokenUsage:
    """Token-käyttö"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float

@dataclass
class TokenBudget:
    """Token-budjetti"""
    daily_limit: int
    hourly_limit: int
    cost_limit: float
    warning_threshold: float = 0.7
    critical_threshold: float = 0.9

class TokenTracker:
    """Token- ja kustannusseuranta"""
    
    def __init__(
        self,
        budget: TokenBudget,
        window_size: int = 3600  # 1 tunti
    ):
        """
        Alusta seuranta
        
        Args:
            budget: Token-budjetti
            window_size: Seurantaikkuna sekunneissa
        """
        self.budget = budget
        self.window_size = window_size
        
        # Token-käyttöhistoria
        self.usage_history: List[Tuple[datetime, str, TokenUsage]] = []
        
        # Mallikohtaiset kustannukset per 1k tokenia
        self.model_costs = {
            ModelType.GPT4: {
                "input": 0.03,
                "output": 0.06
            },
            ModelType.GPT35: {
                "input": 0.0015,
                "output": 0.002
            },
            ModelType.CLAUDE: {
                "input": 0.0163,
                "output": 0.0163
            },
            ModelType.STARCODER: {
                "input": 0.0015,
                "output": 0.0015
            }
        }
        
        # Metriikat
        self.token_counter = Counter(
            'token_usage_total',
            'Total token usage',
            ['model', 'type']
        )
        self.cost_counter = Counter(
            'api_cost_total',
            'Total API cost in dollars',
            ['model']
        )
        self.budget_gauge = Gauge(
            'budget_usage_ratio',
            'Budget usage ratio',
            ['type']
        )
        self.request_cost = Histogram(
            'request_cost_dollars',
            'Cost per request in dollars',
            ['model']
        )
    
    async def track_usage(
        self,
        task_id: str,
        model: ModelType,
        prompt_tokens: int,
        completion_tokens: int
    ) -> TokenUsage:
        """
        Seuraa token-käyttöä
        
        Args:
            task_id: Tehtävän ID
            model: Käytetty malli
            prompt_tokens: Syötetokenit
            completion_tokens: Vastaustoken
        
        Returns:
            TokenUsage: Token-käyttö ja kustannus
        """
        # Laske kustannus
        cost = self._calculate_cost(
            model,
            prompt_tokens,
            completion_tokens
        )
        
        # Luo käyttöobjekti
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost
        )
        
        # Tallenna historiaan
        now = datetime.now()
        self.usage_history.append((now, model.value, usage))
        
        # Päivitä metriikat
        self.token_counter.labels(
            model=model.value,
            type="prompt"
        ).inc(prompt_tokens)
        
        self.token_counter.labels(
            model=model.value,
            type="completion"
        ).inc(completion_tokens)
        
        self.cost_counter.labels(
            model=model.value
        ).inc(cost)
        
        self.request_cost.labels(
            model=model.value
        ).observe(cost)
        
        # Tarkista budjetit
        await self._check_budgets()
        
        # Siivoa vanha historia
        self._cleanup_history()
        
        return usage
    
    def _calculate_cost(
        self,
        model: ModelType,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """
        Laske kustannus
        
        Args:
            model: Käytetty malli
            prompt_tokens: Syötetokenit
            completion_tokens: Vastaustoken
        
        Returns:
            float: Kustannus dollareissa
        """
        costs = self.model_costs[model]
        
        prompt_cost = (
            prompt_tokens / 1000 *
            costs["input"]
        )
        completion_cost = (
            completion_tokens / 1000 *
            costs["output"]
        )
        
        return prompt_cost + completion_cost
    
    async def _check_budgets(self):
        """Tarkista budjetit"""
        now = datetime.now()
        
        # Tarkista tuntibudjetti
        hour_ago = now - timedelta(hours=1)
        hourly_tokens = sum(
            usage.total_tokens
            for dt, _, usage in self.usage_history
            if dt >= hour_ago
        )
        
        hourly_ratio = hourly_tokens / self.budget.hourly_limit
        self.budget_gauge.labels(type="hourly").set(hourly_ratio)
        
        if hourly_ratio >= self.budget.critical_threshold:
            logger.error(
                f"CRITICAL: Hourly token usage at {hourly_ratio:.1%}"
            )
            raise Exception("Hourly token budget exceeded")
        
        elif hourly_ratio >= self.budget.warning_threshold:
            logger.warning(
                f"WARNING: Hourly token usage at {hourly_ratio:.1%}"
            )
        
        # Tarkista päiväbudjetti
        today = now.date()
        daily_tokens = sum(
            usage.total_tokens
            for dt, _, usage in self.usage_history
            if dt.date() == today
        )
        
        daily_ratio = daily_tokens / self.budget.daily_limit
        self.budget_gauge.labels(type="daily").set(daily_ratio)
        
        if daily_ratio >= self.budget.critical_threshold:
            logger.error(
                f"CRITICAL: Daily token usage at {daily_ratio:.1%}"
            )
            raise Exception("Daily token budget exceeded")
        
        elif daily_ratio >= self.budget.warning_threshold:
            logger.warning(
                f"WARNING: Daily token usage at {daily_ratio:.1%}"
            )
        
        # Tarkista kustannusbudjetti
        daily_cost = sum(
            usage.cost
            for dt, _, usage in self.usage_history
            if dt.date() == today
        )
        
        cost_ratio = daily_cost / self.budget.cost_limit
        self.budget_gauge.labels(type="cost").set(cost_ratio)
        
        if cost_ratio >= self.budget.critical_threshold:
            logger.error(
                f"CRITICAL: Daily cost at {cost_ratio:.1%}"
            )
            raise Exception("Daily cost budget exceeded")
        
        elif cost_ratio >= self.budget.warning_threshold:
            logger.warning(
                f"WARNING: Daily cost at {cost_ratio:.1%}"
            )
    
    def _cleanup_history(self):
        """Siivoa vanha historia"""
        cutoff = datetime.now() - timedelta(
            seconds=self.window_size
        )
        
        self.usage_history = [
            (dt, model, usage)
            for dt, model, usage in self.usage_history
            if dt >= cutoff
        ]
    
    def get_usage_stats(self) -> Dict:
        """
        Hae käyttötilastot
        
        Returns:
            Dict: Käyttötilastot
        """
        stats = {
            "models": {},
            "total": {
                "tokens": 0,
                "cost": 0.0
            }
        }
        
        # Kerää mallikohtaiset tilastot
        for model in ModelType:
            model_usage = [
                usage for dt, m, usage in self.usage_history
                if m == model.value
            ]
            
            if model_usage:
                total_tokens = sum(u.total_tokens for u in model_usage)
                total_cost = sum(u.cost for u in model_usage)
                
                stats["models"][model.value] = {
                    "tokens": total_tokens,
                    "cost": total_cost,
                    "requests": len(model_usage)
                }
                
                stats["total"]["tokens"] += total_tokens
                stats["total"]["cost"] += total_cost
        
        return stats
    
    def generate_report(self) -> str:
        """
        Generoi raportti
        
        Returns:
            str: Markdown-muotoinen raportti
        """
        stats = self.get_usage_stats()
        
        report = """# Token- ja kustannusraportti

## Kokonaiskäyttö

"""
        
        # Lisää kokonaiskäyttö
        total = stats["total"]
        report += f"- Tokenit yhteensä: {total['tokens']:,}\n"
        report += f"- Kustannukset yhteensä: ${total['cost']:.2f}\n\n"
        
        report += "## Mallikohtainen käyttö\n\n"
        
        # Lisää mallikohtaiset tilastot
        for model, usage in stats["models"].items():
            report += f"### {model}\n"
            report += f"- Tokenit: {usage['tokens']:,}\n"
            report += f"- Kustannus: ${usage['cost']:.2f}\n"
            report += f"- Pyynnöt: {usage['requests']}\n"
            report += f"- Keskikustannus: ${usage['cost']/usage['requests']:.4f}/pyyntö\n\n"
        
        return report

async def main():
    """Testaa token-seurantaa"""
    # Alusta budjetti
    budget = TokenBudget(
        daily_limit=1_000_000,    # 1M tokenia/päivä
        hourly_limit=100_000,     # 100k tokenia/tunti
        cost_limit=100.0          # $100/päivä
    )
    
    # Alusta seuranta
    tracker = TokenTracker(budget)
    
    # Simuloi käyttöä
    for i in range(10):
        model = np.random.choice(list(ModelType))
        prompt_tokens = np.random.randint(100, 1000)
        completion_tokens = np.random.randint(50, 500)
        
        usage = await tracker.track_usage(
            f"task_{i}",
            model,
            prompt_tokens,
            completion_tokens
        )
        
        logger.info(
            f"Task {i} used {usage.total_tokens} tokens, "
            f"cost: ${usage.cost:.4f}"
        )
    
    # Tulosta raportti
    print("\nUsage Report:")
    print(tracker.generate_report())

if __name__ == "__main__":
    asyncio.run(main())
