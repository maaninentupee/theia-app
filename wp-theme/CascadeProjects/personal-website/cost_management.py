import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
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
class CostConfig:
    """Kustannuskonfiguraatio"""
    input_cost_per_1k: float
    output_cost_per_1k: float
    min_tokens: int
    max_tokens: int
    batch_size: int

class CostOptimizer:
    """Kustannusten optimoija"""
    
    def __init__(self):
        """Alusta kustannusten optimoija"""
        # Mallien kustannuskonfiguraatiot
        self.cost_configs: Dict[ModelType, CostConfig] = {
            ModelType.GPT4: CostConfig(
                input_cost_per_1k=0.03,
                output_cost_per_1k=0.06,
                min_tokens=1,
                max_tokens=8192,
                batch_size=20
            ),
            ModelType.GPT35: CostConfig(
                input_cost_per_1k=0.0015,
                output_cost_per_1k=0.002,
                min_tokens=1,
                max_tokens=4096,
                batch_size=50
            ),
            ModelType.CLAUDE: CostConfig(
                input_cost_per_1k=0.0163,
                output_cost_per_1k=0.0163,
                min_tokens=1,
                max_tokens=100000,
                batch_size=30
            ),
            ModelType.STARCODER: CostConfig(
                input_cost_per_1k=0.0015,
                output_cost_per_1k=0.0015,
                min_tokens=1,
                max_tokens=8192,
                batch_size=40
            )
        }
        
        # Kustannusmetriikat
        self.total_cost = Counter(
            'api_total_cost_dollars',
            'Total API cost in dollars',
            ['model']
        )
        self.token_usage = Counter(
            'api_token_usage_total',
            'Total token usage',
            ['model', 'type']
        )
        self.cost_per_request = Histogram(
            'api_cost_per_request_dollars',
            'Cost per request in dollars',
            ['model']
        )
        
        # Budjetit ja hälytysrajat
        self.daily_budget = 100.0  # dollaria
        self.warning_threshold = 0.7
        self.critical_threshold = 0.9
        
        # Kustannushistoria
        self.cost_history: List[Tuple[datetime, float]] = []
        
        # Optimointiasetukset
        self.batch_requests = True
        self.use_cheaper_models = True
        self.compress_inputs = True
    
    def select_optimal_model(
        self,
        task_type: str,
        input_tokens: int,
        quality_requirement: float
    ) -> ModelType:
        """
        Valitse optimaalinen malli
        
        Args:
            task_type: Tehtävätyyppi
            input_tokens: Syötetokenien määrä
            quality_requirement: Laatuvaatimus (0-1)
        
        Returns:
            ModelType: Optimaalinen malli
        """
        if quality_requirement >= 0.9:
            # Korkea laatuvaatimus -> käytä GPT-4
            return ModelType.GPT4
        
        elif quality_requirement >= 0.7:
            # Keskitason laatuvaatimus
            if task_type == "code":
                return ModelType.STARCODER
            else:
                return ModelType.CLAUDE
        
        else:
            # Matala laatuvaatimus -> käytä halvinta
            return ModelType.GPT35
    
    def optimize_prompt(
        self,
        prompt: str,
        model: ModelType
    ) -> str:
        """
        Optimoi prompt
        
        Args:
            prompt: Alkuperäinen prompt
            model: Käytettävä malli
        
        Returns:
            str: Optimoitu prompt
        """
        if not self.compress_inputs:
            return prompt
        
        # Poista turhat välilyönnit
        prompt = " ".join(prompt.split())
        
        # Lyhennä kontekstia tarvittaessa
        config = self.cost_configs[model]
        max_tokens = config.max_tokens
        
        if len(prompt.split()) > max_tokens:
            words = prompt.split()
            prompt = " ".join(words[:max_tokens])
        
        return prompt
    
    def estimate_cost(
        self,
        model: ModelType,
        input_tokens: int,
        estimated_output_tokens: int
    ) -> float:
        """
        Arvioi kustannus
        
        Args:
            model: Käytettävä malli
            input_tokens: Syötetokenit
            estimated_output_tokens: Arvioidut vastaustoken
        
        Returns:
            float: Arvioitu kustannus dollareissa
        """
        config = self.cost_configs[model]
        
        input_cost = (
            input_tokens / 1000 *
            config.input_cost_per_1k
        )
        output_cost = (
            estimated_output_tokens / 1000 *
            config.output_cost_per_1k
        )
        
        return input_cost + output_cost
    
    def track_usage(
        self,
        model: ModelType,
        input_tokens: int,
        output_tokens: int
    ):
        """
        Seuraa käyttöä
        
        Args:
            model: Käytetty malli
            input_tokens: Syötetokenit
            output_tokens: Vastaustoken
        """
        # Päivitä laskurit
        self.token_usage.labels(
            model=model.value,
            type="input"
        ).inc(input_tokens)
        
        self.token_usage.labels(
            model=model.value,
            type="output"
        ).inc(output_tokens)
        
        # Laske kustannus
        cost = self.estimate_cost(
            model,
            input_tokens,
            output_tokens
        )
        
        self.total_cost.labels(
            model=model.value
        ).inc(cost)
        
        self.cost_per_request.labels(
            model=model.value
        ).observe(cost)
        
        # Tallenna historiaan
        self.cost_history.append((datetime.now(), cost))
        
        # Tarkista budjetit
        self._check_budget_alerts()
    
    def _check_budget_alerts(self):
        """Tarkista budjettihälytykset"""
        # Laske päivän kustannukset
        today = datetime.now().date()
        daily_costs = sum(
            cost for dt, cost in self.cost_history
            if dt.date() == today
        )
        
        # Tarkista hälytysrajat
        usage_ratio = daily_costs / self.daily_budget
        
        if usage_ratio >= self.critical_threshold:
            logger.error(
                f"CRITICAL: Daily budget usage at {usage_ratio:.1%}"
            )
            # Pakota halvemmat mallit
            self.use_cheaper_models = True
            
        elif usage_ratio >= self.warning_threshold:
            logger.warning(
                f"WARNING: Daily budget usage at {usage_ratio:.1%}"
            )
    
    def get_cost_analysis(self) -> Dict:
        """
        Hae kustannusanalyysi
        
        Returns:
            Dict: Kustannusanalyysi
        """
        analysis = {
            "total_cost": {},
            "token_usage": {},
            "cost_per_request": {},
            "budget_status": {}
        }
        
        # Kokonaiskustannukset malleittain
        for model in ModelType:
            total = self.total_cost.labels(
                model=model.value
            )._value.get()
            analysis["total_cost"][model.value] = total
        
        # Token-käyttö
        for model in ModelType:
            input_tokens = self.token_usage.labels(
                model=model.value,
                type="input"
            )._value.get()
            output_tokens = self.token_usage.labels(
                model=model.value,
                type="output"
            )._value.get()
            
            analysis["token_usage"][model.value] = {
                "input": input_tokens,
                "output": output_tokens
            }
        
        # Kustannus per pyyntö
        for model in ModelType:
            count = self.cost_per_request.labels(
                model=model.value
            )._sum.get()
            if count > 0:
                avg_cost = (
                    self.cost_per_request.labels(model=model.value)
                    ._sum.get() / count
                )
                analysis["cost_per_request"][model.value] = avg_cost
        
        # Budjettitilanne
        today = datetime.now().date()
        daily_costs = sum(
            cost for dt, cost in self.cost_history
            if dt.date() == today
        )
        
        analysis["budget_status"] = {
            "daily_budget": self.daily_budget,
            "current_usage": daily_costs,
            "usage_ratio": daily_costs / self.daily_budget
        }
        
        return analysis
    
    def generate_cost_report(self) -> str:
        """
        Generoi kustannusraportti
        
        Returns:
            str: Markdown-muotoinen raportti
        """
        analysis = self.get_cost_analysis()
        
        report = """# API-kustannusraportti

## Kokonaiskustannukset

"""
        
        # Lisää kokonaiskustannukset
        for model, cost in analysis["total_cost"].items():
            report += f"- {model}: ${cost:.2f}\n"
        
        report += "\n## Token-käyttö\n\n"
        
        # Lisää token-käyttö
        for model, usage in analysis["token_usage"].items():
            report += f"### {model}\n"
            report += f"- Input: {usage['input']:,} tokens\n"
            report += f"- Output: {usage['output']:,} tokens\n\n"
        
        report += "## Kustannus per pyyntö\n\n"
        
        # Lisää keskikustannukset
        for model, cost in analysis["cost_per_request"].items():
            report += f"- {model}: ${cost:.4f}\n"
        
        report += "\n## Budjettitilanne\n\n"
        
        # Lisää budjettitilanne
        budget = analysis["budget_status"]
        report += f"- Päiväbudjetti: ${budget['daily_budget']:.2f}\n"
        report += f"- Käytetty: ${budget['current_usage']:.2f}\n"
        report += f"- Käyttöaste: {budget['usage_ratio']:.1%}\n"
        
        return report

async def main():
    """Testaa kustannusten optimointia"""
    optimizer = CostOptimizer()
    
    # Testaa mallin valintaa
    model = optimizer.select_optimal_model(
        "text",
        1000,
        0.8
    )
    logger.info(f"Selected model: {model}")
    
    # Testaa promptin optimointia
    prompt = "This is a test prompt " * 100
    optimized = optimizer.optimize_prompt(prompt, model)
    logger.info(
        f"Optimized prompt length: {len(optimized.split())}"
    )
    
    # Simuloi käyttöä
    for _ in range(10):
        optimizer.track_usage(model, 100, 50)
    
    # Tulosta raportti
    report = optimizer.generate_cost_report()
    print("\nCost Report:")
    print(report)

if __name__ == "__main__":
    asyncio.run(main())
