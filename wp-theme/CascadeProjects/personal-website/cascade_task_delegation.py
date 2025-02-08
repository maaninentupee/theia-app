"""
Cascade tehtävien delegointi.
Tämä moduuli vastaa tehtävien delegoinnista eri AI-malleille.
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime
import asyncio
import anthropic
from anthropic import Client
from dotenv import load_dotenv
from cascade_config import CascadeConfig

# Konfiguroi lokitus
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TaskResult:
    """Tehtävän suorituksen tulos"""
    success: bool
    result: Optional[str]
    model: str
    execution_time: float
    token_usage: Dict[str, int]
    error: Optional[str] = None

class TaskDelegator:
    """Tehtävien delegointi eri malleille"""
    
    def __init__(self):
        """Alusta tehtävien delegointi"""
        self.config = CascadeConfig()
        
        # Varmista että konfiguraatio on validi
        if not self.config.validate_config():
            raise ValueError("Virheellinen konfiguraatio")
        
        # Alusta API clientit
        self._init_clients()
        
        # Alusta asynkroninen sessio
        self.session = None
        self._init_rate_limiters()
    
    def _init_clients(self):
        """Alusta API clientit"""
        if self.config.anthropic.enabled:
            self.anthropic_client = Client(api_key=self.config.anthropic.api_key)
        else:
            logger.warning("Anthropic API ei ole käytössä")
    
    def _init_rate_limiters(self):
        """Alusta rate limiterit"""
        if self.config.anthropic.enabled:
            self.anthropic_limiter = asyncio.Semaphore(
                self.config.anthropic.rate_limits['requests_per_minute']
            )
    
    async def delegate_task_to_claude_opus(
        self,
        task_details: Dict[str, Any],
        system_prompt: Optional[str] = None,
        retry_count: int = 0
    ) -> TaskResult:
        """
        Delegoi tehtävä Claude-3 Opus -mallille
        
        Args:
            task_details: Tehtävän tiedot
            system_prompt: Valinnainen system prompt
            retry_count: Uudelleenyritysten määrä
        
        Returns:
            TaskResult: Tehtävän tulos
        """
        if not self.config.anthropic.enabled:
            return TaskResult(
                success=False,
                result=None,
                model="claude-3-opus",
                execution_time=0,
                token_usage={},
                error="Anthropic API ei ole käytössä"
            )
        
        start_time = time.time()
        
        try:
            # Hae mallin asetukset
            model_config = self.config.get_model_config('claude-3-opus')
            if not model_config:
                raise ValueError("Claude-3 Opus mallin asetuksia ei löydy")
            
            async with self.anthropic_limiter:
                # Valmistele parametrit
                params = {
                    "model": model_config['id'],
                    "max_tokens": task_details.get(
                        "max_tokens",
                        model_config['max_tokens']
                    ),
                    "temperature": task_details.get("temperature", 0.7),
                    "messages": [
                        {"role": "user", "content": task_details["prompt"]}
                    ]
                }
                
                # Lisää system prompt jos annettu
                if system_prompt:
                    params["system"] = system_prompt
                
                # Suorita API-kutsu
                response = await self.anthropic_client.messages.create(**params)
                
                execution_time = time.time() - start_time
                
                return TaskResult(
                    success=True,
                    result=response.content[0].text,
                    model="claude-3-opus",
                    execution_time=execution_time,
                    token_usage={
                        "prompt_tokens": response.usage.input_tokens,
                        "completion_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                    }
                )
        
        except anthropic.RateLimitError:
            # Odota ja yritä uudelleen
            if retry_count < self.config.anthropic.retry_settings['max_retries']:
                delay = (
                    self.config.anthropic.retry_settings['initial_delay'] *
                    (self.config.anthropic.retry_settings['backoff_factor'] ** retry_count)
                )
                await asyncio.sleep(min(
                    delay,
                    self.config.anthropic.retry_settings['max_delay']
                ))
                return await self.delegate_task_to_claude_opus(
                    task_details,
                    system_prompt,
                    retry_count + 1
                )
            
            error = "Rate limit ylitetty"
        
        except Exception as e:
            error = str(e)
            logger.error(f"Virhe Claude-3 Opus API-kutsussa: {error}")
        
        execution_time = time.time() - start_time
        
        return TaskResult(
            success=False,
            result=None,
            model="claude-3-opus",
            execution_time=execution_time,
            token_usage={},
            error=error
        )
    
    async def batch_delegate_tasks(
        self,
        tasks: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> List[TaskResult]:
        """
        Delegoi tehtävät eränä
        
        Args:
            tasks: Lista tehtäviä
            system_prompt: Valinnainen system prompt
        
        Returns:
            List[TaskResult]: Lista tuloksia
        """
        if not self.config.anthropic.batch_mode:
            logger.warning("Batch-tila ei ole käytössä")
            return []
        
        # Jaa tehtävät eriin
        batch_size = self.config.anthropic.batch_limit
        batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
        
        all_results = []
        
        for batch in batches:
            # Suorita tehtävät rinnakkain
            tasks = [
                self.delegate_task_to_claude_opus(task, system_prompt)
                for task in batch
            ]
            
            # Odota kaikki vastaukset
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Käsittele virheet
            for result in batch_results:
                if isinstance(result, Exception):
                    all_results.append(TaskResult(
                        success=False,
                        result=None,
                        model="claude-3-opus",
                        execution_time=0,
                        token_usage={},
                        error=str(result)
                    ))
                else:
                    all_results.append(result)
            
            # Odota rate limit cooldown
            await asyncio.sleep(1.0)  # Kiinteä 1 sekunnin cooldown
        
        return all_results

async def main():
    """Testaa tehtävien delegointia"""
    delegator = TaskDelegator()
    
    # Testaa yksittäistä tehtävää
    task = {
        "prompt": "Selitä miten tekoäly voi tehostaa ohjelmistokehitystä.",
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    result = await delegator.delegate_task_to_claude_opus(
        task,
        system_prompt="Olet asiantunteva tekoälyassistentti."
    )
    
    print("\nYksittäisen tehtävän tulos:")
    print(f"Onnistui: {result.success}")
    if result.success:
        print(f"Vastaus: {result.result[:200]}...")
        print(f"Suoritusaika: {result.execution_time:.2f}s")
        print(f"Tokenien käyttö: {result.token_usage}")
    else:
        print(f"Virhe: {result.error}")
    
    # Testaa eräajoa
    tasks = [
        {"prompt": "Mitä on koneoppiminen?"},
        {"prompt": "Miten neuroverkkot toimivat?"},
        {"prompt": "Selitä luonnollisen kielen käsittely."}
    ]
    
    results = await delegator.batch_delegate_tasks(
        tasks,
        system_prompt="Selitä asiat yksinkertaisesti."
    )
    
    print("\nEräajon tulokset:")
    for i, result in enumerate(results):
        print(f"\nTehtävä {i+1}:")
        print(f"Onnistui: {result.success}")
        if result.success:
            print(f"Vastaus: {result.result[:100]}...")
            print(f"Suoritusaika: {result.execution_time:.2f}s")
        else:
            print(f"Virhe: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())
