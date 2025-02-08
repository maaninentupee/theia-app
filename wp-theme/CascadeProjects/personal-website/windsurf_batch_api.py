"""
Windsurf Batch API integraatio.
Tämä moduuli vastaa Windsurf IDE:n Batch API -integraatiosta Claude-3 mallille.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime
import anthropic
from anthropic import Client
from dotenv import load_dotenv
from windsurf_integration import WindsurfIntegration

# Konfiguroi lokitus
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BatchTask:
    """Batch API tehtävä"""
    prompt: str
    max_tokens: int = 3000
    temperature: float = 0.7
    top_p: float = 0.9
    request_id: Optional[str] = None
    system_prompt: Optional[str] = None

@dataclass
class BatchResponse:
    """Batch API vastaus"""
    success: bool
    result: Optional[str]
    error: Optional[str] = None
    request_id: Optional[str] = None
    execution_time: float = 0.0
    token_usage: Optional[Dict[str, int]] = None

class WindsurfBatchAPI:
    """Windsurf Batch API käsittelijä"""
    
    def __init__(self):
        """Alusta Batch API"""
        self.integration = WindsurfIntegration()
        self.client = None
        self.rate_limiter = None
        
        if self.integration.anthropic:
            self.client = Client(api_key=self.integration.anthropic.api_key)
            self.rate_limiter = asyncio.Semaphore(5)  # Max 5 rinnakkaista pyyntöä
    
    async def _execute_single_task(
        self,
        task: BatchTask,
        retry_count: int = 0
    ) -> BatchResponse:
        """
        Suorita yksittäinen tehtävä
        
        Args:
            task: Tehtävä
            retry_count: Uudelleenyritysten määrä
        
        Returns:
            BatchResponse: Tehtävän tulos
        """
        if not self.client:
            return BatchResponse(
                success=False,
                result=None,
                error="Claude-3 ei ole käytössä",
                request_id=task.request_id
            )
        
        start_time = datetime.now()
        
        try:
            async with self.rate_limiter:
                # Valmistele parametrit
                params = {
                    "model": self.integration.anthropic.default_model,
                    "messages": [{"role": "user", "content": task.prompt}],
                    "max_tokens": task.max_tokens,
                    "temperature": task.temperature,
                    "top_p": task.top_p
                }
                
                if task.system_prompt:
                    params["system"] = task.system_prompt
                
                # Suorita API-kutsu
                response = await self.client.messages.create(**params)
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return BatchResponse(
                    success=True,
                    result=response.content[0].text,
                    request_id=task.request_id,
                    execution_time=execution_time,
                    token_usage={
                        "prompt_tokens": response.usage.input_tokens,
                        "completion_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                    }
                )
        
        except anthropic.RateLimitError:
            # Odota ja yritä uudelleen
            if retry_count < 3:  # Max 3 uudelleenyritystä
                await asyncio.sleep(2 ** retry_count)  # Exponentiaalinen backoff
                return await self._execute_single_task(task, retry_count + 1)
            
            error = "Rate limit ylitetty"
        
        except Exception as e:
            error = str(e)
            logger.error(f"Virhe tehtävän suorituksessa: {error}")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return BatchResponse(
            success=False,
            result=None,
            error=error,
            request_id=task.request_id,
            execution_time=execution_time
        )
    
    async def handle_batch_requests(
        self,
        batch_tasks: List[Union[Dict[str, Any], BatchTask]],
        system_prompt: Optional[str] = None
    ) -> List[BatchResponse]:
        """
        Käsittele eräajo
        
        Args:
            batch_tasks: Lista tehtäviä
            system_prompt: Valinnainen system prompt
        
        Returns:
            List[BatchResponse]: Lista tuloksia
        """
        if not self.client:
            return [
                BatchResponse(
                    success=False,
                    result=None,
                    error="Claude-3 ei ole käytössä"
                )
                for _ in batch_tasks
            ]
        
        # Muunna dict-tehtävät BatchTask-olioiksi
        tasks = []
        for i, task in enumerate(batch_tasks):
            if isinstance(task, dict):
                tasks.append(BatchTask(
                    prompt=task["prompt"],
                    max_tokens=task.get("max_tokens", 3000),
                    temperature=task.get("temperature", 0.7),
                    top_p=task.get("top_p", 0.9),
                    request_id=task.get("request_id", f"req_{i}"),
                    system_prompt=system_prompt
                ))
            else:
                tasks.append(task)
        
        # Jaa tehtävät eriin
        batch_size = self.integration.anthropic.batch_limit
        batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
        
        all_responses = []
        
        for batch in batches:
            # Suorita tehtävät rinnakkain
            tasks = [self._execute_single_task(task) for task in batch]
            
            # Odota kaikki vastaukset
            batch_responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Käsittele virheet
            for response in batch_responses:
                if isinstance(response, Exception):
                    all_responses.append(BatchResponse(
                        success=False,
                        result=None,
                        error=str(response)
                    ))
                else:
                    all_responses.append(response)
            
            # Odota rate limit cooldown
            await asyncio.sleep(1.0)
        
        return all_responses

async def main():
    """Testaa Batch API:a"""
    batch_api = WindsurfBatchAPI()
    
    # Testitehtävät
    tasks = [
        {
            "prompt": "Selitä mitä on tekoäly.",
            "max_tokens": 1000,
            "temperature": 0.7
        },
        {
            "prompt": "Miten koneoppiminen toimii?",
            "max_tokens": 1500,
            "temperature": 0.8
        },
        {
            "prompt": "Kerro neuroverkoista.",
            "max_tokens": 2000,
            "temperature": 0.6
        }
    ]
    
    print("\nSuoritetaan eräajo...")
    responses = await batch_api.handle_batch_requests(
        tasks,
        system_prompt="Selitä asiat yksinkertaisesti."
    )
    
    print("\nTulokset:")
    for i, response in enumerate(responses):
        print(f"\nTehtävä {i+1}:")
        print(f"Onnistui: {response.success}")
        if response.success:
            print(f"Vastaus: {response.result[:200]}...")
            print(f"Suoritusaika: {response.execution_time:.2f}s")
            print(f"Tokenien käyttö: {response.token_usage}")
        else:
            print(f"Virhe: {response.error}")

if __name__ == "__main__":
    asyncio.run(main())
