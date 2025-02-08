"""
Windsurf integraation testit.
Tämä moduuli testaa Windsurf IDE:n integraatiota Claude-3 mallille.
"""

import os
import json
import logging
import asyncio
import unittest
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from windsurf_batch_api import WindsurfBatchAPI, BatchTask, BatchResponse
from windsurf_integration import WindsurfIntegration
from windsurf_commands import WindsurfCommands

# Konfiguroi lokitus
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AsyncTestCase(unittest.TestCase):
    """Asynkroninen test case"""
    
    def setUp(self):
        """Alusta testi"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Siivoa testi"""
        self.loop.close()
    
    def run_async(self, coro):
        """Suorita coroutine"""
        return self.loop.run_until_complete(coro)

class TestWindsurfIntegration(AsyncTestCase):
    """Windsurf integraation testit"""
    
    @classmethod
    def setUpClass(cls):
        """Alusta testit"""
        # Lataa ympäristömuuttujat
        load_dotenv()
        
        # Alusta API:t
        cls.batch_api = WindsurfBatchAPI()
        cls.commands = WindsurfCommands()
        cls.integration = WindsurfIntegration()
    
    def test_windsurf_config(self):
        """Testaa Windsurf konfiguraatio"""
        logger.info("Testataan Windsurf konfiguraatiota...")
        
        # Tarkista Anthropic asetukset
        self.assertIsNotNone(self.integration.anthropic)
        self.assertEqual(
            self.integration.anthropic.default_model,
            "claude-3-opus-20240229"
        )
        self.assertEqual(self.integration.anthropic.batch_limit, 15)
        
        logger.info("Windsurf konfiguraatio OK")
    
    def test_single_api_call(self):
        """Testaa yksittäinen API-kutsu"""
        logger.info("Testataan yksittäistä API-kutsua...")
        
        # Testaa koodin täydennystä
        code = """def analyze_ethics(model: str, data: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"Analyze AI model ethics.\"\"\"
"""
        
        completion = self.run_async(self.commands.complete_code(
            code=code,
            language="python"
        ))
        
        self.assertIsNotNone(completion)
        self.assertIn("def analyze_ethics", completion)
        
        logger.info(f"Koodin täydennys:\n{completion}")
        
        # Testaa koodin selitystä
        explanation = self.run_async(self.commands.explain_code(
            code=completion,
            language="python"
        ))
        
        self.assertIsNotNone(explanation)
        self.assertIn("analyze_ethics", explanation)
        
        logger.info(f"Koodin selitys:\n{explanation}")
    
    def test_batch_api_integration(self):
        """Testaa Batch API integraatio"""
        logger.info("Testataan Batch API integraatiota...")
        
        # Luo testitehtävät
        tasks = [
            BatchTask(
                prompt="Selitä miten tekoäly voi parantaa koodin laatua.",
                max_tokens=1000,
                temperature=0.7,
                request_id="task_1"
            ),
            BatchTask(
                prompt="Anna esimerkkejä tekoälyn käytöstä ohjelmistokehityksessä.",
                max_tokens=1200,
                temperature=0.8,
                request_id="task_2"
            ),
            BatchTask(
                prompt="Miten tekoäly voi auttaa koodin dokumentoinnissa?",
                max_tokens=1500,
                temperature=0.6,
                request_id="task_3"
            )
        ]
        
        # Suorita eräajo
        start_time = datetime.now()
        responses = self.run_async(self.batch_api.handle_batch_requests(tasks))
        execution_time = (datetime.now() - start_time).total_seconds()
        
        self.assertEqual(len(responses), len(tasks))
        self.assertGreater(execution_time, 1.0)  # Rate limiting toimii
        
        # Tarkista tulokset
        for i, response in enumerate(responses):
            logger.info(f"\nTehtävä {i+1} ({tasks[i].request_id}):")
            logger.info(f"Onnistui: {response.success}")
            
            if response.success:
                self.assertIsNotNone(response.result)
                self.assertGreater(response.execution_time, 0)
                self.assertIsNotNone(response.token_usage)
                
                logger.info(f"Vastaus: {response.result[:200]}...")
                logger.info(f"Suoritusaika: {response.execution_time:.2f}s")
                logger.info(f"Tokenien käyttö: {response.token_usage}")
            else:
                logger.error(f"Virhe: {response.error}")
    
    def test_ide_integration(self):
        """Testaa IDE integraatio"""
        logger.info("Testataan IDE integraatiota...")
        
        # Tarkista IDE komennot
        commands = self.integration.register_ide_commands()
        self.assertGreater(len(commands), 0)
        
        # Tarkista pikanäppäimet
        keybindings = self.integration.register_keybindings()
        self.assertGreater(len(keybindings), 0)
        
        # Tarkista kontekstivalikko
        menu_items = self.integration.register_context_menu()
        self.assertGreater(len(menu_items), 0)
        
        logger.info("IDE integraatio OK")
    
    def test_error_handling(self):
        """Testaa virheenkäsittely"""
        logger.info("Testataan virheenkäsittelyä...")
        
        # Testaa virheellistä API-kutsua
        code = "x" * 100000  # Liian pitkä koodi
        completion = self.run_async(self.commands.complete_code(
            code=code,
            language="python"
        ))
        
        self.assertIsNone(completion)
        
        # Testaa virheellistä eräajoa
        tasks = [
            BatchTask(
                prompt="x" * 100000,  # Liian pitkä prompt
                max_tokens=1000
            )
        ]
        
        responses = self.run_async(self.batch_api.handle_batch_requests(tasks))
        
        self.assertEqual(len(responses), 1)
        self.assertFalse(responses[0].success)
        self.assertIsNotNone(responses[0].error)
        
        logger.info("Virheenkäsittely OK")
    
    def test_performance(self):
        """Testaa suorituskyky"""
        logger.info("Testataan suorituskykyä...")
        
        # Testaa rinnakkaista suoritusta
        tasks = [
            BatchTask(
                prompt=f"Tehtävä {i}",
                max_tokens=1000,
                request_id=f"perf_task_{i}"
            )
            for i in range(10)
        ]
        
        start_time = datetime.now()
        responses = self.run_async(self.batch_api.handle_batch_requests(tasks))
        execution_time = (datetime.now() - start_time).total_seconds()
        
        self.assertEqual(len(responses), len(tasks))
        
        # Laske keskimääräinen suoritusaika
        success_times = [
            r.execution_time for r in responses
            if r.success and r.execution_time > 0
        ]
        
        if success_times:
            avg_time = sum(success_times) / len(success_times)
            logger.info(f"Keskimääräinen suoritusaika: {avg_time:.2f}s")
        
        logger.info(f"Kokonaissuoritusaika: {execution_time:.2f}s")
        logger.info("Suorituskykytesti OK")

if __name__ == "__main__":
    unittest.main(verbosity=2)
