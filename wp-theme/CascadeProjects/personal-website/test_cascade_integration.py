"""
Cascade integraation testit.
Tämä moduuli testaa Cascade-integraation toimintaa Claude-3 mallilla.
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

class TestCascadeIntegration(AsyncTestCase):
    """Cascade integraation testit"""
    
    @classmethod
    def setUpClass(cls):
        """Alusta testit"""
        # Lataa ympäristömuuttujat
        load_dotenv()
        
        # Alusta API:t
        cls.batch_api = WindsurfBatchAPI()
        cls.commands = WindsurfCommands()
        cls.integration = WindsurfIntegration()
    
    def test_single_task(self):
        """Testaa yksittäinen tehtävä"""
        logger.info("Testataan yksittäistä tehtävää...")
        
        task = BatchTask(
            prompt="Selitä tekoälyn eettiset vaikutukset ohjelmistokehityksessä.",
            max_tokens=1500,
            temperature=0.7,
            request_id="test_single_task"
        )
        
        response = self.run_async(self.batch_api._execute_single_task(task))
        
        self.assertTrue(response.success)
        self.assertIsNotNone(response.result)
        self.assertGreater(response.execution_time, 0)
        self.assertIsNotNone(response.token_usage)
        
        logger.info(f"Tehtävän tulos: {response.result[:200]}...")
        logger.info(f"Suoritusaika: {response.execution_time:.2f}s")
        logger.info(f"Tokenien käyttö: {response.token_usage}")
    
    def test_batch_processing(self):
        """Testaa eräajo"""
        logger.info("Testataan eräajoa...")
        
        tasks = [
            {
                "prompt": "Selitä tekoälyn eettiset vaikutukset.",
                "max_tokens": 1000,
                "temperature": 0.7
            },
            {
                "prompt": "Miten tekoäly vaikuttaa ohjelmistokehitykseen?",
                "max_tokens": 1200,
                "temperature": 0.8
            },
            {
                "prompt": "Mitä haasteita tekoälyn käyttöön liittyy?",
                "max_tokens": 1500,
                "temperature": 0.6
            }
        ]
        
        responses = self.run_async(self.batch_api.handle_batch_requests(
            tasks,
            system_prompt="Selitä asiat yksinkertaisesti ja käytännönläheisesti."
        ))
        
        self.assertEqual(len(responses), len(tasks))
        
        for i, response in enumerate(responses):
            logger.info(f"\nTehtävä {i+1}:")
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
    
    def test_code_completion(self):
        """Testaa koodin täydennys"""
        logger.info("Testataan koodin täydennystä...")
        
        code = """def analyze_ai_ethics(model_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
        
        completion = self.run_async(self.commands.complete_code(code, "python"))
        
        self.assertIsNotNone(completion)
        logger.info(f"Täydennetty koodi:\n{completion}")
    
    def test_code_explanation(self):
        """Testaa koodin selitys"""
        logger.info("Testataan koodin selitystä...")
        
        code = """def analyze_ai_ethics(model_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"Analyze ethical implications of AI model usage.\"\"\"
    results = {}
    
    # Check data privacy
    results['privacy_score'] = check_data_privacy(data)
    
    # Evaluate bias
    results['bias_metrics'] = evaluate_model_bias(model_type, data)
    
    # Assess transparency
    results['transparency'] = calculate_transparency_score(model_type)
    
    # Generate recommendations
    results['recommendations'] = generate_ethical_recommendations(results)
    
    return results"""
        
        explanation = self.run_async(self.commands.explain_code(code, "python"))
        
        self.assertIsNotNone(explanation)
        logger.info(f"Koodin selitys:\n{explanation}")
    
    def test_rate_limiting(self):
        """Testaa rate limiting"""
        logger.info("Testataan rate limitingiä...")
        
        # Luo 10 tehtävää
        tasks = [
            {
                "prompt": f"Tehtävä {i}",
                "max_tokens": 1000
            }
            for i in range(10)
        ]
        
        start_time = datetime.now()
        responses = self.run_async(self.batch_api.handle_batch_requests(tasks))
        execution_time = (datetime.now() - start_time).total_seconds()
        
        self.assertEqual(len(responses), len(tasks))
        self.assertGreater(execution_time, 1.0)  # Rate limiting toimii
        
        logger.info(f"Eräajon suoritusaika: {execution_time:.2f}s")
    
    def test_error_handling(self):
        """Testaa virheenkäsittely"""
        logger.info("Testataan virheenkäsittelyä...")
        
        # Virheellinen tehtävä (liian pitkä prompt)
        task = BatchTask(
            prompt="x" * 100000,  # Liian pitkä prompt
            max_tokens=1000
        )
        
        response = self.run_async(self.batch_api._execute_single_task(task))
        
        self.assertFalse(response.success)
        self.assertIsNotNone(response.error)
        
        logger.info(f"Virhe: {response.error}")

if __name__ == "__main__":
    unittest.main(verbosity=2)
