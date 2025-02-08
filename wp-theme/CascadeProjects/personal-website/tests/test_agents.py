import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pytest
from dotenv import load_dotenv

from task_delegation import (
    CascadeAgent,
    ModelType,
    Task,
    TaskType,
    TokenEstimator,
    BatchOptimizer
)

# Konfiguroi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestConfig:
    """Testauskonfiguraatio"""
    
    def __init__(self):
        """Alusta konfiguraatio"""
        self.config_path = Path("config.json")
        self.config = self._load_config()
        
        # Lataa API-avaimet
        load_dotenv()
    
    def _load_config(self) -> Dict:
        """Lataa konfiguraatio"""
        with open(self.config_path) as f:
            return json.load(f)
    
    def get_model_config(self, model: str) -> Dict:
        """Hae mallin konfiguraatio"""
        return self.config["models"][model]

class TestScenarios:
    """Testiskenaariot"""
    
    def __init__(self):
        """Alusta skenaariot"""
        self.scenarios = {
            TaskType.ANALYSIS: [
                "Analyze the impact of artificial intelligence on modern software development.",
                "Compare and contrast different approaches to implementing microservices.",
                "Evaluate the benefits and drawbacks of serverless architectures."
            ],
            TaskType.CODE: [
                "Write a Python function to implement binary search.",
                "Create a React component for a responsive navigation menu.",
                "Implement a simple REST API using FastAPI."
            ],
            TaskType.QUICK: [
                "What is dependency injection?",
                "Explain the CAP theorem.",
                "Define continuous integration."
            ],
            TaskType.CONTEXTUAL: [
                "How does this code handle error cases?",
                "What improvements could be made to the current architecture?",
                "How can we optimize the database queries?"
            ]
        }
        
        self.validation_criteria = {
            TaskType.ANALYSIS: {
                "min_words": 200,
                "required_sections": ["advantages", "disadvantages", "conclusion"],
                "quality_threshold": 0.8
            },
            TaskType.CODE: {
                "must_compile": True,
                "test_cases_required": True,
                "documentation_required": True
            },
            TaskType.QUICK: {
                "max_words": 100,
                "conciseness_score": 0.7
            },
            TaskType.CONTEXTUAL: {
                "context_usage": 0.6,
                "relevance_score": 0.8
            }
        }

@pytest.fixture
def config():
    """Konfiguraatio fixture"""
    return TestConfig()

@pytest.fixture
def cascade_agent():
    """Cascade-agentti fixture"""
    return CascadeAgent()

@pytest.fixture
def scenarios():
    """Testiskenaariot fixture"""
    return TestScenarios()

class TestModelPerformance:
    """Mallien suorituskykytestit"""
    
    @pytest.mark.asyncio
    async def test_model_basic_completion(
        self,
        config: TestConfig,
        cascade_agent: CascadeAgent
    ):
        """Testaa peruskompletointi"""
        for model_name, model_config in config.config["models"].items():
            task = Task(
                type=TaskType.QUICK,
                content="What is Python?",
                priority=0.5
            )
            
            start_time = time.time()
            result = await cascade_agent.delegate_task(task)
            duration = time.time() - start_time
            
            assert result, f"No result from {model_name}"
            assert duration < 10, f"{model_name} took too long"
            assert len(result) > 50, f"{model_name} response too short"
    
    @pytest.mark.asyncio
    async def test_model_token_limits(
        self,
        config: TestConfig,
        cascade_agent: CascadeAgent
    ):
        """Testaa token-rajoitukset"""
        for model_name, model_config in config.config["models"].items():
            # Luo pitkä tehtävä
            long_text = "test " * 1000
            task = Task(
                type=TaskType.ANALYSIS,
                content=long_text,
                priority=0.8
            )
            
            result = await cascade_agent.delegate_task(task)
            tokens = TokenEstimator().estimate_tokens(result)
            
            assert tokens <= model_config["max_output_tokens"], \
                f"{model_name} exceeded token limit"
    
    @pytest.mark.asyncio
    async def test_model_error_handling(
        self,
        config: TestConfig,
        cascade_agent: CascadeAgent
    ):
        """Testaa virheenkäsittely"""
        invalid_tasks = [
            Task(type=TaskType.QUICK, content="", priority=0.5),
            Task(type=TaskType.CODE, content="!@#$%^", priority=0.5),
            Task(type=TaskType.ANALYSIS, content="a" * 10000, priority=0.5)
        ]
        
        for task in invalid_tasks:
            try:
                result = await cascade_agent.delegate_task(task)
                assert "error" in result.lower(), \
                    "Invalid task should return error"
            except Exception as e:
                assert str(e), "Error should have message"

class TestTaskDelegation:
    """Tehtävien delegointitestit"""
    
    @pytest.mark.asyncio
    async def test_task_type_routing(
        self,
        cascade_agent: CascadeAgent,
        scenarios: TestScenarios
    ):
        """Testaa tehtävätyyppien reititys"""
        for task_type, prompts in scenarios.scenarios.items():
            for prompt in prompts:
                task = Task(
                    type=task_type,
                    content=prompt,
                    priority=0.7
                )
                
                # Tarkista mallin valinta
                model = cascade_agent._select_model(task)
                config = cascade_agent.model_configs[model]
                
                assert task_type in config["task_types"], \
                    f"Wrong model {model} for {task_type}"
    
    @pytest.mark.asyncio
    async def test_priority_handling(
        self,
        cascade_agent: CascadeAgent
    ):
        """Testaa prioriteettien käsittely"""
        priorities = [0.1, 0.5, 0.9]
        task_type = TaskType.ANALYSIS
        
        for priority in priorities:
            task = Task(
                type=task_type,
                content="Test priority handling",
                priority=priority
            )
            
            model = cascade_agent._select_model(task)
            config = cascade_agent.model_configs[model]
            
            assert priority >= config["min_priority"], \
                f"Priority {priority} too low for {model}"

class TestBatchProcessing:
    """Eräkäsittelytestit"""
    
    @pytest.mark.asyncio
    async def test_batch_optimization(
        self,
        cascade_agent: CascadeAgent
    ):
        """Testaa eräkäsittelyn optimointi"""
        optimizer = BatchOptimizer()
        
        # Luo testierä
        tasks = [
            Task(
                type=TaskType.QUICK,
                content=f"Quick question {i}",
                priority=0.5
            )
            for i in range(5)
        ]
        
        for task in tasks:
            # Jaa tehtävä osiin
            subtasks = optimizer.split_task(
                task,
                max_tokens=1000,
                overlap=50
            )
            
            # Tarkista jako
            if len(task.content) > 1000:
                assert len(subtasks) > 1, "Long task not split"
                
                # Tarkista päällekkäisyys
                for i in range(len(subtasks) - 1):
                    t1 = subtasks[i].content
                    t2 = subtasks[i + 1].content
                    overlap = optimizer._find_overlap(t1, t2)
                    assert overlap > 0, "No overlap between parts"

def main():
    """Suorita testit"""
    pytest.main([__file__, "-v"])

if __name__ == "__main__":
    main()
