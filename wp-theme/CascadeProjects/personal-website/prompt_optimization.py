import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Konfiguroi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptTemplate(Enum):
    """Prompt-mallit"""
    
    # GPT-4 mallit
    GPT4_ANALYSIS = """[ANALYSIS REQUEST]
Context: {context}
Objective: Provide a comprehensive analysis
Focus Areas:
{focus_points}

Analysis Request: {prompt}

Expected Output Format:
1. Key Findings
2. Detailed Analysis
3. Recommendations
4. References"""

    GPT4_CODE = """[CODE GENERATION REQUEST]
Requirements:
{requirements}

Technical Context:
{context}

Implementation Request: {prompt}

Expected Deliverables:
1. Working Code
2. Documentation
3. Test Cases
4. Performance Considerations"""

    # Starcoder mallit
    STARCODER_IMPLEMENTATION = """# Code Implementation Task
'''
Requirements:
{requirements}

Context:
{context}
'''

# Task: {prompt}
# Generate production-ready code with:
# 1. Type hints
# 2. Error handling
# 3. Documentation
# 4. Tests

def implement():
    '''Implementation goes here'''"""

    STARCODER_REVIEW = """# Code Review Task
'''
Context:
{context}

Focus Areas:
- Performance
- Security
- Best Practices
'''

# Review Request: {prompt}"""

    # Claude mallit
    CLAUDE_RESEARCH = """### Research Analysis Request

Background:
{context}

Research Objectives:
{objectives}

Primary Question:
{prompt}

Expected Structure:
1. Executive Summary
2. Methodology
3. Findings
4. Conclusions
5. References"""

    CLAUDE_OPTIMIZATION = """### Optimization Task

Current State:
{context}

Performance Metrics:
{metrics}

Optimization Target:
{prompt}

Requirements:
1. Quantifiable Improvements
2. Implementation Plan
3. Risk Analysis"""

@dataclass
class PromptConfig:
    """Prompt-konfiguraatio"""
    template: PromptTemplate
    max_context_tokens: int
    focus_areas: List[str]
    quality_threshold: float
    required_elements: List[str]

class PromptOptimizer:
    """Prompt-optimoija"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Alusta optimoija
        
        Args:
            config_path: Konfiguraatiotiedoston polku
        """
        self.config = self._load_config(config_path)
        self.vectorizer = TfidfVectorizer()
        
        # Mallikohtaiset konfiguraatiot
        self.model_configs = {
            "gpt-4": PromptConfig(
                template=PromptTemplate.GPT4_ANALYSIS,
                max_context_tokens=6000,
                focus_areas=[
                    "technical_depth",
                    "practical_implications",
                    "future_considerations"
                ],
                quality_threshold=0.9,
                required_elements=[
                    "context",
                    "objectives",
                    "constraints"
                ]
            ),
            "starcoder": PromptConfig(
                template=PromptTemplate.STARCODER_IMPLEMENTATION,
                max_context_tokens=2000,
                focus_areas=[
                    "performance",
                    "security",
                    "maintainability"
                ],
                quality_threshold=0.85,
                required_elements=[
                    "requirements",
                    "test_cases",
                    "documentation"
                ]
            ),
            "claude": PromptConfig(
                template=PromptTemplate.CLAUDE_RESEARCH,
                max_context_tokens=8000,
                focus_areas=[
                    "comprehensive_analysis",
                    "evidence_based",
                    "practical_applications"
                ],
                quality_threshold=0.95,
                required_elements=[
                    "background",
                    "methodology",
                    "conclusions"
                ]
            )
        }
        
        # Optimointimetriikat
        self.optimization_history: Dict[str, List[Dict]] = {}
    
    def _load_config(self, path: str) -> Dict:
        """Lataa konfiguraatio"""
        with open(path) as f:
            return json.load(f)
    
    def optimize_prompt(
        self,
        prompt: str,
        model: str,
        context: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> Tuple[str, Dict]:
        """
        Optimoi prompt
        
        Args:
            prompt: Alkuperäinen prompt
            model: Mallin nimi
            context: Konteksti
            task_type: Tehtävätyyppi
        
        Returns:
            Tuple[str, Dict]: Optimoitu prompt ja metriikat
        """
        config = self.model_configs[model]
        
        # Analysoi prompt
        analysis = self._analyze_prompt(prompt)
        
        # Valitse sopiva template
        template = self._select_template(
            model,
            task_type,
            analysis
        )
        
        # Optimoi konteksti
        if context:
            context = self._optimize_context(
                context,
                config.max_context_tokens
            )
        
        # Generoi focus-alueet
        focus_points = self._generate_focus_points(
            prompt,
            config.focus_areas
        )
        
        # Rakenna optimoitu prompt
        optimized = template.value.format(
            prompt=prompt,
            context=context or "",
            focus_points="\n".join(focus_points),
            requirements=self._extract_requirements(prompt),
            objectives=self._extract_objectives(prompt),
            metrics=self._get_performance_metrics(model)
        )
        
        # Validoi ja paranna
        optimized = self._enhance_prompt(
            optimized,
            config
        )
        
        # Kerää metriikat
        metrics = {
            "original_length": len(prompt),
            "optimized_length": len(optimized),
            "focus_areas": len(focus_points),
            "quality_score": self._calculate_quality(
                optimized,
                config
            )
        }
        
        # Päivitä historia
        self._update_history(model, prompt, optimized, metrics)
        
        return optimized, metrics
    
    def _analyze_prompt(self, prompt: str) -> Dict:
        """
        Analysoi prompt
        
        Args:
            prompt: Prompt
        
        Returns:
            Dict: Analyysi
        """
        # Muunna teksti vektoreiksi
        vectors = self.vectorizer.fit_transform([prompt])
        
        # Analysoi avainsanat
        keywords = dict(
            zip(
                self.vectorizer.get_feature_names_out(),
                vectors.toarray()[0]
            )
        )
        
        return {
            "complexity": self._calculate_complexity(prompt),
            "keywords": keywords,
            "sentiment": self._analyze_sentiment(prompt),
            "technical_terms": self._extract_technical_terms(prompt)
        }
    
    def _select_template(
        self,
        model: str,
        task_type: Optional[str],
        analysis: Dict
    ) -> PromptTemplate:
        """
        Valitse sopiva template
        
        Args:
            model: Malli
            task_type: Tehtävätyyppi
            analysis: Prompt-analyysi
        
        Returns:
            PromptTemplate: Valittu template
        """
        if model == "gpt-4":
            if task_type == "code":
                return PromptTemplate.GPT4_CODE
            return PromptTemplate.GPT4_ANALYSIS
        
        elif model == "starcoder":
            if analysis["complexity"] > 0.7:
                return PromptTemplate.STARCODER_IMPLEMENTATION
            return PromptTemplate.STARCODER_REVIEW
        
        elif model == "claude":
            if "optimization" in analysis["keywords"]:
                return PromptTemplate.CLAUDE_OPTIMIZATION
            return PromptTemplate.CLAUDE_RESEARCH
        
        raise ValueError(f"Unknown model: {model}")
    
    def _optimize_context(
        self,
        context: str,
        max_tokens: int
    ) -> str:
        """
        Optimoi konteksti
        
        Args:
            context: Konteksti
            max_tokens: Maksimi token-määrä
        
        Returns:
            str: Optimoitu konteksti
        """
        # Estimoi token-määrä
        estimated_tokens = len(context.split()) * 1.3
        
        if estimated_tokens <= max_tokens:
            return context
        
        # Lyhennä kontekstia
        sentences = context.split(". ")
        
        # Laske lauseiden tärkeys
        importance = self._calculate_sentence_importance(sentences)
        
        # Valitse tärkeimmät lauseet
        selected = []
        total_tokens = 0
        
        for sentence, score in sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            tokens = len(sentence.split()) * 1.3
            if total_tokens + tokens > max_tokens:
                break
            
            selected.append(sentence)
            total_tokens += tokens
        
        return ". ".join(selected)
    
    def _generate_focus_points(
        self,
        prompt: str,
        focus_areas: List[str]
    ) -> List[str]:
        """
        Generoi focus-alueet
        
        Args:
            prompt: Prompt
            focus_areas: Focus-alueet
        
        Returns:
            List[str]: Generoidut focus-alueet
        """
        points = []
        
        for area in focus_areas:
            if area in self._extract_technical_terms(prompt):
                points.append(
                    f"- {area.replace('_', ' ').title()}: "
                    f"Analyze and optimize specifically for {area}"
                )
        
        return points or [
            f"- {area.replace('_', ' ').title()}"
            for area in focus_areas[:3]
        ]
    
    def _enhance_prompt(
        self,
        prompt: str,
        config: PromptConfig
    ) -> str:
        """
        Paranna promptia
        
        Args:
            prompt: Prompt
            config: Konfiguraatio
        
        Returns:
            str: Parannettu prompt
        """
        # Tarkista vaaditut elementit
        missing = [
            elem for elem in config.required_elements
            if elem not in prompt.lower()
        ]
        
        if missing:
            prompt += "\n\nRequired Elements:\n" + "\n".join(
                f"- Include {elem.replace('_', ' ')}"
                for elem in missing
            )
        
        # Lisää laatuvaatimukset
        if config.quality_threshold > 0.8:
            prompt += f"\n\nQuality Requirements:\n"
            prompt += f"- Minimum quality score: {config.quality_threshold}\n"
            prompt += "- Include references and citations\n"
            prompt += "- Provide concrete examples"
        
        return prompt
    
    def _calculate_quality(
        self,
        prompt: str,
        config: PromptConfig
    ) -> float:
        """
        Laske promptin laatu
        
        Args:
            prompt: Prompt
            config: Konfiguraatio
        
        Returns:
            float: Laatupisteet (0-1)
        """
        score = 0.0
        
        # Tarkista vaaditut elementit
        elements_present = sum(
            1 for elem in config.required_elements
            if elem in prompt.lower()
        )
        score += elements_present / len(config.required_elements) * 0.4
        
        # Tarkista focus-alueet
        focus_present = sum(
            1 for area in config.focus_areas
            if area in prompt.lower()
        )
        score += focus_present / len(config.focus_areas) * 0.3
        
        # Arvioi rakenne
        has_structure = (
            "1." in prompt and
            "2." in prompt and
            "3." in prompt
        )
        score += 0.2 if has_structure else 0.0
        
        # Arvioi selkeys
        score += min(
            len(prompt.split("\n")) / 20,
            0.1
        )
        
        return score
    
    def _update_history(
        self,
        model: str,
        original: str,
        optimized: str,
        metrics: Dict
    ):
        """
        Päivitä optimointihistoria
        
        Args:
            model: Malli
            original: Alkuperäinen prompt
            optimized: Optimoitu prompt
            metrics: Metriikat
        """
        if model not in self.optimization_history:
            self.optimization_history[model] = []
        
        self.optimization_history[model].append({
            "original": original,
            "optimized": optimized,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_optimization_stats(
        self,
        model: Optional[str] = None
    ) -> Dict:
        """
        Hae optimointitilastot
        
        Args:
            model: Malli (jos None, kaikki mallit)
        
        Returns:
            Dict: Tilastot
        """
        if model:
            history = self.optimization_history.get(model, [])
        else:
            history = [
                item
                for items in self.optimization_history.values()
                for item in items
            ]
        
        if not history:
            return {}
        
        quality_scores = [
            item["metrics"]["quality_score"]
            for item in history
        ]
        
        return {
            "total_optimizations": len(history),
            "avg_quality_score": np.mean(quality_scores),
            "avg_length_increase": np.mean([
                item["metrics"]["optimized_length"] /
                item["metrics"]["original_length"]
                for item in history
            ]),
            "success_rate": sum(
                1 for score in quality_scores
                if score > 0.8
            ) / len(quality_scores)
        }

def main():
    """Testaa prompt-optimointia"""
    optimizer = PromptOptimizer()
    
    # Testipromptit
    prompts = {
        "analysis": "Analyze the impact of quantum computing on cryptography",
        "code": "Implement a secure password hashing system",
        "research": "Research the effects of AI on software development"
    }
    
    # Testaa eri malleilla
    for task_type, prompt in prompts.items():
        for model in ["gpt-4", "starcoder", "claude"]:
            optimized, metrics = optimizer.optimize_prompt(
                prompt,
                model,
                task_type=task_type
            )
            
            logger.info(f"\nModel: {model}")
            logger.info(f"Task: {task_type}")
            logger.info(f"Original: {prompt}")
            logger.info(f"Optimized: {optimized}")
            logger.info(f"Metrics: {metrics}")
    
    # Näytä tilastot
    stats = optimizer.get_optimization_stats()
    logger.info(f"\nOverall Stats: {stats}")

if __name__ == "__main__":
    main()
