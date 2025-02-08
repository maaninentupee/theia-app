import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fastapi import FastAPI, WebSocket
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from path_utils import PathManager

# Konfiguroi logging
logging.basicConfig(
    filename="cascade_monitor.log",
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Metriikkatyypit"""
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"
    TOKEN_USAGE = "token_usage"
    COST = "cost"
    QUALITY = "quality"

@dataclass
class TaskMetrics:
    """Tehtävämetriikat"""
    task_id: str
    task_type: str
    model: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    error: Optional[str] = None
    tokens_used: int = 0
    cost: float = 0.0
    quality_score: float = 0.0
    context_size: int = 0
    
    def duration(self) -> float:
        """Laske kesto"""
        if not self.end_time:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

@dataclass
class ModelMetrics:
    """Mallimetriikat"""
    model_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    latencies: List[float] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Laske onnistumisprosentti"""
        if not self.total_requests:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def avg_latency(self) -> float:
        """Laske keskimääräinen latenssi"""
        if not self.latencies:
            return 0.0
        return np.mean(self.latencies)
    
    @property
    def avg_quality(self) -> float:
        """Laske keskimääräinen laatu"""
        if not self.quality_scores:
            return 0.0
        return np.mean(self.quality_scores)

class MetricsCollector:
    """Metriikkojen kerääjä"""
    
    def __init__(self, config_file: str = "config.json"):
        """Alusta kerääjä"""
        self.path_manager = PathManager(config_file)
        self.config = self._load_config()
        
        # Prometheus-mittarit
        self.request_counter = Counter(
            'cascade_requests_total',
            'Total requests',
            ['model', 'task_type', 'status']
        )
        
        self.latency_histogram = Histogram(
            'cascade_request_duration_seconds',
            'Request duration in seconds',
            ['model', 'task_type']
        )
        
        self.token_gauge = Gauge(
            'cascade_token_usage',
            'Token usage',
            ['model']
        )
        
        self.cost_counter = Counter(
            'cascade_cost_total',
            'Total cost',
            ['model']
        )
        
        self.quality_gauge = Gauge(
            'cascade_quality_score',
            'Quality score',
            ['model']
        )
        
        # Sisäiset metriikat
        self.task_metrics: Dict[str, TaskMetrics] = {}
        self.model_metrics: Dict[str, ModelMetrics] = {}
        
        # Käynnistä Prometheus-palvelin
        start_http_server(8000)
    
    def _load_config(self) -> dict:
        """Lataa konfiguraatio"""
        with open(self.path_manager.config_file) as f:
            return json.load(f)
            
    def start_task(
        self,
        task_id: str,
        task_type: str,
        model: str,
        context_size: int
    ):
        """
        Aloita tehtävän seuranta
        
        Args:
            task_id: Tehtävän ID
            task_type: Tehtävätyyppi
            model: Mallin nimi
            context_size: Kontekstin koko
        """
        self.task_metrics[task_id] = TaskMetrics(
            task_id=task_id,
            task_type=task_type,
            model=model,
            start_time=datetime.now(),
            context_size=context_size
        )
        
        if model not in self.model_metrics:
            self.model_metrics[model] = ModelMetrics(model_name=model)
        
        self.model_metrics[model].total_requests += 1
        
        logger.info(
            f"Started task {task_id} | "
            f"Type: {task_type} | "
            f"Model: {model}"
        )
    
    def end_task(
        self,
        task_id: str,
        status: str,
        tokens_used: int,
        cost: float,
        quality_score: float,
        error: Optional[str] = None
    ):
        """
        Lopeta tehtävän seuranta
        
        Args:
            task_id: Tehtävän ID
            status: Status
            tokens_used: Käytetyt tokenit
            cost: Kustannus
            quality_score: Laatupisteet
            error: Virhe
        """
        if task_id not in self.task_metrics:
            logger.error(f"Unknown task ID: {task_id}")
            return
        
        task = self.task_metrics[task_id]
        task.end_time = datetime.now()
        task.status = status
        task.tokens_used = tokens_used
        task.cost = cost
        task.quality_score = quality_score
        task.error = error
        
        model_metrics = self.model_metrics[task.model]
        
        if status == "success":
            model_metrics.successful_requests += 1
        else:
            model_metrics.failed_requests += 1
        
        model_metrics.total_tokens += tokens_used
        model_metrics.total_cost += cost
        model_metrics.latencies.append(task.duration())
        model_metrics.quality_scores.append(quality_score)
        
        # Päivitä Prometheus-mittarit
        self.request_counter.labels(
            task.model,
            task.task_type,
            status
        ).inc()
        
        self.latency_histogram.labels(
            task.model,
            task.task_type
        ).observe(task.duration())
        
        self.token_gauge.labels(task.model).set(tokens_used)
        self.cost_counter.labels(task.model).inc(cost)
        self.quality_gauge.labels(task.model).set(quality_score)
        
        logger.info(
            f"Ended task {task_id} | "
            f"Status: {status} | "
            f"Duration: {task.duration():.2f}s | "
            f"Tokens: {tokens_used} | "
            f"Cost: ${cost:.4f}"
        )
        
        if error:
            logger.error(
                f"Task {task_id} failed: {error}"
            )
    
    def get_metrics(
        self,
        metric_type: MetricType,
        model: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> Dict:
        """
        Hae metriikat
        
        Args:
            metric_type: Metriikkatyyppi
            model: Mallin nimi
            time_window: Aikaikkuna
        
        Returns:
            Dict: Metriikat
        """
        if model and model not in self.model_metrics:
            return {}
        
        models = [model] if model else self.model_metrics.keys()
        now = datetime.now()
        
        metrics = {}
        for model_name in models:
            model_metrics = self.model_metrics[model_name]
            
            if metric_type == MetricType.LATENCY:
                metrics[model_name] = model_metrics.avg_latency
            
            elif metric_type == MetricType.ERROR_RATE:
                metrics[model_name] = (
                    model_metrics.failed_requests /
                    model_metrics.total_requests
                    if model_metrics.total_requests else 0.0
                )
            
            elif metric_type == MetricType.SUCCESS_RATE:
                metrics[model_name] = model_metrics.success_rate
            
            elif metric_type == MetricType.TOKEN_USAGE:
                metrics[model_name] = model_metrics.total_tokens
            
            elif metric_type == MetricType.COST:
                metrics[model_name] = model_metrics.total_cost
            
            elif metric_type == MetricType.QUALITY:
                metrics[model_name] = model_metrics.avg_quality
        
        return metrics
    
    def generate_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        models: Optional[List[str]] = None
    ) -> Dict:
        """
        Generoi raportti
        
        Args:
            start_time: Alkuaika
            end_time: Loppuaika
            models: Mallit
        
        Returns:
            Dict: Raportti
        """
        if not start_time:
            start_time = datetime.now() - timedelta(days=1)
        if not end_time:
            end_time = datetime.now()
        
        report = {
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "models": {},
            "overall": {
                "total_requests": 0,
                "total_cost": 0.0,
                "avg_latency": 0.0,
                "success_rate": 0.0
            }
        }
        
        # Kerää mallikohtaiset metriikat
        for model_name, metrics in self.model_metrics.items():
            if models and model_name not in models:
                continue
            
            report["models"][model_name] = {
                "requests": {
                    "total": metrics.total_requests,
                    "successful": metrics.successful_requests,
                    "failed": metrics.failed_requests
                },
                "performance": {
                    "avg_latency": metrics.avg_latency,
                    "success_rate": metrics.success_rate,
                    "avg_quality": metrics.avg_quality
                },
                "resources": {
                    "total_tokens": metrics.total_tokens,
                    "total_cost": metrics.total_cost
                }
            }
            
            # Päivitä kokonaismetriikat
            report["overall"]["total_requests"] += metrics.total_requests
            report["overall"]["total_cost"] += metrics.total_cost
            report["overall"]["avg_latency"] += metrics.avg_latency
            report["overall"]["success_rate"] += metrics.success_rate
        
        # Keskiarvoista kokonaismetriikat
        num_models = len(report["models"])
        if num_models:
            report["overall"]["avg_latency"] /= num_models
            report["overall"]["success_rate"] /= num_models
        
        return report
    
    def visualize_metrics(
        self,
        metric_type: MetricType,
        models: Optional[List[str]] = None,
        time_window: Optional[timedelta] = None
    ) -> go.Figure:
        """
        Visualisoi metriikat
        
        Args:
            metric_type: Metriikkatyyppi
            models: Mallit
            time_window: Aikaikkuna
        
        Returns:
            go.Figure: Plotly-kuvaaja
        """
        metrics = self.get_metrics(
            metric_type,
            time_window=time_window
        )
        
        if not models:
            models = list(metrics.keys())
        
        df = pd.DataFrame({
            'Model': models,
            'Value': [metrics.get(model, 0) for model in models]
        })
        
        fig = px.bar(
            df,
            x='Model',
            y='Value',
            title=f'{metric_type.value} by Model'
        )
        
        return fig

class DashboardApp:
    """Reaaliaikainen käyttöliittymä"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """
        Alusta dashboard
        
        Args:
            metrics_collector: Metriikkojen kerääjä
        """
        self.app = FastAPI()
        self.metrics = metrics_collector
        self.clients: List[WebSocket] = []
        
        # Määritä reitit
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.clients.append(websocket)
            
            try:
                while True:
                    # Lähetä päivitetyt metriikat
                    metrics = self.metrics.get_metrics(
                        MetricType.LATENCY
                    )
                    await websocket.send_json(metrics)
                    await asyncio.sleep(1)
            
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
            
            finally:
                self.clients.remove(websocket)
        
        @self.app.get("/metrics")
        async def get_metrics(
            metric_type: MetricType,
            model: Optional[str] = None
        ):
            return self.metrics.get_metrics(metric_type, model)
        
        @self.app.get("/report")
        async def get_report(
            start_time: Optional[str] = None,
            end_time: Optional[str] = None,
            models: Optional[List[str]] = None
        ):
            return self.metrics.generate_report(
                start_time=datetime.fromisoformat(start_time)
                if start_time else None,
                end_time=datetime.fromisoformat(end_time)
                if end_time else None,
                models=models
            )
    
    def start(self, port: int = 8080):
        """
        Käynnistä dashboard
        
        Args:
            port: Portti
        """
        import uvicorn
        uvicorn.run(self.app, host="0.0.0.0", port=port)

def main():
    """Testaa monitorointia"""
    collector = MetricsCollector()
    dashboard = DashboardApp(collector)
    
    # Simuloi tehtäviä
    for i in range(10):
        task_id = f"task_{i}"
        model = np.random.choice(["gpt-4", "starcoder", "claude"])
        
        collector.start_task(
            task_id,
            "analysis",
            model,
            1000
        )
        
        time.sleep(np.random.random() * 2)
        
        collector.end_task(
            task_id,
            "success" if np.random.random() > 0.2 else "error",
            np.random.randint(100, 1000),
            np.random.random() * 0.1,
            np.random.random(),
            "Test error" if np.random.random() > 0.8 else None
        )
    
    # Generoi raportti
    report = collector.generate_report()
    print("\nMetrics Report:")
    print(json.dumps(report, indent=2))
    
    # Käynnistä dashboard
    dashboard.start()

if __name__ == "__main__":
    main()
