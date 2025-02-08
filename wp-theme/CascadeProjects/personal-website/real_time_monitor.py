import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import json

import numpy as np
from prometheus_client import (
    Counter, Gauge, Histogram,
    start_http_server, CollectorRegistry
)
import aiohttp
import psutil
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Konfiguroi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Metriikkatyypit"""
    PERFORMANCE = "performance"
    ERROR = "error"
    RESOURCE = "resource"

@dataclass
class Metric:
    """Metriikka"""
    type: MetricType
    name: str
    value: float
    timestamp: datetime
    labels: Optional[Dict] = None

class RealTimeMonitor:
    """Reaaliaikainen seuranta"""
    
    def __init__(
        self,
        metrics_port: int = 8000,
        websocket_port: int = 8001,
        history_size: int = 3600  # 1h
    ):
        """
        Alusta seuranta
        
        Args:
            metrics_port: Prometheus-portti
            websocket_port: WebSocket-portti
            history_size: Historian koko sekunteina
        """
        self.history_size = history_size
        
        # FastAPI-sovellus
        self.app = FastAPI()
        self.setup_routes()
        
        # Prometheus-rekisteri
        self.registry = CollectorRegistry()
        
        # Suorituskykymetriikat
        self.response_time = Histogram(
            'response_time_seconds',
            'Response time in seconds',
            ['endpoint'],
            registry=self.registry
        )
        self.token_usage = Counter(
            'token_usage_total',
            'Token usage',
            ['model'],
            registry=self.registry
        )
        self.task_count = Counter(
            'tasks_total',
            'Total tasks',
            ['status'],
            registry=self.registry
        )
        
        # Virhemetriikat
        self.api_errors = Counter(
            'api_errors_total',
            'API errors',
            ['type'],
            registry=self.registry
        )
        self.rate_limits = Counter(
            'rate_limits_total',
            'Rate limit hits',
            ['endpoint'],
            registry=self.registry
        )
        self.network_errors = Counter(
            'network_errors_total',
            'Network errors',
            ['type'],
            registry=self.registry
        )
        
        # Resurssimetriikat
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        self.network_io = Gauge(
            'network_io_bytes',
            'Network I/O bytes',
            ['direction'],
            registry=self.registry
        )
        
        # Metriikkahistoria
        self.metric_history: List[Metric] = []
        
        # Aktiiviset WebSocket-yhteydet
        self.active_connections: Set[WebSocket] = set()
        
        # Käynnistä palvelimet
        start_http_server(metrics_port, registry=self.registry)
        
        # Käynnistä seuranta
        asyncio.create_task(self._monitor_resources())
    
    def setup_routes(self):
        """Määritä reitit"""
        @self.app.get("/")
        async def get_dashboard():
            """Palauta dashboard"""
            return HTMLResponse(self._generate_dashboard_html())
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket-endpoint"""
            await websocket.accept()
            self.active_connections.add(websocket)
            
            try:
                while True:
                    # Odota dataa
                    data = await websocket.receive_text()
                    
                    # Lähetä päivitys
                    await self._broadcast_metrics()
            
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
            
            finally:
                self.active_connections.remove(websocket)
    
    def _generate_dashboard_html(self) -> str:
        """
        Generoi dashboard HTML
        
        Returns:
            str: Dashboard HTML
        """
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Real-Time Monitoring</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { margin: 0; padding: 20px; font-family: Arial, sans-serif; }
        .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
        .card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric { font-size: 24px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Real-Time Monitoring Dashboard</h1>
    <div class="grid">
        <div class="card">
            <h2>Performance</h2>
            <div id="performance-chart"></div>
        </div>
        <div class="card">
            <h2>Errors</h2>
            <div id="error-chart"></div>
        </div>
        <div class="card">
            <h2>Resources</h2>
            <div id="resource-chart"></div>
        </div>
        <div class="card">
            <h2>Key Metrics</h2>
            <div id="metrics"></div>
        </div>
    </div>
    <script>
        const ws = new WebSocket('ws://localhost:8001/ws');
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            updateCharts(data);
            updateMetrics(data);
        };
        
        function updateCharts(data) {
            // Päivitä kuvaajat
            Plotly.newPlot('performance-chart', data.performance);
            Plotly.newPlot('error-chart', data.errors);
            Plotly.newPlot('resource-chart', data.resources);
        }
        
        function updateMetrics(data) {
            // Päivitä metriikat
            const metrics = document.getElementById('metrics');
            metrics.innerHTML = '';
            
            for (const [name, value] of Object.entries(data.metrics)) {
                metrics.innerHTML += `
                    <div class="metric">
                        <span>${name}:</span>
                        <span>${value}</span>
                    </div>
                `;
            }
        }
    </script>
</body>
</html>
"""
    
    async def track_metric(
        self,
        type: MetricType,
        name: str,
        value: float,
        labels: Optional[Dict] = None
    ):
        """
        Seuraa metriikkaa
        
        Args:
            type: Metriikkatyyppi
            name: Metriikan nimi
            value: Arvo
            labels: Labelit
        """
        # Luo metriikka
        metric = Metric(
            type=type,
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels
        )
        
        # Tallenna historia
        self.metric_history.append(metric)
        
        # Siivoa vanha historia
        cutoff = datetime.now() - timedelta(
            seconds=self.history_size
        )
        self.metric_history = [
            m for m in self.metric_history
            if m.timestamp >= cutoff
        ]
        
        # Päivitä Prometheus
        if type == MetricType.PERFORMANCE:
            if name == "response_time":
                self.response_time.labels(
                    endpoint=labels.get("endpoint", "unknown")
                ).observe(value)
            elif name == "token_usage":
                self.token_usage.labels(
                    model=labels.get("model", "unknown")
                ).inc(value)
            elif name == "task_count":
                self.task_count.labels(
                    status=labels.get("status", "unknown")
                ).inc(value)
        
        elif type == MetricType.ERROR:
            if name == "api_error":
                self.api_errors.labels(
                    type=labels.get("type", "unknown")
                ).inc(value)
            elif name == "rate_limit":
                self.rate_limits.labels(
                    endpoint=labels.get("endpoint", "unknown")
                ).inc(value)
            elif name == "network_error":
                self.network_errors.labels(
                    type=labels.get("type", "unknown")
                ).inc(value)
        
        # Lähetä päivitys
        await self._broadcast_metrics()
    
    async def _monitor_resources(self):
        """Seuraa resursseja"""
        while True:
            try:
                # CPU
                cpu = psutil.cpu_percent()
                self.cpu_usage.set(cpu)
                await self.track_metric(
                    MetricType.RESOURCE,
                    "cpu_usage",
                    cpu
                )
                
                # Muisti
                memory = psutil.virtual_memory()
                self.memory_usage.set(memory.used)
                await self.track_metric(
                    MetricType.RESOURCE,
                    "memory_usage",
                    memory.percent
                )
                
                # Verkko
                net = psutil.net_io_counters()
                self.network_io.labels(
                    direction="sent"
                ).set(net.bytes_sent)
                self.network_io.labels(
                    direction="recv"
                ).set(net.bytes_recv)
                
                await self.track_metric(
                    MetricType.RESOURCE,
                    "network_io",
                    net.bytes_sent + net.bytes_recv
                )
            
            except Exception as e:
                logger.error(f"Resource monitoring error: {str(e)}")
            
            await asyncio.sleep(1)
    
    async def _broadcast_metrics(self):
        """Lähetä metriikat"""
        if not self.active_connections:
            return
        
        # Kerää data
        data = self._prepare_chart_data()
        
        # Lähetä kaikille
        for connection in self.active_connections:
            try:
                await connection.send_text(
                    json.dumps(data)
                )
            except Exception as e:
                logger.error(
                    f"WebSocket send error: {str(e)}"
                )
    
    def _prepare_chart_data(self) -> Dict:
        """
        Valmistele kuvaajadata
        
        Returns:
            Dict: Kuvaajadata
        """
        # Kerää metriikat tyypeittäin
        performance = []
        errors = []
        resources = []
        
        for metric in self.metric_history:
            if metric.type == MetricType.PERFORMANCE:
                performance.append(metric)
            elif metric.type == MetricType.ERROR:
                errors.append(metric)
            else:
                resources.append(metric)
        
        # Luo kuvaajat
        return {
            "performance": [
                go.Scatter(
                    x=[m.timestamp for m in performance],
                    y=[m.value for m in performance],
                    name=m.name
                )
            ],
            "errors": [
                go.Bar(
                    x=[m.timestamp for m in errors],
                    y=[m.value for m in errors],
                    name=m.name
                )
            ],
            "resources": [
                go.Scatter(
                    x=[m.timestamp for m in resources],
                    y=[m.value for m in resources],
                    name=m.name,
                    fill='tozeroy'
                )
            ],
            "metrics": {
                "Total Tasks": sum(
                    m.value for m in performance
                    if m.name == "task_count"
                ),
                "Error Rate": (
                    len(errors) /
                    len(performance)
                    if performance else 0
                ),
                "Avg Response Time": np.mean([
                    m.value for m in performance
                    if m.name == "response_time"
                ] or [0])
            }
        }

async def main():
    """Testaa reaaliaikaista seurantaa"""
    monitor = RealTimeMonitor()
    
    # Simuloi metriikkoja
    while True:
        # Suorituskyky
        await monitor.track_metric(
            MetricType.PERFORMANCE,
            "response_time",
            np.random.uniform(0.1, 2.0),
            {"endpoint": "/api/test"}
        )
        
        await monitor.track_metric(
            MetricType.PERFORMANCE,
            "token_usage",
            np.random.randint(100, 1000),
            {"model": "gpt-3.5-turbo"}
        )
        
        # Virheet
        if np.random.random() < 0.1:
            await monitor.track_metric(
                MetricType.ERROR,
                "api_error",
                1,
                {"type": "rate_limit"}
            )
        
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
