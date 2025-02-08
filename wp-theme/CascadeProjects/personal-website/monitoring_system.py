import asyncio
import logging
import logging.handlers
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import json

import numpy as np
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import aiohttp
import psutil

# Konfiguroi logging
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Pääloggeri
logger = logging.getLogger("monitoring")
logger.setLevel(logging.INFO)

# Tiedostokäsittelijä
file_handler = logging.handlers.RotatingFileHandler(
    f"{LOG_DIR}/performance.log",
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s'
))
logger.addHandler(file_handler)

# Konsolikäsittelijä
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s'
))
logger.addHandler(console_handler)

class MetricType(Enum):
    """Metriikkatyypit"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"

class AlertLevel(Enum):
    """Hälytystasot"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Alert:
    """Hälytys"""
    level: AlertLevel
    message: str
    timestamp: datetime
    data: Optional[Dict] = None

class PerformanceMetrics:
    """Suorituskykymetriikat"""
    
    def __init__(self):
        self.task_duration = Histogram(
            'task_duration_seconds',
            'Task duration',
            ['type', 'status']
        )
        self.task_queue_size = Gauge(
            'task_queue_size',
            'Task queue size',
            ['priority']
        )
        self.worker_utilization = Gauge(
            'worker_utilization_percent',
            'Worker utilization',
            ['pool_type']
        )
        self.batch_size = Histogram(
            'batch_size',
            'Batch size',
            ['strategy']
        )

class ResourceMetrics:
    """Resurssimetriikat"""
    
    def __init__(self):
        self.memory_allocation = Gauge(
            'memory_allocation_bytes',
            'Memory allocation',
            ['type']
        )
        self.io_operations = Counter(
            'io_operations_total',
            'IO operations',
            ['type']
        )
        self.network_traffic = Counter(
            'network_traffic_bytes',
            'Network traffic',
            ['direction']
        )
        self.cache_stats = Gauge(
            'cache_stats',
            'Cache statistics',
            ['operation']
        )

class RealTimeMetrics:
    """Reaaliaikaiset metriikat"""
    
    def __init__(self):
        self.realtime_latency = Histogram(
            'realtime_latency_seconds',
            'Real-time latency',
            ['operation'],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)
        )
        self.realtime_throughput = Counter(
            'realtime_throughput_total',
            'Real-time throughput',
            ['type']
        )
        self.realtime_errors = Counter(
            'realtime_errors_total',
            'Real-time errors',
            ['type']
        )
        self.realtime_queue = Gauge(
            'realtime_queue_current',
            'Real-time queue size',
            ['priority']
        )

class AlertSystem:
    """Hälytysjärjestelmä"""
    
    def __init__(self):
        self.alerts = []
        self.thresholds = {
            'cpu_usage': 80,  # %
            'memory_usage': 85,  # %
            'error_rate': 0.05,  # 5%
            'latency': 1.0  # sekunti
        }
    
    async def check_thresholds(self, metrics: Dict[str, float]):
        """Tarkista kynnysarvot"""
        for metric, value in metrics.items():
            threshold = self.thresholds.get(metric)
            if threshold and value > threshold:
                await self.create_alert(
                    AlertLevel.WARNING,
                    f"{metric} exceeded threshold: {value:.1f} > {threshold}"
                )
    
    async def create_alert(
        self,
        level: AlertLevel,
        message: str,
        data: Optional[Dict] = None
    ):
        """Luo hälytys"""
        alert = Alert(
            level=level,
            message=message,
            timestamp=datetime.now(),
            data=data
        )
        self.alerts.append(alert)
        
        # Loki
        logger.warning(
            f"Alert | {level.value} | {message}",
            extra={"data": data}
        )
        
        # Lähetä hälytys
        if self.alert_webhook:
            await self._send_alert(alert)
    
    async def _send_alert(self, alert: Alert):
        """Lähetä hälytys"""
        async with aiohttp.ClientSession() as session:
            await session.post(
                self.alert_webhook,
                json={
                    "level": alert.level.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "data": alert.data
                }
            )

class PrometheusExporter:
    """Prometheus-viejä"""
    
    def __init__(
        self,
        host: str = 'localhost',
        ports: Dict[str, int] = None
    ):
        """
        Alusta viejä
        
        Args:
            host: Palvelimen osoite
            ports: Portit palveluille
        """
        self.host = host
        self.ports = ports or {
            'tasks': 8000,
            'resources': 8001,
            'alerts': 8002
        }
        
        # HTTP-palvelimet
        self.servers = {}
        
        # Metriikat
        self.export_counter = Counter(
            'metric_exports_total',
            'Metric exports',
            ['type']
        )
        self.export_errors = Counter(
            'export_errors_total',
            'Export errors',
            ['type']
        )
        self.last_export = Gauge(
            'last_export_timestamp',
            'Last export timestamp',
            ['type']
        )
    
    async def start(self):
        """Käynnistä viejä"""
        for service, port in self.ports.items():
            # Käynnistä palvelin
            start_http_server(port, self.host)
            
            logger.info(
                f"Started Prometheus exporter for "
                f"{service} on port {port}"
            )
    
    async def export_metrics(
        self,
        metrics: Dict[str, float],
        service: str
    ):
        """
        Vie metriikat
        
        Args:
            metrics: Metriikat
            service: Palvelu
        """
        try:
            # Päivitä metriikat
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    # Luo tai päivitä mittari
                    gauge = Gauge(
                        f"{service}_{name}",
                        f"{service} {name}",
                        ['realtime']
                    )
                    gauge.labels(
                        realtime='true'
                    ).set(value)
            
            # Päivitä vientilaskuri
            self.export_counter.labels(
                type=service
            ).inc()
            
            # Päivitä aikaleima
            self.last_export.labels(
                type=service
            ).set_to_current_time()
        
        except Exception as e:
            logger.error(
                f"Failed to export metrics for {service}: {str(e)}"
            )
            
            self.export_errors.labels(
                type=service
            ).inc()

class MonitoringSystem:
    """Monitorointijärjestelmä"""
    
    def __init__(
        self,
        metrics_port: int = 8000,
        alert_webhook: Optional[str] = None,
        prometheus_config: Optional[Dict] = None
    ):
        """
        Alusta monitorointi
        
        Args:
            metrics_port: Prometheus-metriikoiden portti
            alert_webhook: Hälytys-webhook URL
            prometheus_config: Prometheus-konfiguraatio
        """
        self.metrics_port = metrics_port
        self.alert_webhook = alert_webhook
        
        # Alustetaan järjestelmät
        self.performance = PerformanceMetrics()
        self.resources = ResourceMetrics()
        self.alerts = AlertSystem()
        self.realtime = RealTimeMetrics()
        
        # Prometheus-viejä
        self.prometheus = PrometheusExporter(
            ports=prometheus_config.get('ports')
            if prometheus_config else None
        )
        
        # Metriikat
        self.api_requests = Counter(
            'api_requests_total',
            'Total API requests',
            ['endpoint', 'status']
        )
        self.api_latency = Histogram(
            'api_latency_seconds',
            'API request latency',
            ['endpoint']
        )
        self.token_usage = Counter(
            'token_usage_total',
            'Token usage',
            ['model', 'type']
        )
        self.api_cost = Counter(
            'api_cost_dollars',
            'API cost in dollars',
            ['model']
        )
        self.error_counter = Counter(
            'errors_total',
            'Total errors',
            ['type']
        )
        
        # Resurssimetriikat
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage'
        )
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes'
        )
        self.disk_usage = Gauge(
            'disk_usage_bytes',
            'Disk usage in bytes'
        )
        
        # Hälytyshistoria
        self.alert_history: List[Alert] = []
        
        # Käynnistä monitorointi
        asyncio.create_task(self.start())
    
    async def start(self):
        """Käynnistä monitorointi"""
        # Käynnistä Prometheus-viejä
        await self.prometheus.start()
        
        # Käynnistä reaaliaikainen seuranta
        asyncio.create_task(self._monitor_realtime())
        
        # Käynnistä resurssiseuranta
        asyncio.create_task(self._monitor_resources())
    
    async def _monitor_realtime(self):
        """Seuraa reaaliaikaisia metriikoita"""
        while True:
            try:
                # Kerää metriikat
                metrics = {
                    'latency': self.realtime.realtime_latency.labels(
                        operation='total'
                    )._value.get(),
                    'throughput': self.realtime.realtime_throughput.labels(
                        type='total'
                    )._value.get(),
                    'errors': self.realtime.realtime_errors.labels(
                        type='total'
                    )._value.get(),
                    'queue': self.realtime.realtime_queue.labels(
                        priority='total'
                    )._value.get()
                }
                
                # Vie metriikat
                await self.prometheus.export_metrics(
                    metrics,
                    'realtime'
                )
                
                # Tarkista kynnysarvot
                if metrics['latency'] > 1.0:  # 1s
                    await self.alerts.create_alert(
                        AlertLevel.WARNING,
                        f"High latency: {metrics['latency']:.2f}s"
                    )
                
                if metrics['errors'] > 100:  # 100 virhettä
                    await self.alerts.create_alert(
                        AlertLevel.ERROR,
                        f"High error rate: {metrics['errors']} errors"
                    )
            
            except Exception as e:
                logger.error(
                    f"Failed to monitor realtime metrics: {str(e)}"
                )
            
            await asyncio.sleep(1)  # 1s välein
    
    async def track_realtime(
        self,
        operation: str,
        latency: float,
        throughput: int = 1,
        errors: int = 0,
        queue_size: Optional[int] = None
    ):
        """
        Seuraa reaaliaikaisia metriikoita
        
        Args:
            operation: Operaatio
            latency: Latenssi sekunteina
            throughput: Läpimeno
            errors: Virheet
            queue_size: Jonon koko
        """
        # Päivitä metriikat
        self.realtime.realtime_latency.labels(
            operation=operation
        ).observe(latency)
        
        self.realtime.realtime_throughput.labels(
            type=operation
        ).inc(throughput)
        
        if errors:
            self.realtime.realtime_errors.labels(
                type=operation
            ).inc(errors)
        
        if queue_size is not None:
            self.realtime.realtime_queue.labels(
                priority=operation
            ).set(queue_size)
        
        # Loki
        logger.info(
            f"Realtime | {operation} | "
            f"Latency: {latency:.3f}s | "
            f"Throughput: {throughput} | "
            f"Errors: {errors} | "
            f"Queue: {queue_size or 'N/A'}"
        )
    
    async def track_api_request(
        self,
        endpoint: str,
        status: str,
        duration: float,
        tokens: Optional[int] = None,
        model: Optional[str] = None,
        cost: Optional[float] = None
    ):
        """
        Seuraa API-kutsua
        
        Args:
            endpoint: API-endpoint
            status: Vastauksen status
            duration: Kesto sekunteina
            tokens: Käytetyt tokenit
            model: Käytetty malli
            cost: Kustannus dollareina
        """
        # Päivitä metriikat
        self.api_requests.labels(
            endpoint=endpoint,
            status=status
        ).inc()
        
        self.api_latency.labels(
            endpoint=endpoint
        ).observe(duration)
        
        if tokens and model:
            self.token_usage.labels(
                model=model,
                type="total"
            ).inc(tokens)
        
        if cost and model:
            self.api_cost.labels(
                model=model
            ).inc(cost)
        
        # Loki
        logger.info(
            f"API Request | {endpoint} | {status} | "
            f"{duration:.2f}s | {tokens or 'N/A'} tokens | "
            f"${cost or 'N/A'}"
        )
    
    async def track_error(
        self,
        error_type: str,
        message: str,
        data: Optional[Dict] = None
    ):
        """
        Seuraa virhettä
        
        Args:
            error_type: Virhetyyppi
            message: Virheilmoitus
            data: Lisätiedot
        """
        # Päivitä metriikat
        self.error_counter.labels(
            type=error_type
        ).inc()
        
        # Loki
        logger.error(
            f"Error | {error_type} | {message}",
            extra={"data": data}
        )
        
        # Luo hälytys
        alert = Alert(
            level=AlertLevel.ERROR,
            message=f"{error_type}: {message}",
            timestamp=datetime.now(),
            data=data
        )
        
        await self._handle_alert(alert)
    
    async def create_alert(
        self,
        level: AlertLevel,
        message: str,
        data: Optional[Dict] = None
    ):
        """
        Luo hälytys
        
        Args:
            level: Hälytystaso
            message: Viesti
            data: Lisätiedot
        """
        alert = Alert(
            level=level,
            message=message,
            timestamp=datetime.now(),
            data=data
        )
        
        await self._handle_alert(alert)
    
    async def _handle_alert(self, alert: Alert):
        """
        Käsittele hälytys
        
        Args:
            alert: Hälytys
        """
        # Tallenna historia
        self.alert_history.append(alert)
        
        # Loki
        log_method = getattr(
            logger,
            alert.level.value,
            logger.info
        )
        log_method(
            f"Alert | {alert.level.value} | {alert.message}",
            extra={"data": alert.data}
        )
        
        # Lähetä webhook
        if self.alert_webhook:
            await self._send_alert_webhook(alert)
    
    async def _send_alert_webhook(self, alert: Alert):
        """
        Lähetä hälytys webhookiin
        
        Args:
            alert: Hälytys
        """
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    self.alert_webhook,
                    json={
                        "level": alert.level.value,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat(),
                        "data": alert.data
                    }
                )
        except Exception as e:
            logger.error(
                f"Failed to send alert webhook: {str(e)}"
            )
    
    async def _monitor_resources(self):
        """Seuraa resurssien käyttöä"""
        while True:
            try:
                # CPU
                cpu_percent = psutil.cpu_percent()
                self.cpu_usage.set(cpu_percent)
                
                if cpu_percent > 90:
                    await self.create_alert(
                        AlertLevel.WARNING,
                        f"High CPU usage: {cpu_percent}%"
                    )
                
                # Muisti
                memory = psutil.virtual_memory()
                self.memory_usage.set(memory.used)
                
                if memory.percent > 90:
                    await self.create_alert(
                        AlertLevel.WARNING,
                        f"High memory usage: {memory.percent}%"
                    )
                
                # Levy
                disk = psutil.disk_usage('/')
                self.disk_usage.set(disk.used)
                
                if disk.percent > 90:
                    await self.create_alert(
                        AlertLevel.WARNING,
                        f"High disk usage: {disk.percent}%"
                    )
            
            except Exception as e:
                logger.error(f"Resource monitoring error: {str(e)}")
            
            await asyncio.sleep(60)  # 1min välein
    
    async def track_performance(
        self,
        task_type: str,
        duration: float,
        status: str,
        queue_size: Optional[int] = None,
        worker_util: Optional[float] = None,
        batch_size: Optional[int] = None
    ):
        """
        Seuraa suorituskykyä
        
        Args:
            task_type: Tehtävätyyppi
            duration: Kesto sekunteina
            status: Status
            queue_size: Jonon koko
            worker_util: Työntekijöiden käyttöaste
            batch_size: Eräkoko
        """
        # Päivitä metriikat
        self.performance.task_duration.labels(
            type=task_type,
            status=status
        ).observe(duration)
        
        if queue_size is not None:
            self.performance.task_queue_size.labels(
                priority="total"
            ).set(queue_size)
        
        if worker_util is not None:
            self.performance.worker_utilization.labels(
                pool_type="total"
            ).set(worker_util)
        
        if batch_size is not None:
            self.performance.batch_size.labels(
                strategy="total"
            ).observe(batch_size)
        
        # Loki
        logger.info(
            f"Performance | {task_type} | {status} | "
            f"{duration:.2f}s | Queue: {queue_size or 'N/A'} | "
            f"Workers: {worker_util or 'N/A'}% | "
            f"Batch: {batch_size or 'N/A'}"
        )
    
    async def track_resources(
        self,
        memory_used: Optional[int] = None,
        io_ops: Optional[int] = None,
        network_bytes: Optional[int] = None,
        cache_hits: Optional[int] = None
    ):
        """
        Seuraa resursseja
        
        Args:
            memory_used: Käytetty muisti tavuina
            io_ops: I/O-operaatiot
            network_bytes: Verkkoliikenne tavuina
            cache_hits: Välimuistiosumat
        """
        # Päivitä metriikat
        if memory_used is not None:
            self.resources.memory_allocation.labels(
                type="used"
            ).set(memory_used)
        
        if io_ops is not None:
            self.resources.io_operations.labels(
                type="total"
            ).inc(io_ops)
        
        if network_bytes is not None:
            self.resources.network_traffic.labels(
                direction="total"
            ).inc(network_bytes)
        
        if cache_hits is not None:
            self.resources.cache_stats.labels(
                operation="hits"
            ).set(cache_hits)
        
        # Loki
        logger.info(
            f"Resources | Memory: {memory_used or 'N/A'} B | "
            f"IO: {io_ops or 'N/A'} ops | "
            f"Network: {network_bytes or 'N/A'} B | "
            f"Cache: {cache_hits or 'N/A'} hits"
        )
        
        # Tarkista kynnysarvot
        if memory_used:
            total_memory = psutil.virtual_memory().total
            memory_percent = (memory_used / total_memory) * 100
            
            await self.alerts.check_thresholds({
                'memory_usage': memory_percent
            })

    def get_stats(self) -> Dict:
        """
        Hae tilastot
        
        Returns:
            Dict: Tilastot
        """
        # Kerää metriikat
        stats = {
            "resources": {
                "cpu": psutil.cpu_percent(),
                "memory": psutil.virtual_memory().percent,
                "disk": psutil.disk_usage('/').percent
            },
            "alerts": {
                level.value: len([
                    a for a in self.alert_history
                    if a.level == level
                ])
                for level in AlertLevel
            }
        }
        
        return stats
    
    def generate_report(self) -> str:
        """
        Generoi raportti
        
        Returns:
            str: Markdown-muotoinen raportti
        """
        stats = self.get_stats()
        
        report = """# Monitorointiraportti

## Resurssien käyttö

"""
        
        for resource, usage in stats["resources"].items():
            report += f"- {resource.upper()}: {usage}%\n"
        
        report += "\n## Hälytykset\n\n"
        
        for level, count in stats["alerts"].items():
            report += f"- {level}: {count}\n"
        
        # Viimeisimmät hälytykset
        report += "\n## Viimeisimmät hälytykset\n\n"
        
        for alert in sorted(
            self.alert_history,
            key=lambda x: x.timestamp,
            reverse=True
        )[:5]:
            report += (
                f"- {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"{alert.level.value} | {alert.message}\n"
            )
        
        return report

async def main():
    """Testaa monitorointia"""
    monitoring = MonitoringSystem()
    
    # Simuloi API-kutsuja
    for i in range(10):
        # Simuloi onnistunut kutsu
        await monitoring.track_api_request(
            endpoint="/api/test",
            status="success",
            duration=np.random.uniform(0.1, 2.0),
            tokens=np.random.randint(100, 1000),
            model="gpt-3.5-turbo",
            cost=np.random.uniform(0.01, 0.1)
        )
        
        # Simuloi virhe
        if i % 3 == 0:
            await monitoring.track_error(
                "api_error",
                "Test error message",
                {"task_id": f"task_{i}"}
            )
        
        await asyncio.sleep(1)
    
    # Tulosta raportti
    print("\nMonitoring Report:")
    print(monitoring.generate_report())

if __name__ == "__main__":
    asyncio.run(main())
