import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# Konfiguroi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Tehtävätyypit"""
    ANALYSIS = "analysis"
    CODE = "code"
    CHAT = "chat"
    SEARCH = "search"

@dataclass
class TaskMetrics:
    """Tehtävän metriikat"""
    duration: float
    token_count: int
    memory_usage: float
    cpu_usage: float
    success: bool

class LoadAnalyzer:
    """Kuormitusanalysaattori"""
    
    def __init__(
        self,
        window_size: int = 3600,  # 1 tunti
        update_interval: int = 60  # 1 minuutti
    ):
        """
        Alusta analysaattori
        
        Args:
            window_size: Analyysi-ikkuna sekunneissa
            update_interval: Päivitysväli sekunneissa
        """
        self.window_size = window_size
        self.update_interval = update_interval
        
        # Tehtävähistoria
        self.task_history: List[Tuple[datetime, str, TaskMetrics]] = []
        
        # Kuormitusmetriikat
        self.load_metrics = {
            "cpu": [],
            "memory": [],
            "tasks": [],
            "tokens": []
        }
        
        # Tehtäväkohtaiset metriikat
        self.task_metrics: Dict[TaskType, List[TaskMetrics]] = {
            task_type: [] for task_type in TaskType
        }
        
        # Prometheus-metriikat
        self.task_counter = Counter(
            'task_count_total',
            'Total number of tasks',
            ['type']
        )
        self.task_duration = Histogram(
            'task_duration_seconds',
            'Task duration in seconds',
            ['type']
        )
        self.resource_usage = Gauge(
            'resource_usage_ratio',
            'Resource usage ratio',
            ['resource']
        )
        
        # Anomaliatunnistin
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
    
    async def track_task(
        self,
        task_type: TaskType,
        metrics: TaskMetrics
    ):
        """
        Seuraa tehtävää
        
        Args:
            task_type: Tehtävätyyppi
            metrics: Tehtävän metriikat
        """
        # Tallenna historiaan
        now = datetime.now()
        self.task_history.append((now, task_type.value, metrics))
        
        # Päivitä metriikat
        self.task_counter.labels(type=task_type.value).inc()
        self.task_duration.labels(type=task_type.value).observe(
            metrics.duration
        )
        
        # Päivitä resurssienkäyttö
        self.resource_usage.labels(resource="cpu").set(
            metrics.cpu_usage
        )
        self.resource_usage.labels(resource="memory").set(
            metrics.memory_usage
        )
        
        # Lisää tehtäväkohtaisiin metriikkoihin
        self.task_metrics[task_type].append(metrics)
        
        # Siivoa vanha historia
        self._cleanup_history()
    
    def _cleanup_history(self):
        """Siivoa vanha historia"""
        cutoff = datetime.now() - timedelta(
            seconds=self.window_size
        )
        
        # Poista vanhat merkinnät
        self.task_history = [
            (dt, type_, metrics)
            for dt, type_, metrics in self.task_history
            if dt >= cutoff
        ]
    
    def analyze_load(self) -> Dict:
        """
        Analysoi kuormitus
        
        Returns:
            Dict: Kuormitusanalyysi
        """
        if not self.task_history:
            return {}
        
        analysis = {
            "current_load": self._analyze_current_load(),
            "patterns": self._analyze_patterns(),
            "resource_usage": self._analyze_resources(),
            "anomalies": self._detect_anomalies(),
            "recommendations": self._generate_recommendations()
        }
        
        return analysis
    
    def _analyze_current_load(self) -> Dict:
        """
        Analysoi nykyinen kuormitus
        
        Returns:
            Dict: Kuormitusanalyysi
        """
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Laske viimeisen minuutin metriikat
        recent_tasks = [
            (dt, type_, metrics)
            for dt, type_, metrics in self.task_history
            if dt >= minute_ago
        ]
        
        if not recent_tasks:
            return {}
        
        # Laske keskiarvot
        avg_metrics = {
            "task_count": len(recent_tasks),
            "cpu_usage": np.mean([
                m.cpu_usage for _, _, m in recent_tasks
            ]),
            "memory_usage": np.mean([
                m.memory_usage for _, _, m in recent_tasks
            ]),
            "token_count": np.mean([
                m.token_count for _, _, m in recent_tasks
            ])
        }
        
        # Laske tyyppikohtaiset määrät
        type_counts = defaultdict(int)
        for _, type_, _ in recent_tasks:
            type_counts[type_] += 1
        
        avg_metrics["type_distribution"] = {
            type_: count / len(recent_tasks)
            for type_, count in type_counts.items()
        }
        
        return avg_metrics
    
    def _analyze_patterns(self) -> Dict:
        """
        Analysoi kuormituskuviot
        
        Returns:
            Dict: Kuvioanalyysi
        """
        if len(self.task_history) < 100:
            return {}
        
        # Muunna aikasarjaksi
        df = pd.DataFrame([
            {
                'timestamp': dt,
                'type': type_,
                'duration': metrics.duration,
                'tokens': metrics.token_count,
                'cpu': metrics.cpu_usage,
                'memory': metrics.memory_usage
            }
            for dt, type_, metrics in self.task_history
        ])
        
        # Ryhmittele 5min välein
        df.set_index('timestamp', inplace=True)
        hourly = df.resample('5T').agg({
            'duration': 'mean',
            'tokens': 'sum',
            'cpu': 'mean',
            'memory': 'mean'
        })
        
        # Tunnista kuviot
        patterns = {
            'peak_hours': self._find_peak_hours(hourly),
            'resource_correlation': self._analyze_correlation(hourly),
            'task_complexity': self._analyze_complexity(df)
        }
        
        return patterns
    
    def _find_peak_hours(self, df: pd.DataFrame) -> List[int]:
        """
        Etsi ruuhkatunnit
        
        Args:
            df: Aikasarjadata
        
        Returns:
            List[int]: Ruuhkatunnit
        """
        # Laske keskimääräinen kuorma tunneittain
        hourly_load = df.groupby(df.index.hour)['cpu'].mean()
        
        # Tunnista ruuhkatunnit (>75% kvantiili)
        threshold = hourly_load.quantile(0.75)
        peak_hours = hourly_load[hourly_load >= threshold].index.tolist()
        
        return peak_hours
    
    def _analyze_correlation(self, df: pd.DataFrame) -> Dict:
        """
        Analysoi resurssien korrelaatio
        
        Args:
            df: Aikasarjadata
        
        Returns:
            Dict: Korrelaatioanalyysi
        """
        # Laske korrelaatiomatriisi
        corr = df.corr()
        
        # Tunnista vahvat korrelaatiot
        strong_corr = {}
        for col1 in corr.columns:
            for col2 in corr.columns:
                if col1 < col2:
                    correlation = corr.loc[col1, col2]
                    if abs(correlation) >= 0.7:
                        strong_corr[f"{col1}_vs_{col2}"] = correlation
        
        return strong_corr
    
    def _analyze_complexity(self, df: pd.DataFrame) -> Dict:
        """
        Analysoi tehtävien kompleksisuus
        
        Args:
            df: Tehtävädata
        
        Returns:
            Dict: Kompleksisuusanalyysi
        """
        # Ryhmittele tehtävätyypin mukaan
        complexity = {}
        for task_type in df['type'].unique():
            task_df = df[df['type'] == task_type]
            
            complexity[task_type] = {
                'avg_duration': task_df['duration'].mean(),
                'avg_tokens': task_df['tokens'].mean(),
                'avg_cpu': task_df['cpu'].mean(),
                'avg_memory': task_df['memory'].mean()
            }
        
        return complexity
    
    def _analyze_resources(self) -> Dict:
        """
        Analysoi resurssien käyttö
        
        Returns:
            Dict: Resurssianalyysi
        """
        if not self.task_history:
            return {}
        
        # Kerää resurssidata
        resources = {
            'cpu': [m.cpu_usage for _, _, m in self.task_history],
            'memory': [m.memory_usage for _, _, m in self.task_history],
            'tokens': [m.token_count for _, _, m in self.task_history]
        }
        
        # Laske tilastot
        stats = {}
        for resource, values in resources.items():
            stats[resource] = {
                'current': values[-1],
                'avg': np.mean(values),
                'max': np.max(values),
                'min': np.min(values),
                'std': np.std(values)
            }
        
        return stats
    
    def _detect_anomalies(self) -> Dict:
        """
        Tunnista anomaliat
        
        Returns:
            Dict: Anomaliat
        """
        if len(self.task_history) < 100:
            return {}
        
        # Kerää data
        data = np.array([
            [m.duration, m.token_count, m.cpu_usage, m.memory_usage]
            for _, _, m in self.task_history
        ])
        
        # Normalisoi
        scaler = StandardScaler()
        normalized = scaler.fit_transform(data)
        
        # Tunnista anomaliat
        predictions = self.anomaly_detector.fit_predict(normalized)
        anomaly_indices = np.where(predictions == -1)[0]
        
        # Analysoi anomaliat
        anomalies = {
            'count': len(anomaly_indices),
            'ratio': len(anomaly_indices) / len(data),
            'details': []
        }
        
        for idx in anomaly_indices[-5:]:  # Viimeiset 5 anomaliaa
            _, type_, metrics = self.task_history[idx]
            anomalies['details'].append({
                'type': type_,
                'metrics': {
                    'duration': metrics.duration,
                    'tokens': metrics.token_count,
                    'cpu': metrics.cpu_usage,
                    'memory': metrics.memory_usage
                }
            })
        
        return anomalies
    
    def _generate_recommendations(self) -> List[str]:
        """
        Generoi suositukset
        
        Returns:
            List[str]: Suositukset
        """
        recommendations = []
        
        # Analysoi nykyinen tila
        current = self._analyze_current_load()
        resources = self._analyze_resources()
        
        # Tarkista resurssien käyttö
        if current.get('cpu_usage', 0) > 0.8:
            recommendations.append(
                "CPU-käyttö korkea: Lisää rinnakkaisia työntekijöitä"
            )
        
        if current.get('memory_usage', 0) > 0.8:
            recommendations.append(
                "Muistinkäyttö korkea: Optimoi muistinhallintaa"
            )
        
        # Tarkista tehtävätyypit
        type_dist = current.get('type_distribution', {})
        for type_, ratio in type_dist.items():
            if ratio > 0.5:
                recommendations.append(
                    f"Tehtävätyyppi {type_} dominoi: "
                    "Harkitse eriytettyä käsittelyä"
                )
        
        # Tarkista trendit
        for resource, stats in resources.items():
            if stats['current'] > stats['avg'] + 2 * stats['std']:
                recommendations.append(
                    f"{resource}-käyttö epätavallisen korkea: "
                    "Tutki mahdollisia ongelmia"
                )
        
        return recommendations

async def main():
    """Testaa kuormitusanalyysiä"""
    analyzer = LoadAnalyzer()
    
    # Simuloi tehtäviä
    for i in range(100):
        metrics = TaskMetrics(
            duration=np.random.exponential(1.0),
            token_count=np.random.randint(100, 1000),
            memory_usage=np.random.uniform(0.1, 0.9),
            cpu_usage=np.random.uniform(0.1, 0.9),
            success=True
        )
        
        task_type = np.random.choice(list(TaskType))
        await analyzer.track_task(task_type, metrics)
        
        if i % 10 == 0:
            # Analysoi kuormitus
            analysis = analyzer.analyze_load()
            
            print("\nKuormitusanalyysi:")
            print(f"Nykyinen kuorma: {analysis.get('current_load', {})}")
            print(f"Anomaliat: {analysis.get('anomalies', {})}")
            print("\nSuositukset:")
            for rec in analysis.get('recommendations', []):
                print(f"- {rec}")
        
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(main())
