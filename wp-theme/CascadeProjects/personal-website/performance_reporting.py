import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader

# Konfiguroi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportType(Enum):
    """Raporttityypit"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

class MetricCategory(Enum):
    """Metriikkakategoriat"""
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    COST = "cost"
    QUALITY = "quality"

class Report:
    """Raportti"""
    
    def __init__(
        self,
        report_type: ReportType,
        start_date: datetime,
        end_date: datetime,
        metrics: Dict,
        improvements: Dict,
        recommendations: List[str]
    ):
        self.report_type = report_type
        self.start_date = start_date
        self.end_date = end_date
        self.metrics = metrics
        self.improvements = improvements
        self.recommendations = recommendations
        self.generated_at = datetime.now()

class PerformanceReporter:
    """Suorituskykyraportoija"""
    
    def __init__(
        self,
        metrics_collector,
        template_path: str = "templates"
    ):
        """
        Alusta raportoija
        
        Args:
            metrics_collector: Metriikoiden kerääjä
            template_path: HTML-templatejen polku
        """
        self.metrics = metrics_collector
        self.template_env = Environment(
            loader=FileSystemLoader(template_path)
        )
        
        # Raporttien tallennus
        self.reports: Dict[str, Report] = {}
        
        # Parannusten seuranta
        self.baseline_metrics = {}
        self.improvement_thresholds = {
            MetricCategory.PERFORMANCE: 0.1,  # 10% parannus
            MetricCategory.RELIABILITY: 0.05,  # 5% parannus
            MetricCategory.COST: 0.15         # 15% säästö
        }
    
    async def generate_report(
        self,
        report_type: ReportType,
        categories: Optional[List[MetricCategory]] = None
    ) -> Report:
        """
        Generoi raportti
        
        Args:
            report_type: Raporttityyppi
            categories: Metriikkakategoriat
        
        Returns:
            Report: Generoitu raportti
        """
        # Määritä aikaväli
        end_date = datetime.now()
        if report_type == ReportType.DAILY:
            start_date = end_date - timedelta(days=1)
        elif report_type == ReportType.WEEKLY:
            start_date = end_date - timedelta(weeks=1)
        elif report_type == ReportType.MONTHLY:
            start_date = end_date - timedelta(days=30)
        else:
            start_date = end_date - timedelta(days=90)
        
        # Kerää metriikat
        metrics = await self._collect_metrics(
            start_date,
            end_date,
            categories
        )
        
        # Analysoi parannukset
        improvements = self._analyze_improvements(metrics)
        
        # Generoi suositukset
        recommendations = self._generate_recommendations(
            metrics,
            improvements
        )
        
        # Luo raportti
        report = Report(
            report_type,
            start_date,
            end_date,
            metrics,
            improvements,
            recommendations
        )
        
        # Tallenna raportti
        report_id = f"{report_type.value}_{end_date.strftime('%Y%m%d')}"
        self.reports[report_id] = report
        
        return report
    
    async def _collect_metrics(
        self,
        start_date: datetime,
        end_date: datetime,
        categories: Optional[List[MetricCategory]]
    ) -> Dict:
        """
        Kerää metriikat
        
        Args:
            start_date: Alkupäivä
            end_date: Loppupäivä
            categories: Kategoriat
        
        Returns:
            Dict: Metriikat
        """
        if not categories:
            categories = list(MetricCategory)
        
        metrics = {}
        for category in categories:
            if category == MetricCategory.PERFORMANCE:
                metrics[category.value] = {
                    "latency": await self._get_latency_metrics(
                        start_date,
                        end_date
                    ),
                    "throughput": await self._get_throughput_metrics(
                        start_date,
                        end_date
                    )
                }
            
            elif category == MetricCategory.RELIABILITY:
                metrics[category.value] = {
                    "success_rate": await self._get_success_rate(
                        start_date,
                        end_date
                    ),
                    "error_rate": await self._get_error_rate(
                        start_date,
                        end_date
                    ),
                    "uptime": await self._get_uptime(
                        start_date,
                        end_date
                    )
                }
            
            elif category == MetricCategory.COST:
                metrics[category.value] = {
                    "token_usage": await self._get_token_usage(
                        start_date,
                        end_date
                    ),
                    "api_costs": await self._get_api_costs(
                        start_date,
                        end_date
                    ),
                    "cost_per_request": await self._get_cost_per_request(
                        start_date,
                        end_date
                    )
                }
            
            elif category == MetricCategory.QUALITY:
                metrics[category.value] = {
                    "accuracy": await self._get_accuracy_metrics(
                        start_date,
                        end_date
                    ),
                    "satisfaction": await self._get_satisfaction_metrics(
                        start_date,
                        end_date
                    )
                }
        
        return metrics
    
    def _analyze_improvements(self, metrics: Dict) -> Dict:
        """
        Analysoi parannukset
        
        Args:
            metrics: Metriikat
        
        Returns:
            Dict: Parannukset
        """
        improvements = {}
        
        for category, category_metrics in metrics.items():
            if not self.baseline_metrics.get(category):
                # Aseta baseline
                self.baseline_metrics[category] = category_metrics
                continue
            
            baseline = self.baseline_metrics[category]
            threshold = self.improvement_thresholds.get(
                MetricCategory(category),
                0.1
            )
            
            category_improvements = {}
            for metric, value in category_metrics.items():
                if metric not in baseline:
                    continue
                
                baseline_value = baseline[metric]
                if isinstance(value, dict):
                    # Rekursiivinen analyysi
                    sub_improvements = {}
                    for sub_metric, sub_value in value.items():
                        if sub_metric in baseline_value:
                            change = (
                                sub_value - baseline_value[sub_metric]
                            ) / baseline_value[sub_metric]
                            if abs(change) >= threshold:
                                sub_improvements[sub_metric] = change
                    
                    if sub_improvements:
                        category_improvements[metric] = sub_improvements
                
                else:
                    change = (
                        value - baseline_value
                    ) / baseline_value
                    if abs(change) >= threshold:
                        category_improvements[metric] = change
            
            if category_improvements:
                improvements[category] = category_improvements
        
        return improvements
    
    def _generate_recommendations(
        self,
        metrics: Dict,
        improvements: Dict
    ) -> List[str]:
        """
        Generoi suositukset
        
        Args:
            metrics: Metriikat
            improvements: Parannukset
        
        Returns:
            List[str]: Suositukset
        """
        recommendations = []
        
        # Analysoi suorituskyky
        if "performance" in metrics:
            perf = metrics["performance"]
            if perf["latency"].get("avg", 0) > 1.0:
                recommendations.append(
                    "Optimoi latenssia välimuistin avulla"
                )
            if perf["throughput"].get("avg", 0) < 100:
                recommendations.append(
                    "Lisää rinnakkaista prosessointia"
                )
        
        # Analysoi luotettavuus
        if "reliability" in metrics:
            rel = metrics["reliability"]
            if rel["error_rate"].get("avg", 0) > 0.01:
                recommendations.append(
                    "Paranna virheenkäsittelyä"
                )
            if rel["uptime"].get("avg", 0) < 0.999:
                recommendations.append(
                    "Implementoi automaattinen palautuminen"
                )
        
        # Analysoi kustannukset
        if "cost" in metrics:
            cost = metrics["cost"]
            if cost["cost_per_request"].get("avg", 0) > 0.01:
                recommendations.append(
                    "Optimoi token-käyttöä"
                )
        
        return recommendations
    
    def export_report(
        self,
        report: Report,
        format: str = "html"
    ) -> str:
        """
        Vie raportti
        
        Args:
            report: Raportti
            format: Formaatti (html/json/md)
        
        Returns:
            str: Viety raportti
        """
        if format == "html":
            template = self.template_env.get_template(
                "report_template.html"
            )
            return template.render(report=report)
        
        elif format == "json":
            return json.dumps({
                "type": report.report_type.value,
                "period": {
                    "start": report.start_date.isoformat(),
                    "end": report.end_date.isoformat()
                },
                "metrics": report.metrics,
                "improvements": report.improvements,
                "recommendations": report.recommendations,
                "generated_at": report.generated_at.isoformat()
            }, indent=2)
        
        elif format == "md":
            return self._generate_markdown(report)
        
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _generate_markdown(self, report: Report) -> str:
        """
        Generoi Markdown-raportti
        
        Args:
            report: Raportti
        
        Returns:
            str: Markdown-raportti
        """
        md = f"""# Suorituskykyraportti ({report.report_type.value})

## Yhteenveto
- Aikaväli: {report.start_date.strftime('%Y-%m-%d')} - {report.end_date.strftime('%Y-%m-%d')}
- Generoitu: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}

## Metriikat

"""
        # Lisää metriikat
        for category, metrics in report.metrics.items():
            md += f"### {category.title()}\n"
            for metric, value in metrics.items():
                if isinstance(value, dict):
                    md += f"#### {metric.title()}\n"
                    for sub_metric, sub_value in value.items():
                        md += f"- {sub_metric}: {sub_value}\n"
                else:
                    md += f"- {metric}: {value}\n"
            md += "\n"
        
        # Lisää parannukset
        if report.improvements:
            md += "## Parannukset\n\n"
            for category, improvements in report.improvements.items():
                md += f"### {category.title()}\n"
                for metric, change in improvements.items():
                    if isinstance(change, dict):
                        md += f"#### {metric.title()}\n"
                        for sub_metric, sub_change in change.items():
                            md += f"- {sub_metric}: {sub_change:+.1%}\n"
                    else:
                        md += f"- {metric}: {change:+.1%}\n"
                md += "\n"
        
        # Lisää suositukset
        if report.recommendations:
            md += "## Suositukset\n\n"
            for rec in report.recommendations:
                md += f"- {rec}\n"
        
        return md
    
    def visualize_metrics(
        self,
        report: Report,
        category: MetricCategory
    ) -> go.Figure:
        """
        Visualisoi metriikat
        
        Args:
            report: Raportti
            category: Kategoria
        
        Returns:
            go.Figure: Plotly-kuvaaja
        """
        metrics = report.metrics.get(category.value, {})
        
        if not metrics:
            return None
        
        # Muunna data DataFrame-muotoon
        data = []
        for metric, value in metrics.items():
            if isinstance(value, dict):
                for sub_metric, sub_value in value.items():
                    data.append({
                        'Metric': f"{metric}_{sub_metric}",
                        'Value': sub_value
                    })
            else:
                data.append({
                    'Metric': metric,
                    'Value': value
                })
        
        df = pd.DataFrame(data)
        
        # Luo kuvaaja
        fig = px.bar(
            df,
            x='Metric',
            y='Value',
            title=f'{category.value} Metrics'
        )
        
        return fig

async def main():
    """Testaa raportointia"""
    from monitoring import MetricsCollector
    
    # Alusta komponentit
    metrics = MetricsCollector()
    reporter = PerformanceReporter(metrics)
    
    # Generoi testidataa
    for i in range(10):
        metrics.start_task(
            f"task_{i}",
            "analysis",
            "gpt-4",
            1000
        )
        await asyncio.sleep(0.1)
        metrics.end_task(
            f"task_{i}",
            "success",
            500,
            0.1,
            0.95
        )
    
    # Generoi raportti
    report = await reporter.generate_report(
        ReportType.DAILY
    )
    
    # Vie eri formaateissa
    print("\nHTML Report:")
    print(reporter.export_report(report, "html")[:500])
    
    print("\nJSON Report:")
    print(reporter.export_report(report, "json"))
    
    print("\nMarkdown Report:")
    print(reporter.export_report(report, "md"))
    
    # Visualisoi
    fig = reporter.visualize_metrics(
        report,
        MetricCategory.PERFORMANCE
    )
    fig.show()

if __name__ == "__main__":
    asyncio.run(main())
