import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
import json

import numpy as np
from prometheus_client import Counter, Gauge, Histogram
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Konfiguroi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Hälytysten vakavuustasot"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertType(Enum):
    """Hälytystyypit"""
    TOKEN_LIMIT = "token_limit"
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    COST = "cost"

@dataclass
class AlertRule:
    """Hälytyssääntö"""
    type: AlertType
    severity: AlertSeverity
    threshold: float
    window: int  # sekunteina
    cooldown: int  # sekunteina
    description: str
    callback: Optional[Callable] = None

@dataclass
class Alert:
    """Hälytys"""
    rule: AlertRule
    value: float
    timestamp: datetime
    context: Optional[Dict] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class AlertSystem:
    """Hälytysjärjestelmä"""
    
    def __init__(
        self,
        slack_webhook: Optional[str] = None,
        email_config: Optional[Dict] = None
    ):
        """
        Alusta hälytysjärjestelmä
        
        Args:
            slack_webhook: Slack webhook URL
            email_config: Sähköpostikonfiguraatio
        """
        self.slack_webhook = slack_webhook
        self.email_config = email_config
        
        # Hälytyssäännöt
        self.rules: List[AlertRule] = self._create_default_rules()
        
        # Aktiiviset hälytykset
        self.active_alerts: Dict[str, Alert] = {}
        
        # Hälytyshistoria
        self.alert_history: List[Alert] = []
        
        # Viimeisimmät hälytysajat
        self.last_alerts: Dict[AlertType, datetime] = {}
        
        # Metriikat
        self.alert_counter = Counter(
            'alerts_total',
            'Total alerts',
            ['type', 'severity']
        )
        self.active_alerts_gauge = Gauge(
            'active_alerts',
            'Active alerts',
            ['type']
        )
        self.alert_duration = Histogram(
            'alert_duration_seconds',
            'Alert duration in seconds',
            ['type']
        )
    
    def _create_default_rules(self) -> List[AlertRule]:
        """
        Luo oletussäännöt
        
        Returns:
            List[AlertRule]: Hälytyssäännöt
        """
        return [
            AlertRule(
                type=AlertType.TOKEN_LIMIT,
                severity=AlertSeverity.ERROR,
                threshold=0.9,  # 90% käyttö
                window=300,     # 5min
                cooldown=600,   # 10min
                description="Token usage exceeded limit"
            ),
            AlertRule(
                type=AlertType.RATE_LIMIT,
                severity=AlertSeverity.WARNING,
                threshold=5,    # 5 rate limitiä
                window=60,      # 1min
                cooldown=300,   # 5min
                description="Rate limit hits detected"
            ),
            AlertRule(
                type=AlertType.PERFORMANCE,
                severity=AlertSeverity.WARNING,
                threshold=2.0,  # 2s vasteaika
                window=300,     # 5min
                cooldown=600,   # 10min
                description="High response time detected"
            ),
            AlertRule(
                type=AlertType.RESOURCE,
                severity=AlertSeverity.CRITICAL,
                threshold=90,   # 90% käyttö
                window=60,      # 1min
                cooldown=300,   # 5min
                description="High resource usage detected"
            ),
            AlertRule(
                type=AlertType.COST,
                severity=AlertSeverity.ERROR,
                threshold=100,  # $100/päivä
                window=86400,   # 24h
                cooldown=3600,  # 1h
                description="Daily cost limit exceeded"
            )
        ]
    
    async def check_alert(
        self,
        type: AlertType,
        value: float,
        context: Optional[Dict] = None
    ):
        """
        Tarkista hälytys
        
        Args:
            type: Hälytystyyppi
            value: Arvo
            context: Konteksti
        """
        # Etsi sääntö
        rule = next(
            (r for r in self.rules if r.type == type),
            None
        )
        
        if not rule:
            return
        
        # Tarkista cooldown
        now = datetime.now()
        last_alert = self.last_alerts.get(type)
        
        if last_alert and (
            now - last_alert
        ).total_seconds() < rule.cooldown:
            return
        
        # Tarkista kynnysarvo
        if value >= rule.threshold:
            # Luo hälytys
            alert = Alert(
                rule=rule,
                value=value,
                timestamp=now,
                context=context
            )
            
            # Tallenna hälytys
            alert_id = f"{type.value}_{now.timestamp()}"
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            self.last_alerts[type] = now
            
            # Päivitä metriikat
            self.alert_counter.labels(
                type=type.value,
                severity=rule.severity.value
            ).inc()
            
            self.active_alerts_gauge.labels(
                type=type.value
            ).inc()
            
            # Lähetä ilmoitukset
            await self._send_notifications(alert)
            
            # Suorita callback
            if rule.callback:
                await rule.callback(alert)
    
    async def resolve_alert(
        self,
        alert_id: str,
        resolution_context: Optional[Dict] = None
    ):
        """
        Merkitse hälytys ratkaistuksi
        
        Args:
            alert_id: Hälytyksen ID
            resolution_context: Ratkaisukonteksti
        """
        if alert_id not in self.active_alerts:
            return
        
        alert = self.active_alerts[alert_id]
        alert.resolved = True
        alert.resolved_at = datetime.now()
        
        if resolution_context:
            alert.context = {
                **(alert.context or {}),
                "resolution": resolution_context
            }
        
        # Päivitä metriikat
        self.active_alerts_gauge.labels(
            type=alert.rule.type.value
        ).dec()
        
        duration = (
            alert.resolved_at - alert.timestamp
        ).total_seconds()
        
        self.alert_duration.labels(
            type=alert.rule.type.value
        ).observe(duration)
        
        # Poista aktiivisista
        del self.active_alerts[alert_id]
        
        # Lähetä ratkaisu-ilmoitus
        await self._send_resolution_notification(alert)
    
    async def _send_notifications(self, alert: Alert):
        """
        Lähetä hälytykset
        
        Args:
            alert: Hälytys
        """
        # Luo viesti
        message = self._format_alert_message(alert)
        
        # Loki
        log_method = getattr(
            logger,
            alert.rule.severity.value,
            logger.info
        )
        log_method(message)
        
        # Slack
        if self.slack_webhook:
            await self._send_slack_alert(alert, message)
        
        # Sähköposti
        if self.email_config:
            await self._send_email_alert(alert, message)
    
    def _format_alert_message(self, alert: Alert) -> str:
        """
        Muotoile hälytysviesti
        
        Args:
            alert: Hälytys
        
        Returns:
            str: Muotoiltu viesti
        """
        message = f"""🚨 {alert.rule.severity.value.upper()} ALERT 🚨

Type: {alert.rule.type.value}
Description: {alert.rule.description}
Value: {alert.value}
Threshold: {alert.rule.threshold}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        if alert.context:
            message += "\nContext:\n"
            for key, value in alert.context.items():
                message += f"- {key}: {value}\n"
        
        return message
    
    async def _send_slack_alert(
        self,
        alert: Alert,
        message: str
    ):
        """
        Lähetä Slack-hälytys
        
        Args:
            alert: Hälytys
            message: Viesti
        """
        try:
            # Värikoodit
            colors = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ffd700",
                AlertSeverity.ERROR: "#ff4500",
                AlertSeverity.CRITICAL: "#ff0000"
            }
            
            # Luo payload
            payload = {
                "attachments": [{
                    "color": colors[alert.rule.severity],
                    "title": f"Alert: {alert.rule.type.value}",
                    "text": message,
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert.rule.severity.value,
                            "short": True
                        },
                        {
                            "title": "Value",
                            "value": str(alert.value),
                            "short": True
                        }
                    ]
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                await session.post(
                    self.slack_webhook,
                    json=payload
                )
        
        except Exception as e:
            logger.error(f"Slack notification error: {str(e)}")
    
    async def _send_email_alert(
        self,
        alert: Alert,
        message: str
    ):
        """
        Lähetä sähköpostihälytys
        
        Args:
            alert: Hälytys
            message: Viesti
        """
        try:
            # Luo viesti
            msg = MIMEMultipart()
            msg["Subject"] = (
                f"[{alert.rule.severity.value.upper()}] "
                f"{alert.rule.type.value} Alert"
            )
            msg["From"] = self.email_config["from"]
            msg["To"] = self.email_config["to"]
            
            # HTML-sisältö
            html = f"""
            <html>
                <body>
                    <h2>{alert.rule.type.value} Alert</h2>
                    <p><strong>Severity:</strong> {alert.rule.severity.value}</p>
                    <p><strong>Description:</strong> {alert.rule.description}</p>
                    <p><strong>Value:</strong> {alert.value}</p>
                    <p><strong>Threshold:</strong> {alert.rule.threshold}</p>
                    <p><strong>Time:</strong> {alert.timestamp}</p>
                </body>
            </html>
            """
            
            msg.attach(MIMEText(html, "html"))
            
            # Lähetä
            with smtplib.SMTP(
                self.email_config["smtp_host"],
                self.email_config["smtp_port"]
            ) as server:
                if self.email_config.get("use_tls"):
                    server.starttls()
                
                if "username" in self.email_config:
                    server.login(
                        self.email_config["username"],
                        self.email_config["password"]
                    )
                
                server.send_message(msg)
        
        except Exception as e:
            logger.error(f"Email notification error: {str(e)}")
    
    async def _send_resolution_notification(self, alert: Alert):
        """
        Lähetä ratkaisuilmoitus
        
        Args:
            alert: Hälytys
        """
        message = f"""✅ ALERT RESOLVED ✅

Type: {alert.rule.type.value}
Description: {alert.rule.description}
Duration: {(alert.resolved_at - alert.timestamp).total_seconds():.1f}s
"""
        
        if alert.context and "resolution" in alert.context:
            message += "\nResolution:\n"
            for key, value in alert.context["resolution"].items():
                message += f"- {key}: {value}\n"
        
        # Loki
        logger.info(message)
        
        # Slack
        if self.slack_webhook:
            try:
                payload = {
                    "attachments": [{
                        "color": "#36a64f",
                        "title": "Alert Resolved",
                        "text": message
                    }]
                }
                
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        self.slack_webhook,
                        json=payload
                    )
            
            except Exception as e:
                logger.error(
                    f"Slack resolution notification error: {str(e)}"
                )
    
    def get_stats(self) -> Dict:
        """
        Hae tilastot
        
        Returns:
            Dict: Tilastot
        """
        stats = {
            "active_alerts": len(self.active_alerts),
            "total_alerts": len(self.alert_history),
            "by_type": {},
            "by_severity": {}
        }
        
        # Laske tyypeittäin
        for alert in self.alert_history:
            type_name = alert.rule.type.value
            severity_name = alert.rule.severity.value
            
            if type_name not in stats["by_type"]:
                stats["by_type"][type_name] = 0
            stats["by_type"][type_name] += 1
            
            if severity_name not in stats["by_severity"]:
                stats["by_severity"][severity_name] = 0
            stats["by_severity"][severity_name] += 1
        
        return stats
    
    def generate_report(self) -> str:
        """
        Generoi raportti
        
        Returns:
            str: Markdown-muotoinen raportti
        """
        stats = self.get_stats()
        
        report = """# Hälytysraportti

## Yhteenveto

"""
        
        report += f"- Aktiivisia hälytyksiä: {stats['active_alerts']}\n"
        report += f"- Hälytyksiä yhteensä: {stats['total_alerts']}\n\n"
        
        report += "## Hälytykset tyypeittäin\n\n"
        
        for type_name, count in stats["by_type"].items():
            report += f"- {type_name}: {count}\n"
        
        report += "\n## Hälytykset vakavuuksittain\n\n"
        
        for severity_name, count in stats["by_severity"].items():
            report += f"- {severity_name}: {count}\n"
        
        report += "\n## Aktiiviset hälytykset\n\n"
        
        for alert in self.active_alerts.values():
            report += f"### {alert.rule.type.value}\n"
            report += f"- Vakavuus: {alert.rule.severity.value}\n"
            report += f"- Kuvaus: {alert.rule.description}\n"
            report += f"- Arvo: {alert.value}\n"
            report += f"- Aika: {alert.timestamp}\n\n"
        
        return report

async def main():
    """Testaa hälytysjärjestelmää"""
    # Alusta hälytysjärjestelmä
    alert_system = AlertSystem(
        slack_webhook="https://hooks.slack.com/services/xxx"
    )
    
    # Simuloi hälytyksiä
    for i in range(10):
        # Token-käyttö
        await alert_system.check_alert(
            AlertType.TOKEN_LIMIT,
            np.random.uniform(0.8, 1.0),
            {"task_id": f"task_{i}"}
        )
        
        # Suorituskyky
        await alert_system.check_alert(
            AlertType.PERFORMANCE,
            np.random.uniform(1.5, 3.0),
            {"endpoint": "/api/test"}
        )
        
        # Resurssit
        await alert_system.check_alert(
            AlertType.RESOURCE,
            np.random.uniform(80, 100),
            {"resource": "cpu"}
        )
        
        await asyncio.sleep(1)
    
    # Tulosta raportti
    print("\nAlert Report:")
    print(alert_system.generate_report())

if __name__ == "__main__":
    asyncio.run(main())
