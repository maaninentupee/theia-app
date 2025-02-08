import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
import json

import aiohttp
from prometheus_client import Counter, Gauge, Histogram

# Konfiguroi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertType(Enum):
    """Hälytystyypit"""
    TOKEN_LIMIT = "token_limit"
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"
    CUSTOM = "custom"

class AlertSeverity(Enum):
    """Hälytysvakavuudet"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """Hälytysten tilat"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SILENCED = "silenced"

@dataclass
class Alert:
    """Hälytys"""
    id: str
    type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime
    status: AlertStatus
    task_id: Optional[str] = None
    data: Optional[Dict] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution: Optional[str] = None

class AlertRule:
    """Hälytyssääntö"""
    
    def __init__(
        self,
        type: AlertType,
        severity: AlertSeverity,
        condition: Callable,
        message_template: str,
        cooldown: int = 300,  # 5min
        auto_resolve: bool = False
    ):
        """
        Alusta sääntö
        
        Args:
            type: Hälytystyyppi
            severity: Vakavuus
            condition: Ehto
            message_template: Viestipohja
            cooldown: Viilennysaika sekunteina
            auto_resolve: Automaattinen ratkaisu
        """
        self.type = type
        self.severity = severity
        self.condition = condition
        self.message_template = message_template
        self.cooldown = cooldown
        self.auto_resolve = auto_resolve
        
        # Viimeisin hälytys
        self.last_alert: Optional[datetime] = None

class AlertManager:
    """Hälytysten hallinta"""
    
    def __init__(
        self,
        slack_webhook: Optional[str] = None,
        email_config: Optional[Dict] = None,
        pagerduty_key: Optional[str] = None
    ):
        """
        Alusta hallinta
        
        Args:
            slack_webhook: Slack webhook URL
            email_config: Sähköpostikonfiguraatio
            pagerduty_key: PagerDuty API-avain
        """
        self.slack_webhook = slack_webhook
        self.email_config = email_config
        self.pagerduty_key = pagerduty_key
        
        # Hälytykset ja säännöt
        self.alerts: Dict[str, Alert] = {}
        self.rules: Dict[AlertType, AlertRule] = {}
        
        # Metriikat
        self.alert_counter = Counter(
            'alerts_total',
            'Total alerts',
            ['type', 'severity']
        )
        self.active_alerts = Gauge(
            'active_alerts',
            'Active alerts',
            ['type', 'severity']
        )
        self.alert_latency = Histogram(
            'alert_latency_seconds',
            'Alert latency',
            ['type']
        )
        
        # Alusta oletussäännöt
        self._init_default_rules()
    
    def _init_default_rules(self):
        """Alusta oletussäännöt"""
        # Token-raja
        self.add_rule(
            AlertRule(
                type=AlertType.TOKEN_LIMIT,
                severity=AlertSeverity.ERROR,
                condition=lambda x: x.get('tokens_used', 0) > x.get('token_limit', 0),
                message_template="Task {task_id} exceeded token limit: {tokens_used}/{token_limit}"
            )
        )
        
        # API-virhe
        self.add_rule(
            AlertRule(
                type=AlertType.API_ERROR,
                severity=AlertSeverity.ERROR,
                condition=lambda x: x.get('status_code', 200) >= 500,
                message_template="API error in task {task_id}: {error_message}"
            )
        )
        
        # Pyyntöraja
        self.add_rule(
            AlertRule(
                type=AlertType.RATE_LIMIT,
                severity=AlertSeverity.WARNING,
                condition=lambda x: x.get('rate_limited', False),
                message_template="Rate limit hit for {api_endpoint}"
            )
        )
        
        # Aikakatkaisu
        self.add_rule(
            AlertRule(
                type=AlertType.TIMEOUT,
                severity=AlertSeverity.WARNING,
                condition=lambda x: x.get('duration', 0) > x.get('timeout', 30),
                message_template="Task {task_id} timed out after {duration}s"
            )
        )
        
        # Muisti
        self.add_rule(
            AlertRule(
                type=AlertType.MEMORY,
                severity=AlertSeverity.CRITICAL,
                condition=lambda x: x.get('memory_usage', 0) > 90,
                message_template="High memory usage: {memory_usage}%",
                auto_resolve=True
            )
        )
        
        # CPU
        self.add_rule(
            AlertRule(
                type=AlertType.CPU,
                severity=AlertSeverity.WARNING,
                condition=lambda x: x.get('cpu_usage', 0) > 80,
                message_template="High CPU usage: {cpu_usage}%",
                auto_resolve=True
            )
        )
    
    def add_rule(self, rule: AlertRule):
        """
        Lisää sääntö
        
        Args:
            rule: Sääntö
        """
        self.rules[rule.type] = rule
    
    async def check_alert(
        self,
        type: AlertType,
        data: Dict[str, Any]
    ) -> Optional[Alert]:
        """
        Tarkista hälytys
        
        Args:
            type: Hälytystyyppi
            data: Tiedot
        
        Returns:
            Optional[Alert]: Hälytys
        """
        rule = self.rules.get(type)
        if not rule:
            return None
        
        # Tarkista viilennys
        if rule.last_alert and (
            datetime.now() - rule.last_alert
        ).total_seconds() < rule.cooldown:
            return None
        
        # Tarkista ehto
        if not rule.condition(data):
            return None
        
        # Luo hälytys
        alert = Alert(
            id=f"alert_{len(self.alerts)}",
            type=type,
            severity=rule.severity,
            message=rule.message_template.format(**data),
            timestamp=datetime.now(),
            status=AlertStatus.ACTIVE,
            task_id=data.get('task_id'),
            data=data
        )
        
        # Tallenna hälytys
        self.alerts[alert.id] = alert
        rule.last_alert = alert.timestamp
        
        # Päivitä metriikat
        self.alert_counter.labels(
            type=type.value,
            severity=rule.severity.value
        ).inc()
        
        self.active_alerts.labels(
            type=type.value,
            severity=rule.severity.value
        ).inc()
        
        # Lähetä ilmoitukset
        await self._send_notifications(alert)
        
        return alert
    
    async def resolve_alert(
        self,
        alert_id: str,
        resolution: str,
        resolver: Optional[str] = None
    ):
        """
        Ratkaise hälytys
        
        Args:
            alert_id: Hälytyksen ID
            resolution: Ratkaisu
            resolver: Ratkaisija
        """
        alert = self.alerts.get(alert_id)
        if not alert:
            return
        
        # Päivitä tila
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        alert.resolution = resolution
        alert.acknowledged_by = resolver
        
        # Päivitä metriikat
        self.active_alerts.labels(
            type=alert.type.value,
            severity=alert.severity.value
        ).dec()
        
        # Loki
        logger.info(
            f"Alert {alert_id} resolved by {resolver}: {resolution}"
        )
    
    async def acknowledge_alert(
        self,
        alert_id: str,
        acknowledger: str
    ):
        """
        Kuittaa hälytys
        
        Args:
            alert_id: Hälytyksen ID
            acknowledger: Kuittaaja
        """
        alert = self.alerts.get(alert_id)
        if not alert:
            return
        
        # Päivitä tila
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = acknowledger
        
        # Loki
        logger.info(
            f"Alert {alert_id} acknowledged by {acknowledger}"
        )
    
    async def silence_alert(
        self,
        alert_id: str,
        duration: int,
        silencer: str
    ):
        """
        Hiljennä hälytys
        
        Args:
            alert_id: Hälytyksen ID
            duration: Kesto sekunteina
            silencer: Hiljentäjä
        """
        alert = self.alerts.get(alert_id)
        if not alert:
            return
        
        # Päivitä tila
        alert.status = AlertStatus.SILENCED
        alert.acknowledged_by = silencer
        
        # Ajasta palautus
        asyncio.create_task(
            self._unsilence_alert(
                alert_id,
                duration
            )
        )
        
        # Loki
        logger.info(
            f"Alert {alert_id} silenced by {silencer} "
            f"for {duration}s"
        )
    
    async def _unsilence_alert(
        self,
        alert_id: str,
        duration: int
    ):
        """
        Palauta hälytys
        
        Args:
            alert_id: Hälytyksen ID
            duration: Kesto sekunteina
        """
        await asyncio.sleep(duration)
        
        alert = self.alerts.get(alert_id)
        if not alert or alert.status != AlertStatus.SILENCED:
            return
        
        # Palauta tila
        alert.status = AlertStatus.ACTIVE
        alert.acknowledged_by = None
        
        # Loki
        logger.info(f"Alert {alert_id} unsilenced")
    
    async def _send_notifications(self, alert: Alert):
        """
        Lähetä ilmoitukset
        
        Args:
            alert: Hälytys
        """
        # Slack
        if self.slack_webhook:
            await self._send_slack_alert(alert)
        
        # Sähköposti
        if self.email_config:
            await self._send_email_alert(alert)
        
        # PagerDuty
        if (
            self.pagerduty_key and
            alert.severity in (
                AlertSeverity.ERROR,
                AlertSeverity.CRITICAL
            )
        ):
            await self._send_pagerduty_alert(alert)
    
    async def _send_slack_alert(self, alert: Alert):
        """
        Lähetä Slack-hälytys
        
        Args:
            alert: Hälytys
        """
        async with aiohttp.ClientSession() as session:
            await session.post(
                self.slack_webhook,
                json={
                    "text": (
                        f"*{alert.severity.value.upper()} ALERT*\n"
                        f"Type: {alert.type.value}\n"
                        f"Message: {alert.message}\n"
                        f"Time: {alert.timestamp.isoformat()}"
                    )
                }
            )
    
    async def _send_email_alert(self, alert: Alert):
        """
        Lähetä sähköpostihälytys
        
        Args:
            alert: Hälytys
        """
        # TODO: Implementoi sähköpostilähetys
        pass
    
    async def _send_pagerduty_alert(self, alert: Alert):
        """
        Lähetä PagerDuty-hälytys
        
        Args:
            alert: Hälytys
        """
        async with aiohttp.ClientSession() as session:
            await session.post(
                "https://events.pagerduty.com/v2/enqueue",
                json={
                    "routing_key": self.pagerduty_key,
                    "event_action": "trigger",
                    "payload": {
                        "summary": alert.message,
                        "severity": alert.severity.value,
                        "source": "cascade",
                        "custom_details": alert.data
                    }
                }
            )
    
    def get_active_alerts(
        self,
        type: Optional[AlertType] = None,
        severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """
        Hae aktiiviset hälytykset
        
        Args:
            type: Hälytystyyppi
            severity: Vakavuus
        
        Returns:
            List[Alert]: Hälytykset
        """
        alerts = [
            a for a in self.alerts.values()
            if a.status == AlertStatus.ACTIVE
        ]
        
        if type:
            alerts = [a for a in alerts if a.type == type]
        
        if severity:
            alerts = [
                a for a in alerts
                if a.severity == severity
            ]
        
        return sorted(
            alerts,
            key=lambda x: x.timestamp,
            reverse=True
        )
    
    def get_alert_stats(self) -> Dict:
        """
        Hae tilastot
        
        Returns:
            Dict: Tilastot
        """
        stats = {
            "total": len(self.alerts),
            "by_type": {},
            "by_severity": {},
            "by_status": {}
        }
        
        # Laske tyypit
        for alert in self.alerts.values():
            # Tyyppi
            type_stats = stats["by_type"].setdefault(
                alert.type.value,
                {"total": 0, "active": 0}
            )
            type_stats["total"] += 1
            if alert.status == AlertStatus.ACTIVE:
                type_stats["active"] += 1
            
            # Vakavuus
            sev_stats = stats["by_severity"].setdefault(
                alert.severity.value,
                {"total": 0, "active": 0}
            )
            sev_stats["total"] += 1
            if alert.status == AlertStatus.ACTIVE:
                sev_stats["active"] += 1
            
            # Tila
            status_stats = stats["by_status"].setdefault(
                alert.status.value,
                0
            )
            stats["by_status"][
                alert.status.value
            ] += 1
        
        return stats

async def main():
    """Testaa hälytysjärjestelmää"""
    # Alusta hallinta
    manager = AlertManager(
        slack_webhook="https://hooks.slack.com/..."
    )
    
    # Testaa hälytyksiä
    await manager.check_alert(
        AlertType.TOKEN_LIMIT,
        {
            "task_id": "task_1",
            "tokens_used": 1500,
            "token_limit": 1000
        }
    )
    
    await manager.check_alert(
        AlertType.API_ERROR,
        {
            "task_id": "task_2",
            "status_code": 500,
            "error_message": "Internal Server Error"
        }
    )
    
    await manager.check_alert(
        AlertType.MEMORY,
        {
            "memory_usage": 95,
            "process_id": 1234
        }
    )
    
    # Tulosta tilastot
    print("\nAlert Stats:")
    print(json.dumps(
        manager.get_alert_stats(),
        indent=2
    ))
    
    # Tulosta aktiiviset
    print("\nActive Alerts:")
    for alert in manager.get_active_alerts():
        print(
            f"- [{alert.severity.value}] "
            f"{alert.type.value}: {alert.message}"
        )

if __name__ == "__main__":
    asyncio.run(main())
