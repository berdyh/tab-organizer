"""
Alert Manager - Manages alerting and notification system for container and service failures.
Provides comprehensive alerting with multiple notification channels and alert correlation.
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from enum import Enum

import httpx
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from config import MonitoringSettings
from logging_config import get_logger

logger = get_logger("alert_manager")


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class Alert:
    """Alert data structure."""
    
    def __init__(self, alert_type: str, severity: AlertSeverity, message: str,
                 service: str = None, **context):
        self.id = self._generate_id(alert_type, service, message)
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.service = service
        self.context = context
        self.status = AlertStatus.ACTIVE
        self.created_at = time.time()
        self.updated_at = time.time()
        self.acknowledged_at = None
        self.resolved_at = None
        self.notification_count = 0
        self.last_notification = None
    
    def _generate_id(self, alert_type: str, service: str, message: str) -> str:
        """Generate unique alert ID."""
        content = f"{alert_type}:{service}:{message}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "id": self.id,
            "alert_type": self.alert_type,
            "severity": self.severity.value,
            "message": self.message,
            "service": self.service,
            "context": self.context,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "acknowledged_at": self.acknowledged_at,
            "resolved_at": self.resolved_at,
            "notification_count": self.notification_count,
            "last_notification": self.last_notification
        }


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, settings: MonitoringSettings):
        self.settings = settings
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules = self._initialize_alert_rules()
        self.notification_channels = self._initialize_notification_channels()
        self.suppression_rules: Set[str] = set()
        self.alert_history: List[Dict[str, Any]] = []
    
    def _initialize_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize alert rules based on thresholds."""
        return {
            "high_cpu": {
                "threshold": self.settings.alert_thresholds["cpu_percent"],
                "severity": AlertSeverity.HIGH,
                "cooldown_minutes": 5
            },
            "high_memory": {
                "threshold": self.settings.alert_thresholds["memory_percent"],
                "severity": AlertSeverity.HIGH,
                "cooldown_minutes": 5
            },
            "high_disk": {
                "threshold": self.settings.alert_thresholds["disk_percent"],
                "severity": AlertSeverity.CRITICAL,
                "cooldown_minutes": 10
            },
            "slow_response": {
                "threshold": self.settings.alert_thresholds["response_time_ms"],
                "severity": AlertSeverity.MEDIUM,
                "cooldown_minutes": 3
            },
            "high_error_rate": {
                "threshold": self.settings.alert_thresholds["error_rate_percent"],
                "severity": AlertSeverity.HIGH,
                "cooldown_minutes": 5
            },
            "service_down": {
                "threshold": self.settings.alert_thresholds["service_down_minutes"],
                "severity": AlertSeverity.CRITICAL,
                "cooldown_minutes": 1
            }
        }
    
    def _initialize_notification_channels(self) -> List[str]:
        """Initialize available notification channels."""
        channels = []
        
        if "console" in self.settings.notification_channels:
            channels.append("console")
        
        if "webhook" in self.settings.notification_channels and self.settings.webhook_url:
            channels.append("webhook")
        
        if "slack" in self.settings.notification_channels and self.settings.slack_webhook_url:
            channels.append("slack")
        
        if "email" in self.settings.notification_channels and self.settings.email_smtp_server:
            channels.append("email")
        
        return channels    

    async def start_alert_processing(self):
        """Start the alert processing loop."""
        logger.info("Starting alert processing")
        
        while True:
            try:
                # Process pending notifications
                await self._process_notifications()
                
                # Clean up old alerts
                await self._cleanup_old_alerts()
                
                # Wait before next processing cycle
                await asyncio.sleep(30)  # Process every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Alert processing error", error=str(e))
                await asyncio.sleep(30)
    
    async def create_alert(self, alert_type: str, severity: AlertSeverity, 
                          message: str, service: str = None, **context) -> str:
        """Create a new alert."""
        alert = Alert(alert_type, severity, message, service, **context)
        
        # Check if alert already exists
        if alert.id in self.alerts:
            existing_alert = self.alerts[alert.id]
            existing_alert.updated_at = time.time()
            existing_alert.context.update(context)
            logger.info("Alert updated", alert_id=alert.id, alert_type=alert_type)
            return alert.id
        
        # Check suppression rules
        if self._is_suppressed(alert):
            logger.info("Alert suppressed", alert_id=alert.id, alert_type=alert_type)
            return alert.id
        
        # Add alert
        self.alerts[alert.id] = alert
        
        # Log alert creation
        logger.log_alert(alert_type, severity.value, message, 
                        service=service, alert_id=alert.id, **context)
        
        # Schedule notification
        await self._schedule_notification(alert)
        
        # Add to history
        self.alert_history.append({
            "action": "created",
            "alert": alert.to_dict(),
            "timestamp": time.time()
        })
        
        return alert.id
    
    def _is_suppressed(self, alert: Alert) -> bool:
        """Check if alert should be suppressed."""
        # Check global suppression rules
        suppression_key = f"{alert.alert_type}:{alert.service}"
        return suppression_key in self.suppression_rules
    
    async def _schedule_notification(self, alert: Alert):
        """Schedule notification for an alert."""
        # Check cooldown period
        rule = self.alert_rules.get(alert.alert_type, {})
        cooldown_minutes = rule.get("cooldown_minutes", 5)
        
        if (alert.last_notification and 
            time.time() - alert.last_notification < cooldown_minutes * 60):
            return  # Still in cooldown period
        
        # Send notification
        await self._send_notification(alert)
        
        # Update notification tracking
        alert.notification_count += 1
        alert.last_notification = time.time()
    
    async def _send_notification(self, alert: Alert):
        """Send notification through all configured channels."""
        notification_tasks = []
        
        for channel in self.notification_channels:
            if channel == "console":
                notification_tasks.append(self._send_console_notification(alert))
            elif channel == "webhook":
                notification_tasks.append(self._send_webhook_notification(alert))
            elif channel == "slack":
                notification_tasks.append(self._send_slack_notification(alert))
            elif channel == "email":
                notification_tasks.append(self._send_email_notification(alert))
        
        # Send all notifications concurrently
        if notification_tasks:
            await asyncio.gather(*notification_tasks, return_exceptions=True)
    
    async def _send_console_notification(self, alert: Alert):
        """Send console notification."""
        logger.warning(f"🚨 ALERT: {alert.message}",
                      alert_id=alert.id,
                      severity=alert.severity.value,
                      service=alert.service,
                      **alert.context)
    
    async def _send_webhook_notification(self, alert: Alert):
        """Send webhook notification."""
        if not self.settings.webhook_url:
            return
        
        try:
            payload = {
                "alert": alert.to_dict(),
                "timestamp": time.time()
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    self.settings.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    logger.info("Webhook notification sent", alert_id=alert.id)
                else:
                    logger.error("Webhook notification failed",
                               alert_id=alert.id,
                               status_code=response.status_code)
        
        except Exception as e:
            logger.error("Webhook notification error", alert_id=alert.id, error=str(e))
    
    async def _send_slack_notification(self, alert: Alert):
        """Send Slack notification."""
        if not self.settings.slack_webhook_url:
            return
        
        try:
            # Format Slack message
            color = {
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.HIGH: "warning",
                AlertSeverity.MEDIUM: "warning",
                AlertSeverity.LOW: "good",
                AlertSeverity.INFO: "good"
            }.get(alert.severity, "warning")
            
            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"🚨 {alert.severity.value.upper()} Alert",
                    "text": alert.message,
                    "fields": [
                        {"title": "Service", "value": alert.service or "System", "short": True},
                        {"title": "Alert Type", "value": alert.alert_type, "short": True},
                        {"title": "Time", "value": datetime.fromtimestamp(alert.created_at).strftime("%Y-%m-%d %H:%M:%S"), "short": True}
                    ],
                    "footer": f"Alert ID: {alert.id}"
                }]
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    self.settings.slack_webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    logger.info("Slack notification sent", alert_id=alert.id)
                else:
                    logger.error("Slack notification failed",
                               alert_id=alert.id,
                               status_code=response.status_code)
        
        except Exception as e:
            logger.error("Slack notification error", alert_id=alert.id, error=str(e))
    
    async def _send_email_notification(self, alert: Alert):
        """Send email notification."""
        if not self.settings.email_smtp_server or not self.settings.email_recipients:
            return
        
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.settings.email_username
            msg['To'] = ', '.join(self.settings.email_recipients)
            msg['Subject'] = f"🚨 {alert.severity.value.upper()} Alert: {alert.alert_type}"
            
            # Email body
            body = f"""
Alert Details:
- ID: {alert.id}
- Type: {alert.alert_type}
- Severity: {alert.severity.value.upper()}
- Service: {alert.service or 'System'}
- Message: {alert.message}
- Time: {datetime.fromtimestamp(alert.created_at).strftime('%Y-%m-%d %H:%M:%S')}

Context:
{json.dumps(alert.context, indent=2)}

This is an automated alert from the Web Scraping Tool Monitoring System.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.settings.email_smtp_server, self.settings.email_smtp_port)
            server.starttls()
            server.login(self.settings.email_username, self.settings.email_password)
            server.send_message(msg)
            server.quit()
            
            logger.info("Email notification sent", alert_id=alert.id)
        
        except Exception as e:
            logger.error("Email notification error", alert_id=alert.id, error=str(e))
    
    async def _process_notifications(self):
        """Process pending notifications."""
        current_time = time.time()
        
        for alert in self.alerts.values():
            if alert.status != AlertStatus.ACTIVE:
                continue
            
            # Check if notification is due
            rule = self.alert_rules.get(alert.alert_type, {})
            cooldown_minutes = rule.get("cooldown_minutes", 5)
            
            if (not alert.last_notification or 
                current_time - alert.last_notification >= cooldown_minutes * 60):
                await self._schedule_notification(alert)
    
    async def _cleanup_old_alerts(self):
        """Clean up old resolved alerts."""
        current_time = time.time()
        cutoff_time = current_time - (24 * 60 * 60)  # 24 hours
        
        alerts_to_remove = []
        for alert_id, alert in self.alerts.items():
            if (alert.status == AlertStatus.RESOLVED and 
                alert.resolved_at and alert.resolved_at < cutoff_time):
                alerts_to_remove.append(alert_id)
        
        for alert_id in alerts_to_remove:
            del self.alerts[alert_id]
        
        if alerts_to_remove:
            logger.info("Cleaned up old alerts", count=len(alerts_to_remove))
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = time.time()
        alert.updated_at = time.time()
        
        # Add to history
        self.alert_history.append({
            "action": "acknowledged",
            "alert_id": alert_id,
            "timestamp": time.time()
        })
        
        logger.info("Alert acknowledged", alert_id=alert_id)
        return True
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = time.time()
        alert.updated_at = time.time()
        
        # Add to history
        self.alert_history.append({
            "action": "resolved",
            "alert_id": alert_id,
            "timestamp": time.time()
        })
        
        logger.info("Alert resolved", alert_id=alert_id)
        return True
    
    async def suppress_alert_type(self, alert_type: str, service: str = None):
        """Suppress alerts of a specific type."""
        suppression_key = f"{alert_type}:{service}" if service else f"{alert_type}:*"
        self.suppression_rules.add(suppression_key)
        logger.info("Alert type suppressed", alert_type=alert_type, service=service)
    
    async def unsuppress_alert_type(self, alert_type: str, service: str = None):
        """Remove suppression for alert type."""
        suppression_key = f"{alert_type}:{service}" if service else f"{alert_type}:*"
        self.suppression_rules.discard(suppression_key)
        logger.info("Alert type unsuppressed", alert_type=alert_type, service=service)
    
    async def get_all_alerts(self) -> List[Dict[str, Any]]:
        """Get all alerts."""
        return [alert.to_dict() for alert in self.alerts.values()]
    
    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get only active alerts."""
        return [alert.to_dict() for alert in self.alerts.values() 
                if alert.status == AlertStatus.ACTIVE]
    
    async def get_alert_by_id(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Get specific alert by ID."""
        alert = self.alerts.get(alert_id)
        return alert.to_dict() if alert else None
    
    async def get_alert_history(self) -> List[Dict[str, Any]]:
        """Get alert history."""
        return self.alert_history[-100:]  # Last 100 entries
    
    async def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        total_alerts = len(self.alerts)
        active_alerts = sum(1 for alert in self.alerts.values() 
                           if alert.status == AlertStatus.ACTIVE)
        acknowledged_alerts = sum(1 for alert in self.alerts.values() 
                                if alert.status == AlertStatus.ACKNOWLEDGED)
        resolved_alerts = sum(1 for alert in self.alerts.values() 
                            if alert.status == AlertStatus.RESOLVED)
        
        # Count by severity
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = sum(
                1 for alert in self.alerts.values() 
                if alert.severity == severity and alert.status == AlertStatus.ACTIVE
            )
        
        # Count by service
        service_counts = {}
        for alert in self.alerts.values():
            if alert.status == AlertStatus.ACTIVE:
                service = alert.service or "system"
                service_counts[service] = service_counts.get(service, 0) + 1
        
        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "acknowledged_alerts": acknowledged_alerts,
            "resolved_alerts": resolved_alerts,
            "severity_breakdown": severity_counts,
            "service_breakdown": service_counts,
            "suppression_rules": list(self.suppression_rules),
            "notification_channels": self.notification_channels
        }
    
    async def reload_config(self):
        """Reload configuration."""
        self.settings = MonitoringSettings()
        self.alert_rules = self._initialize_alert_rules()
        self.notification_channels = self._initialize_notification_channels()
        logger.info("Alert manager configuration reloaded")
    
    async def close(self):
        """Clean up resources."""
        logger.info("Alert manager closed")