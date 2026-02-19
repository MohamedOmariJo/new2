"""
=============================================================================
🔔 نظام الإشعارات المتعدد القنوات
=============================================================================
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta  # ✅ إضافة timedelta المفقودة
import json
import os
from enum import Enum

from config.settings import Config
from utils.logger import logger


class NotificationPriority(Enum):
    """أولويات الإشعارات"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    CRITICAL = "critical"


class NotificationChannel(Enum):
    """قنوات الإرسال"""
    IN_APP = "in_app"
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    LOG = "log"


class Notification:
    """تمثيل للإشعار"""
    
    def __init__(self, title: str, message: str, 
                 priority: NotificationPriority = NotificationPriority.INFO,
                 channels: List[NotificationChannel] = None,
                 metadata: Dict = None):
        self.id = self._generate_id()
        self.title = title
        self.message = message
        self.priority = priority
        self.channels = channels or [NotificationChannel.IN_APP, NotificationChannel.LOG]
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.sent_at = None
        self.status = "pending"
        self.retry_count = 0
    
    def _generate_id(self) -> str:
        """توليد معرف فريد للإشعار"""
        import uuid
        return str(uuid.uuid4())
    
    def to_dict(self) -> Dict:
        """تحويل الإشعار إلى قاموس"""
        return {
            'id': self.id,
            'title': self.title,
            'message': self.message,
            'priority': self.priority.value,
            'channels': [c.value for c in self.channels],
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'sent_at': self.sent_at.isoformat() if self.sent_at else None,
            'status': self.status,
            'retry_count': self.retry_count
        }


class NotificationProvider:
    """مزود خدمة الإشعارات الأساسي"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_enabled = True
        self.stats = {
            'total_sent': 0,
            'total_failed': 0,
            'last_sent': None
        }
    
    def send(self, notification: Notification) -> bool:
        """إرسال الإشعار"""
        raise NotImplementedError
    
    def can_send(self, notification: Notification) -> bool:
        """التحقق من إمكانية الإرسال"""
        return self.is_enabled
    
    def update_stats(self, success: bool):
        """تحديث إحصائيات المزود"""
        self.stats['last_sent'] = datetime.now()
        if success:
            self.stats['total_sent'] += 1
        else:
            self.stats['total_failed'] += 1
    
    def get_stats(self) -> Dict:
        """الحصول على إحصائيات المزود"""
        return self.stats.copy()


class InAppProvider(NotificationProvider):
    """مزود الإشعارات داخل التطبيق"""
    
    def __init__(self):
        super().__init__('in_app')
        self.notifications_history: List[Notification] = []
        self.max_history = 100
    
    def send(self, notification: Notification) -> bool:
        """إرسال إشعار داخل التطبيق"""
        try:
            logger.logger.info(f"🔔 إشعار داخل التطبيق: {notification.title}")
            
            self.notifications_history.append(notification)
            
            if len(self.notifications_history) > self.max_history:
                self.notifications_history = self.notifications_history[-self.max_history:]
            
            notification.sent_at = datetime.now()
            notification.status = "sent"
            self.update_stats(True)
            
            return True
            
        except Exception as e:
            logger.logger.error(f"❌ فشل إرسال إشعار داخل التطبيق: {e}")
            notification.status = "failed"
            self.update_stats(False)
            return False
    
    def get_recent_notifications(self, limit: int = 20) -> List[Notification]:
        """الحصول على أحدث الإشعارات"""
        return self.notifications_history[-limit:]


class EmailProvider(NotificationProvider):
    """مزود الإشعارات عبر البريد الإلكتروني"""
    
    def __init__(self, smtp_server: str = None, smtp_port: int = 587,
                 username: str = None, password: str = None):
        super().__init__('email')
        
        self.smtp_server = smtp_server or os.getenv('EMAIL_SMTP_SERVER', '')
        self.smtp_port = smtp_port
        self.username = username or os.getenv('EMAIL_USER', '')
        self.password = password or os.getenv('EMAIL_PASSWORD', '')
        
        if not all([self.smtp_server, self.username, self.password]):
            self.is_enabled = False
            logger.logger.warning("⚠️ مزود البريد الإلكتروني معطل - إعدادات غير مكتملة")
    
    def send(self, notification: Notification) -> bool:
        """إرسال إشعار عبر البريد الإلكتروني"""
        if not self.is_enabled:
            return False
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[Jordan Lottery] {notification.title}"
            msg['From'] = self.username
            msg['To'] = notification.metadata.get('recipient', self.username)
            
            text = (
                f"{notification.title}\n"
                f"{'=' * len(notification.title)}\n\n"
                f"{notification.message}\n\n"
                f"تاريخ الإرسال: {datetime.now().strftime(Config.DATETIME_FORMAT)}\n"
                f"الأولوية: {notification.priority.value}"
            )
            
            priority_colors = {
                NotificationPriority.SUCCESS: '#10b981',
                NotificationPriority.WARNING: '#f59e0b',
                NotificationPriority.ERROR: '#ef4444',
                NotificationPriority.CRITICAL: '#7f1d1d',
                NotificationPriority.INFO: '#3b82f6',
            }
            header_color = priority_colors.get(notification.priority, '#3b82f6')
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                    .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ 
                        background-color: {header_color};
                        color: white;
                        padding: 15px;
                        border-radius: 8px;
                        margin-bottom: 20px;
                    }}
                    .content {{ padding: 20px; background-color: #f9fafb; border-radius: 8px; }}
                    .footer {{ margin-top: 20px; font-size: 12px; color: #6b7280; text-align: center; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h2>{notification.title}</h2>
                    </div>
                    <div class="content">
                        <p>{notification.message.replace(chr(10), '<br>')}</p>
                    </div>
                    <div class="footer">
                        <p>Jordan Lottery AI Pro - {datetime.now().strftime(Config.DATETIME_FORMAT)}</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            part1 = MIMEText(text, 'plain')
            part2 = MIMEText(html, 'html')
            msg.attach(part1)
            msg.attach(part2)
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            notification.sent_at = datetime.now()
            notification.status = "sent"
            self.update_stats(True)
            
            logger.logger.info(f"📧 تم إرسال إشعار بالبريد: {notification.title}")
            return True
            
        except Exception as e:
            logger.logger.error(f"❌ فشل إرسال إشعار بالبريد: {e}")
            notification.status = "failed"
            self.update_stats(False)
            return False


class LogProvider(NotificationProvider):
    """مزود الإشعارات عبر السجلات"""
    
    def __init__(self):
        super().__init__('log')
    
    def send(self, notification: Notification) -> bool:
        """تسجيل الإشعار في السجلات"""
        try:
            log_level = {
                NotificationPriority.INFO: 'info',
                NotificationPriority.WARNING: 'warning',
                NotificationPriority.ERROR: 'error',
                NotificationPriority.SUCCESS: 'info',
                NotificationPriority.CRITICAL: 'critical'
            }.get(notification.priority, 'info')
            
            log_message = f"🔔 {notification.title}: {notification.message}"
            getattr(logger.logger, log_level)(log_message, extra={
                'notification_id': notification.id,
                'priority': notification.priority.value,
                'metadata': notification.metadata
            })
            
            notification.sent_at = datetime.now()
            notification.status = "sent"
            self.update_stats(True)
            
            return True
            
        except Exception as e:
            logger.logger.error(f"❌ فشل تسجيل الإشعار: {e}")
            notification.status = "failed"
            self.update_stats(False)
            return False


class NotificationSystem:
    """نظام الإشعارات الرئيسي"""
    
    def __init__(self):
        self.providers: Dict[NotificationChannel, NotificationProvider] = {}
        self.notifications_queue: List[Notification] = []
        self.notifications_history: List[Notification] = []
        self.max_history = 1000
        self.retry_limit = 3
        
        self._initialize_providers()
    
    def _initialize_providers(self):
        """تهيئة جميع مزودي الإشعارات"""
        self.providers[NotificationChannel.IN_APP] = InAppProvider()
        
        email_provider = EmailProvider()
        if email_provider.is_enabled:
            self.providers[NotificationChannel.EMAIL] = email_provider
        
        self.providers[NotificationChannel.LOG] = LogProvider()
        
        logger.logger.info("🔔 نظام الإشعارات مهيأ", extra={
            'providers_count': len(self.providers),
            'providers': [k.value for k in self.providers.keys()]
        })
    
    def send(self, title: str, message: str, 
            priority = None,
            channels: List[NotificationChannel] = None,
            metadata: Dict = None) -> Dict[str, Any]:
        """
        إرسال إشعار.
        ✅ إصلاح: priority يمكن أن يكون NotificationPriority أو string أو None
        """
        op_id = logger.start_operation('send_notification', {
            'title': title,
        })
        
        try:
            # ✅ تحويل priority من string إلى NotificationPriority إذا لزم الأمر
            if priority is None:
                priority_enum = NotificationPriority.INFO
            elif isinstance(priority, str):
                try:
                    priority_enum = NotificationPriority(priority.lower())
                except ValueError:
                    priority_enum = NotificationPriority.INFO
            elif isinstance(priority, NotificationPriority):
                priority_enum = priority
            else:
                priority_enum = NotificationPriority.INFO
            
            notification = Notification(
                title=title,
                message=message,
                priority=priority_enum,
                channels=channels or [NotificationChannel.IN_APP, NotificationChannel.LOG],
                metadata=metadata or {}
            )
            
            self.notifications_queue.append(notification)
            
            result = self._process_notification(notification)
            
            self._add_to_history(notification)
            
            logger.end_operation(op_id, 'completed', {
                'notification_id': notification.id,
                'status': notification.status,
                'channels_used': result
            })
            
            return {
                'notification_id': notification.id,
                'status': notification.status,
                'channels': result,
                'created_at': notification.created_at
            }
            
        except Exception as e:
            logger.end_operation(op_id, 'failed', {'error': str(e)})
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _process_notification(self, notification: Notification) -> Dict[str, bool]:
        """معالجة إرسال الإشعار عبر جميع القنوات"""
        results = {}
        
        for channel in notification.channels:
            if channel in self.providers:
                provider = self.providers[channel]
                
                if not provider.can_send(notification):
                    results[channel.value] = False
                    continue
                
                success = False
                for attempt in range(self.retry_limit):
                    try:
                        success = provider.send(notification)
                        if success:
                            break
                        
                        notification.retry_count += 1
                        logger.logger.warning(
                            f"⚠️ إعادة محاولة إرسال الإشعار {notification.id} "
                            f"عبر {channel.value} (المحاولة {attempt + 1})"
                        )
                        
                    except Exception as e:
                        logger.logger.error(
                            f"❌ خطأ في إرسال الإشعار عبر {channel.value}: {e}"
                        )
                
                results[channel.value] = success
                
                if not success:
                    notification.status = "partially_failed"
            
            else:
                results[channel.value] = False
                logger.logger.warning(f"⚠️ قناة إشعار غير معروفة: {channel.value}")
        
        if all(results.values()):
            notification.status = "sent"
        elif any(results.values()):
            notification.status = "partially_sent"
        else:
            notification.status = "failed"
        
        return results
    
    def _add_to_history(self, notification: Notification):
        """إضافة الإشعار إلى التاريخ"""
        self.notifications_history.append(notification)
        
        if len(self.notifications_history) > self.max_history:
            self.notifications_history = self.notifications_history[-self.max_history:]
    
    def get_notifications(self, limit: int = 50, 
                         priority: NotificationPriority = None,
                         status: str = None) -> List[Dict]:
        """الحصول على الإشعارات"""
        filtered = self.notifications_history.copy()
        
        if priority:
            filtered = [n for n in filtered if n.priority == priority]
        
        if status:
            filtered = [n for n in filtered if n.status == status]
        
        filtered.sort(key=lambda x: x.created_at, reverse=True)
        filtered = filtered[:limit]
        
        return [n.to_dict() for n in filtered]
    
    def get_provider_stats(self) -> Dict[str, Dict]:
        """الحصول على إحصائيات جميع المزودين"""
        return {
            channel.value: provider.get_stats()
            for channel, provider in self.providers.items()
        }
    
    def send_bulk(self, notifications: List[Dict]) -> List[Dict]:
        """إرسال إشعارات جماعية"""
        results = []
        
        for notification_data in notifications:
            priority_val = notification_data.get('priority', 'info')
            try:
                priority_enum = NotificationPriority(priority_val)
            except ValueError:
                priority_enum = NotificationPriority.INFO
            
            channels_raw = notification_data.get('channels', ['in_app'])
            channels = []
            for c in channels_raw:
                try:
                    channels.append(NotificationChannel(c))
                except ValueError:
                    pass
            
            result = self.send(
                title=notification_data.get('title', ''),
                message=notification_data.get('message', ''),
                priority=priority_enum,
                channels=channels or None,
                metadata=notification_data.get('metadata', {})
            )
            results.append(result)
        
        return results
    
    def schedule_notification(self, title: str, message: str, 
                            send_time: datetime,
                            priority: NotificationPriority = NotificationPriority.INFO,
                            channels: List[NotificationChannel] = None,
                            metadata: Dict = None) -> str:
        """جدولة إشعار للوقت المستقبلي"""
        notification_id = f"scheduled_{datetime.now().timestamp()}"
        
        logger.logger.info(f"📅 جدولة إشعار: {title} للوقت {send_time}", extra={
            'notification_id': notification_id,
            'send_time': send_time.isoformat(),
            'priority': priority.value
        })
        
        return notification_id
    
    def clear_notifications(self, older_than_days: int = 30):
        """✅ إصلاح: timedelta مستوردة الآن بشكل صحيح"""
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        old_count = len(self.notifications_history)
        self.notifications_history = [
            n for n in self.notifications_history 
            if n.created_at > cutoff_date
        ]
        new_count = len(self.notifications_history)
        
        logger.logger.info("🧹 مسح الإشعارات القديمة", extra={
            'old_count': old_count,
            'new_count': new_count,
            'removed_count': old_count - new_count,
            'cutoff_date': cutoff_date.isoformat()
        })
    
    def export_notifications(self, format: str = 'json') -> str:
        """تصدير الإشعارات"""
        notifications_data = [n.to_dict() for n in self.notifications_history]
        
        if format == 'json':
            return json.dumps(notifications_data, ensure_ascii=False, indent=2)
        elif format == 'csv':
            import csv
            import io
            
            if not notifications_data:
                return ''
            
            output_buffer = io.StringIO()
            writer = csv.DictWriter(output_buffer, fieldnames=notifications_data[0].keys())
            writer.writeheader()
            writer.writerows(notifications_data)
            return output_buffer.getvalue()
        else:
            raise ValueError(f"تنسيق غير معروف: {format}")


# وظائف مساعدة للاستخدام السريع
def notify_info(title: str, message: str, metadata: Dict = None):
    """إرسال إشعار معلومات"""
    notification_system = NotificationSystem()
    return notification_system.send(
        title=title,
        message=message,
        priority=NotificationPriority.INFO,
        metadata=metadata
    )


def notify_success(title: str, message: str, metadata: Dict = None):
    """إرسال إشعار نجاح"""
    notification_system = NotificationSystem()
    return notification_system.send(
        title=title,
        message=message,
        priority=NotificationPriority.SUCCESS,
        metadata=metadata
    )


def notify_warning(title: str, message: str, metadata: Dict = None):
    """إرسال إشعار تحذير"""
    notification_system = NotificationSystem()
    return notification_system.send(
        title=title,
        message=message,
        priority=NotificationPriority.WARNING,
        metadata=metadata
    )


def notify_error(title: str, message: str, metadata: Dict = None):
    """إرسال إشعار خطأ"""
    notification_system = NotificationSystem()
    return notification_system.send(
        title=title,
        message=message,
        priority=NotificationPriority.ERROR,
        metadata=metadata
    )
