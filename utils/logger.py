"""
=============================================================================
📝 نظام Logging متقدم مع Rotation والتتبع
=============================================================================
"""

import logging
import logging.handlers
import logging.config
import json
from datetime import datetime
from typing import Dict, Any, Optional
from config.settings import Config
import os

class AppLogger:
    """نظام Logging احترافي مع features متقدمة"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # إنشاء مجلد السجلات
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
        
        # تهيئة نظام Logging
        self._setup_logging()
        
        # إعدادات خاصة
        self.operation_stack = []
        self.performance_records = {}
        
        self._initialized = True
    
    def _setup_logging(self):
        """إعداد نظام Logging"""
        logging.config.dictConfig(Config.get_logging_config())
        self.logger = logging.getLogger('lottery')
    
    def start_operation(self, operation_name: str, metadata: Optional[Dict] = None):
        """بدء عملية جديدة مع تتبع"""
        operation_id = f"{operation_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        self.operation_stack.append({
            'id': operation_id,
            'name': operation_name,
            'start_time': datetime.now(),
            'metadata': metadata or {}
        })
        
        self.logger.info(f"🔧 بدء العملية: {operation_name}", extra={
            'operation_id': operation_id,
            'metadata': metadata
        })
        
        return operation_id
    
    def end_operation(self, operation_id: str, status: str = "completed", 
                     metrics: Optional[Dict] = None):
        """إنهاء عملية مع التسجيل"""
        for op in reversed(self.operation_stack):
            if op['id'] == operation_id:
                duration = (datetime.now() - op['start_time']).total_seconds()
                
                log_data = {
                    'operation_id': operation_id,
                    'operation_name': op['name'],
                    'duration_seconds': round(duration, 3),
                    'status': status,
                    'metrics': metrics or {},
                    'metadata': op['metadata']
                }
                
                if status == "completed":
                    self.logger.info(f"✅ اكتملت العملية: {op['name']} ({duration:.2f} ثانية)", 
                                   extra=log_data)
                elif status == "failed":
                    self.logger.error(f"❌ فشلت العملية: {op['name']}", extra=log_data)
                elif status == "skipped":
                    self.logger.info(f"⏭️ تخطيت العملية: {op['name']}", extra=log_data)
                else:
                    self.logger.warning(f"⚠️ حالة غير معروفة للعملية: {op['name']}", extra=log_data)
                
                # حفظ في سجل الأداء
                self.performance_records[operation_id] = log_data
                
                # إزالة من المكدس
                self.operation_stack.remove(op)
                break
    
    def log_generation(self, constraints: Dict, ticket_count: int, 
                      duration: float, success_count: int):
        """تسجيل عملية توليد تذاكر"""
        self.logger.info("🎰 توليد التذاكر", extra={
            'operation': 'ticket_generation',
            'constraints': constraints,
            'requested_count': ticket_count,
            'generated_count': success_count,
            'duration_seconds': round(duration, 3),
            'success_rate': round(success_count / ticket_count * 100, 2) if ticket_count > 0 else 0,
            'efficiency': round(success_count / duration, 2) if duration > 0 else 0
        })
    
    def log_prediction(self, model_name: str, accuracy: float, 
                      confidence: float, features_used: list):
        """تسجيل عملية توقع"""
        self.logger.info("🔮 تنبؤ الذكاء الاصطناعي", extra={
            'operation': 'ai_prediction',
            'model': model_name,
            'accuracy': round(accuracy, 4),
            'confidence': round(confidence, 4),
            'features_count': len(features_used),
            'features': features_used[:10]  # أول 10 features فقط
        })
    
    def log_anomaly(self, number: int, z_score: float, 
                   expected: float, actual: float):
        """تسجيل شذوذ تم اكتشافه"""
        severity = "HIGH" if abs(z_score) > 3 else "MEDIUM" if abs(z_score) > 2 else "LOW"
        
        self.logger.warning(f"⚠️ اكتشاف شذوذ: الرقم {number}", extra={
            'operation': 'anomaly_detection',
            'number': number,
            'z_score': round(z_score, 2),
            'expected_frequency': round(expected, 2),
            'actual_frequency': actual,
            'deviation': round(abs(actual - expected), 2),
            'severity': severity
        })
    
    def log_security_event(self, event_type: str, user_ip: str = "", 
                          details: Dict = None):
        """تسجيل أحداث أمنية"""
        security_levels = {
            'login_attempt': 'INFO',
            'failed_validation': 'WARNING',
            'suspicious_activity': 'ERROR',
            'data_tampering': 'CRITICAL'
        }
        
        level = security_levels.get(event_type, 'INFO')
        log_method = getattr(self.logger, level.lower())
        
        log_method(f"🔒 حدث أمني: {event_type}", extra={
            'operation': 'security',
            'event_type': event_type,
            'user_ip': user_ip,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        })
    
    def get_performance_report(self) -> Dict[str, Any]:
        """تقرير أداء مفصل"""
        report = {
            'total_operations': len(self.performance_records),
            'successful_operations': sum(1 for r in self.performance_records.values() 
                                        if r['status'] == 'completed'),
            'failed_operations': sum(1 for r in self.performance_records.values() 
                                    if r['status'] == 'failed'),
            'average_duration': 0,
            'operations_by_type': {}
        }
        
        if self.performance_records:
            durations = [r['duration_seconds'] for r in self.performance_records.values()]
            report['average_duration'] = round(sum(durations) / len(durations), 3)
            report['max_duration'] = round(max(durations), 3)
            report['min_duration'] = round(min(durations), 3)
        
        # تجميع حسب النوع
        for record in self.performance_records.values():
            op_name = record['operation_name']
            if op_name not in report['operations_by_type']:
                report['operations_by_type'][op_name] = {
                    'count': 0,
                    'total_duration': 0,
                    'success_rate': 0
                }
            
            report['operations_by_type'][op_name]['count'] += 1
            report['operations_by_type'][op_name]['total_duration'] += record['duration_seconds']
        
        # حساب معدلات النجاح
        for op_name, stats in report['operations_by_type'].items():
            successful = sum(1 for r in self.performance_records.values() 
                           if r['operation_name'] == op_name and r['status'] == 'completed')
            stats['success_rate'] = round(successful / stats['count'] * 100, 2) if stats['count'] > 0 else 0
            stats['avg_duration'] = round(stats['total_duration'] / stats['count'], 3)
        
        return report
    
    def export_logs(self, days: int = 7) -> str:
        """تصدير السجلات لعدد معين من الأيام"""
        cutoff_date = datetime.now().timestamp() - (days * 86400)
        log_file = os.path.join(Config.LOGS_DIR, 'app.log')
        
        filtered_logs = []
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        # تحليل timestamp من السجل
                        log_time_str = line.split(' - ')[0]
                        # ✅ استخدام Config.DATETIME_FORMAT الموجود الآن
                        log_time = datetime.strptime(log_time_str.strip(), Config.DATETIME_FORMAT)
                        
                        if log_time.timestamp() >= cutoff_date:
                            filtered_logs.append(line.strip())
                    except Exception:
                        continue
        
        os.makedirs(Config.EXPORT_DIR, exist_ok=True)
        export_file = os.path.join(Config.EXPORT_DIR, f'logs_export_{datetime.now().strftime("%Y%m%d")}.json')
        
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump({
                'export_date': datetime.now().isoformat(),
                'days_covered': days,
                'log_count': len(filtered_logs),
                'logs': filtered_logs
            }, f, ensure_ascii=False, indent=2)
        
        return export_file

# Singleton instance
logger = AppLogger()
