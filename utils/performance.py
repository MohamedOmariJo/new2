"""
=============================================================================
⚡ نظام قياس وتحسين الأداء
=============================================================================
"""

import time
import psutil
import tracemalloc
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np
from contextlib import contextmanager
from collections import defaultdict

from config.settings import Config
from utils.logger import logger


class PerformanceBenchmark:
    """نظام متقدم لقياس وتحسين الأداء"""
    
    def __init__(self):
        self.metrics = {}
        self.history = []
        self.max_history_size = 100
        
        # إحصائيات النظام
        self.system_stats = {
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'system_start_time': datetime.now()
        }
    
    def start_monitoring(self, operation_name: str):
        """بدء مراقبة أداء عملية"""
        # ✅ Config.ENABLE_PROFILING موجود الآن في settings.py
        if Config.ENABLE_PROFILING:
            tracemalloc.start()
        
        self.metrics[operation_name] = {
            'start_time': time.time(),
            'start_memory': psutil.Process().memory_info().rss,
            'start_cpu': psutil.cpu_percent(interval=None),
            'start_malloc': tracemalloc.get_traced_memory()[0] if Config.ENABLE_PROFILING else 0
        }
        
        return operation_name
    
    def stop_monitoring(self, operation_name: str) -> Dict[str, Any]:
        """إنهاء مراقبة أداء عملية"""
        if operation_name not in self.metrics:
            return {}
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        end_cpu = psutil.cpu_percent(interval=None)
        
        start_data = self.metrics[operation_name]
        
        metrics = {
            'duration_seconds': round(end_time - start_data['start_time'], 4),
            'memory_used_bytes': end_memory - start_data['start_memory'],
            'memory_used_mb': round((end_memory - start_data['start_memory']) / (1024**2), 2),
            'cpu_usage_percent': round(end_cpu - start_data['start_cpu'], 2),
            'throughput': 0
        }
        
        if Config.ENABLE_PROFILING:
            try:
                current, peak = tracemalloc.get_traced_memory()
                metrics['malloc_current_mb'] = round(current / 1024 / 1024, 2)
                metrics['malloc_peak_mb'] = round(peak / 1024 / 1024, 2)
                metrics['malloc_increase_mb'] = round(
                    (current - start_data['start_malloc']) / 1024 / 1024, 2
                )
                tracemalloc.stop()
            except Exception:
                pass
        
        # حساب Throughput إذا كان هناك حجم مخرجات
        if 'output_size' in start_data:
            if metrics['duration_seconds'] > 0:
                metrics['throughput'] = round(start_data['output_size'] / metrics['duration_seconds'], 2)
        
        # إضافة إلى التاريخ
        record = {
            'operation': operation_name,
            'timestamp': datetime.now(),
            **metrics
        }
        
        self.history.append(record)
        
        # الاحتفاظ فقط بأحدث السجلات
        if len(self.history) > self.max_history_size:
            self.history = self.history[-self.max_history_size:]
        
        # التحقق من حدود الأداء
        self._check_performance_limits(operation_name, metrics)
        
        # تنظيف
        del self.metrics[operation_name]
        
        return metrics
    
    def _check_performance_limits(self, operation_name: str, metrics: Dict):
        """التحقق من تجاوز حدود الأداء"""
        warnings_list = []
        
        # ✅ Config.MAX_MEMORY_USAGE_MB و Config.MAX_CPU_PERCENT موجودان الآن
        if metrics['memory_used_mb'] > Config.MAX_MEMORY_USAGE_MB:
            warnings_list.append(f"استخدام ذاكرة مرتفع: {metrics['memory_used_mb']}MB")
        
        if metrics['cpu_usage_percent'] > Config.MAX_CPU_PERCENT:
            warnings_list.append(f"استخدام CPU مرتفع: {metrics['cpu_usage_percent']}%")
        
        if warnings_list:
            logger.logger.warning(f"⚠️ حدود أداء تجاوزت في {operation_name}", extra={
                'operation': operation_name,
                'metrics': metrics,
                'warnings': warnings_list
            })
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """مدير سياق لمراقبة العمليات"""
        op_id = self.start_monitoring(operation_name)
        try:
            yield op_id
        finally:
            self.stop_monitoring(operation_name)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """الحصول على إحصائيات النظام الحالية"""
        cpu_percent = psutil.cpu_percent(interval=0.5)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_freq = psutil.cpu_freq()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'percent': cpu_percent,
                'count': self.system_stats['cpu_count'],
                'frequency': cpu_freq.current if cpu_freq else None
            },
            'memory': {
                'total_gb': self.system_stats['total_memory_gb'],
                'available_gb': round(memory.available / (1024**3), 2),
                'used_percent': memory.percent,
                'used_gb': round(memory.used / (1024**3), 2)
            },
            'disk': {
                'total_gb': round(disk.total / (1024**3), 2),
                'used_gb': round(disk.used / (1024**3), 2),
                'free_gb': round(disk.free / (1024**3), 2),
                'percent': disk.percent
            },
            'process': {
                'memory_mb': round(psutil.Process().memory_info().rss / (1024**2), 2),
                'cpu_percent': psutil.Process().cpu_percent(interval=0.1),
                'threads': psutil.Process().num_threads()
            }
        }
    
    def get_performance_report(self, operation_filter: Optional[str] = None) -> Dict[str, Any]:
        """تقرير أداء مفصل"""
        if operation_filter:
            filtered_history = [h for h in self.history if h['operation'] == operation_filter]
        else:
            filtered_history = self.history
        
        if not filtered_history:
            return {
                'message': 'لا توجد بيانات أداء',
                'duration': {'min': 0, 'max': 0, 'avg': 0, 'std': 0},
                'memory': {'min_mb': 0, 'max_mb': 0, 'avg_mb': 0, 'total_mb': 0},
                'operations_by_type': {},
                'trend_analysis': {},
                'recommendations': []
            }
        
        # تحليل إحصائي
        durations = [h['duration_seconds'] for h in filtered_history]
        memory_used = [h['memory_used_mb'] for h in filtered_history]
        
        report = {
            'summary': {
                'total_operations': len(filtered_history),
                'time_period': {
                    'first': filtered_history[0]['timestamp'].isoformat(),
                    'last': filtered_history[-1]['timestamp'].isoformat()
                }
            },
            'duration': {
                'min': round(min(durations), 4),
                'max': round(max(durations), 4),
                'avg': round(np.mean(durations), 4),
                'median': round(np.median(durations), 4),
                'std': round(np.std(durations), 4),
                'p95': round(np.percentile(durations, 95), 4)
            },
            'memory': {
                'min_mb': round(min(memory_used), 2),
                'max_mb': round(max(memory_used), 2),
                'avg_mb': round(np.mean(memory_used), 2),
                'total_mb': round(sum(memory_used), 2)
            },
            'operations_by_type': self._group_by_operation(filtered_history),
            'trend_analysis': self._analyze_trends(filtered_history),
            'recommendations': self._generate_recommendations(filtered_history)
        }
        
        return report
    
    def _group_by_operation(self, history: List[Dict]) -> Dict[str, Any]:
        """تجميع البيانات حسب نوع العملية"""
        operations = {}
        
        for record in history:
            op_name = record['operation']
            if op_name not in operations:
                operations[op_name] = {
                    'count': 0,
                    'total_duration': 0,
                    'total_memory': 0,
                    'durations': [],
                    'memories': []
                }
            
            operations[op_name]['count'] += 1
            operations[op_name]['total_duration'] += record['duration_seconds']
            operations[op_name]['total_memory'] += record['memory_used_mb']
            operations[op_name]['durations'].append(record['duration_seconds'])
            operations[op_name]['memories'].append(record['memory_used_mb'])
        
        # حساب الإحصائيات
        for op_name, data in operations.items():
            data['avg_duration'] = round(data['total_duration'] / data['count'], 4)
            data['avg_memory'] = round(data['total_memory'] / data['count'], 2)
            data['duration_std'] = round(np.std(data['durations']), 4) if data['durations'] else 0
            data['memory_std'] = round(np.std(data['memories']), 2) if data['memories'] else 0
        
        return operations
    
    def _analyze_trends(self, history: List[Dict]) -> Dict[str, Any]:
        """تحليل الاتجاهات في الأداء"""
        if len(history) < 3:
            return {'message': 'بيانات غير كافية لتحليل الاتجاهات'}
        
        daily_stats = defaultdict(lambda: {'count': 0, 'total_duration': 0, 'total_memory': 0})
        
        for record in history:
            day = record['timestamp'].strftime('%Y-%m-%d')
            daily_stats[day]['count'] += 1
            daily_stats[day]['total_duration'] += record['duration_seconds']
            daily_stats[day]['total_memory'] += record['memory_used_mb']
        
        days = sorted(daily_stats.keys())
        avg_durations = [daily_stats[d]['total_duration'] / daily_stats[d]['count'] for d in days]
        avg_memories = [daily_stats[d]['total_memory'] / daily_stats[d]['count'] for d in days]
        
        if len(avg_durations) >= 2:
            duration_trend = np.polyfit(range(len(avg_durations)), avg_durations, 1)[0]
            memory_trend = np.polyfit(range(len(avg_memories)), avg_memories, 1)[0]
        else:
            duration_trend = memory_trend = 0.0
        
        return {
            'days_analyzed': len(days),
            'duration_trend': round(float(duration_trend), 6),
            'memory_trend': round(float(memory_trend), 6),
            'is_improving': duration_trend < 0 and memory_trend < 0,
            'needs_attention': duration_trend > 0.001 or memory_trend > 0.1
        }
    
    def _generate_recommendations(self, history: List[Dict]) -> List[str]:
        """توليد توصيات لتحسين الأداء"""
        recommendations = []
        
        slow_operations = [
            (r['operation'], r['duration_seconds'])
            for r in history if r['duration_seconds'] > 5.0
        ]
        
        if slow_operations:
            slow_operations.sort(key=lambda x: x[1], reverse=True)
            top_slow = slow_operations[:3]
            recommendations.append(
                f"🔧 العمليات البطيئة: {', '.join([f'{op}({dur:.1f}s)' for op, dur in top_slow])}"
            )
        
        memory_hungry = [
            (r['operation'], r['memory_used_mb'])
            for r in history if r['memory_used_mb'] > 100
        ]
        
        if memory_hungry:
            memory_hungry.sort(key=lambda x: x[1], reverse=True)
            top_memory = memory_hungry[:3]
            recommendations.append(
                f"💾 عمليات تستهلك ذاكرة عالية: {', '.join([f'{op}({mem:.0f}MB)' for op, mem in top_memory])}"
            )
        
        if not recommendations:
            avg_duration = np.mean([h['duration_seconds'] for h in history])
            if avg_duration > 2.0:
                recommendations.append("⚡ متوسط وقت التنفيذ مرتفع، فكر في تحسين الخوارزميات")
            
            avg_memory = np.mean([h['memory_used_mb'] for h in history])
            if avg_memory > 50:
                recommendations.append("🧹 متوسط استخدام الذاكرة مرتفع، فكر في تحسين إدارة الذاكرة")
        
        if not recommendations:
            recommendations.append("✅ الأداء ضمن النطاق المقبول")
        
        return recommendations
    
    def compare_with_baseline(self, baseline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """مقارنة الأداء الحالي مع خط أساس"""
        current_metrics = self.get_performance_report()
        
        comparison = {
            'duration_change': 0.0,
            'memory_change': 0.0,
            'improvement': False,
            'regression': False,
            'details': {}
        }
        
        if 'duration' in current_metrics and 'duration' in baseline_metrics:
            current_avg = current_metrics['duration']['avg']
            baseline_avg = baseline_metrics['duration']['avg']
            if baseline_avg != 0:
                comparison['duration_change'] = round((current_avg - baseline_avg) / baseline_avg * 100, 2)
        
        if 'memory' in current_metrics and 'memory' in baseline_metrics:
            current_avg = current_metrics['memory']['avg_mb']
            baseline_avg = baseline_metrics['memory']['avg_mb']
            if baseline_avg != 0:
                comparison['memory_change'] = round((current_avg - baseline_avg) / baseline_avg * 100, 2)
        
        comparison['improvement'] = (
            comparison['duration_change'] < -5 and
            comparison['memory_change'] < -10
        )
        
        comparison['regression'] = (
            comparison['duration_change'] > 10 or
            comparison['memory_change'] > 20
        )
        
        comparison['details'] = {
            'current_avg_duration': current_metrics.get('duration', {}).get('avg', 0),
            'baseline_avg_duration': baseline_metrics.get('duration', {}).get('avg', 0),
            'current_avg_memory': current_metrics.get('memory', {}).get('avg_mb', 0),
            'baseline_avg_memory': baseline_metrics.get('memory', {}).get('avg_mb', 0)
        }
        
        return comparison
