"""
=============================================================================
🎰 مولد التذاكر الذكي المحسّن مع تحسينات الأداء
=============================================================================
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
import random
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import Config
from utils.logger import logger
from utils.performance import PerformanceBenchmark
from core.analyzer import AdvancedAnalyzer


class SmartGenerator:
    """مولد تذاكر ذكي مع فلاتر متقدمة وتحسينات أداء"""
    
    def __init__(self, analyzer: AdvancedAnalyzer):
        self.analyzer = analyzer
        self.benchmark = PerformanceBenchmark()
        self.cache: Dict[str, List[List[int]]] = {}
    
    def generate_tickets(
        self,
        count: int,
        size: int = 6,
        constraints: Optional[Dict] = None,
        use_cache: bool = True
    ) -> List[List[int]]:
        """توليد تذاكر مع فلاتر محسنة - يبحث في كل التذاكر الممكنة حتى يجد العدد المطلوب"""
        
        if constraints is None:
            constraints = {}
        
        cache_key = self._generate_cache_key(count, size, constraints)
        if use_cache and cache_key in self.cache:
            logger.logger.info(f"🎯 استخدام Cache للتوليد - مفتاح: {cache_key[:50]}...")
            return self.cache[cache_key].copy()
        
        op_id = logger.start_operation('ticket_generation', {
            'count': count,
            'size': size,
            'constraints': constraints
        })
        
        try:
            with self.benchmark.monitor_operation('generation'):
                pool = self._prepare_number_pool(constraints)
                
                if len(pool) < size:
                    error_msg = f"❌ عدد الأرقام المتاحة ({len(pool)}) أقل من حجم التذكرة ({size})"
                    logger.logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # ✅ البحث الشامل: يبحث في كل التذاكر الممكنة حتى يجد العدد المطلوب
                tickets = self._exhaustive_search(pool, size, count, constraints)
                
                if use_cache and len(tickets) > 0:
                    self.cache[cache_key] = tickets.copy()
                    self._clean_cache()
                
                logger.end_operation(op_id, 'completed', {
                    'generated_count': len(tickets),
                    'success_rate': round(len(tickets) / count * 100, 2) if count > 0 else 0,
                    'cache_used': use_cache,
                    'cache_key': cache_key[:30]
                })
                
                return tickets
                
        except Exception as e:
            logger.end_operation(op_id, 'failed', {'error': str(e)})
            raise
    
    def _exhaustive_search(self, pool: List[int], size: int, 
                           count: int, constraints: Dict) -> List[List[int]]:
        """
        بحث شامل: يبحث في كل التوليفات الممكنة حتى يجد العدد المطلوب.
        يستخدم استراتيجية ذكية: عشوائي أولاً، ثم استنزاف شامل إذا لزم.
        """
        import math
        
        total_combinations = math.comb(len(pool), size)
        tickets_set: Set[tuple] = set()
        
        # ✅ المرحلة 1: توليد عشوائي سريع (للشروط غير الصارمة جداً)
        max_random_attempts = max(count * 5000, 50000)
        attempts = 0
        
        while len(tickets_set) < count and attempts < max_random_attempts:
            attempts += 1
            ticket = tuple(sorted(random.sample(pool, size)))
            
            if (self._satisfies_basic_constraints(list(ticket), constraints) and
                    self._satisfies_advanced_constraints(list(ticket), constraints)):
                tickets_set.add(ticket)
        
        # ✅ المرحلة 2: إذا لم نجد الكمية المطلوبة، نبحث استنزافياً في كل التوليفات
        if len(tickets_set) < count and total_combinations <= 2_000_000:
            logger.logger.info(f"🔍 بدء البحث الشامل في {total_combinations:,} توليفة ممكنة...")
            
            # خلط الـ pool لتنويع النتائج
            shuffled_pool = pool.copy()
            random.shuffle(shuffled_pool)
            
            for combo in itertools.combinations(shuffled_pool, size):
                ticket = tuple(sorted(combo))
                if ticket in tickets_set:
                    continue
                    
                if (self._satisfies_basic_constraints(list(ticket), constraints) and
                        self._satisfies_advanced_constraints(list(ticket), constraints)):
                    tickets_set.add(ticket)
                    
                if len(tickets_set) >= count:
                    break
        
        elif len(tickets_set) < count and total_combinations > 2_000_000:
            # ✅ المرحلة 3: للأرقام الكثيرة، نزيد المحاولات العشوائية بشكل كبير
            logger.logger.info(f"🔍 البحث الموسع... ({total_combinations:,} توليفة ممكنة)")
            extra_attempts = 0
            max_extra = count * 100_000
            
            while len(tickets_set) < count and extra_attempts < max_extra:
                extra_attempts += 1
                ticket = tuple(sorted(random.sample(pool, size)))
                
                if (self._satisfies_basic_constraints(list(ticket), constraints) and
                        self._satisfies_advanced_constraints(list(ticket), constraints)):
                    tickets_set.add(ticket)
        
        return [list(t) for t in list(tickets_set)[:count]]
    
    def _prepare_number_pool(self, constraints: Dict) -> List[int]:
        """تحضير مجموعة الأرقام مع تطبيق الاستبعاد"""
        pool = list(range(Config.MIN_NUMBER, Config.MAX_NUMBER + 1))
        
        if 'exclude' in constraints:
            exclude_set = set(constraints['exclude'])
            pool = [n for n in pool if n not in exclude_set]
        
        if constraints.get('filter_low_freq', False):
            freq_values = list(self.analyzer.freq.values())
            if freq_values:
                avg_freq = np.mean(freq_values)
                pool = [n for n in pool if self.analyzer.freq.get(n, 0) >= avg_freq * 0.5]
        
        return pool
    
    def _generate_small_batch(self, pool: List[int], size: int, 
                            count: int, constraints: Dict) -> List[List[int]]:
        """توليد دفعات صغيرة (<= 10)"""
        return self._exhaustive_search(pool, size, count, constraints)
    
    def _generate_medium_batch(self, pool: List[int], size: int, 
                             count: int, constraints: Dict) -> List[List[int]]:
        """توليد دفعات متوسطة (<= 100)"""
        return self._exhaustive_search(pool, size, count, constraints)
    
    def _generate_large_batch(self, pool: List[int], size: int, 
                            count: int, constraints: Dict) -> List[List[int]]:
        """توليد دفعات كبيرة (> 100)"""
        return self._exhaustive_search(pool, size, count, constraints)
    
    def _generate_batch_parallel(self, pool: List[int], size: int, 
                               batch_size: int, constraints: Dict) -> List[List[int]]:
        """توليد دفعة بالتوازي"""
        batch_tickets = []
        
        for _ in range(batch_size):
            if len(pool) < size:
                break
            ticket = sorted(random.sample(pool, size))
            
            if (self._satisfies_basic_constraints(ticket, constraints) and 
                self._satisfies_advanced_constraints(ticket, constraints)):
                batch_tickets.append(ticket)
        
        return batch_tickets
    
    def _filter_batch_vectorized(self, batch: np.ndarray, constraints: Dict) -> np.ndarray:
        """تصفية الدفعة باستخدام vectorization"""
        if batch.size == 0:
            return batch
        
        mask = np.ones(len(batch), dtype=bool)
        
        if 'sum_range' in constraints:
            min_sum, max_sum = constraints['sum_range']
            row_sums = batch.sum(axis=1)
            mask &= (row_sums >= min_sum) & (row_sums <= max_sum)
        
        if 'odd' in constraints:
            target_odd = constraints['odd']
            odd_counts = (batch % 2).sum(axis=1)
            mask &= (odd_counts == target_odd)
        
        return batch[mask]
    
    def _satisfies_basic_constraints(self, ticket: List[int], constraints: Dict) -> bool:
        """التحقق من القيود الأساسية"""
        if 'odd' in constraints:
            odd_count = sum(1 for n in ticket if n % 2)
            if odd_count != constraints['odd']:
                return False
        
        if 'sum_range' in constraints:
            min_sum, max_sum = constraints['sum_range']
            ticket_sum = sum(ticket)
            if not (min_sum <= ticket_sum <= max_sum):
                return False
        
        if 'fixed' in constraints:
            fixed_set = set(constraints['fixed'])
            if not fixed_set.issubset(set(ticket)):
                return False
        
        return True
    
    def _satisfies_advanced_constraints(self, ticket: List[int], constraints: Dict) -> bool:
        """التحقق من القيود المتقدمة"""
        if 'consecutive' in constraints:
            consec_count = sum(1 for i in range(len(ticket)-1) 
                             if ticket[i+1] - ticket[i] == 1)
            if consec_count != constraints['consecutive']:
                return False
        
        if 'shadows' in constraints:
            shadows_count = sum(1 for c in Counter([n % 10 for n in ticket]).values() 
                              if c > 1)
            if shadows_count != constraints['shadows']:
                return False
        
        if 'hot_min' in constraints:
            hot_count = len(set(ticket) & self.analyzer.hot)
            if hot_count < constraints['hot_min']:
                return False
        
        if 'cold_max' in constraints:
            cold_count = len(set(ticket) & self.analyzer.cold)
            if cold_count > constraints['cold_max']:
                return False
        
        if 'last_match' in constraints:
            match_count = len(set(ticket) & self.analyzer.last_draw)
            if match_count != constraints['last_match']:
                return False
        
        return True
    
    def _apply_advanced_filters(self, tickets: List[List[int]], constraints: Dict) -> List[List[int]]:
        """تطبيق فلاتر متقدمة بعد التوليد"""
        return [
            ticket for ticket in tickets
            if self._satisfies_advanced_constraints(ticket, constraints)
        ]
    
    def generate_markov_based(self, count: int, size: int = 6) -> List[List[int]]:
        """توليد تذاكر بناءً على Markov"""
        op_id = logger.start_operation('markov_generation', {
            'count': count,
            'size': size
        })
        
        try:
            with self.benchmark.monitor_operation('markov_generation'):
                tickets = []
                last_nums = sorted(list(self.analyzer.last_draw))
                
                for _ in range(count):
                    predictions = self.analyzer.get_markov_prediction(last_nums, top_n=15)
                    
                    if not predictions:
                        ticket = sorted(random.sample(range(Config.MIN_NUMBER, Config.MAX_NUMBER + 1), size))
                    else:
                        # ✅ إصلاح: تحويل إلى قوائم فوراً وتوسيع الأوزان بشكل صحيح
                        cand_nums = [num for num, _ in predictions]
                        cand_weights = [w for _, w in predictions]
                        
                        # تكملة القائمة إذا لم تكن كافية
                        if len(cand_nums) < size:
                            all_nums = set(range(Config.MIN_NUMBER, Config.MAX_NUMBER + 1))
                            remaining = list(all_nums - set(cand_nums))
                            random.shuffle(remaining)
                            needed = size * 2 - len(cand_nums)
                            extra_nums = remaining[:needed]
                            # منح الأرقام الإضافية وزناً منخفضاً
                            avg_weight = float(np.mean(cand_weights)) if cand_weights else 0.1
                            extra_weights = [avg_weight * 0.1] * len(extra_nums)
                            cand_nums = cand_nums + extra_nums
                            cand_weights = cand_weights + extra_weights
                        
                        # تطبيع الأوزان
                        total_w = sum(cand_weights)
                        if total_w > 0:
                            cand_weights = [w / total_w for w in cand_weights]
                        else:
                            cand_weights = [1.0 / len(cand_nums)] * len(cand_nums)
                        
                        # ✅ التأكد من أن الأوزان بنفس طول المرشحين تماماً
                        n = len(cand_nums)
                        weights_arr = np.array(cand_weights[:n])
                        weights_arr = weights_arr / weights_arr.sum()  # إعادة تطبيع
                        
                        selected = np.random.choice(
                            cand_nums[:n],
                            size=min(size, n),
                            replace=False,
                            p=weights_arr
                        )
                        ticket = sorted(selected.tolist())
                        
                        # إذا كان الحجم أقل من المطلوب، نكمل عشوائياً
                        if len(ticket) < size:
                            remaining = list(set(range(Config.MIN_NUMBER, Config.MAX_NUMBER + 1)) - set(ticket))
                            ticket.extend(random.sample(remaining, size - len(ticket)))
                            ticket = sorted(ticket)
                    
                    if ticket not in tickets:
                        tickets.append(ticket)
                
                logger.end_operation(op_id, 'completed', {
                    'generated_count': len(tickets),
                    'markov_used': len(predictions) > 0 if 'predictions' in dir() else False
                })
                
                return tickets
                
        except Exception as e:
            logger.end_operation(op_id, 'failed', {'error': str(e)})
            raise
    
    def _generate_cache_key(self, count: int, size: int, constraints: Dict) -> str:
        """توليد مفتاح Cache فريد"""
        import hashlib
        import json
        
        # ✅ التأكد من أن البيانات قابلة للتسلسل إلى JSON
        safe_constraints = {}
        for k, v in constraints.items():
            if isinstance(v, set):
                safe_constraints[k] = sorted(list(v))
            elif isinstance(v, (list, tuple)):
                safe_constraints[k] = list(v)
            else:
                safe_constraints[k] = v
        
        data = {
            'count': count,
            'size': size,
            'constraints': safe_constraints,
            'analyzer_hash': hash(str(sorted(self.analyzer.freq.items())))
        }
        
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _clean_cache(self):
        """تنظيف Cache القديم"""
        max_cache_size = 100
        
        if len(self.cache) > max_cache_size:
            keys_to_remove = list(self.cache.keys())[:len(self.cache) - max_cache_size]
            for key in keys_to_remove:
                del self.cache[key]
    
    def generate_with_ml(self, count: int, size: int = 6, 
                        model_name: str = 'random_forest') -> List[List[int]]:
        """توليد تذاكر باستخدام تنبؤات ML"""
        op_id = logger.start_operation('ml_generation', {
            'count': count,
            'size': size,
            'model': model_name
        })
        
        try:
            with self.benchmark.monitor_operation('ml_generation'):
                tickets = []
                
                for _ in range(count):
                    ticket = self._generate_ml_inspired_ticket(size)
                    if ticket not in tickets:
                        tickets.append(ticket)
                
                logger.end_operation(op_id, 'completed', {
                    'generated_count': len(tickets),
                    'model_used': model_name
                })
                
                return tickets
                
        except Exception as e:
            logger.end_operation(op_id, 'failed', {'error': str(e)})
            raise
    
    def _generate_ml_inspired_ticket(self, size: int) -> List[int]:
        """توليد تذكرة مستوحاة من تنبؤات ML"""
        pool = list(range(Config.MIN_NUMBER, Config.MAX_NUMBER + 1))
        
        weights = np.ones(len(pool))
        for i, num in enumerate(pool):
            if num in self.analyzer.hot:
                weights[i] = 2.0
            elif num in self.analyzer.cold:
                weights[i] = 0.5
        
        weights = weights / weights.sum()
        
        ticket = np.random.choice(
            pool,
            size=size,
            replace=False,
            p=weights
        )
        
        return sorted(ticket.tolist())
    
    def get_generation_stats(self) -> Dict:
        """الحصول على إحصائيات التوليد"""
        return {
            'cache_size': len(self.cache),
            'performance_stats': self.benchmark.get_performance_report('generation'),
            'generator_info': {
                'class': self.__class__.__name__,
                'analyzer_initialized': self.analyzer is not None,
                'methods_available': [
                    'generate_tickets',
                    'generate_markov_based',
                    'generate_with_ml'
                ]
            }
        }
