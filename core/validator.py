"""
=============================================================================
✅ نظام تحقق متقدم مع فحص تناقضات القيود
=============================================================================
"""

import re
import random
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from config.settings import Config
from utils.logger import logger


class ConstraintType(Enum):
    """أنواع القيود"""
    NUMERIC_RANGE = "numeric_range"
    COUNT = "count"
    SET_INCLUSION = "set_inclusion"
    SET_EXCLUSION = "set_exclusion"
    PATTERN = "pattern"
    RELATIONAL = "relational"


@dataclass
class Constraint:
    """تمثيل للقيد مع معلومات التحقق"""
    name: str
    type: ConstraintType
    value: object
    severity: str  # error, warning, info
    description: str
    dependencies: List[str] = field(default_factory=list)


class ConstraintValidator:
    """متحقق متقدم من تناقضات القيود"""
    
    def __init__(self):
        self.constraints_registry: Dict[str, Constraint] = {}
        self.initialize_constraints()
    
    def initialize_constraints(self):
        """تهيئة جميع القيود المعروفة"""
        self.constraints_registry['ticket_size'] = Constraint(
            name='ticket_size',
            type=ConstraintType.NUMERIC_RANGE,
            value=(6, 10),
            severity='error',
            description='حجم التذكرة يجب أن يكون بين 6 و10 أرقام'
        )
        
        self.constraints_registry['sum_range'] = Constraint(
            name='sum_range',
            type=ConstraintType.NUMERIC_RANGE,
            value=(20, 200),
            severity='error',
            description='نطاق مجموع الأرقام',
            dependencies=['ticket_size']
        )
        
        self.constraints_registry['odd_count'] = Constraint(
            name='odd_count',
            type=ConstraintType.COUNT,
            value=None,
            severity='error',
            description='عدد الأرقام الفردية',
            dependencies=['ticket_size']
        )
    
    def validate_constraints(self, constraints: Dict, ticket_size: int = 6) -> Tuple[bool, List[str]]:
        """التحقق من تناقضات القيود"""
        issues = []
        warnings_list = []
        
        self._update_dynamic_constraints(ticket_size)
        
        # 1. فحص الأرقام الثابتة مع الفردي/الزوجي
        if 'odd' in constraints and 'fixed' in constraints:
            fixed_nums = constraints['fixed']
            odd_fixed = sum(1 for n in fixed_nums if n % 2)
            
            if odd_fixed > constraints['odd']:
                issues.append(
                    f"❌ الأرقام الثابتة تحتوي على {odd_fixed} رقم فردي "
                    f"لكن المطلوب {constraints['odd']} فقط"
                )
            
            max_possible_odd = ticket_size - len(fixed_nums) + odd_fixed
            if constraints['odd'] > max_possible_odd:
                issues.append(
                    f"❌ لا يمكن تحقيق {constraints['odd']} رقم فردي "
                    f"مع {len(fixed_nums)} رقم ثابت ({odd_fixed} فردي منهم)"
                )
        
        # 2. فحص نطاق المجموع
        if 'sum_range' in constraints:
            min_sum, max_sum = constraints['sum_range']
            
            theoretical_min = self._calculate_min_sum(ticket_size, constraints)
            theoretical_max = self._calculate_max_sum(ticket_size, constraints)
            
            if min_sum < theoretical_min:
                issues.append(
                    f"❌ المجموع الأدنى {min_sum} أقل من الممكن نظرياً {theoretical_min}"
                )
            
            if max_sum > theoretical_max:
                issues.append(
                    f"❌ المجموع الأقصى {max_sum} أكبر من الممكن نظرياً {theoretical_max}"
                )
            
            if min_sum > max_sum:
                issues.append(
                    f"❌ المجموع الأدنى {min_sum} أكبر من الأقصى {max_sum}"
                )
        
        # 3. فحص تناقض Hot/Cold
        if 'hot_min' in constraints and 'cold_max' in constraints:
            hot_min = constraints['hot_min']
            cold_max = constraints['cold_max']
            
            if hot_min + cold_max > ticket_size:
                warnings_list.append(
                    f"⚠️ مجموع الحد الأدنى للساخن ({hot_min}) والحد الأقصى للبارد ({cold_max}) "
                    f"أكبر من حجم التذكرة ({ticket_size})"
                )
        
        # 4. فحص الأرقام المستبعدة مع الثابتة
        if 'exclude' in constraints and 'fixed' in constraints:
            fixed_set = set(constraints['fixed'])
            exclude_set = set(constraints['exclude'])
            conflict = fixed_set.intersection(exclude_set)
            
            if conflict:
                issues.append(
                    f"❌ تناقض: الأرقام {sorted(conflict)} مدرجة كـثابتة ومستبعدة معاً"
                )
        
        # 5. فحص القيود المعقدة
        if self._has_complex_constraints(constraints):
            feasibility_check = self._check_constraints_feasibility(constraints, ticket_size)
            if not feasibility_check[0]:
                issues.append(f"❌ القيود مستحيلة: {feasibility_check[1]}")
        
        if issues or warnings_list:
            logger.log_security_event('failed_validation', details={
                'constraints': {
                    k: list(v) if isinstance(v, set) else v 
                    for k, v in constraints.items()
                },
                'issues': issues,
                'warnings': warnings_list,
                'ticket_size': ticket_size
            })
        
        return len(issues) == 0, issues + warnings_list
    
    def _calculate_min_sum(self, ticket_size: int, constraints: Dict) -> int:
        """حساب أقل مجموع ممكن مع القيود"""
        numbers = list(range(Config.MIN_NUMBER, Config.MAX_NUMBER + 1))
        
        if 'exclude' in constraints:
            numbers = [n for n in numbers if n not in constraints['exclude']]
        
        fixed_sum = 0
        if 'fixed' in constraints:
            fixed_nums = constraints['fixed']
            fixed_sum = sum(fixed_nums)
            numbers = [n for n in numbers if n not in fixed_nums]
            remaining_slots = ticket_size - len(fixed_nums)
        else:
            remaining_slots = ticket_size
        
        remaining_slots = max(0, remaining_slots)
        remaining_numbers = sorted(numbers)[:remaining_slots]
        
        return fixed_sum + sum(remaining_numbers)
    
    def _calculate_max_sum(self, ticket_size: int, constraints: Dict) -> int:
        """حساب أكبر مجموع ممكن مع القيود"""
        numbers = list(range(Config.MIN_NUMBER, Config.MAX_NUMBER + 1))
        
        if 'exclude' in constraints:
            numbers = [n for n in numbers if n not in constraints['exclude']]
        
        fixed_sum = 0
        if 'fixed' in constraints:
            fixed_nums = constraints['fixed']
            fixed_sum = sum(fixed_nums)
            numbers = [n for n in numbers if n not in fixed_nums]
            remaining_slots = ticket_size - len(fixed_nums)
        else:
            remaining_slots = ticket_size
        
        remaining_slots = max(0, remaining_slots)
        remaining_numbers = sorted(numbers, reverse=True)[:remaining_slots]
        
        return fixed_sum + sum(remaining_numbers)
    
    def _has_complex_constraints(self, constraints: Dict) -> bool:
        """التحقق إذا كانت هناك قيود معقدة"""
        complex_keys = {'odd', 'consecutive', 'shadows', 'last_match', 'hot_min', 'cold_max'}
        return any(key in constraints for key in complex_keys)
    
    def _check_constraints_feasibility(self, constraints: Dict, ticket_size: int) -> Tuple[bool, str]:
        """فحص إمكانية تحقيق القيود معاً - محسّن لتجنب الرفض الخاطئ"""
        # ✅ زيادة عدد المحاولات لتجنب الرفض الخاطئ للقيود النادرة
        attempts = 5000
        success_count = 0
        
        for _ in range(attempts):
            ticket = self._generate_random_ticket(ticket_size, constraints)
            if ticket and self._ticket_satisfies_constraints(ticket, constraints):
                success_count += 1
        
        success_rate = success_count / attempts
        
        # ✅ خفض العتبة: نرفض فقط ما هو مستحيل فعلاً (< 0.02%)
        # المولد يستطيع البحث الشامل حتى للقيود النادرة
        if success_rate < 0.0002:
            return False, f"القيود مستحيلة نظرياً (معدل النجاح {success_rate:.2%})"
        else:
            return True, f"معدل النجاح {success_rate:.1%}"
    
    def _generate_random_ticket(self, size: int, constraints: Dict) -> List[int]:
        """توليد تذكرة عشوائية مع بعض القيود"""
        pool = list(range(Config.MIN_NUMBER, Config.MAX_NUMBER + 1))
        
        if 'exclude' in constraints:
            pool = [n for n in pool if n not in constraints['exclude']]
        
        ticket = []
        if 'fixed' in constraints:
            ticket = list(constraints['fixed'])
            pool = [n for n in pool if n not in ticket]
        
        remaining = size - len(ticket)
        if remaining > 0 and len(pool) >= remaining:
            ticket.extend(random.sample(pool, remaining))
        
        return sorted(ticket)
    
    def _ticket_satisfies_constraints(self, ticket: List[int], constraints: Dict) -> bool:
        """التحقق إذا كانت التذكرة تحقق القيود"""
        if not ticket or len(ticket) != len(set(ticket)):
            return False
        
        if 'sum_range' in constraints:
            ticket_sum = sum(ticket)
            min_sum, max_sum = constraints['sum_range']
            if not (min_sum <= ticket_sum <= max_sum):
                return False
        
        if 'odd' in constraints:
            odd_count = sum(1 for n in ticket if n % 2)
            if odd_count != constraints['odd']:
                return False
        
        if 'consecutive' in constraints:
            consec_count = sum(1 for i in range(len(ticket)-1) if ticket[i+1] - ticket[i] == 1)
            if consec_count != constraints['consecutive']:
                return False
        
        return True
    
    def _update_dynamic_constraints(self, ticket_size: int):
        """تحديث القيود الديناميكية بناءً على حجم التذكرة"""
        min_sum_possible = sum(range(Config.MIN_NUMBER, Config.MIN_NUMBER + ticket_size))
        max_sum_possible = sum(range(Config.MAX_NUMBER - ticket_size + 1, Config.MAX_NUMBER + 1))
        
        self.constraints_registry['sum_range'].value = (min_sum_possible, max_sum_possible)
        self.constraints_registry['odd_count'].value = (0, ticket_size)
    
    def suggest_constraint_relaxation(self, constraints: Dict, issues: List[str]) -> Dict:
        """اقتراح تخفيف للقيود"""
        suggestions = {}
        
        if 'sum_range' in constraints:
            min_sum, max_sum = constraints['sum_range']
            range_width = max_sum - min_sum
            new_min = max(20, int(min_sum - range_width * 0.2))
            new_max = min(200, int(max_sum + range_width * 0.2))
            suggestions['sum_range'] = (new_min, new_max)
        
        if 'odd' in constraints:
            odd_value = constraints['odd']
            ticket_size = 6
            suggestions['odd'] = [
                max(0, odd_value - 1),
                odd_value,
                min(ticket_size, odd_value + 1)
            ]
        
        if 'consecutive' in constraints:
            consec_value = constraints['consecutive']
            suggestions['consecutive'] = [
                max(0, consec_value - 1),
                consec_value,
                min(5, consec_value + 1)
            ]
        
        return suggestions


class AdvancedValidator:
    """متحقق محسّن مع جميع الميزات"""
    
    def __init__(self):
        self.constraint_validator = ConstraintValidator()
        self.min_number = Config.MIN_NUMBER
        self.max_number = Config.MAX_NUMBER
    
    def validate_numbers(self, text: str) -> List[int]:
        """التحقق من الأرقام المدخلة واستخراجها"""
        if not text or not text.strip():
            return []
        
        numbers = []
        number_pattern = r'\d+'
        matches = re.findall(number_pattern, text)
        
        for match in matches:
            try:
                num = int(match)
                if self.min_number <= num <= self.max_number:
                    numbers.append(num)
            except ValueError:
                continue
        
        numbers = sorted(list(set(numbers)))
        
        return numbers
    
    def validate_with_constraints(self, text: str, constraints: Dict = None) -> Tuple[List[int], List[str]]:
        """التحقق مع فحص القيود"""
        numbers = self.validate_numbers(text)
        
        if not numbers and text.strip():
            return [], ["❌ لا توجد أرقام صالحة"]
        
        if constraints:
            is_valid, constraint_issues = self.constraint_validator.validate_constraints(
                constraints, len(numbers) if numbers else Config.DEFAULT_TICKET_SIZE
            )
            
            if not is_valid:
                return numbers, constraint_issues
            
            if numbers:
                ticket_issues = self._validate_ticket_against_constraints(numbers, constraints)
                if ticket_issues:
                    return numbers, ticket_issues
        
        return numbers, []
    
    def _validate_ticket_against_constraints(self, ticket: List[int], constraints: Dict) -> List[str]:
        """التحقق من تطابق التذكرة مع القيود"""
        issues = []
        
        if 'sum_range' in constraints:
            min_sum, max_sum = constraints['sum_range']
            ticket_sum = sum(ticket)
            if not (min_sum <= ticket_sum <= max_sum):
                issues.append(f"❌ المجموع {ticket_sum} خارج النطاق {min_sum}-{max_sum}")
        
        if 'odd' in constraints:
            odd_count = sum(1 for n in ticket if n % 2)
            if odd_count != constraints['odd']:
                issues.append(f"❌ عدد الفردي {odd_count} لا يساوي المطلوب {constraints['odd']}")
        
        if 'consecutive' in constraints:
            sorted_ticket = sorted(ticket)
            consec_count = sum(1 for i in range(len(sorted_ticket)-1) 
                             if sorted_ticket[i+1] - sorted_ticket[i] == 1)
            if consec_count != constraints['consecutive']:
                issues.append(f"❌ المتتاليات {consec_count} لا تساوي المطلوب {constraints['consecutive']}")
        
        return issues
