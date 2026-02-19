"""
=============================================================================
🧠 نظام Machine Learning متقدم للتنبؤ
=============================================================================
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
from itertools import chain   # ✅ إضافة استيراد chain المفقود
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

from config.settings import Config
from utils.logger import logger
from utils.performance import PerformanceBenchmark


class LotteryPredictor:
    """نظام تنبؤ متقدم باستخدام Machine Learning"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.is_trained = False
        self.benchmark = PerformanceBenchmark()
        
        self._initialize_models()
    
    def _initialize_models(self):
        """تهيئة نماذج ML مختلفة"""
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        for model_name in self.models:
            self.scalers[model_name] = StandardScaler()
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """إعداد features متقدمة من البيانات"""
        operation_id = logger.start_operation('feature_preparation', {
            'total_draws': len(df),
            'models_count': len(self.models)
        })
        
        features_list = []
        labels_list = []
        
        try:
            for i in range(len(df) - 2):
                current = df.iloc[i]['numbers']
                next_draw = df.iloc[i + 1]['numbers']
                future_draw = df.iloc[i + 2]['numbers']
                
                # ✅ chain مستوردة الآن بشكل صحيح
                freq_counter = Counter(list(chain.from_iterable(df.iloc[:i+1]['numbers'])))
                
                basic_features = [
                    *sorted(current),
                    sum(current),
                    sum(1 for n in current if n % 2),
                    sum(1 for j in range(len(current)-1) if current[j+1] - current[j] == 1),
                    current[-1] - current[0] if current else 0,
                    float(np.mean(current)),
                    float(np.std(current))
                ]
                
                statistical_features = [
                    float(np.mean([freq_counter.get(n, 0) for n in current])),
                    float(np.std([freq_counter.get(n, 0) for n in current])),
                    len(set(current) & set(next_draw)),
                ]
                
                pattern_features = [
                    len(set([n % 10 for n in current])),
                    sum(1 for n in current if self._is_prime(n)),
                    self._calculate_balance(current)
                ]
                
                feature_vector = basic_features + statistical_features + pattern_features
                features_list.append(feature_vector)
                
                # Label: هل يظهر الرقم في السحب المستقبلي؟
                # ✅ إصلاح: نضيف label واحد فقط لكل صف (نتوقع المجموع الكلي)
                # وليس 32 label لكل صف (كان يسبب عدم تطابق بين X و y)
                for num in range(1, Config.MAX_NUMBER + 1):
                    label = 1 if num in future_draw else 0
                    labels_list.append(label)
            
            # ✅ يجب أن يكون X بحجم (n_samples * 32) و y بحجم (n_samples * 32)
            # نكرر كل صف features 32 مرة ليتطابق مع 32 label
            if features_list:
                expanded_features = []
                for fv in features_list:
                    for _ in range(Config.MAX_NUMBER):
                        expanded_features.append(fv)
                features_array = np.array(expanded_features)
            else:
                features_array = np.array([]).reshape(0, 0)
            
            labels_array = np.array(labels_list)
            
            logger.end_operation(operation_id, 'completed', {
                'features_shape': features_array.shape,
                'labels_shape': labels_array.shape,
            })
            
            return features_array, labels_array
            
        except Exception as e:
            logger.end_operation(operation_id, 'failed', {'error': str(e)})
            raise
    
    def train(self, df: pd.DataFrame, model_name: str = 'random_forest') -> Dict:
        """تدريب النموذج المحدد"""
        operation_id = logger.start_operation('model_training', {
            'model': model_name,
            'data_size': len(df)
        })
        
        try:
            self.benchmark.start_monitoring(f'train_{model_name}')
            
            X, y = self.prepare_features(df)
            
            if X.shape[0] < 10:
                raise ValueError(f"بيانات غير كافية للتدريب: {X.shape[0]} عينة فقط")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            X_train_scaled = self.scalers[model_name].fit_transform(X_train)
            X_test_scaled = self.scalers[model_name].transform(X_test)
            
            model = self.models[model_name]
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = model.feature_importances_
            
            self._save_model(model_name)
            
            metrics = self.benchmark.stop_monitoring(f'train_{model_name}')
            
            result = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'cv_scores': cv_scores.tolist(),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'feature_importance': self.feature_importance.get(model_name, np.array([])).tolist()
            }
            
            logger.end_operation(operation_id, 'completed', {
                'accuracy': round(accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'cv_mean': round(float(cv_scores.mean()), 4),
                **metrics
            })
            
            logger.log_prediction(
                model_name=model_name,
                accuracy=accuracy,
                confidence=precision,
                features_used=[f'feature_{i}' for i in range(X.shape[1])]
            )
            
            self.is_trained = True
            
            return result
            
        except Exception as e:
            logger.end_operation(operation_id, 'failed', {'error': str(e)})
            raise
    
    def predict(self, current_numbers: List[int], df: pd.DataFrame, 
                top_n: int = 10, model_name: str = 'random_forest') -> List[Tuple[int, float]]:
        """التنبؤ بالأرقام التالية"""
        if not self.is_trained or model_name not in self.models:
            raise ValueError(f"النموذج {model_name} غير مدرب")
        
        operation_id = logger.start_operation('prediction', {
            'model': model_name,
            'current_numbers': current_numbers,
            'top_n': top_n
        })
        
        try:
            self.benchmark.start_monitoring(f'predict_{model_name}')
            
            feature_vector = self._prepare_single_features(current_numbers, df)
            scaled_features = self.scalers[model_name].transform([feature_vector])
            
            predictions = []
            model = self.models[model_name]
            
            for num in range(1, Config.MAX_NUMBER + 1):
                if num in current_numbers:
                    continue
                
                try:
                    prob = model.predict_proba(scaled_features)[0][1]
                except (IndexError, AttributeError):
                    prob = 0.5
                
                predictions.append((num, float(prob)))
            
            predictions.sort(key=lambda x: x[1], reverse=True)
            top_predictions = predictions[:top_n]
            
            metrics = self.benchmark.stop_monitoring(f'predict_{model_name}')
            
            logger.end_operation(operation_id, 'completed', {
                'top_predictions': top_predictions,
                'highest_probability': top_predictions[0][1] if top_predictions else 0,
                **metrics
            })
            
            return top_predictions
            
        except Exception as e:
            logger.end_operation(operation_id, 'failed', {'error': str(e)})
            raise
    
    def _prepare_single_features(self, numbers: List[int], df: pd.DataFrame) -> np.ndarray:
        """تحضير features لسحب واحد"""
        sorted_nums = sorted(numbers)
        
        basic_features = [
            *sorted_nums,
            sum(numbers),
            sum(1 for n in numbers if n % 2),
            sum(1 for i in range(len(sorted_nums)-1) if sorted_nums[i+1] - sorted_nums[i] == 1),
            sorted_nums[-1] - sorted_nums[0] if len(sorted_nums) > 1 else 0,
            float(np.mean(numbers)),
            float(np.std(numbers))
        ]
        
        # ✅ chain مستوردة بشكل صحيح
        freq_counter = Counter(list(chain.from_iterable(df['numbers'])))
        statistical_features = [
            float(np.mean([freq_counter.get(n, 0) for n in numbers])),
            float(np.std([freq_counter.get(n, 0) for n in numbers])),
            0.0
        ]
        
        pattern_features = [
            len(set([n % 10 for n in numbers])),
            sum(1 for n in numbers if self._is_prime(n)),
            self._calculate_balance(numbers)
        ]
        
        return np.array(basic_features + statistical_features + pattern_features)
    
    def _is_prime(self, n: int) -> bool:
        """التحقق إذا كان الرقم أولياً"""
        if n < 2:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
    
    def _calculate_balance(self, numbers: List[int]) -> float:
        """حساب درجة التوازن"""
        if len(numbers) < 2:
            return 1.0
        
        first_half = sum(1 for n in numbers if n <= 16)
        second_half = len(numbers) - first_half
        balance = 1 - abs(first_half - second_half) / len(numbers)
        
        return balance
    
    def _save_model(self, model_name: str):
        """حفظ النموذج للاستخدام المستقبلي"""
        # ✅ Config.MODELS_DIR موجود الآن في settings.py
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        
        model_path = os.path.join(Config.MODELS_DIR, f'{model_name}.pkl')
        scaler_path = os.path.join(Config.MODELS_DIR, f'{model_name}_scaler.pkl')
        
        joblib.dump(self.models[model_name], model_path)
        joblib.dump(self.scalers[model_name], scaler_path)
        
        logger.logger.info(f"💾 تم حفظ النموذج {model_name}")
    
    def load_model(self, model_name: str):
        """تحميل نموذج محفوظ"""
        model_path = os.path.join(Config.MODELS_DIR, f'{model_name}.pkl')
        scaler_path = os.path.join(Config.MODELS_DIR, f'{model_name}_scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.models[model_name] = joblib.load(model_path)
            self.scalers[model_name] = joblib.load(scaler_path)
            self.is_trained = True
            logger.logger.info(f"✅ تم تحميل النموذج {model_name}")
        else:
            raise FileNotFoundError(f"ملفات النموذج {model_name} غير موجودة")
    
    def ensemble_predict(self, current_numbers: List[int], df: pd.DataFrame,
                        top_n: int = 10) -> List[Tuple[int, float]]:
        """تنبؤ باستخدام Ensemble من عدة نماذج"""
        all_predictions = []
        
        for model_name in self.models:
            try:
                predictions = self.predict(current_numbers, df, top_n=20, model_name=model_name)
                all_predictions.append(predictions)
            except Exception as e:
                logger.logger.warning(f"فشل التنبؤ بالنموذج {model_name}: {e}")
                continue
        
        if not all_predictions:
            return []
        
        combined_scores: Counter = Counter()
        
        for predictions in all_predictions:
            for num, prob in predictions:
                combined_scores[num] += prob
        
        for num in combined_scores:
            combined_scores[num] /= len(all_predictions)
        
        final_predictions = combined_scores.most_common(top_n)
        
        return [(num, float(score)) for num, score in final_predictions]


class RecommendationEngine:
    """نظام توصيات ذكي يعتمد على تعلم تفضيلات المستخدم"""
    
    def __init__(self):
        self.user_profiles: Dict[str, Dict] = {}
        self.collaborative_matrix = None
    
    def learn_preferences(self, user_id: str, selected_tickets: List[List[int]], 
                         rejected_tickets: List[List[int]] = None):
        """تعلم تفضيلات المستخدم"""
        profile = {
            'selected_patterns': self._extract_patterns(selected_tickets),
            'preferred_numbers': self._get_common_numbers(selected_tickets),
            'avoided_numbers': self._get_common_numbers(rejected_tickets) if rejected_tickets else set(),
            'sum_preference': self._get_sum_preference(selected_tickets),
            'odd_even_preference': self._get_odd_even_preference(selected_tickets),
            'learning_strength': min(1.0, len(selected_tickets) / 10)
        }
        
        self.user_profiles[user_id] = profile
        
        logger.logger.info(f"🎯 تعلم تفضيلات المستخدم {user_id}", extra={
            'selected_tickets': len(selected_tickets),
            'preferred_numbers_count': len(profile['preferred_numbers']),
            'learning_strength': profile['learning_strength']
        })
    
    def recommend(self, user_id: str, base_tickets: List[List[int]], 
                 count: int = 5) -> List[List[int]]:
        """توليد توصيات مخصصة"""
        if user_id not in self.user_profiles:
            return base_tickets[:count]
        
        profile = self.user_profiles[user_id]
        recommendations = []
        
        for base_ticket in base_tickets[:max(10, count * 2)]:
            customized = self._customize_ticket(base_ticket, profile)
            if customized and customized not in recommendations:
                recommendations.append(customized)
                if len(recommendations) >= count:
                    break
        
        # إذا لم تكفِ التوصيات، أكمل من التذاكر الأساسية
        if len(recommendations) < count:
            for ticket in base_tickets:
                if ticket not in recommendations:
                    recommendations.append(ticket)
                    if len(recommendations) >= count:
                        break
        
        return recommendations[:count]
    
    def _extract_patterns(self, tickets: List[List[int]]) -> Dict:
        """استخراج الأنماط من التذاكر"""
        if not tickets:
            return {}
        
        patterns: Dict[str, list] = {
            'consecutive_range': [],
            'shadow_range': [],
            'sum_range': [],
            'odd_range': []
        }
        
        for ticket in tickets:
            patterns['consecutive_range'].append(
                sum(1 for i in range(len(ticket)-1) if ticket[i+1] - ticket[i] == 1)
            )
            patterns['shadow_range'].append(
                sum(1 for c in Counter([n % 10 for n in ticket]).values() if c > 1)
            )
            patterns['sum_range'].append(sum(ticket))
            patterns['odd_range'].append(sum(1 for n in ticket if n % 2))
        
        result = {}
        for key in patterns:
            vals = patterns[key]
            if vals:
                result[key] = {
                    'min': min(vals),
                    'max': max(vals),
                    'avg': float(np.mean(vals))
                }
            else:
                result[key] = {'min': 0, 'max': 0, 'avg': 0.0}
        
        return result
    
    def _get_common_numbers(self, tickets: List[List[int]]) -> Set[int]:
        """الحصول على الأرقام المشتركة"""
        if not tickets:
            return set()
        
        counter: Counter = Counter()
        for ticket in tickets:
            counter.update(ticket)
        
        threshold = len(tickets) * 0.3
        return {num for num, count in counter.items() if count >= threshold}
    
    def _get_sum_preference(self, tickets: List[List[int]]) -> Dict:
        """تحديد تفضيل المجموع"""
        if not tickets:
            return {'min': 20, 'max': 200, 'avg': 100.0, 'std': 0.0}
        
        sums = [sum(t) for t in tickets]
        return {
            'min': min(sums),
            'max': max(sums),
            'avg': float(np.mean(sums)),
            'std': float(np.std(sums))
        }
    
    def _get_odd_even_preference(self, tickets: List[List[int]]) -> Dict:
        """تحديد تفضيل الفردي/الزوجي"""
        if not tickets:
            return {'min_odd': 0, 'max_odd': 6, 'avg_odd': 3.0, 'preferred_odd': 3}
        
        odd_counts = [sum(1 for n in t if n % 2) for t in tickets]
        return {
            'min_odd': min(odd_counts),
            'max_odd': max(odd_counts),
            'avg_odd': float(np.mean(odd_counts)),
            'preferred_odd': int(np.round(np.mean(odd_counts)))
        }
    
    def _customize_ticket(self, base_ticket: List[int], profile: Dict) -> List[int]:
        """تخصيص التذكرة بناءً على التفضيلات"""
        ticket = base_ticket.copy()
        
        preferred = profile.get('preferred_numbers', set())
        avoided = profile.get('avoided_numbers', set())
        
        for i in range(len(ticket)):
            if ticket[i] in avoided and preferred:
                for pref_num in preferred:
                    if pref_num not in ticket and Config.MIN_NUMBER <= pref_num <= Config.MAX_NUMBER:
                        ticket[i] = pref_num
                        break
        
        odd_pref = profile.get('odd_even_preference', {})
        target_odd = odd_pref.get('preferred_odd', 3)
        current_odd = sum(1 for n in ticket if n % 2)
        
        if current_odd > target_odd:
            odd_indices = [i for i, n in enumerate(ticket) if n % 2]
            changes_needed = current_odd - target_odd
            
            for idx in odd_indices[:changes_needed]:
                candidate = ticket[idx] + 1
                if candidate > Config.MAX_NUMBER:
                    candidate = ticket[idx] - 1
                if Config.MIN_NUMBER <= candidate <= Config.MAX_NUMBER and candidate not in ticket:
                    ticket[idx] = candidate
        
        elif current_odd < target_odd:
            even_indices = [i for i, n in enumerate(ticket) if n % 2 == 0]
            changes_needed = target_odd - current_odd
            
            for idx in even_indices[:changes_needed]:
                candidate = ticket[idx] + 1
                if candidate > Config.MAX_NUMBER:
                    candidate = ticket[idx] - 1
                if Config.MIN_NUMBER <= candidate <= Config.MAX_NUMBER and candidate not in ticket:
                    ticket[idx] = candidate
        
        # ضبط المجموع
        sum_pref = profile.get('sum_preference', {})
        current_sum = sum(ticket)
        target_sum = int(sum_pref.get('avg', current_sum))
        
        if abs(current_sum - target_sum) > 10 and len(ticket) > 0:
            diff = target_sum - current_sum
            adjustment_per_num = diff // len(ticket)
            
            if abs(adjustment_per_num) > 0:
                for i in range(len(ticket)):
                    new_val = ticket[i] + adjustment_per_num
                    if Config.MIN_NUMBER <= new_val <= Config.MAX_NUMBER and new_val not in ticket:
                        ticket[i] = new_val
        
        # التأكد من عدم وجود تكرار وأن الأرقام ضمن النطاق
        ticket = sorted(list(set([
            n for n in ticket 
            if Config.MIN_NUMBER <= n <= Config.MAX_NUMBER
        ])))
        
        return ticket
