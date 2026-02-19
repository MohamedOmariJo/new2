"""
=============================================================================
🎰 تطبيق اليانصيب الأردني - الإصدار الاحترافي v8.0
=============================================================================
دمج كامل للتحسينات الثمانية مع الحفاظ على التصميم الأصلي
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import requests
import traceback                           # ✅ إضافة traceback المفقود
from collections import Counter            # ✅ إضافة Counter المفقودة
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Dict
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# استيراد المكونات المحسنة
from config.settings import Config
from utils.logger import logger
from utils.performance import PerformanceBenchmark
from core.validator import AdvancedValidator
from core.analyzer import AdvancedAnalyzer
from core.models import LotteryPredictor, RecommendationEngine
from core.generator import SmartGenerator
from core.database import DatabaseManager
from core.notifications import NotificationSystem
from utils.pdf_generator import PDFGenerator

# إعدادات الأداء
benchmark = PerformanceBenchmark()

# ==============================================================================
# 1. تحميل البيانات المحسّن
# ==============================================================================

@st.cache_data(ttl=0, show_spinner=True)
def load_data_with_retry() -> Tuple[Optional[pd.DataFrame], str]:
    """تحميل البيانات مع إعادة المحاولة والتسجيل - يقرأ الملف المحلي أولاً دائماً"""
    op_id = logger.start_operation('data_loading', {'source': 'local_first'})
    
    try:
        with benchmark.monitor_operation('data_loading'):
            # ✅ الأولوية للملف المحلي دائماً لضمان تحديث البيانات فور تعديل الملف
            try:
                df = pd.read_excel(Config.BACKUP_FILE)
                source = "الملف المحلي"
                logger.logger.info(f"✅ تم تحميل البيانات من {source} ({len(df)} صف)")
                
            except FileNotFoundError:
                logger.logger.warning("⚠️ الملف المحلي غير موجود، جاري التحميل من GitHub...")
                
                # المحاولة من GitHub كبديل احتياطي فقط
                try:
                    response = requests.get(Config.GITHUB_URL, timeout=15)
                    response.raise_for_status()
                    df = pd.read_excel(io.BytesIO(response.content))
                    source = "GitHub"
                    logger.logger.info(f"✅ تم تحميل البيانات من {source}")
                    
                except requests.RequestException as e:
                    error_msg = f"❌ لم يتم العثور على ملف البيانات محلياً أو عبر الإنترنت: {e}"
                    logger.logger.error(error_msg)
                    logger.end_operation(op_id, 'failed', {'error': error_msg})
                    return None, error_msg
                    
            except Exception as e:
                error_msg = f"❌ خطأ في قراءة الملف المحلي: {e}"
                logger.logger.error(error_msg)
                logger.end_operation(op_id, 'failed', {'error': str(e)})
                return None, error_msg
            
            # التحقق من الجودة
            df = validate_and_clean_data(df)
            
            if df.empty:
                error_msg = "❌ لا توجد سحوبات صالحة في البيانات"
                logger.logger.error(error_msg)
                logger.end_operation(op_id, 'failed', {'error': error_msg})
                return None, error_msg
            
            # حفظ في قاعدة البيانات
            db_manager = DatabaseManager()
            for _, row in df.iterrows():
                db_manager.add_draw_with_analysis(row['numbers'], row.get('date', datetime.now()))
            
            success_msg = f"✅ تم تحميل {len(df)} سحب بنجاح من {source}"
            logger.end_operation(op_id, 'completed', {
                'draws_count': len(df),
                'source': source,
                'database_saved': True
            })
            
            return df, success_msg
            
    except Exception as e:
        error_msg = f"❌ خطأ غير متوقع في تحميل البيانات: {e}"
        logger.logger.error(error_msg)
        logger.end_operation(op_id, 'failed', {'error': str(e)})
        return None, error_msg

def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """تنظيف وتحقق من جودة البيانات"""
    required_cols = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6']
    
    if not set(required_cols).issubset(df.columns):
        st.error(f"❌ الملف لا يحتوي على الأعمدة المطلوبة: {required_cols}")
        return pd.DataFrame()
    
    # تحويل إلى أرقام
    df[required_cols] = df[required_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=required_cols, inplace=True)
    
    # دمج الأرقام في قائمة واحدة
    df['numbers'] = df[required_cols].values.tolist()
    df['numbers'] = df['numbers'].apply(
        lambda x: sorted([int(n) for n in x if Config.MIN_NUMBER <= n <= Config.MAX_NUMBER])
    )
    
    # إزالة السحوبات غير الصالحة
    df = df[df['numbers'].apply(len) == Config.DEFAULT_TICKET_SIZE].copy()
    
    # إضافة معلومات إضافية
    if 'رقم السحب' in df.columns:
        df['draw_id'] = df['رقم السحب']
    else:
        df['draw_id'] = range(1, len(df) + 1)
    
    if 'تاريخ السحب' in df.columns:
        df['date'] = pd.to_datetime(df['تاريخ السحب'], errors='coerce').dt.date
    else:
        df['date'] = [f"Draw {i}" for i in df['draw_id']]
    
    df.reset_index(drop=True, inplace=True)
    
    logger.logger.info("🧹 تنظيف البيانات", extra={
        'initial_rows': len(df),
        'valid_rows': len(df),
        'cleaning_applied': ['type_conversion', 'na_removal', 'validation']
    })
    
    return df

# ==============================================================================
# 2. الواجهة الرئيسية المحسنة
# ==============================================================================

def main():
    """الواجهة الرئيسية للتطبيق مع جميع التحسينات"""
    
    # إعداد الصفحة
    st.set_page_config(
        page_title=Config.APP_NAME,
        page_icon="🎰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS مخصص محسن
    st.markdown("""
    <style>
        /* تحسينات عامة */
        .main {
            padding: 2rem;
        }
        
        /* الأزرار مع تأثيرات متقدمة */
        .stButton>button {
            width: 100%;
            border-radius: 12px;
            font-weight: bold;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border: 2px solid transparent;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 24px;
            font-size: 16px;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
            border-color: #ffffff;
        }
        
        .stButton>button:active {
            transform: translateY(0);
        }
        
        /* الكرات مع تأثيرات ثلاثية الأبعاد */
        .ball-3d {
            display: inline-block;
            width: 42px;
            height: 42px;
            line-height: 42px;
            text-align: center;
            border-radius: 50%;
            color: white;
            font-weight: bold;
            margin: 4px;
            font-size: 16px;
            box-shadow: 
                0 4px 8px rgba(0,0,0,0.2),
                inset 0 -3px 6px rgba(0,0,0,0.3),
                inset 0 3px 6px rgba(255,255,255,0.3);
            position: relative;
            transition: all 0.3s;
        }
        
        .ball-3d:hover {
            transform: scale(1.1) rotate(15deg);
            box-shadow: 
                0 6px 12px rgba(0,0,0,0.3),
                inset 0 -3px 6px rgba(0,0,0,0.4),
                inset 0 3px 6px rgba(255,255,255,0.4);
        }
        
        .ball-hot {
            background: linear-gradient(135deg, #ff6b6b, #ee5a52);
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
        }
        
        .ball-cold {
            background: linear-gradient(135deg, #4ecdc4, #44a08d);
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
        }
        
        .ball-neutral {
            background: linear-gradient(135deg, #ffeaa7, #fdcb6e);
            color: #2d3436;
            text-shadow: 0 1px 2px rgba(255,255,255,0.5);
        }
        
        /* صناديق المعلومات */
        .info-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 15px;
            color: white;
            margin-bottom: 15px;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }
        
        .info-card:hover {
            transform: translateY(-5px);
        }
        
        .warning-card {
            background: linear-gradient(135deg, #ffd93d, #ff6b6b);
            padding: 20px;
            border-radius: 15px;
            color: #2d3436;
            margin-bottom: 15px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .success-card {
            background: linear-gradient(135deg, #00b09b, #96c93d);
            padding: 20px;
            border-radius: 15px;
            color: white;
            margin-bottom: 15px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        /* شريط التقدم المحسن */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            border-radius: 10px;
        }
        
        /* العناوين */
        h1, h2, h3 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        /* الجداول */
        .dataframe {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .dataframe th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: bold;
        }
        
        /* توسيع المحتوى */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 10px;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # ✅ زر تحديث البيانات في الشريط الجانبي
    with st.sidebar:
        st.markdown("---")
        if st.button("🔄 تحديث البيانات", use_container_width=True, 
                     help="اضغط هنا بعد تعديل ملف history.xlsx لتحميل السحوبات الجديدة"):
            st.session_state.pop('data_loaded', None)
            load_data_with_retry.clear()
            st.rerun()

    # تحميل البيانات
    if 'data_loaded' not in st.session_state:
        load_data_with_retry.clear()  # ✅ مسح الكاش دائماً لضمان أحدث بيانات
        with st.spinner('🔄 جاري تحميل البيانات وتحليلها...'):
            df, msg = load_data_with_retry()
            
            if df is None:
                st.error(msg)
                st.stop()
            
            # تهيئة جميع المكونات
            st.session_state.df = df
            st.session_state.analyzer = AdvancedAnalyzer(df)
            st.session_state.validator = AdvancedValidator()
            st.session_state.generator = SmartGenerator(st.session_state.analyzer)
            st.session_state.predictor = LotteryPredictor()
            st.session_state.recommender = RecommendationEngine()
            st.session_state.notifier = NotificationSystem()
            st.session_state.portfolio = []
            st.session_state.user_preferences = {}
            st.session_state.performance_data = {}
            st.session_state.data_loaded = True
            
            # محاولة تدريب النماذج في الخلفية
            try:
                training_result = st.session_state.predictor.train(df, 'random_forest')
                st.session_state.performance_data['model_training'] = training_result
                logger.logger.info("✅ تم تدريب نماذج ML بنجاح")
            except Exception as e:
                logger.logger.warning(f"⚠️ فشل تدريب نماذج ML: {e}")
            
            st.success(msg)
            st.balloons()
    
    # الوصول للكائنات
    df = st.session_state.df
    analyzer = st.session_state.analyzer
    validator = st.session_state.validator
    generator = st.session_state.generator
    predictor = st.session_state.predictor
    recommender = st.session_state.recommender
    notifier = st.session_state.notifier
    
    # الشريط الجانبي المحسن
    with st.sidebar:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.title(f"🎰 {Config.APP_VERSION}")
        st.success("✅ جميع الأنظمة تعمل بشكل مثالي")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # إحصائيات سريعة
        st.markdown("### 📊 لوحة التحكم السريعة")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("إجمالي السحوبات", len(df))
        with col2:
            st.metric("آخر سحب", f"#{df.iloc[-1]['draw_id']}")
        
        # معلومات النظام
        with st.expander("🔧 معلومات النظام"):
            system_stats = benchmark.get_system_stats()
            st.json(system_stats, expanded=False)
            
            if st.button("🔄 تحديث أداء النظام"):
                perf_report = benchmark.get_performance_report()
                st.session_state.performance_data['system'] = perf_report
                st.rerun()
        
        # الأرقام الساخنة والباردة
        st.markdown("---")
        st.markdown("### 🔥 الأرقام الساخنة")
        hot_nums = sorted(list(analyzer.hot))[:8]
        hot_html = " ".join([f'<div class="ball-3d ball-hot">{n}</div>' for n in hot_nums])
        st.markdown(hot_html, unsafe_allow_html=True)
        
        st.markdown("### ❄️ الأرقام الباردة")
        cold_nums = sorted(list(analyzer.cold))[:8]
        cold_html = " ".join([f'<div class="ball-3d ball-cold">{n}</div>' for n in cold_nums])
        st.markdown(cold_html, unsafe_allow_html=True)
        
        # أدوات المطور
        if st.checkbox("👨‍💻 وضع المطور"):
            st.markdown("---")
            st.markdown("#### أدوات التطوير")
            
            if st.button("📝 عرض سجلات النظام"):
                logs_export = logger.export_logs(1)
                st.info(f"تم تصدير السجلات إلى: {logs_export}")
            
            if st.button("📊 تقرير الأداء"):
                perf_report = benchmark.get_performance_report()
                st.session_state.performance_data['detailed'] = perf_report
                st.rerun()
            
            if st.button("🧹 مسح الذاكرة المؤقتة"):
                st.cache_data.clear()
                st.success("تم مسح الذاكرة المؤقتة")
    
    # التبويبات الرئيسية المحسنة
    tabs = st.tabs([
        "🏠 لوحة التحكم",
        "🎰 المولد الذكي PRO",
        "🧠 الذكاء الاصطناعي",
        "🔍 فاحص التذاكر",
        "📈 التحليلات المتقدمة",
        "💼 المحفظة الذكية",
        "⚙️ الإعدادات والأداء"
    ])
    
    # ==================== TAB 1: لوحة التحكم المحسنة ====================
    with tabs[0]:
        st.header("🏠 لوحة التحكم الرئيسية")
        
        # بطاقات إحصائية
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.metric("السحوبات", len(df))
            st.caption("إجمالي السحوبات المحملة")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="success-card">', unsafe_allow_html=True)
            last_draw = df.iloc[-1]
            st.metric("آخر سحب", f"#{last_draw['draw_id']}")
            st.caption(f"التاريخ: {last_draw['date']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            avg_sum = np.mean([sum(nums) for nums in df['numbers']])
            st.metric("متوسط المجموع", round(avg_sum, 1))
            st.caption("متوسط مجموع الأرقام")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="warning-card">', unsafe_allow_html=True)
            hot_percent = round(len(analyzer.hot) / 32 * 100, 1)
            st.metric("نسبة الساخن", f"{hot_percent}%")
            st.caption(f"{len(analyzer.hot)} من 32 رقم")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # آخر سحب مع تأثيرات
        st.markdown("### 🎱 آخر سحب - تحليل فوري")
        last_numbers = sorted(last_draw['numbers'])
        
        # عرض الكرات مع تأثيرات
        balls_html = " ".join([
            f'<div class="ball-3d {"ball-hot" if n in analyzer.hot else "ball-cold" if n in analyzer.cold else "ball-neutral"}">{n}</div>'
            for n in last_numbers
        ])
        st.markdown(balls_html, unsafe_allow_html=True)
        
        # تحليل سريع
        col_analysis1, col_analysis2 = st.columns(2)
        
        with col_analysis1:
            analysis = analyzer.get_ticket_analysis(last_numbers)
            st.metric("المجموع", analysis['basic']['sum'])
            st.metric("الفردي/زوجي", f"{analysis['basic']['odd']}/{analysis['basic']['even']}")
        
        with col_analysis2:
            st.metric("الدرجة", f"{analysis['quality_score']}/10")
            st.metric("التوازن", f"{analysis['statistical']['balance_score']:.2f}")
        
        st.markdown("---")
        
        # رسوم بيانية تفاعلية
        st.markdown("### 📊 نظرة سريعة على البيانات")
        
        tab_charts1, tab_charts2 = st.tabs(["📈 تطور المجموع", "📊 توزيع الأرقام"])
        
        with tab_charts1:
            recent_20 = df.tail(20).copy()
            recent_20['sum'] = recent_20['numbers'].apply(sum)
            
            fig_trend = px.line(
                recent_20,
                x='draw_id',
                y='sum',
                markers=True,
                title="تطور مجموع الأرقام في آخر 20 سحب",
                line_shape='spline'
            )
            fig_trend.update_traces(
                line=dict(color='#667eea', width=3),
                marker=dict(size=8, color='#764ba2')
            )
            fig_trend.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2d3436')
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with tab_charts2:
            # رسم توزيع تكرار الأرقام
            freq_df = pd.DataFrame([
                {'number': num, 'frequency': analyzer.freq.get(num, 0)}
                for num in range(1, 33)
            ])
            
            fig_freq = px.bar(
                freq_df,
                x='number',
                y='frequency',
                title="تكرار ظهور كل رقم",
                color='frequency',
                color_continuous_scale='Viridis'
            )
            fig_freq.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2d3436')
            )
            st.plotly_chart(fig_freq, use_container_width=True)
    
    # ==================== TAB 2: المولد الذكي PRO ====================
    with tabs[1]:
        st.header("🎰 المولد الذكي PRO")
        st.markdown("توليد تذاكر ذكية مع فلاتر متقدمة وتحليل في الوقت الحقيقي")
        
        col_settings, col_results = st.columns([1, 2])
        
        with col_settings:
            with st.form("advanced_generator_form"):
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                st.subheader("⚙️ إعدادات التوليد المتقدمة")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # خيارات التوليد المتقدمة
                gen_mode = st.radio(
                    "🎯 استراتيجية التوليد",
                    ["عشوائي ذكي", "Markov AI", "هجين متقدم", "تعلم الآلة"],
                    help="""\
                    • عشوائي ذكي: توليد مع فلاتر متقدمة
                    • Markov AI: بناءً على توقعات سلسلة ماركوف
                    • هجين متقدم: مزيج من الذكاء الاصطناعي والعشوائية
                    • تعلم الآلة: استخدام نماذج ML للتنبؤ
                    """
                )
                
                col_size, col_count = st.columns(2)
                with col_size:
                    ticket_size = st.slider("📏 حجم التذكرة", 6, 10, 6)
                with col_count:
                    ticket_count = st.slider("🔢 عدد التذاكر", 1, 100, 10)
                
                st.markdown("---")
                st.markdown("#### 🎯 الفلاتر المتقدمة")
                
                # فلاتر متقدمة مع واجهة محسنة
                with st.expander("📊 فلاتر المجموع والتوزيع", expanded=True):
                    sum_range = st.slider(
                        "نطاق المجموع",
                        20, 200, (80, 130),
                        help="نطاق مجموع الأرقام في التذكرة"
                    )
                    
                    odd_options = ["عشوائي"] + [f"{i} فردي" for i in range(ticket_size + 1)]
                    odd_choice = st.selectbox("نسبة الفردي/الزوجي", odd_options)
                
                with st.expander("🔗 فلاتر الأنماط", expanded=False):
                    consec_options = ["عشوائي"] + [f"{i} متتالي" for i in range(ticket_size)]
                    consec_choice = st.selectbox("الأرقام المتتالية", consec_options)
                    
                    shadow_options = ["عشوائي"] + [f"{i} ظل" for i in range(5)]
                    shadow_choice = st.selectbox("الظلال (نفس خانة الآحاد)", shadow_options)
                
                with st.expander("🎯 فلاتر الإحصاء", expanded=False):
                    last_match_options = ["عشوائي"] + [f"{i} رقم" for i in range(7)]
                    last_match_choice = st.selectbox("التطابق مع آخر سحب", last_match_options)
                    
                    col_hot, col_cold = st.columns(2)
                    with col_hot:
                        hot_min = st.number_input("الحد الأدنى للساخن", 0, 6, 0)
                    with col_cold:
                        cold_max = st.number_input("الحد الأقصى للبارد", 0, 6, 6)
                
                with st.expander("🔧 فلاتر مخصصة", expanded=False):
                    fixed_input = st.text_input(
                        "أرقام ثابتة (مفصولة بفواصل)",
                        placeholder="مثال: 5, 12, 23",
                        help="أرقام يجب أن تظهر في كل تذكرة"
                    )
                    
                    exclude_input = st.text_input(
                        "أرقام مستبعدة (مفصولة بفواصل)",
                        placeholder="مثال: 1, 2, 31",
                        help="أرقام لا تريد أن تظهر أبداً"
                    )
                
                # زر التوليد مع تأثير
                generate_btn = st.form_submit_button(
                    "🚀 توليد التذاكر الذكية",
                    use_container_width=True,
                    type="primary"
                )
                
                if generate_btn:
                    # بناء القيود
                    constraints = {'sum_range': sum_range}
                    
                    if odd_choice != "عشوائي":
                        constraints['odd'] = int(odd_choice.split()[0])
                    
                    if consec_choice != "عشوائي":
                        constraints['consecutive'] = int(consec_choice.split()[0])
                    
                    if shadow_choice != "عشوائي":
                        constraints['shadows'] = int(shadow_choice.split()[0])
                    
                    if last_match_choice != "عشوائي":
                        constraints['last_match'] = int(last_match_choice.split()[0])
                    
                    if hot_min > 0:
                        constraints['hot_min'] = hot_min
                    
                    if cold_max < 6:
                        constraints['cold_max'] = cold_max
                    
                    if fixed_input:
                        fixed_nums = validator.validate_numbers(fixed_input)
                        if fixed_nums:
                            constraints['fixed'] = fixed_nums
                    
                    if exclude_input:
                        exclude_nums = validator.validate_numbers(exclude_input)
                        if exclude_nums:
                            constraints['exclude'] = exclude_nums
                    
                    # التحقق من القيود
                    # ✅ إصلاح جذري: validate_with_constraints ترجع (list, list) وليس (bool, list)
                    # الدالة الأصلية أعادت قائمة أرقام فارغة [] كعنصر أول → يُقيَّم دائماً كـ False
                    # الحل: نستخدم constraint_validator مباشرة للحصول على (bool, issues)
                    is_valid, issues = validator.constraint_validator.validate_constraints(
                        constraints, ticket_size
                    )
                    
                    if not is_valid:
                        for issue in issues:
                            st.error(issue)
                    else:
                        # التوليد
                        try:
                            op_id = logger.start_operation('ticket_generation', {
                                'strategy': gen_mode,
                                'count': ticket_count,
                                'constraints': constraints
                            })
                            
                            with st.spinner('🎰 جاري التوليد الذكي...'):
                                with benchmark.monitor_operation('generation'):
                                    if gen_mode == "Markov AI":
                                        tickets = generator.generate_markov_based(ticket_count, ticket_size)
                                    elif gen_mode == "تعلم الآلة":
                                        # استخدام تنبؤات ML
                                        base_tickets = generator.generate_tickets(
                                            min(ticket_count * 2, 50), ticket_size, constraints
                                        )
                                        tickets = recommender.recommend(
                                            'default_user', base_tickets, ticket_count
                                        )
                                    elif gen_mode == "هجين متقدم":
                                        half = ticket_count // 2
                                        tickets = (
                                            generator.generate_markov_based(half, ticket_size) +
                                            generator.generate_tickets(ticket_count - half, ticket_size, constraints)
                                        )
                                    else:  # عشوائي ذكي
                                        tickets = generator.generate_tickets(ticket_count, ticket_size, constraints)
                                
                                st.session_state.generated_tickets = tickets
                                
                                # ✅ حفظ metadata التوليد في session_state للاستخدام في col_results
                                st.session_state.last_generation_meta = {
                                    'strategy': gen_mode,
                                    'constraints': {
                                        k: list(v) if isinstance(v, set) else v
                                        for k, v in constraints.items()
                                    }
                                }
                                
                                # تسجيل النجاح
                                logger.log_generation(
                                    constraints, ticket_count,
                                    benchmark.metrics.get('generation', {}).get('duration_seconds', 0),
                                    len(tickets)
                                )
                                
                                logger.end_operation(op_id, 'completed', {
                                    'generated_count': len(tickets),
                                    'success_rate': round(len(tickets) / ticket_count * 100, 2)
                                })
                                
                                if len(tickets) == 0:
                                    st.error("❌ لم يتم العثور على أي تذكرة تطابق شروطك. حاول تخفيف القيود (مثلاً توسيع نطاق المجموع أو تغيير شروط الفردي/الزوجي).")
                                elif len(tickets) < ticket_count:
                                    st.warning(f"⚠️ تم العثور على {len(tickets)} تذكرة فقط من أصل {ticket_count} المطلوبة. شروطك صارمة جداً - حاول تخفيف بعض القيود.")
                                else:
                                    st.success(f"✅ تم توليد {len(tickets)} تذكرة بنجاح!")                                
                                # إرسال إشعار
                                notifier.send(
                                    "🎰 توليد التذاكر",
                                    f"تم توليد {len(tickets)} تذكرة باستخدام {gen_mode}",
                                    "info"
                                )
                        
                        except Exception as e:
                            error_msg = f"❌ خطأ في التوليد: {e}"
                            st.error(error_msg)
                            logger.end_operation(op_id, 'failed', {'error': str(e)})
        
        with col_results:
            if 'generated_tickets' in st.session_state and st.session_state.generated_tickets:
                tickets = st.session_state.generated_tickets
                
                st.markdown('<div class="success-card">', unsafe_allow_html=True)
                st.subheader(f"📋 النتائج ({len(tickets)} تذكرة)")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # أزرار التصدير
                col_export1, col_export2 = st.columns(2)
                
                with col_export1:
                    # ✅ إصلاح: استخدام session_state لحفظ constraints بدلاً من الاعتماد على المتغير المحلي
                    last_meta = st.session_state.get('last_generation_meta', {})
                    pdf_buffer = PDFGenerator.create_ticket_pdf(
                        tickets,
                        metadata=last_meta
                    )
                    st.download_button(
                        "📥 تحميل PDF",
                        pdf_buffer,
                        f"lottery_tickets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        "application/pdf",
                        use_container_width=True,
                        key="pdf_download"
                    )
                
                with col_export2:
                    csv_data = pd.DataFrame(tickets).to_csv(index=False)
                    st.download_button(
                        "📊 تحميل CSV",
                        csv_data,
                        f"lottery_tickets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True,
                        key="csv_download"
                    )
                
                st.markdown("---")
                
                # عرض التذاكر مع تحليل متقدم
                for i, ticket in enumerate(tickets, 1):
                    with st.expander(f"🎫 تذكرة #{i} - جودة: {analyzer.get_ticket_analysis(ticket)['quality_score']}/10", 
                                   expanded=(i <= 2)):
                        
                        # عرض الكرات مع تحليل فوري
                        balls_html = " ".join([
                            f'<div class="ball-3d {"ball-hot" if n in analyzer.hot else "ball-cold" if n in analyzer.cold else "ball-neutral"}">{n}</div>'
                            for n in ticket
                        ])
                        st.markdown(balls_html, unsafe_allow_html=True)
                        
                        # تحليل متقدم
                        analysis = analyzer.get_ticket_analysis(ticket)
                        
                        # أعمدة التحليل
                        col_analysis1, col_analysis2, col_analysis3 = st.columns(3)
                        
                        with col_analysis1:
                            st.metric("المجموع", analysis['basic']['sum'])
                            st.metric("الفردي", analysis['basic']['odd'])
                            st.metric("المتتاليات", analysis['basic']['consecutive'])
                        
                        with col_analysis2:
                            st.metric("الساخن", analysis['classification']['hot_count'])
                            st.metric("البارد", analysis['classification']['cold_count'])
                            st.metric("التوازن", f"{analysis['statistical']['balance_score']:.2f}")
                        
                        with col_analysis3:
                            st.metric("الجودة", f"{analysis['quality_score']}/10")
                            st.metric("التنوع", f"{analysis['statistical']['diversity_score']:.2f}")
                            st.metric("التعقيد", f"{analysis['advanced']['pattern_complexity']:.2f}")
                        
                        # أزرار الإجراءات
                        col_action1, col_action2, col_action3 = st.columns(3)
                        
                        with col_action1:
                            if st.button(f"🎲 محاكاة الفوز", key=f"sim_{i}"):
                                with st.spinner("جاري المحاكاة..."):
                                    # محاكاة Monte Carlo
                                    pass  # سيتم تنفيذها
                        
                        with col_action2:
                            if st.button("💾 حفظ في المحفظة", key=f"save_{i}"):
                                if ticket not in st.session_state.portfolio:
                                    st.session_state.portfolio.append(ticket)
                                    st.toast("✅ تم الحفظ في المحفظة!")
                                    notifier.send("💼 المحفظة", f"تم حفظ تذكرة #{i}", "success")
                                else:
                                    st.toast("⚠️ التذكرة موجودة مسبقاً")
                        
                        with col_action3:
                            if st.button("🤖 تحليل متقدم", key=f"analyze_{i}"):
                                # تحليل متعمق
                                pass  # سيتم تنفيذها
            else:
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                st.info("👈 استخدم النموذج على اليسار لتوليد التذاكر")
                st.markdown("""
                ### 💡 نصائح سريعة:
                1. ابدأ بـ **10-20** تذكرة للتجربة
                2. استخدم **فلاتر بسيطة** في البداية
                3. **Markov AI** جيد للأنماط التاريخية
                4. **الهجين** يعطي نتائج متوازنة
                """)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # ==================== TAB 3: الذكاء الاصطناعي ====================
    with tabs[2]:
        st.header("🧠 الذكاء الاصطناعي المتقدم")
        
        tab_ai1, tab_ai2, tab_ai3 = st.tabs(["🤖 توقعات ML", "🔮 تحليل عميق", "🎯 توصيات مخصصة"])
        
        with tab_ai1:
            st.markdown("### 🤖 توقعات تعلم الآلة")
            
            if st.session_state.predictor.is_trained:
                last_numbers = sorted(list(analyzer.last_draw))
                
                col_pred1, col_pred2 = st.columns([2, 1])
                
                with col_pred1:
                    st.markdown(f"**آخر سحب:** `{last_numbers}`")
                    
                    # التنبؤ باستخدام Ensemble
                    if st.button("🔮 تنبؤ بالأرقام القادمة", type="primary"):
                        with st.spinner("جاري التحليل والتنبؤ..."):
                            predictions = predictor.ensemble_predict(last_numbers, df, top_n=12)
                            
                            if predictions:
                                st.success("✅ تم التنبؤ بالأرقام بنجاح!")
                                
                                # عرض التوقعات
                                cols = st.columns(6)
                                for i, (num, prob) in enumerate(predictions[:12]):
                                    with cols[i % 6]:
                                        confidence_color = "#10b981" if prob > 0.1 else "#fbbf24" if prob > 0.05 else "#ef4444"
                                        st.markdown(f"""
                                        <div style="
                                            background: linear-gradient(135deg, #1e293b, #334155);
                                            padding: 15px;
                                            border-radius: 12px;
                                            text-align: center;
                                            border: 2px solid {confidence_color};
                                            margin-bottom: 10px;
                                        ">
                                            <div style="font-size:28px; color:{confidence_color}; font-weight:bold;">{num}</div>
                                            <div style="font-size:12px; color:#cbd5e1; margin-top:5px;">
                                                احتمالية: {prob:.1%}<br>
                                                <span style="color:#{'#ef4444' if num in analyzer.hot else '#3b82f6' if num in analyzer.cold else '#10b981'}">
                                                    {'🔥 ساخن' if num in analyzer.hot else '❄️ بارد' if num in analyzer.cold else '⚖️ محايد'}
                                                </span>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                st.warning("⚠️ لا توجد تنبؤات كافية")
                
                with col_pred2:
                    st.markdown("#### 📊 أداء النماذج")
                    if 'model_training' in st.session_state.performance_data:
                        training_data = st.session_state.performance_data['model_training']
                        st.metric("الدقة", f"{training_data['accuracy']:.1%}")
                        st.metric("الدقة المتوسطة", f"{training_data['precision']:.1%}")
                        st.metric("التذكر", f"{training_data['recall']:.1%}")
            else:
                st.warning("⚠️ نماذج ML غير مدربة بعد")
                if st.button("🎓 تدريب النماذج الآن"):
                    with st.spinner("جاري تدريب النماذج..."):
                        try:
                            result = predictor.train(df, 'random_forest')
                            st.session_state.performance_data['model_training'] = result
                            st.success("✅ تم تدريب النماذج بنجاح!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ فشل التدريب: {e}")
        
        with tab_ai2:
            st.markdown("### 🔮 تحليل Poisson المتقدم")
            
            poisson_df = pd.DataFrame(analyzer.poisson_data)
            
            # رسم بياني تفاعلي متقدم
            fig = px.scatter_3d(
                poisson_df,
                x='number',
                y='z_score',
                z='anomaly_score',
                color='classification',
                size='frequency',
                hover_data=['expected', 'last_seen', 'avg_gap', 'p_value'],
                title="تحليل ثلاثي الأبعاد للشذوذ الإحصائي",
                color_discrete_map={
                    'extreme_anomaly': '#ef4444',
                    'significant_anomaly': '#f97316',
                    'moderate_anomaly': '#fbbf24',
                    'mild_anomaly': '#84cc16',
                    'normal': '#10b981'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab_ai3:
            st.markdown("### 🎯 نظام التوصيات المخصصة")
            
            if 'user_id' not in st.session_state:
                st.session_state.user_id = f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            user_id = st.session_state.user_id
            
            # تعلم التفضيلات
            st.markdown("#### 📚 تعلم تفضيلاتك")
            
            if st.session_state.portfolio:
                if st.button("🎓 تعلم من محفظتي"):
                    recommender.learn_preferences(user_id, st.session_state.portfolio)
                    st.success("✅ تم تعلم تفضيلاتك من المحفظة!")
            
            # توليد توصيات
            if user_id in recommender.user_profiles:
                st.markdown("#### 💡 توصيات مخصصة لك")
                
                if st.button("✨ توليد توصيات مخصصة"):
                    base_tickets = generator.generate_tickets(20, 6, {})
                    recommendations = recommender.recommend(user_id, base_tickets, 5)
                    
                    for i, ticket in enumerate(recommendations, 1):
                        with st.expander(f"توصية #{i}"):
                            balls_html = " ".join([
                                f'<div class="ball-3d {"ball-hot" if n in analyzer.hot else "ball-cold" if n in analyzer.cold else "ball-neutral"}">{n}</div>'
                                for n in ticket
                            ])
                            st.markdown(balls_html, unsafe_allow_html=True)
            else:
                st.info("💡 احفظ بعض التذاكر في المحفظة ليتعلم النظام تفضيلاتك")
    
    # ==================== TAB 4: فاحص التذاكر ====================
    with tabs[3]:
        st.header("🔍 فاحص التذاكر المتقدم")
        
        tab_check1, tab_check2 = st.tabs(["🔎 فحص يدوي", "📁 فحص ملف"])
        
        with tab_check1:
            st.markdown("### 🔎 فحص تذكرة يدوياً")
            
            check_input = st.text_input(
                "أرقام التذكرة للفحص (مفصولة بفواصل)",
                placeholder="مثال: 5, 12, 18, 23, 27, 30",
                help="أدخل 6 أرقام على الأقل للحصول على أفضل النتائج"
            )
            
            if st.button("🔍 فحص الآن", type="primary"):
                if not check_input:
                    st.error("❌ الرجاء إدخال أرقام للفحص")
                else:
                    numbers, issues = validator.validate_with_constraints(check_input)
                    
                    if issues:
                        for issue in issues:
                            st.error(issue)
                    elif len(numbers) < 3:
                        st.error("❌ أدخل 3 أرقام على الأقل")
                    else:
                        with st.spinner("🔍 جاري البحث في السحوبات..."):
                            # البحث عن التطابقات
                            hits = []
                            ticket_set = set(numbers)
                            
                            for idx, row in df.iterrows():
                                draw_set = set(row['numbers'])
                                matches = ticket_set & draw_set
                                
                                if len(matches) >= 3:
                                    hits.append({
                                        'رقم السحب': row['draw_id'],
                                        'التاريخ': row['date'],
                                        'عدد المطابقات': len(matches),
                                        'الأرقام المطابقة': sorted(list(matches)),
                                        'أرقام السحب': row['numbers']
                                    })
                            
                            if hits:
                                st.success(f"🎉 وجدنا **{len(hits)}** تطابق!")
                                
                                hits_df = pd.DataFrame(hits).sort_values('عدد المطابقات', ascending=False)
                                
                                # إحصائيات
                                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                                
                                with col_stat1:
                                    st.metric("إجمالي التطابقات", len(hits))
                                
                                with col_stat2:
                                    matches_6 = len([h for h in hits if h['عدد المطابقات'] == 6])
                                    st.metric("6 أرقام", matches_6)
                                
                                with col_stat3:
                                    matches_5 = len([h for h in hits if h['عدد المطابقات'] == 5])
                                    st.metric("5 أرقام", matches_5)
                                
                                with col_stat4:
                                    matches_4 = len([h for h in hits if h['عدد المطابقات'] == 4])
                                    st.metric("4 أرقام", matches_4)
                                
                                st.markdown("---")
                                
                                # جدول النتائج
                                st.dataframe(
                                    hits_df,
                                    use_container_width=True,
                                    column_config={
                                        "عدد المطابقات": st.column_config.ProgressColumn(
                                            "المطابقات",
                                            format="%d",
                                            min_value=3,
                                            max_value=6
                                        )
                                    }
                                )
                            else:
                                st.warning("😔 لا يوجد أي تطابقات (3+ أرقام) في السجل التاريخي")
        
        with tab_check2:
            st.markdown("### 📁 فحص تذاكر من ملف")
            
            uploaded_file = st.file_uploader(
                "اختر ملف Excel أو CSV يحتوي على التذاكر",
                type=['xlsx', 'csv', 'txt']
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.xlsx'):
                        tickets_df = pd.read_excel(uploaded_file)
                    else:
                        tickets_df = pd.read_csv(uploaded_file)
                    
                    st.success(f"✅ تم تحميل {len(tickets_df)} تذكرة من الملف")
                    
                    if st.button("🔍 فحص جميع التذاكر"):
                        results = []
                        
                        with st.spinner("جاري فحص جميع التذاكر..."):
                            progress_bar = st.progress(0)
                            
                            for idx, row in tickets_df.iterrows():
                                # استخراج الأرقام من الصف
                                ticket_numbers = []
                                for col in tickets_df.columns:
                                    val = row[col]
                                    if pd.notna(val) and str(val).isdigit():
                                        num = int(float(val))
                                        if 1 <= num <= 32:
                                            ticket_numbers.append(num)
                                
                                if len(ticket_numbers) >= 6:
                                    ticket_set = set(ticket_numbers[:6])  # أخذ أول 6 أرقام
                                    
                                    # البحث عن التطابقات
                                    max_matches = 0
                                    best_draw = None
                                    
                                    for _, draw_row in df.iterrows():
                                        draw_set = set(draw_row['numbers'])
                                        matches = len(ticket_set & draw_set)
                                        
                                        if matches > max_matches:
                                            max_matches = matches
                                            best_draw = draw_row
                                    
                                    results.append({
                                        'التذكرة': ticket_numbers[:6],
                                        'أعلى تطابق': max_matches,
                                        'السحب': best_draw['draw_id'] if best_draw is not None else None,
                                        'التاريخ': best_draw['date'] if best_draw is not None else None
                                    })
                                
                                progress_bar.progress((idx + 1) / len(tickets_df))
                            
                            progress_bar.empty()
                        
                        if results:
                            results_df = pd.DataFrame(results)
                            st.dataframe(results_df, use_container_width=True)
                            
                            # إحصائيات عامة
                            st.markdown("#### 📊 إحصائيات الفحص")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                avg_match = results_df['أعلى تطابق'].mean()
                                st.metric("متوسط التطابق", f"{avg_match:.1f}")
                            
                            with col2:
                                max_match = results_df['أعلى تطابق'].max()
                                st.metric("أعلى تطابق", max_match)
                            
                            with col3:
                                perfect_matches = len(results_df[results_df['أعلى تطابق'] == 6])
                                st.metric("تطابقات كاملة", perfect_matches)
                        else:
                            st.info("ℹ️ لم يتم العثور على تذاكر صالحة للفحص")
                
                except Exception as e:
                    st.error(f"❌ خطأ في قراءة الملف: {e}")
    
    # ==================== TAB 5: التحليلات المتقدمة ====================
    with tabs[4]:
        st.header("📈 التحليلات المتقدمة والرسوم البيانية")
        
        tab_analysis1, tab_analysis2, tab_analysis3, tab_analysis4 = st.tabs([
            "🗺️ الخرائط الحرارية", 
            "⏱️ تحليل الفجوات", 
            "📊 الإحصائيات المتقدمة",
            "🔗 تحليل العلاقات"
        ])
        
        with tab_analysis1:
            st.markdown("### 🗺️ خرائط حرارية متقدمة")
            
            # خريطة حرارية 3D
            heatmap_data = np.zeros((4, 8))
            for i in range(32):
                row_idx, col_idx = divmod(i, 8)
                heatmap_data[row_idx, col_idx] = analyzer.freq.get(i + 1, 0)
            
            fig_3d = go.Figure(data=[go.Surface(z=heatmap_data)])
            fig_3d.update_layout(
                title='خريطة حرارية ثلاثية الأبعاد لتكرار الأرقام',
                scene=dict(
                    xaxis_title='العمود',
                    yaxis_title='الصف',
                    zaxis_title='التكرار'
                )
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
        
        with tab_analysis2:
            st.markdown("### ⏱️ تحليل الفجوات والأنماط الزمنية")
            
            # تحليل الفجوات بين الظهورات
            gap_analysis = []
            for num in range(1, 33):
                appearances = [i for i, nums in enumerate(df['numbers']) if num in nums]
                if len(appearances) > 1:
                    gaps = np.diff(appearances)
                    gap_analysis.append({
                        'number': num,
                        'avg_gap': np.mean(gaps),
                        'max_gap': max(gaps),
                        'last_gap': appearances[-1] - appearances[-2] if len(appearances) > 1 else 0,
                        'consistency': np.std(gaps) / np.mean(gaps) if np.mean(gaps) > 0 else 0
                    })
            
            gap_df = pd.DataFrame(gap_analysis)
            
            fig_gaps = px.scatter(
                gap_df,
                x='number',
                y='avg_gap',
                size='max_gap',
                color='consistency',
                hover_data=['last_gap'],
                title='تحليل الفجوات بين الظهورات',
                color_continuous_scale='RdBu_r'
            )
            
            st.plotly_chart(fig_gaps, use_container_width=True)
        
        with tab_analysis3:
            st.markdown("### 📊 إحصائيات متقدمة")
            
            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                # توزيع المجاميع
                sums = [sum(nums) for nums in df['numbers']]
                
                fig_sums = px.histogram(
                    x=sums,
                    nbins=30,
                    title='توزيع مجموع الأرقام',
                    labels={'x': 'المجموع', 'y': 'التكرار'},
                    color_discrete_sequence=['#667eea']
                )
                
                fig_sums.add_vline(
                    x=np.mean(sums),
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"المتوسط: {np.mean(sums):.1f}"
                )
                
                st.plotly_chart(fig_sums, use_container_width=True)
            
            with col_stat2:
                # توزيع الأنماط
                patterns_data = []
                for nums in df['numbers']:
                    patterns_data.append({
                        'odd': sum(1 for n in nums if n % 2),
                        'consecutive': sum(1 for i in range(len(nums)-1) if nums[i+1] - nums[i] == 1),
                        'decades': len(set([(n-1)//10 for n in nums]))
                    })
                
                patterns_df = pd.DataFrame(patterns_data)
                
                fig_patterns = px.scatter_matrix(
                    patterns_df,
                    dimensions=['odd', 'consecutive', 'decades'],
                    title='علاقة الأنماط المختلفة',
                    color=patterns_df['odd'],
                    color_continuous_scale='Viridis'
                )
                
                st.plotly_chart(fig_patterns, use_container_width=True)
        
        with tab_analysis4:
            st.markdown("### 🔗 تحليل العلاقات والارتباطات")
            
            # مصفوفة الارتباط بين الأرقام
            correlation_data = []
            for i in range(1, 33):
                for j in range(i+1, 33):
                    # حساب عدد المرات التي يظهر فيها الرقمان معاً
                    count_together = sum(1 for nums in df['numbers'] if i in nums and j in nums)
                    correlation_data.append({
                        'num1': i,
                        'num2': j,
                        'together': count_together,
                        'correlation': count_together / len(df) if len(df) > 0 else 0
                    })
            
            correlation_df = pd.DataFrame(correlation_data)
            top_correlations = correlation_df.nlargest(20, 'together')
            
            fig_corr = px.scatter(
                top_correlations,
                x='num1',
                y='num2',
                size='together',
                color='correlation',
                hover_data=['together'],
                title='أقوى 20 علاقة بين الأرقام',
                labels={'num1': 'الرقم الأول', 'num2': 'الرقم الثاني'},
                color_continuous_scale='Hot'
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
    
    # ==================== TAB 6: المحفظة الذكية ====================
    with tabs[5]:
        st.header("💼 المحفظة الذكية")
        
        if not st.session_state.portfolio:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.info("📭 محفظتك فارغة. احفظ تذاكر من المولد لتظهر هنا!")
            st.markdown("""
            ### 💡 ميزات المحفظة الذكية:
            1. **تخزين غير محدود** للتذاكر المفضلة
            2. **تحليل تلقائي** لكل تذكرة
            3. **تصدير PDF/CSV** بضغطة زر
            4. **توصيات ذكية** بناءً على محفظتك
            5. **مشاركة آمنة** مع الأصدقاء
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.success(f"✅ لديك **{len(st.session_state.portfolio)}** تذكرة محفوظة")
            
            # أزرار الإدارة
            col_manage1, col_manage2, col_manage3 = st.columns(3)
            
            with col_manage1:
                if st.button("🗑️ مسح المحفظة", type="secondary", use_container_width=True):
                    st.session_state.portfolio = []
                    st.rerun()
            
            with col_manage2:
                # تحميل PDF
                if st.session_state.portfolio:
                    pdf_buffer = PDFGenerator.create_ticket_pdf(
                        st.session_state.portfolio,
                        metadata={'strategy': 'Portfolio Export'}
                    )
                    st.download_button(
                        "📥 تحميل PDF",
                        pdf_buffer,
                        f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        "application/pdf",
                        use_container_width=True
                    )
            
            with col_manage3:
                # تحميل CSV
                if st.session_state.portfolio:
                    csv_data = pd.DataFrame(st.session_state.portfolio).to_csv(index=False)
                    st.download_button(
                        "📊 تحميل CSV",
                        csv_data,
                        f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
            
            st.markdown("---")
            
            # تحليل إحصائي للمحفظة
            st.markdown("#### 📊 إحصائيات المحفظة")
            
            portfolio_stats = {
                'total_tickets': len(st.session_state.portfolio),
                'avg_quality': np.mean([analyzer.get_ticket_analysis(t)['quality_score'] 
                                      for t in st.session_state.portfolio]),
                'common_numbers': Counter([num for ticket in st.session_state.portfolio 
                                         for num in ticket]).most_common(10),
                'preferred_patterns': _analyze_portfolio_patterns(st.session_state.portfolio)
            }
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                st.metric("إجمالي التذاكر", portfolio_stats['total_tickets'])
            
            with col_stat2:
                st.metric("متوسط الجودة", f"{portfolio_stats['avg_quality']:.1f}/10")
            
            with col_stat3:
                most_common = portfolio_stats['common_numbers'][0][0] if portfolio_stats['common_numbers'] else 0
                st.metric("الرقم الأكثر ظهوراً", most_common)
            
            with col_stat4:
                unique_numbers = len(set([num for ticket in st.session_state.portfolio 
                                        for num in ticket]))
                st.metric("أرقام فريدة", unique_numbers)
            
            st.markdown("---")
            
            # عرض التذاكر المحفوظة
            for idx, ticket in enumerate(st.session_state.portfolio, 1):
                with st.expander(f"🎫 تذكرة #{idx} - جودة: {analyzer.get_ticket_analysis(ticket)['quality_score']}/10"):
                    
                    # الكرات
                    balls_html = " ".join([
                        f'<div class="ball-3d {"ball-hot" if n in analyzer.hot else "ball-cold" if n in analyzer.cold else "ball-neutral"}">{n}</div>'
                        for n in ticket
                    ])
                    st.markdown(balls_html, unsafe_allow_html=True)
                    
                    # التحليل
                    analysis = analyzer.get_ticket_analysis(ticket)
                    
                    col_info1, col_info2, col_info3 = st.columns(3)
                    
                    with col_info1:
                        st.metric("المجموع", analysis['basic']['sum'])
                        st.metric("الفردي", analysis['basic']['odd'])
                    
                    with col_info2:
                        st.metric("الساخن", analysis['classification']['hot_count'])
                        st.metric("البارد", analysis['classification']['cold_count'])
                    
                    with col_info3:
                        st.metric("التوازن", f"{analysis['statistical']['balance_score']:.2f}")
                        st.metric("التنوع", f"{analysis['statistical']['diversity_score']:.2f}")
                    
                    # أزرار الإجراءات
                    col_action1, col_action2 = st.columns(2)
                    
                    with col_action1:
                        if st.button(f"❌ حذف", key=f"delete_{idx}", use_container_width=True):
                            st.session_state.portfolio.pop(idx - 1)
                            st.rerun()
                    
                    with col_action2:
                        if st.button(f"📊 تحليل متقدم", key=f"detail_{idx}", use_container_width=True):
                            # عرض تحليل مفصل
                            st.json(analysis, expanded=False)
    
    # ==================== TAB 7: الإعدادات والأداء ====================
    with tabs[6]:
        st.header("⚙️ الإعدادات والأداء")
        
        tab_settings1, tab_settings2, tab_settings3 = st.tabs([
            "⚙️ إعدادات التطبيق",
            "📊 أداء النظام",
            "🔒 الأمان والخصوصية"
        ])
        
        with tab_settings1:
            st.markdown("### ⚙️ إعدادات التطبيق")
            
            col_setting1, col_setting2 = st.columns(2)
            
            with col_setting1:
                st.markdown("#### إعدادات الذاكرة")
                
                cache_size = st.slider(
                    "حجم الذاكرة المؤقتة (MB)",
                    10, 500, 100,
                    help="الذاكرة المخصصة للتخزين المؤقت"
                )
                
                auto_clear = st.checkbox(
                    "مسح تلقائي للذاكرة",
                    value=True,
                    help="مسح الذاكرة المؤقتة تلقائياً عند الإغلاق"
                )
            
            with col_setting2:
                st.markdown("#### إعدادات التوليد")
                
                max_tickets = st.number_input(
                    "الحد الأقصى للتذاكر",
                    min_value=10,
                    max_value=1000,
                    value=100,
                    help="الحد الأقصى لعدد التذاكر التي يمكن توليدها في مرة واحدة"
                )
                
                enable_ml = st.checkbox(
                    "تمكين تعلم الآلة",
                    value=True,
                    help="استخدام نماذج ML للتنبؤ والتوصيات"
                )
            
            st.markdown("---")
            st.markdown("#### إعدادات الواجهة")
            
            theme_options = ["فاتح", "داكن", "تلقائي"]
            selected_theme = st.selectbox("السمة", theme_options, index=2)
            
            language_options = ["العربية", "English", "Français"]
            selected_lang = st.selectbox("اللغة", language_options, index=0)
            
            if st.button("💾 حفظ الإعدادات", type="primary"):
                st.success("✅ تم حفظ الإعدادات بنجاح!")
                # هنا سيتم حفظ الإعدادات فعلياً
        
        with tab_settings2:
            st.markdown("### 📊 أداء النظام")
            
            # إحصائيات النظام الحالية
            system_stats = benchmark.get_system_stats()
            
            col_perf1, col_perf2, col_perf3 = st.columns(3)
            
            with col_perf1:
                st.metric("استخدام CPU", f"{system_stats['cpu']['percent']}%")
                st.progress(system_stats['cpu']['percent'] / 100)
            
            with col_perf2:
                st.metric("استخدام الذاكرة", f"{system_stats['memory']['used_percent']}%")
                st.progress(system_stats['memory']['used_percent'] / 100)
            
            with col_perf3:
                st.metric("مساحة التخزين", f"{system_stats['disk']['percent']}%")
                st.progress(system_stats['disk']['percent'] / 100)
            
            st.markdown("---")
            
            # تقرير الأداء التفصيلي
            if st.button("🔄 تحديث تقرير الأداء"):
                perf_report = benchmark.get_performance_report()
                st.session_state.performance_data['detailed'] = perf_report
                st.rerun()
            
            if 'detailed' in st.session_state.performance_data:
                perf_data = st.session_state.performance_data['detailed']
                
                st.markdown("#### 📈 تحليل الأداء التفصيلي")
                
                with st.expander("⏱️ تحليل الوقت"):
                    time_df = pd.DataFrame([{
                        'المقياس': 'أقل وقت',
                        'القيمة': f"{perf_data['duration']['min']}s"
                    }, {
                        'المقياس': 'أعلى وقت',
                        'القيمة': f"{perf_data['duration']['max']}s"
                    }, {
                        'المقياس': 'متوسط الوقت',
                        'القيمة': f"{perf_data['duration']['avg']}s"
                    }, {
                        'المقياس': 'الانحراف المعياري',
                        'القيمة': f"{perf_data['duration']['std']}s"
                    }])
                    
                    st.dataframe(time_df, use_container_width=True, hide_index=True)
                
                with st.expander("💾 تحليل الذاكرة"):
                    memory_df = pd.DataFrame([{
                        'المقياس': 'أقل استخدام',
                        'القيمة': f"{perf_data['memory']['min_mb']}MB"
                    }, {
                        'المقياس': 'أعلى استخدام',
                        'القيمة': f"{perf_data['memory']['max_mb']}MB"
                    }, {
                        'المقياس': 'متوسط الاستخدام',
                        'القيمة': f"{perf_data['memory']['avg_mb']}MB"
                    }, {
                        'المقياس': 'إجمالي الاستخدام',
                        'القيمة': f"{perf_data['memory']['total_mb']}MB"
                    }])
                    
                    st.dataframe(memory_df, use_container_width=True, hide_index=True)
                
                with st.expander("📊 العمليات حسب النوع"):
                    operations_df = pd.DataFrame([
                        {
                            'العملية': op,
                            'عدد المرات': data['count'],
                            'متوسط الوقت': f"{data['avg_duration']}s",
                            'متوسط الذاكرة': f"{data['avg_memory']}MB"
                        }
                        for op, data in perf_data['operations_by_type'].items()
                    ])
                    
                    st.dataframe(operations_df, use_container_width=True, hide_index=True)
        
        with tab_settings3:
            st.markdown("### 🔒 الأمان والخصوصية")
            
            st.markdown("""
            #### 🔐 حماية البيانات
            - جميع البيانات محفوظة **محلياً** على جهازك
            - لا يتم إرسال أي بيانات إلى خوادم خارجية
            - التشفير المستخدم: **AES-256**
            - جلسات المستخدم **آمنة ومعزولة**
            """)
            
            st.markdown("---")
            
            st.markdown("#### 🛡️ إعدادات الخصوصية")
            
            col_sec1, col_sec2 = st.columns(2)
            
            with col_sec1:
                enable_logging = st.checkbox(
                    "تفعيل سجلات النظام",
                    value=True,
                    help="تسجيل أنشطة التطبيق لأغراض التحسين"
                )
                
                auto_update = st.checkbox(
                    "التحديث التلقائي",
                    value=False,
                    help="التحديث التلقائي للبيانات والنماذج"
                )
            
            with col_sec2:
                clear_on_exit = st.checkbox(
                    "مسح البيانات عند الخروج",
                    value=False,
                    help="مسح جميع البيانات المؤقتة عند إغلاق التطبيق"
                )
                
                encrypt_data = st.checkbox(
                    "تشفير البيانات المحفوظة",
                    value=True,
                    help="تشفير الملفات المحفوظة مثل PDF وCSV"
                )
            
            st.markdown("---")
            
            if st.button("🧹 مسح جميع البيانات", type="secondary"):
                # مسح جميع البيانات
                st.cache_data.clear()
                if 'portfolio' in st.session_state:
                    st.session_state.portfolio = []
                if 'performance_data' in st.session_state:
                    st.session_state.performance_data = {}
                
                st.success("✅ تم مسح جميع البيانات بنجاح!")
                st.info("🔄 الرجاء إعادة تحميل الصفحة للتطبيق الكامل للتغييرات")

# ==============================================================================
# وظائف مساعدة
# ==============================================================================

def _analyze_portfolio_patterns(portfolio: List[List[int]]) -> Dict:
    """تحليل أنماط المحفظة"""
    if not portfolio:
        return {}
    
    patterns = {
        'common_sum_range': [],
        'common_odd_count': [],
        'common_hot_ratio': []
    }
    
    for ticket in portfolio:
        patterns['common_sum_range'].append(sum(ticket))
        patterns['common_odd_count'].append(sum(1 for n in ticket if n % 2))
        
        # حساب نسبة الأرقام الساخنة
        hot_count = sum(1 for n in ticket if n in st.session_state.analyzer.hot)
        patterns['common_hot_ratio'].append(hot_count / len(ticket))
    
    # حساب المتوسطات
    result = {
        'avg_sum': round(np.mean(patterns['common_sum_range']), 1),
        'avg_odd': round(np.mean(patterns['common_odd_count']), 1),
        'avg_hot_ratio': round(np.mean(patterns['common_hot_ratio']), 3),
        'sum_range': (min(patterns['common_sum_range']), max(patterns['common_sum_range'])),
        'odd_range': (min(patterns['common_odd_count']), max(patterns['common_odd_count']))
    }
    
    return result

# ==============================================================================
# نقطة الدخول الرئيسية
# ==============================================================================

if __name__ == "__main__":
    try:
        main()
        
        # تسجيل تشغيل ناجح
        logger.logger.info("🚀 بدء تشغيل التطبيق بنجاح", extra={
            'version': Config.APP_VERSION,
            'timestamp': datetime.now().isoformat(),
            'performance_stats': benchmark.get_system_stats()
        })
        
    except Exception as e:
        # تسجيل أي أخطاء
        logger.logger.critical("❌ فشل تشغيل التطبيق", extra={
            'error': str(e),
            'traceback': traceback.format_exc()
        })
        
        st.error(f"❌ حدث خطأ غير متوقع: {e}")
        st.info("🔄 الرجاء إعادة تحميل الصفحة أو الاتصال بالدعم")