#!/usr/bin/env python
# coding: utf-8

# In[2]:


import plotly.express as px
import re
import requests
import urllib.parse
from collections import defaultdict
from datetime import datetime
import streamlit as st
import pandas as pd
import os
import time # لاستخدامه مع st.toast
import streamlit.components.v1 as components # لاستخدام iframe
from analyse_sentiment import predict_sentiment, predict_emotion, labels_map_sentiment, labels_map_emotion
import matplotlib.pyplot as plt
import base64

import re
# import google.generativeai as genai # تم التعليق لأننا نستخدم دالة محلية الآن
# from nicegui import ui # تم التعليق لعدم استخدامها مباشرة في هذا الإصدار

# --- إعدادات أساسية ---
POSTS_FILE = "facebook_posts.csv"
COMMENTS_FILE = "facebook_comments.csv"
LOGO_PATH = "logo.jpg" # <-- تأكد أن هذا المسار صحيح أو استخدم رابط URL
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
GOOGLE_SEARCH_DELAY = 2

class FacebookTrendAnalyzer:
    def __init__(self):
        self.posts_df = None
        self.fb_trends = None
        self.veracity_df = None

    def extract_hashtags(self, text):
        """Extract hashtags including Arabic characters"""
        if pd.isna(text):
            return []
        return [tag.lower() for tag in re.findall(r'#([^\s!@#$%^&*(),.?":{}|<>]+)', str(text))]

    def calculate_trends(self, posts_df):
        """Calculate trends from both title and article_title"""
        trend_data = defaultdict(lambda: {
            'posts': 0, 'likes': 0, 'comments': 0, 'shares': 0, 'examples': []
        })

        for _, post in posts_df.iterrows():
            # Combine title and article_title for hashtag search
            text_to_scan = f"{post.get('title', '')} {post.get('article_title', '')}"
            for tag in self.extract_hashtags(text_to_scan):
                data = trend_data[tag]
                data['posts'] += 1
                data['likes'] += post.get('likes_count', 0)
                data['comments'] += post.get('comments_count', 0)
                data['shares'] += post.get('shares_count', 0)
                data['examples'].append(text_to_scan[:50] + "...")

        # Rest of the function remains the same...

        if not trend_data:
            return []

        max_engagement = max(
            (d['likes'] * 0.5 + d['comments'] * 0.3 + d['shares'] * 0.2)
            for d in trend_data.values()
        )

        trends = []
        for tag, data in trend_data.items():
            raw_score = (data['likes'] * 0.5 + data['comments'] * 0.3 + data['shares'] * 0.2)
            norm_score = (raw_score / max_engagement * 100) if max_engagement > 0 else 0

            trends.append({
                'hashtag': f"#{tag}",
                'score': round(norm_score, 1),
                'likes': data['likes'],
                'comments': data['comments'],
                'shares': data['shares'],
                'post_count': data['posts'],
                'examples': ", ".join(data['examples'][:3]),
                'google_mentions': 0
            })

        return sorted(trends, key=lambda x: -x['score'])[:50]

    def safe_google_search(self, query, retry=0):
        """Robust Google search with retries"""
        try:
            url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            patterns = [
                r"([\d,\.]+)\s+results",
                r"About\s([\d,\.]+)\s+results",
                r"([\d,\.]+)\s+نتيجة"
            ]

            for pattern in patterns:
                match = re.search(pattern, response.text)  # Changed from response.title to response.text
                if match:
                    count = match.group(1).replace(",", "").replace(".", "")
                    return int(count) if count.isdigit() else 0
            return 0

        except requests.exceptions.RequestException as e:
            if retry < MAX_RETRIES:
                time.sleep(2 ** retry)
                return self.safe_google_search(query, retry + 1)
            return 0

    def calculate_veracity_scores(self, trends):
        """Calculate comprehensive veracity scores"""
        results = []
        progress_bar = st.progress(0)

        for i, trend in enumerate(trends):
            query = trend['hashtag'].lstrip('#')
            mentions = self.safe_google_search(query)

            internal = trend['score']
            external = min(mentions / 1000, 100)
            veracity = round(internal * 0.6 + external * 0.4, 1)

            results.append({
                'Trend': query,
                'Veracity Score': veracity,
                'Internal Score': internal,
                'Google Mentions': mentions,
                'FB Likes': trend['likes'],
                'FB Comments': trend['comments'],
                'FB Shares': trend['shares'],
                'Post Count': trend['post_count']
            })

            progress_bar.progress((i + 1) / len(trends))
            if i < len(trends) - 1:
                time.sleep(GOOGLE_SEARCH_DELAY)

        progress_bar.empty()
        return pd.DataFrame(results).sort_values('Veracity Score', ascending=False)

    def display_dashboard(self, trends, veracity_df):
        """Interactive dashboard with visualizations"""
        st.title("📊 Advanced Facebook Trends Analysis")

        cols = st.columns(4)
        cols[0].metric("Total Trends", len(trends))
        cols[1].metric("Avg Veracity", f"{veracity_df['Veracity Score'].mean():.1f}")
        cols[2].metric("High Confidence", len(veracity_df[veracity_df['Veracity Score'] > 60]))

        st.subheader("Top Performing Trends")
        st.dataframe(
            veracity_df.head(20)[[
                'Trend', 'Veracity Score', 'Internal Score',
                'Post Count'
            ]],
            height=600
        )

        st.subheader("Trend Analysis")
        tab1, tab2 = st.tabs(["Engagement Breakdown", "Veracity Analysis"])

        with tab1:
            fig = px.bar(
                pd.DataFrame(trends).head(15),
                x='hashtag',
                y=['likes', 'comments', 'shares'],
                title='Top 15 Trends by Engagement',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig = px.scatter(
                veracity_df,
                x='Internal Score',
                y='Veracity Score',
                size='Google Mentions',
                color='Post Count',
                hover_name='Trend',
                title='Veracity vs Internal Score'
            )
            st.plotly_chart(fig, use_container_width=True)
# --- إعداد Gemini (اختياري، إذا كنت ستستخدمه) ---
# حاول تحميل المفتاح من متغير بيئة أولاً
# GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_FALLBACK_API_KEY_HERE") # استبدل بالمفتاح الفعلي كاحتياطي
# if GEMINI_API_KEY != "YOUR_FALLBACK_API_KEY_HERE":
#     try:
#         genai.configure(api_key=GEMINI_API_KEY)
#         model = genai.GenerativeModel("gemini-1.5-pro")
#         gemini_available = True
#     except Exception as e:
#         st.warning(f"فشل إعداد Gemini: {e}. سيتم تعطيل ميزة التلخيص.", icon="⚠️")
#         gemini_available = False
#         model = None # تأكد من تعيينه إلى None
# else:
#     st.warning("لم يتم العثور على مفتاح Gemini API. سيتم تعطيل ميزة التلخيص.", icon="⚠️")
#     gemini_available = False
#     model = None

# --- دالة تلخيص بديلة/محلية (إذا لم يتوفر Gemini) ---
def set_category(category):
    """تحديث الفئة المختارة في حالة الجلسة"""
    st.session_state.selected_category = category
    st.session_state.show_gallery = False
    st.session_state.search_term = ""

def get_base64_encoded_image(image_path):
    """تحويل الصورة إلى تنسيق base64 لعرضها مباشرة في HTML"""
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
def summarize_arabic_local(text, max_sentences=3):
    """Summarize Arabic text locally (simple approach)."""
    if not text or not isinstance(text, str) or len(text.strip()) < 50:
        return "النص قصير جدًا أو غير متاح للتلخيص"
    try:
        # تقسيم بسيط إلى جمل (قد لا يكون مثالياً للعربية)
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
        if not sentences:
             # إذا فشل التقسيم بالنقطة، حاول بالأسطر الجديدة
             sentences = [s.strip() for s in text.split('\n') if len(s.strip()) > 10]
        # أخذ أول جملتين أو ثلاث جمل مهمة (بافتراض الأهمية في البداية)
        summary_sentences = sentences[:max_sentences]
        summary = ". ".join(summary_sentences)
        if summary:
            return summary + "." if not summary.endswith('.') else summary
        else:
            # إذا لم نجد جمل، نعيد جزء من النص الأصلي
            return text[:250] + "..." if len(text) > 250 else text
    except Exception as e:
        return f"حدث خطأ أثناء التلخيص المحلي: {str(e)}"

# --- استخدام دالة التلخيص (تعتمد على توفر Gemini) ---
def summarize_arabic(text):
    # if gemini_available and model:
    #     prompt = f"قم بتلخيص هذا النص العربي بطريقة واضحة ومختصرة مع الحفاظ على المعنى الأساسي:\n\n{text}"
    #     try:
    #         # زيادة الوقت المستقطع المحتمل إذا كانت الاستجابات بطيئة
    #         response = model.generate_content(prompt, request_options={'timeout': 120})
    #         return response.text
    #     except Exception as e:
    #         st.warning(f"خطأ في التلخيص باستخدام Gemini: {e}. سيتم استخدام التلخيص المحلي.", icon="⚠️")
    #         return summarize_arabic_local(text) # العودة للتلخيص المحلي عند الفشل
    # else:
        # استخدام التلخيص المحلي إذا لم يكن Gemini متاحًا أو فشل إعداده
        return summarize_arabic_local(text)

# ---------------------------
# تحميل البيانات (بدون تغيير جوهري)
# ---------------------------
# @st.cache_data # استخدام الكاش لتحسين الأداء
def load_data():
    try:
        if not os.path.exists(POSTS_FILE): st.error(f"ملف {POSTS_FILE} غير موجود."); return None, None
        if not os.path.exists(COMMENTS_FILE): st.error(f"ملف {COMMENTS_FILE} غير موجود."); return None, None
        try: posts_df = pd.read_csv(POSTS_FILE); comments_df = pd.read_csv(COMMENTS_FILE)
        except UnicodeDecodeError:
            try: posts_df = pd.read_csv(POSTS_FILE, encoding='utf-8'); comments_df = pd.read_csv(COMMENTS_FILE, encoding='utf-8')
            except Exception as e: st.error(f"خطأ قراءة (ترميز): {e}"); return None, None
        except pd.errors.EmptyDataError: st.error("أحد ملفات البيانات فارغ."); return None, None
        except Exception as e: st.error(f"خطأ غير متوقع عند قراءة CSV: {e}"); return None, None

        required_post_cols = ['post_id', 'article_title', 'short_description', 'image', 'video_link', 'article_link', 'likes_count', 'comments_count']
        required_comment_cols = ['post_id', 'text', 'likes']

        # التحقق من الأعمدة المطلوبة في المنشورات
        missing_post_cols = [col for col in required_post_cols if col not in posts_df.columns]
        if missing_post_cols: st.error(f"ملف المنشورات يفتقد الأعمدة: {', '.join(missing_post_cols)}"); return None, None

        # التحقق من الأعمدة المطلوبة في التعليقات (مع مرونة لعمود 'author')
        missing_comment_cols = [col for col in required_comment_cols if col not in comments_df.columns]
        if missing_comment_cols:
            if 'author' in missing_comment_cols and len(missing_comment_cols) == 1:
                st.warning("عمود 'author' في التعليقات مفقود، سيتم المتابعة بدونه.", icon="⚠️")
                required_comment_cols.remove('author') # إزالته من القائمة مؤقتًا
                # أعد التحقق بدون author
                missing_comment_cols = [col for col in required_comment_cols if col not in comments_df.columns]
                if missing_comment_cols: # إذا كان هناك أعمدة أخرى مفقودة
                    st.error(f"ملف التعليقات يفتقد أيضًا: {', '.join(missing_comment_cols)}"); return None, None
                else:
                     comments_df['author'] = "غير معروف" # إضافة عمود وهمي
            else: # إذا كانت أعمدة أخرى غير author مفقودة
                st.error(f"ملف التعليقات يفتقد الأعمدة: {', '.join(missing_comment_cols)}"); return None, None

        # تحويل أنواع البيانات وتنظيف القيم المفقودة
        for col in ['likes_count', 'comments_count']:
            posts_df[col] = pd.to_numeric(posts_df[col], errors='coerce').fillna(0).astype(int)
        if 'likes' in comments_df.columns:
            comments_df['likes'] = pd.to_numeric(comments_df['likes'], errors='coerce').fillna(0).astype(int)
        else:
             # إذا كان عمود likes غير موجود أصلاً (على الرغم من التحقق السابق)
            comments_df['likes'] = 0

        for col in ['article_title', 'short_description', 'image', 'video_link', 'article_link']:
             posts_df[col] = posts_df[col].fillna('').astype(str)
        for col in ['author', 'text']:
             if col in comments_df.columns:
                 comments_df[col] = comments_df[col].fillna('').astype(str)
             elif col == 'text': # يجب أن يكون عمود النص موجودًا
                 st.error("عمود 'text' للتعليقات مفقود وهو ضروري!"); return None, None

        # التأكد من وجود 'post_id' للربط
        if 'post_id' not in posts_df.columns or 'post_id' not in comments_df.columns:
            st.error("عمود 'post_id' للربط بين المنشورات والتعليقات غير موجود في أحد الملفين."); return None, None

        # توحيد نوع 'post_id' كنص
        posts_df['post_id'] = posts_df['post_id'].astype(str)
        comments_df['post_id'] = comments_df['post_id'].astype(str)

        return posts_df, comments_df

    except FileNotFoundError as fnf_error:
        st.error(f"خطأ: لم يتم العثور على الملف - {fnf_error}")
        return None, None
    except pd.errors.ParserError as parse_error:
        st.error(f"خطأ في تحليل ملف CSV: {parse_error}. تأكد من أن الملف بتنسيق CSV صحيح.")
        return None, None
    except KeyError as key_error:
        st.error(f"خطأ: عمود مفقود في البيانات - {key_error}. تأكد من صحة أسماء الأعمدة.")
        return None, None
    except Exception as e:
        st.error(f"خطأ غير متوقع أثناء تحميل البيانات: {e}")
        st.exception(e) # طباعة تتبع الخطأ الكامل في الطرفية للمساعدة في التصحيح
        return None, None

# ---------------------------
# تصنيف المنشورات (بدون تغيير)
# ---------------------------
# @st.cache_data # استخدام الكاش لتحسين الأداء
def categorize_posts(posts_df):
    if 'article_title' not in posts_df.columns:
        posts_df['category'] = "أخرى"
        return posts_df

    # قائمة الفئات والكلمات المفتاحية (يمكن توسيعها)
    categories = {
        "كرة القدم": ["كرة القدم", "الدوري", "كأس", "مباراة", "فريق", "ريال مدريد", "برشلونة", "ليفربول", "مانشستر", "تشيلسي", "يوفنتوس", "ميلان", "بايرن", "هدف", "لاعب", "الملعب", "بطل", "نهائي", "تشامبيونز ليج", "يورو", "رونالدو", "ميسي"],
        "كرة السلة": ["كرة السلة", "nba", "الدوري الاميركي", "سلة", "ليكرز", "واريورز", "ليبرون", "كيري", "جيمس", "كوبي"],
        "التنس": ["التنس", "ويمبلدون", "رولان غاروس", "فلاشينغ ميدوز", "أستراليا المفتوحة", "نادال", "فيدرر", "ديوكوفيتش", "مضرب"],
        "كرة اليد": ["كرة اليد", "بطولة العالم لكرة اليد"],
        "فورمولا 1": ["فورمولا 1", "فورمولا وان", "f1", "سباق سيارات", "هاميلتون", "فيرستابن", "فيراري", "مرسيدس", "ريد بول", "حلبة"],
        "ألعاب قوى": ["ألعاب قوى", "عدو", "قفز", "رمي", "ماراثون", "أولمبياد"],
        "رياضات أخرى": ["سباحة", "ملاكمة", "مصارعة", "جولف", "دراجات"] # فئة أوسع
    }

    def detect_category(text):
        best_match = "أخرى" # الافتراضي
        if isinstance(text, str) and text.strip():
            text_lower = " " + text.lower() + " " # إضافة مسافات للتأكد من تطابق الكلمات الكاملة
            matched_cats = []

            # البحث عن تطابق كلمة كاملة أولاً (أكثر دقة)
            for cat, keywords in categories.items():
                if any(f" {kw.lower()} " in text_lower for kw in keywords):
                    matched_cats.append(cat)

            # إذا وجدنا تطابقات دقيقة، نختار الأول (أو يمكن تطبيق منطق آخر)
            if matched_cats:
                return matched_cats[0]

            # إذا لم نجد تطابقاً دقيقاً، نبحث عن وجود الكلمة المفتاحية كجزء من كلمة أخرى (أقل دقة)
            for cat, keywords in categories.items():
                 if any(kw.lower() in text_lower for kw in keywords):
                     matched_cats.append(cat)

            # إذا وجدنا تطابقات جزئية، نختار الأول
            if matched_cats:
                return matched_cats[0]

        return best_match # إذا لم يتم العثور على أي تطابق

    # تطبيق الدالة على عمود العنوان
    posts_df['category'] = posts_df['article_title'].apply(detect_category)
    return posts_df


# ---------------------------
# حساب البوستات الأكثر شهرة (بدون تغيير)
# ---------------------------
# @st.cache_data # استخدام الكاش لتحسين الأداء
def get_top_posts(posts_df):
    # Calculate popularity score based on likes, comments, and shares with weights
    # likes: 50%, comments: 30%, shares: 20%

    # Convert columns to numeric if they aren't already
    for col in ['likes_count', 'comments_count', 'shares_count']:
        if col in posts_df.columns:
            posts_df[col] = pd.to_numeric(posts_df[col], errors='coerce').fillna(0)

    # Calculate popularity score with the new formula
    if all(col in posts_df.columns for col in ['likes_count', 'comments_count', 'shares_count']):
        # All metrics are available
        posts_df['popularity_score'] = (
                posts_df['likes_count'] * 0.5 +
                posts_df['comments_count'] * 0.3 +
                posts_df['shares_count'] * 0.2
        )
    elif all(col in posts_df.columns for col in ['likes_count', 'comments_count']):
        # Only likes and comments are available
        posts_df['popularity_score'] = posts_df['likes_count'] * 0.7 + posts_df['comments_count'] * 0.3
    elif 'likes_count' in posts_df.columns:
        # Only likes are available
        posts_df['popularity_score'] = posts_df['likes_count']
    else:
        # No engagement metrics available
        posts_df['popularity_score'] = 0

    # Sort posts by popularity score in descending order
    if not posts_df.empty:
        return posts_df.sort_values('popularity_score', ascending=False)
    else:
        return posts_df# إعادة DataFrame فارغ إذا كان الإدخال فارغًا

# ---------------------------
# التحقق من صحة الرابط (بدون تغيير)
# ---------------------------
def is_valid_url(url):
    """Checks if a string is a valid HTTP/HTTPS URL."""
    return isinstance(url, str) and url.strip().startswith(('http://', 'https://'))

def is_valid_media_link(link):
    """Checks if a link is likely a valid, non-empty media link (image/video)."""
    if not isinstance(link, str): return False
    link_cleaned = link.strip().lower()
    # قائمة القيم التي تعتبر غير صالحة أو فارغة
    invalid_values = ['unavailable', 'no video', '', 'nan', 'none', 'null']
    if link_cleaned in invalid_values: return False
    # يجب أن يبدأ بـ http أو https
    if not link_cleaned.startswith(('http://', 'https://')): return False
    # يمكنك إضافة تحققات أخرى هنا إذا لزم الأمر (مثل امتدادات الملفات)
    return True

# ---------------------------
# تطبيق CSS مع تحسينات التصميم والـ Sidebar
# ---------------------------
def apply_css():
    st.markdown('''
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.rtl.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;900&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary-color: #4CAF50; /* Green */
                --primary-dark: #388E3C;
                --primary-light: #81C784;
                --secondary-color: #263238; /* Dark Grey */
                --accent-color: #FFC107; /* Amber */
                --text-primary: #ECEFF1; /* Light Grey */
                --text-secondary: #B0BEC5; /* Medium Grey */
                --bg-dark: #121212; /* Very Dark Grey */
                --bg-darker: #0a0a0a; /* Almost Black */
                --bg-card: #1E1E1E; /* Slightly Lighter Dark Grey */
                --bg-card-hover: #2a2a2a;
                --bg-sidebar: #181818; /* Dark for Sidebar */
                --border-color: #37474F; /* Bluish Grey */
                --border-radius: 10px;
                --box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
                --transition: all 0.3s ease-in-out;
            }

              /* --- General Styles --- */
              body, .stApp {
                font-family: 'Tajawal', sans-serif;
                background-color: var(--bg-dark);
                color: var(--text-primary);
                direction: rtl; /* Right-to-left layout */
                font-size: 18px; /* تكبير الخط العام */
  
            }
            h1 {
               font-size: 36px;
            }
            h2 {
               font-size: 32px;
            }
            h3 {
               font-size: 28px;
            }
            h4 {
               font-size: 24px;
            } 
            h5 {
               font-size: 20px;
            } 
            h6 {
               font-size: 18px;
            }
            h1, h2, h3, h4, h5, h6 {
                color: var(--primary-light);
                font-weight: 700;
                font-size: 30px;
            }
            a {
                color: var(--primary-light);
                text-decoration: none;
                transition: var(--transition);
            }
            a:hover {
                color: var(--primary-color);
                text-decoration: underline;
            }

            hr {
                border-top: 1px solid var(--border-color);
                opacity: 0.5;
            }

            /* --- Sidebar Styles --- */
            [data-testid="stSidebar"] {
                background: linear-gradient(to bottom, var(--bg-sidebar), #212121); /* Gradient background */
                border-left: 2px solid var(--primary-color);
                box-shadow: 5px 0px 15px rgba(0, 0, 0, 0.5);
            }

            .sidebar-title {
                font-size: 2rem;
                font-weight: 1000;
                color: var(--primary-color);
                text-align: center;
                margin-bottom: 20px;
                padding: 10px 0;
                border-bottom: 1px solid var(--border-color);
                 background: linear-gradient(to right, #4CAF50, #A5D6A7);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .sidebar-logo-container {
                display: flex;          /* يجعل هذا العنصر حاوية مرنة (flex container) */
                justify-content: center; /* يوسّط العناصر داخل الحاوية أفقيًا */
                margin: 20px 0 30px 0; /* هوامش فوق وتحت لتحديد المسافة */
            }

            .sidebar-logo {
                width: 250px;           /* عرض اللوجو */
                height: 250px;          /* ارتفاع اللوجو (مساوي للعرض للحصول على دائرة مثالية) */
                border-radius: 50%;     /* يجعل الحواف دائرية تمامًا */
                object-fit: cover;      /* يضمن أن الصورة تغطي المساحة المحددة بدون تشويه */
                box-shadow: 0 4px 10px rgba(0,0,0,0.2); /* ظل خفيف */
                transition: all 0.3s ease; /* تأثير انتقال سلس عند التمرير */
            }

            [data-testid="stSidebar"] .stButton>button {
                /* Sidebar specific button style - More subtle */
                background-color: transparent;
                color: var(--text-secondary);
                border: 1px solid var(--border-color);
                border-radius: var(--border-radius);
                padding: 10px 15px;
                margin: 8px 0; /* Vertical spacing */
                font-weight: 500;
                width: 100%; /* Full width */
                text-align: right;
                display: flex;
                align-items: center;
                justify-content: flex-start; /* Align text to the right */
                gap: 10px; /* Space between icon and text */
            }

            [data-testid="stSidebar"] .stButton>button:hover {
                background-color: rgba(76, 175, 80, 0.1); /* Light green highlight on hover */
                color: var(--primary-light);
                border-color: var(--primary-color);
                box-shadow: none; /* Remove general button shadow */
                transform: none; /* Remove general button transform */
            }

            [data-testid="stSidebar"] .stButton>button i {
                 color: var(--primary-color); /* Icon color */
                 font-size: 1.1em;
            }

            /* --- Main Content Styles --- */
            .main-content { /* Add a class to the main container if needed */
                padding: 20px;
            }

            /* --- Enhanced Button Styles (General) --- */
            .stButton>button {
                border-radius: 50px !important; /* Pill shape */
                font-weight: bold !important;
                background: linear-gradient(to right, var(--primary-color), var(--primary-dark)) !important;
                color: white !important;
                border: none !important; /* Remove border, rely on gradient/shadow */
                padding: 12px 25px !important; /* More padding */
                margin: 5px !important; /* Consistent margin */
                transition: var(--transition) !important;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3); /* Subtle shadow */
                cursor: pointer;
                display: inline-flex; /* Align icon and text */
                align-items: center;
                justify-content: center;
                gap: 8px; /* Space between icon and text */
                line-height: 1.1; /* Adjust line height */
            }

            .stButton>button:hover {
                background: linear-gradient(to right, var(--primary-dark), #2E7D32) !important; /* Darker gradient on hover */
                color: white !important;
                transform: translateY(-3px) scale(1.03); /* Lift and slightly enlarge */
                box-shadow: 0 7px 14px rgba(0, 0, 0, 0.4); /* Stronger shadow on hover */
            }

            .stButton>button:active {
                transform: translateY(0px) scale(1); /* Press down effect */
                box-shadow: 0 3px 6px rgba(0, 0, 0, 0.3); /* Slightly reduced shadow */
            }

             /* --- Category Buttons Specific Style (Keep them distinct) --- */
            .stButton[key*="cat_btn_"]>button { /* Target category buttons by key prefix */
                background: var(--secondary-color) !important; /* Dark grey background */
                color: var(--primary-light) !important; /* Light green text */
                border: 1px solid var(--primary-color) !important; /* Green border */
                padding: 10px 18px !important;
                font-weight: 500 !important;
                border-radius: var(--border-radius) !important; /* Less rounded than main buttons */
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            }

            .stButton[key*="cat_btn_"]>button:hover {
                background: var(--primary-color) !important; /* Green background on hover */
                color: white !important;
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
            }

             /* --- Post Card Styles --- */
            .post-card {
                background: linear-gradient(135deg, var(--bg-card), var(--bg-darker));
                border-radius: var(--border-radius);
                padding: 25px;
                margin-bottom: 30px;
                box-shadow: var(--box-shadow);
                transition: var(--transition);
                border: 1px solid var(--border-color);
                position: relative; /* Needed for potential absolute positioned elements inside */
                overflow: hidden; /* Hide overflow */
            }

            .post-card:hover {
                transform: translateY(-5px) scale(1.01); /* Subtle lift and scale */
                box-shadow: 0 10px 20px rgba(76, 175, 80, 0.25); /* Enhanced shadow on hover */
                border-color: var(--primary-color);
            }

            .post-card h3 {
                color: var(--primary-light);
                font-size: 1.6rem; /* Slightly larger title */
                font-weight: 700;
                margin-bottom: 15px;
                line-height: 1.4;
            }
             /* Style for smaller post titles */
            .post-card strong {
                 color: var(--primary-light);
                 font-size: 1.1rem;
                 font-weight: 700;
                 display: block; /* Ensure it takes its own line */
                 margin-bottom: 8px;
            }

            .post-card p {
                color: var(--text-secondary);
                font-size: 1rem;
                line-height: 1.7; /* Increased line height for readability */
                margin-bottom: 20px;
            }

            .post-card img {
                width: 100%;
                max-height: 400px; /* Limit image height */
                object-fit: cover; /* Cover the area nicely */
                border-radius: var(--border-radius);
                margin: 15px 0;
                border: 1px solid var(--border-color);
                transition: var(--transition);
                opacity: 0.9; /* Slightly transparent */
            }

            .post-card:hover img {
                opacity: 1; /* Full opacity on hover */
                transform: scale(1.03); /* Slight zoom on image */
            }

            .post-stats { /* Renamed from 'stats' */
                text-align: center;
                color: var(--text-secondary);
                border-top: 1px solid var(--border-color);
                margin-top: 20px;
                padding-top: 15px;
                font-size: 0.9rem;
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 15px; /* More space between stats */
                flex-wrap: wrap; /* Allow wrapping on small screens */
            }

            .post-stats span {
                display: flex;
                align-items: center;
                gap: 6px; /* Space between icon and number */
            }

            .post-stats i {
                color: var(--accent-color); /* Amber color for icons */
                font-size: 1.1em;
            }
            .category-label {
                background-color: rgba(76, 175, 80, 0.15);
                padding: 3px 8px;
                border-radius: 5px;
                color: var(--primary-light);
                font-size: 0.85em;
            }

            /* --- Comment Card Styles --- */
            .comment-card {
                background-color: rgba(38, 50, 56, 0.5); /* Semi-transparent dark grey */
                border-right: 5px solid var(--primary-color); /* Thicker border */
                padding: 18px;
                margin-bottom: 15px;
                border-radius: var(--border-radius);
                color: var(--text-primary);
                box-shadow: 0 3px 6px rgba(0, 0, 0, 0.25);
                transition: var(--transition);
                 backdrop-filter: blur(3px); /* Frosted glass effect (if supported) */
            }

            .comment-card:hover {
                transform: translateX(-5px); /* Slide slightly left on hover */
                box-shadow: 0 6px 12px rgba(76, 175, 80, 0.2);
                 border-right-color: var(--accent-color); /* Change border color on hover */
            }
            .comment-card p {
                margin-bottom: 10px; /* Space between text and stats */
                line-height: 1.6;
                color: var(--text-primary);
            }
            .comment-card small {
                color: var(--primary-light);
                font-size: 0.9em;
                display: flex;
                align-items: center;
                gap: 8px; /* More space for comment stats */
                margin-top: 10px;
            }

            .comment-card small i {
                color: var(--accent-color);
            }

             /* --- Search Input --- */
            .stTextInput input {
                background-color: var(--bg-card) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: var(--border-radius) !important;
                color: var(--text-primary) !important;
                padding: 12px 20px !important;
                font-size: 1rem !important;
                transition: var(--transition) !important;
                font-family: 'Tajawal', sans-serif !important;
            }

            .stTextInput input:focus {
                border-color: var(--primary-color) !important;
                outline: none !important;
                box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.3) !important;
            }

            .stTextInput label { /* Style the label */
                 color: var(--text-secondary);
                 font-size: 0.95rem;
                 padding-bottom: 5px; /* Add space below label */
             }

            /* --- Details Page Specific --- */
            .details-title {
                 text-align: center;
                 margin-bottom: 25px;
                 font-size: 2.2rem;
                 color: var(--primary-color);
            }
             .details-description {
                 background-color: var(--bg-card);
                 padding: 20px;
                 border-radius: var(--border-radius);
                 margin-bottom: 20px;
                 line-height: 1.8;
                 color: var(--text-secondary);
             }
             .details-stats {
                 /* Similar to post-stats but maybe slightly different context */
                 text-align: center;
                 color: var(--text-secondary);
                 border-top: 1px solid var(--border-color);
                 border-bottom: 1px solid var(--border-color);
                 margin: 25px 0;
                 padding: 15px 0;
                 font-size: 1rem;
                 display: flex;
                 justify-content: center;
                 align-items: center;
                 gap: 20px;
                 flex-wrap: wrap;
            }
            .details-stats span { display: flex; align-items: center; gap: 8px; }
            .details-stats i { color: var(--accent-color); }

             .back-button-container { /* Container for the back button */
                 margin-bottom: 20px;
             }
             /* Style the back button differently */
             .stButton[key="back_to_main"]>button {
                 background: transparent !important;
                 color: var(--primary-light) !important;
                 border: 1px solid var(--primary-color) !important;
                 padding: 8px 20px !important;
                 border-radius: var(--border-radius) !important;
                 font-weight: 500 !important;
                 box-shadow: none !important;
             }
            .stButton[key="back_to_main"]>button:hover {
                 background: rgba(76, 175, 80, 0.1) !important;
                 transform: none !important; /* Disable lift effect */
            }

            /* Style analysis/summary buttons */
            .stButton[key="generate_summary"]>button,
            .stButton[key="run_analysis"]>button,
            .stButton[key="analyze_topics"]>button {
                background: linear-gradient(to right, var(--accent-color), #FFB300) !important; /* Amber gradient */
                color: #111 !important; /* Dark text for contrast */
            }
            .stButton[key="generate_summary"]>button:hover,
            .stButton[key="run_analysis"]>button:hover,
            .stButton[key="analyze_topics"]>button:hover {
                 background: linear-gradient(to right, #FFB300, #FF8F00) !important; /* Darker amber gradient */
            }


             /* --- Media Query for Responsiveness --- */
            @media (max-width: 768px) {
                .post-card h3 { font-size: 1.4rem; }
                .post-card p { font-size: 0.95rem; }
                .details-title { font-size: 1.8rem; }
                /* Adjust columns or layout for smaller screens if needed */
                /* Example: stack columns */
                 .stButton>button { padding: 10px 20px !important; font-size: 0.9rem;} /* Slightly smaller buttons */
            }
             @media (max-width: 992px) { /* Target typical sidebar collapse breakpoint */
                 [data-testid="stSidebar"] {
                      /* Styles for when sidebar might be collapsed or in overlay mode */
                 }
             }

        </style>
    ''', unsafe_allow_html=True)

# ---------------------------
# دالة عرض الغاليري (بدون تغيير)
# ---------------------------
def render_trends_analysis():
    analyzer = FacebookTrendAnalyzer()
    st.title("📈 تحليل الترندات")

    if st.button("تحليل الترندات من البيانات الحالية"):
        posts_df, _ = load_data()
        if posts_df is not None:
            with st.spinner("جارٍ تحليل البيانات..."):
                trends = analyzer.calculate_trends(posts_df)
                if trends:
                    veracity_df = analyzer.calculate_veracity_scores(trends)
                    analyzer.display_dashboard(trends, veracity_df)

                    # Add download button
                    csv = veracity_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "تحميل التقرير",
                        data=csv,
                        file_name="facebook_trends_report.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("لم يتم العثور على ترندات في البيانات الحالية")
        else:
            st.error("فشل تحميل البيانات")
def render_gallery(posts_df):
    st.markdown("#### معرض الصور")
    posts_with_images = posts_df[posts_df['image'].apply(is_valid_media_link)].copy()  # <-- أضف .copy() هنا

    # تأكد من وجود عمود popularity_score قبل الترتيب
    if 'popularity_score' not in posts_with_images.columns:
        # إذا لم يكن موجوداً، قم بإنشائه بنفس الطريقة المستخدمة في get_top_posts
        if 'likes_count' in posts_with_images.columns and 'comments_count' in posts_with_images.columns:
            posts_with_images['popularity_score'] = posts_with_images['likes_count'] * 0.7 + posts_with_images['comments_count'] * 0.3
        elif 'likes_count' in posts_with_images.columns:
            posts_with_images['popularity_score'] = posts_with_images['likes_count']
        else:
            posts_with_images['popularity_score'] = 0

    # الآن يمكنك الترتيب بأمان
    posts_with_images.sort_values('popularity_score', ascending=False, inplace=True)

    if not posts_with_images.empty:
        image_count = 0
        max_gallery_images = 15
        gallery_cols = st.columns(2)
        col_idx = 0
        for _, post in posts_with_images.iterrows():
            if image_count < max_gallery_images:
                with gallery_cols[col_idx % 2]:
                    img_src = post['image']
                    title_alt = post.get('article_title', 'صورة')[:30]
                    st.image(
                        img_src,
                        caption=f"{title_alt}...",
                        use_container_width=True,
                        output_format="auto"
                    )
                image_count += 1
                col_idx += 1
            else:
                break
        if len(posts_with_images) > image_count:
            st.caption(f"عرض أول {image_count} صورة من أصل {len(posts_with_images)}...")
    else:
        st.info("لا توجد صور متاحة في الوقت الحالي.")
    st.markdown("---") # Separator

# ---------------------------
# دالة عرض القائمة الرئيسية (مع البحث والتحديثات)
# ---------------------------
def render_main_feed(posts_df, comments_df):
    st.markdown("## <i class='fas fa-newspaper'></i> آخر أخبار الرياضة", unsafe_allow_html=True)

    # --- Category Buttons ---
    st.markdown("#### <i class='fas fa-tags'></i> اختر القسم:", unsafe_allow_html=True)
    unique_categories = sorted([cat for cat in posts_df['category'].unique() if cat != "أخرى"])
    if "أخرى" in posts_df['category'].unique():
        unique_categories.append("أخرى")
    categories_to_display = ["كل الفئات"] + unique_categories

    category_style = """
       <style>
          .category-btn {
               font-family: 'Cairo', 'Arial', 'Droid Arabic Naskh', sans-serif !important;
               font-size: 4em !important;
               font-weight: bold !important;
               letter-spacing: 0 !important;
             }
          .category-btn:hover {
               background-color: #4CAF50 !important;
            }
      </style>
      """


    # Display category buttons more dynamically
    cols = st.columns(len(categories_to_display))
    for i, cat in enumerate(categories_to_display):
         with cols[i]:
             # Use a more descriptive key including the category name
             button_key = f"cat_btn_{cat.replace(' ', '_').replace('1', '_one')}" # Make key more robust
             if st.button(cat, key=button_key, use_container_width=True):
                 st.session_state.selected_category = cat
                 st.session_state.show_gallery = False # Reset gallery view on category change
                 st.session_state.search_term = "" # Reset search term on category change
                 st.rerun()

    st.markdown(f"##### <i class='fas fa-eye'></i> تعرض الآن: {st.session_state.selected_category}", unsafe_allow_html=True)
    st.divider()

    # --- Main Layout (Posts and Top Comments) ---
    # Using columns for layout flexibility
    center_col, right_col = st.columns([3, 1.5]) # Main feed, Top comments

    # --- Filtering Posts ---
    if st.session_state.selected_category == "كل الفئات":
        filtered_posts = posts_df.copy()
    else:
        filtered_posts = posts_df[posts_df['category'] == st.session_state.selected_category].copy()

    # Apply search filter if a search term exists
    search_term = st.session_state.get('search_term', "") # Use .get for safety
    if search_term:
        search_term_lower = search_term.lower()
        # Combine search conditions using boolean indexing for efficiency
        search_condition = (
            filtered_posts['article_title'].str.lower().str.contains(search_term_lower, na=False) |
            filtered_posts['short_description'].str.lower().str.contains(search_term_lower, na=False)
        )
        filtered_posts = filtered_posts[search_condition]

    # Sort the filtered posts by popularity
    if not filtered_posts.empty:
        # Ensure popularity score exists before sorting
        if 'popularity_score' not in filtered_posts.columns:
            # Recalculate if missing (shouldn't happen with current flow, but safety check)
             filtered_posts = get_top_posts(filtered_posts)
        filtered_posts = filtered_posts.sort_values('popularity_score', ascending=False)
        top_cat_post = filtered_posts.head(1)
        other_posts = filtered_posts.iloc[1:]
    else:
        top_cat_post = pd.DataFrame() # Empty DataFrame if no results
        other_posts = pd.DataFrame()

    # --- Center Column: Display Posts ---
    with center_col:
        if search_term:
             st.info(f"🔍 نتائج البحث عن: '{search_term}' ضمن قسم '{st.session_state.get('selected_category', 'الكل')}'")

        if filtered_posts.empty:
            if search_term:
                st.warning(f"لم يتم العثور على نتائج تطابق بحثك '{search_term}'. جرب كلمة أخرى أو قسمًا آخر.")
            else:
                st.warning(f"لا توجد منشورات لعرضها في قسم '{st.session_state.selected_category}'.")
        else:
            # --- Display Top Post ---
            if not top_cat_post.empty:
                st.markdown("### <i class='fas fa-star'></i> المنشور الأبرز", unsafe_allow_html=True)
                post = top_cat_post.iloc[0]
                with st.container(): # Use container for better card structure
                    st.markdown("<div class='post-card top-post-card'>", unsafe_allow_html=True) # Add a class if specific styling needed

                    st.markdown(f"<h3>{post.get('article_title', 'N/A')}</h3>", unsafe_allow_html=True)
                    desc = post.get('short_description', '')
                    st.markdown(f"<p>{desc[:250]}{'...' if len(desc) > 250 else ''}</p>", unsafe_allow_html=True) # Show more description

                    image_url = post.get('image')
                    video_url = post.get('video_link')
                    if is_valid_media_link(image_url):
                        # Use markdown for more control over image style if needed
                        st.markdown(f'<img src="{image_url}" alt="صورة للمنشور" onerror="this.style.display=\'none\'">', unsafe_allow_html=True)
                    if is_valid_media_link(video_url):
                         st.markdown("<small style='color: var(--accent-color);'><i class='fas fa-video'></i> <i>يحتوي على فيديو</i></small>", unsafe_allow_html=True)


                    # --- Stats and Actions for Top Post ---
                    stats_html = f"""
                    <div class="post-stats">
                        <span><i class="fas fa-thumbs-up"></i> {post.get('likes_count', 0)}</span>
                        <span><i class="fas fa-comments"></i> {post.get('comments_count', 0)}</span>
                        <span class="category-label">{post.get('category', 'N/A')}</span>
                    </div>
                    """
                    st.markdown(stats_html, unsafe_allow_html=True)

                    # Buttons row
                    btn_cols = st.columns(2)
                    with btn_cols[0]:
                        # Use icon in button
                        if st.button("📄 عرض التفاصيل", key=f"details_{post.get('post_id')}", help="عرض تفاصيل هذا المنشور وتحليل التعليقات", use_container_width=True):
                            st.session_state.view = 'details'
                            st.session_state.selected_post_id = post.get('post_id')
                            st.rerun()
                    with btn_cols[1]:
                        article_link = post.get('article_link')
                        if is_valid_url(article_link):
                            # Use markdown for a styled link button if st.button isn't enough
                             st.link_button("🔗 المقال الأصلي", url=article_link, use_container_width=True)
                        else:
                            st.button("🔗 المقال الأصلي", disabled=True, use_container_width=True, help="الرابط غير متوفر")


                    st.markdown("</div>", unsafe_allow_html=True) # Close post-card div

            # --- Display Other Posts ---
            if not other_posts.empty:
                st.markdown("---")
                st.markdown("### <i class='far fa-list-alt'></i> منشورات أخرى", unsafe_allow_html=True)
                num_other_posts = 10 # Number of other posts to show
                for _, post in other_posts.head(num_other_posts).iterrows():
                     with st.container(): # Container for each smaller post
                        st.markdown("<div class='post-card'>", unsafe_allow_html=True) # Use the same card class

                        # Title and Short Description
                        st.markdown(f"<strong>{post.get('article_title', 'N/A')}</strong>", unsafe_allow_html=True)
                        desc_other = post.get('short_description', '')
                        st.markdown(f"<p style='font-size: 0.9em; color: var(--text-secondary);'>{desc_other[:150]}{'...' if len(desc_other) > 150 else ''}</p>", unsafe_allow_html=True)

                        # Optional: Smaller Image for other posts
                        image_url_other = post.get('image')
                        if is_valid_media_link(image_url_other):
                             st.markdown(f'<img src="{image_url_other}" alt="صورة مصغرة" style="max-height: 150px; width:auto; margin: 5px 0; border-radius: 5px; display: block; margin-left: auto; margin-right: auto;">', unsafe_allow_html=True)


                        # Stats for other posts
                        stats_other_html = f"""
                        <div class="post-stats" style="font-size: 0.85em; padding-top: 10px; margin-top: 10px;">
                            <span><i class="fas fa-thumbs-up"></i> {post.get('likes_count', 0)}</span>
                            <span><i class="fas fa-comments"></i> {post.get('comments_count', 0)}</span>
                             <span class="category-label" style="font-size: 0.9em;">{post.get('category', 'N/A')}</span>
                        </div>"""
                        st.markdown(stats_other_html, unsafe_allow_html=True)

                        # Buttons for other posts (smaller or just details)
                        btn_cols_other = st.columns(2)
                        with btn_cols_other[0]:
                             if st.button("📄 التفاصيل", key=f"details_{post.get('post_id')}", help="عرض التفاصيل", use_container_width=True):
                                 st.session_state.view = 'details'
                                 st.session_state.selected_post_id = post.get('post_id')
                                 st.rerun()
                        with btn_cols_other[1]:
                             article_link_other = post.get('article_link')
                             if is_valid_url(article_link_other):
                                st.link_button("🔗 المصدر", url=article_link_other, use_container_width=True)
                             else:
                                st.button("🔗 المصدر", disabled=True, use_container_width=True)


                        st.markdown("</div>", unsafe_allow_html=True) # Close post-card div

                if len(other_posts) > num_other_posts:
                    st.caption(f"عرض أول {num_other_posts} منشور من {len(other_posts)}...")

    # --- Right Column: Top Comments for the *Displayed Top Post* ---
    with right_col:
        st.markdown("### <i class='fas fa-comment-dots'></i> أبرز تعليقات المنشور الأبرز", unsafe_allow_html=True)
        if not top_cat_post.empty:
            # st.markdown("<small>(على المنشور الأبرز المعروض في الأعلى)</small>", unsafe_allow_html=True)
            top_post_id_filtered = top_cat_post.iloc[0].get('post_id')

            # Check if required comment columns exist
            required_cols = ['post_id', 'text', 'likes']
            if top_post_id_filtered and all(col in comments_df.columns for col in required_cols):
                # Filter comments, sort by likes, take top N
                post_comments = comments_df[comments_df['post_id'] == str(top_post_id_filtered)].copy() # Ensure post_id is string for comparison
                if not post_comments.empty:
                    post_comments = post_comments.sort_values('likes', ascending=False).head(7) # Get top 7 comments
                    st.markdown("---")
                    for _, comment in post_comments.iterrows():
                        text = comment.get('text', '')
                        likes = comment.get('likes', 0)
                        # author = comment.get('author', 'غير معروف') # Get author if available
                        # Display comment with author and likes
                        st.markdown(f"""
                        <div class="comment-card">
                            <p>{text}</p>
                            <small>
                                <i class="fas fa-thumbs-up"></i> {likes}
                            </small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("لا توجد تعليقات لعرضها لهذا المنشور.")
            elif not all(col in comments_df.columns for col in required_cols):
                 st.warning(f"ملف التعليقات يفتقد لبعض الأعمدة المطلوبة ({', '.join(required_cols)}) لعرض أبرز التعليقات.")
            else:
                 # Should not happen if top_cat_post is not empty, but safety check
                 st.info("لم يتم تحديد منشور أبرز لعرض تعليقاته.")

        elif search_term and filtered_posts.empty:
             st.info("لا يوجد منشور أبرز بسبب نتائج البحث الحالية.")
        else:
            st.info("اختر قسماً أو منشوراً لعرض أبرز التعليقات.")


# ---------------------------
# دالة عرض صفحة التفاصيل (مع التحسينات)
# ---------------------------
def render_details_page(post_id, posts_df, comments_df):
    # --- Back Button ---
    # Place it prominently at the top
    st.markdown('<div class="back-button-container">', unsafe_allow_html=True)
    if st.button("⬅️ العودة للقائمة الرئيسية", key="back_to_main"):
        st.session_state.view = 'main'
        st.session_state.selected_post_id = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Get Post Details ---
    post_details = posts_df[posts_df['post_id'] == str(post_id)] # Ensure comparison with string
    if post_details.empty:
        st.error("لم يتم العثور على تفاصيل المنشور المحدد. ربما تم حذفه أو تغيير معرفه.")
        st.warning("سيتم إعادتك للقائمة الرئيسية.")
        time.sleep(2)
        st.session_state.view = 'main'
        st.session_state.selected_post_id = None
        st.rerun()
        st.stop() # Stop execution for this run

    post = post_details.iloc[0]

    # --- Main Details Container ---
    with st.container():
        st.markdown('<div class="details-container">', unsafe_allow_html=True) # Optional class for overall container

        # --- Title ---
        st.markdown(f"<h2 class='details-title'>{post.get('article_title', 'N/A')}</h2>", unsafe_allow_html=True)

        # --- Description and Summary ---
        # Use long description if available, otherwise short description
        description = post.get('long_description', post.get('short_description', ''))
        if not description: # Handle case where both might be missing
             description = "لا يتوفر وصف لهذا المنشور."

        st.markdown("### <i class='fas fa-file-alt'></i> الوصف الكامل", unsafe_allow_html=True)
        st.markdown(f"<div class='details-description'>{description}</div>", unsafe_allow_html=True)

        # --- AI Summary Section (Conditional) ---
        # Check length before offering summary
        if len(description.split()) > 40: # Only offer summary for longer texts
            st.markdown("### <i class='fas fa-brain'></i> ملخص النص", unsafe_allow_html=True)
            if st.button("💡 إنشاء ملخص", key="generate_summary"):
                # Use spinner for better UX
                with st.spinner("⏳ جارٍ إنشاء الملخص... قد يستغرق بعض الوقت."):
                    summary = summarize_arabic(description) # Call the summary function
                    # Display the summary in a styled box
                    st.markdown(
                        f"""
                        <div style='background-color: rgba(76, 175, 80, 0.1);
                                    padding: 15px;
                                    border-radius: var(--border-radius);
                                    border-left: 4px solid var(--primary-color);
                                    margin-top: 10px;'>
                            <strong>ملخص:</strong><br>
                            {summary}
                        </div>
                        """,
                        unsafe_allow_html=True)
            st.divider() # Separator after summary section

        # --- Media Display (Image and Video) ---
        media_cols = st.columns(2) # Display image and video side-by-side if both exist
        image_displayed = False
        video_displayed = False

        with media_cols[0]:
            image_url = post.get('image')
            if is_valid_media_link(image_url):
                st.markdown("#### <i class='fas fa-image'></i> الصورة", unsafe_allow_html=True)
                st.image(image_url, caption=f"صورة: {post.get('article_title', '')[:50]}...", use_container_width=True)
                image_displayed = True

        with media_cols[1]:
            video_url = post.get('video_link')
            article_link_for_video = post.get("article_link", "#")
            article_link_text = f'<a href="{article_link_for_video}" target="_blank" rel="noopener noreferrer">رابط المقال الأصلي</a>' if is_valid_url(article_link_for_video) else "المقال الأصلي (الرابط غير متاح)"

            if is_valid_media_link(video_url):
                st.markdown("#### <i class='fas fa-video'></i> الفيديو", unsafe_allow_html=True)
                # Specific handling for Dailymotion iframe
                if 'dailymotion.com/player' in video_url or 'dai.ly' in video_url:
                     # Extract video ID if it's a share link like dai.ly/xyz
                     if 'dai.ly' in video_url:
                         try:
                             # Basic extraction, might need refinement
                             video_id = video_url.split('/')[-1].split('?')[0]
                             embed_url = f"https://www.dailymotion.com/embed/video/{video_id}"
                             video_url = embed_url # Replace with embed URL
                         except Exception:
                             st.warning("لم نتمكن من تحويل رابط Dailymotion المختصر. سنحاول استخدام الرابط كما هو.")

                     try:
                        iframe_height = 315 # Standard 16:9 aspect ratio based on default width
                        # Use a wrapper div for potential future styling
                        st.markdown('<div class="iframe-video-wrapper">', unsafe_allow_html=True)
                        components.iframe(video_url, height=iframe_height, scrolling=False)
                        st.markdown('</div>', unsafe_allow_html=True)
                        video_displayed = True
                     except Exception as e:
                        st.error(f"حدث خطأ أثناء محاولة تضمين فيديو Dailymotion: {e}")
                        st.markdown(
                            f'''<div class="video-error-box" style='border: 1px solid var(--accent-color); padding: 10px; border-radius: var(--border-radius); background-color: rgba(255, 193, 7, 0.1);'>
                                <small><i class="fas fa-exclamation-triangle"></i> يمكنك محاولة مشاهدة الفيديو مباشرة عبر {article_link_text}.</small>
                            </div>''', unsafe_allow_html=True)
                else: # Handle other video links (e.g., MP4, YouTube if link is direct)
                    try:
                        st.video(video_url)
                        video_displayed = True
                    except Exception as video_error:
                        st.warning(f"تعذر عرض الفيديو مباشرة (قد يكون بتنسيق غير مدعوم).")
                        st.markdown(
                            f'''<div class="video-error-box" style='border: 1px solid var(--accent-color); padding: 10px; border-radius: var(--border-radius); background-color: rgba(255, 193, 7, 0.1);'>
                                <small><i class="fas fa-link"></i> حاول فتح رابط الفيديو مباشرة: <a href="{video_url}" target="_blank" rel="noopener noreferrer">اضغط هنا</a></small><br>
                                <small><i class="fas fa-newspaper"></i> أو قم بزيارة {article_link_text}.</small>
                            </div>''', unsafe_allow_html=True)
            # Handle cases where video_link exists but isn't a standard URL (e.g., just text saying 'video')
            elif video_url and video_url not in ['unavailable', 'no video', '', 'nan', 'none', 'null']:
                 st.markdown("#### <i class='fas fa-video'></i> الفيديو", unsafe_allow_html=True)
                 st.info(f"ℹ️ تم العثور على إشارة لوجود فيديو، ولكن لا يمكن عرضه مباشرة هنا.")
                 st.markdown(
                        f'''<div class="video-error-box" style='border: 1px solid #17a2b8; padding: 10px; border-radius: var(--border-radius); background-color: rgba(23, 162, 184, 0.1);'>
                             <small><i class="fas fa-info-circle"></i> أفضل طريقة لمشاهدة هذا الفيديو هي غالبًا عبر زيارة {article_link_text}.</small>
                         </div>''', unsafe_allow_html=True)

        # Add divider if media was displayed
        if image_displayed or video_displayed:
             st.divider()

        # --- Statistics and Original Article Link ---
        st.markdown("### <i class='fas fa-chart-bar'></i> إحصائيات وروابط", unsafe_allow_html=True)
        stats_html = f"""
        <div class="details-stats">
            <span><i class="fas fa-thumbs-up"></i> {post.get('likes_count', 0)} إعجاب</span>
            <span><i class="fas fa-comments"></i> {post.get('comments_count', 0)} تعليق</span>
            <span class="category-label"><i class="fas fa-tag"></i> {post.get('category', 'N/A')}</span>
        </div>
        """
        st.markdown(stats_html, unsafe_allow_html=True)

        article_link = post.get('article_link')
        if is_valid_url(article_link):
             # Use st.link_button for a consistent look
             st.link_button("🔗 زيارة المقال الأصلي للمزيد من المعلومات", url=article_link, use_container_width=True)
        else:
             st.info("رابط المقال الأصلي غير متوفر لهذا المنشور.")

        st.divider()

        # --- Sentiment and Topic Analysis Section ---
        st.markdown("### <i class='fas fa-smile-beam'></i><i class='fas fa-angry'></i> تحليل محتوى التعليقات", unsafe_allow_html=True)

        # Get comments for this post
        post_comments_df = comments_df[comments_df['post_id'] == str(post_id)].copy()
        post_comments_list = post_comments_df['text'].dropna().tolist()

        if not post_comments_list:
             st.info("لا توجد تعليقات متاحة لهذا المنشور لتحليلها.")
        else:
            analysis_tabs = st.tabs(["📊 تحليل المشاعر", "🎭 تحليل العواطف", "📝 تحليل المواضيع"])

            with analysis_tabs[0]: # Sentiment Analysis
                st.markdown("#### تحليل المشاعر (إيجابي/سلبي/محايد)")
                if st.button("تحليل المشاعر الآن", key="run_sentiment_analysis"):
                    with st.spinner("⏳ جارٍ تحليل مشاعر التعليقات..."):
                         try:
                            # Ensure the sentiment analysis module and functions are available
                            from analyse_sentiment import predict_sentiment, labels_map_sentiment
                            predictions = predict_sentiment(post_comments_list)
                            if predictions is not None:
                                df_results = pd.DataFrame({
                                    # "التعليق": post_comments_list, # Keep it concise, show stats
                                    "التصنيف": [labels_map_sentiment.get(p.item(), "غير معروف") for p in predictions]
                                })

                                # Show statistics
                                st.subheader("إحصائيات المشاعر")
                                stats = df_results['التصنيف'].value_counts().reset_index()
                                stats.columns = ['المشاعر', 'العدد']

                                # Display stats and chart side-by-side
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    st.dataframe(stats, use_container_width=True)
                                with col2:
                                    try:
                                        fig, ax = plt.subplots()
                                        # Use matplotlib directly for more control if needed
                                        ax.bar(stats['المشاعر'], stats['العدد'], color=['#4CAF50', '#FFC107', '#F44336']) # Green, Amber, Red
                                        plt.xticks(rotation=0)
                                        st.pyplot(fig)
                                    except Exception as plot_err:
                                        st.error(f"خطأ في رسم المخطط: {plot_err}")
                                        st.bar_chart(stats.set_index('المشاعر')) # Fallback to st.bar_chart
                            else:
                                st.error("فشل تحليل المشاعر.")
                         except ImportError:
                             st.error("الوحدة 'analyse_sentiment' غير موجودة. لا يمكن إجراء تحليل المشاعر.")
                         except Exception as e:
                             st.error(f"خطأ غير متوقع أثناء تحليل المشاعر: {e}")

            with analysis_tabs[1]:  # Emotion Analysis
                st.markdown("#### تحليل العواطف (فرح، حزن، غضب...)")
                if st.button("تحليل العواطف الآن", key="run_emotion_analysis"):
                    with st.spinner("⏳ جارٍ تحليل عواطف التعليقات..."):
                        try:
                            from analyse_sentiment import predict_sentiment, predict_emotion, labels_map_sentiment, \
                                labels_map_emotion

                            # D'abord analyser les sentiments
                            sentiment_predictions = predict_sentiment(post_comments_list)

                            # Filtrer les commentaires non-neutres et avec relation
                            comments_to_analyze = []
                            sentiment_labels = []
                            for comment, pred in zip(post_comments_list, sentiment_predictions):
                                sentiment = labels_map_sentiment.get(pred.item(), "")
                                if sentiment not in ["neutre", "no_relation"]:
                                    comments_to_analyze.append(comment)
                                    sentiment_labels.append(sentiment)

                            if not comments_to_analyze:
                                st.warning("جميع التعليقات محايدة أو بدون علاقة - لا يوجد ما لتحليله")
                            else:
                                # Analyser seulement les commentaires filtrés
                                emotion_predictions = predict_emotion(comments_to_analyze)

                                # Préparer les résultats complets
                                full_emotion_results = ["غير مطبق" for _ in post_comments_list]
                                emo_idx = 0
                                for i, sentiment in enumerate(sentiment_predictions):
                                    if labels_map_sentiment.get(sentiment.item(), "") not in ["neutre", "no_relation"]:
                                        if emo_idx < len(emotion_predictions):
                                            full_emotion_results[i] = labels_map_emotion.get(
                                                emotion_predictions[emo_idx].item(), "غير معروف")
                                            emo_idx += 1

                                # Afficher les statistiques
                                df_results = pd.DataFrame({
                                    "التعليق": post_comments_list,
                                    "المشاعر": [labels_map_sentiment.get(p.item(), "غير معروف") for p in
                                                sentiment_predictions],
                                    "العاطفة": full_emotion_results
                                })

                                # Filtrer pour les statistiques (uniquement les commentaires analysés)
                                stats_df = df_results[df_results['العاطفة'] != "غير مطبق"]

                                if not stats_df.empty:
                                    st.subheader("إحصائيات العواطف (للتعليقات غير المحايدة فقط)")
                                    stats = stats_df['العاطفة'].value_counts().reset_index()
                                    stats.columns = ['العاطفة', 'العدد']

                                    col1, col2 = st.columns([1, 2])
                                    with col1:
                                        st.dataframe(stats, use_container_width=True)
                                    with col2:
                                        try:
                                            fig, ax = plt.subplots(figsize=(8, 5))
                                            ax.pie(stats['العدد'], labels=stats['العاطفة'], autopct='%1.1f%%',
                                                   startangle=90)
                                            st.pyplot(fig)
                                        except Exception as plot_err:
                                            st.error(f"خطأ في رسم المخطط الدائري: {plot_err}")
                                else:
                                    st.info("لا توجد عواطف يمكن تحليلها في التعليقات")

                        except ImportError:
                            st.error("الوحدة 'analyse_sentiment' غير موجودة. لا يمكن إجراء تحليل العواطف.")
                        except Exception as e:
                            st.error(f"خطأ غير متوقع أثناء تحليل العواطف: {e}")

            with analysis_tabs[2]:  # Topic Analysis
                st.markdown("#### تحليل المواضيع الرئيسية في التعليقات")
                min_comments_for_topics = 10  # Increased minimum comments
                if len(post_comments_list) >= min_comments_for_topics:
                    if st.button("تحليل المواضيع الآن", key="analyze_topics"):
                        with st.spinner("⏳ جارٍ تحليل مواضيع التعليقات... (قد يستغرق بعض الوقت)"):
                            try:
                                from topic import analyze_comments_topics
                                result, error = analyze_comments_topics(post_comments_list)

                                if error:
                                    if "No meaningful topics found" in error:
                                        st.warning("⚠️ لم يتم العثور على مواضيع واضحة في التعليقات. قد يكون السبب:")
                                        st.markdown("""
                                        - تنوع التعليقات الشديد بدون أنماط واضحة
                                        - قصر التعليقات أو عدم احتوائها على كلمات دلالية كافية
                                        - استخدام لغة عامية يصعب تحليلها
                                        """)
                                    else:
                                        st.error(f"خطأ في تحليل المواضيع: {error}")
                                elif result:
                                    if "error" in result:
                                        st.warning(result["error"])
                                    else:
                                        # Show topic info table
                                        st.markdown("##### المواضيع المكتشفة والكلمات المرتبطة:")
                                        if "topics" in result and not result["topics"].empty:
                                            st.dataframe(result["topics"], use_container_width=True, hide_index=True)

                                        # Show visualization if available
                                        if "visualization" in result and result["visualization"]:
                                            st.markdown("##### تصور المواضيع:")
                                            st.plotly_chart(result["visualization"], use_container_width=True)

                                        # Show generated hashtags if available
                                        if result.get("hashtags"):
                                            st.markdown("##### وسوم مقترحة:")
                                            st.info(" ".join(result["hashtags"]))
                            except ImportError:
                                st.error("الوحدة 'topic.py' أو الاعتماديات المطلوبة غير موجودة.")
                            except Exception as e:
                                st.error(f"خطأ غير متوقع أثناء تحليل المواضيع: {e}")
                else:
                    st.warning(
                        f"لا توجد تعليقات كافية لتحليل المواضيع (يحتاج إلى {min_comments_for_topics} على الأقل، يوجد {len(post_comments_list)}).")


        st.divider()
        # --- Comments Display Section ---
        st.markdown('### <i class="fas fa-comments"></i> التعليقات على المنشور', unsafe_allow_html=True)
        # Re-fetch or use the previously fetched comments dataframe
        if not post_comments_df.empty:
            # Sort comments by likes (descending) for display
            post_comments_df = post_comments_df.sort_values('likes', ascending=False)

            num_comments_to_show = st.slider("اختر عدد التعليقات لعرضها:", 5, min(50, len(post_comments_df)), 10) # Slider to choose how many comments

            st.markdown(f"##### عرض أفضل {num_comments_to_show} تعليقات (مرتبة حسب الإعجابات):")
            for _, comment in post_comments_df.head(num_comments_to_show).iterrows():
                text = comment.get('text', '')
                likes = comment.get('likes', 0)
                st.markdown(f"""
                <div class="comment-card">
                     <p>{text}</p>
                     <small>
                         <i class="fas fa-thumbs-up"></i> {likes}
                     </small>
                 </div>
                 """, unsafe_allow_html=True)

            if len(post_comments_df) > num_comments_to_show:
                 st.caption(f"يوجد {len(post_comments_df) - num_comments_to_show} تعليقات أخرى...")
        else:
             # This check is slightly redundant due to the check for analysis, but good practice
             st.info("لا توجد تعليقات لعرضها لهذا المنشور.")


        st.markdown('</div>', unsafe_allow_html=True) # Close details-container div

# ---------------------------
# الدالة الرئيسية للتطبيق (مع الـ Sidebar)
# ---------------------------
def main():
    page_title = "كابتن تراند | تحليلات رياضية"
    page_title = "كابتن تراند | تحليلات رياضية"
    page_icon = LOGO_PATH if os.path.exists(LOGO_PATH) else "⚽"  # Fallback icon
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")

    apply_css()  # Apply custom styles

    # --- Initialize Session State Variables ---
    if "view" not in st.session_state: st.session_state.view = "main"
    if "selected_post_id" not in st.session_state: st.session_state.selected_post_id = None
    if "selected_category" not in st.session_state: st.session_state.selected_category = "كل الفئات"
    if "show_gallery" not in st.session_state: st.session_state.show_gallery = False
    if 'search_term' not in st.session_state: st.session_state.search_term = ""
    if "show_trends" not in st.session_state:
        st.session_state.show_trends = False
    # --- Sidebar ---
    with st.sidebar:
        st.markdown('''
          <div class="sidebar-logo-container">
              <img class="sidebar-logo" src="data:image/png;base64,{}" alt="شعار التطبيق">
          </div>
        '''.format(get_base64_encoded_image(LOGO_PATH)), unsafe_allow_html=True)

        # Display Title in Sidebar
        st.markdown(f"<h1 class='sidebar-title'>{page_title}</h1>", unsafe_allow_html=True)
        st.markdown("---")

        # --- Search Bar Moved to Sidebar ---
        st.markdown("### <i class='fas fa-search'></i> بحث سريع", unsafe_allow_html=True)
        # Update search_term in session state directly on input change
        st.session_state.search_term = st.text_input(
            "ابحث في العناوين/الوصف:",
            value=st.session_state.search_term,
            key="search_input_sidebar", # Unique key for sidebar search
            help="اكتب كلمة للبحث في المنشورات المعروضة حالياً"
        )
        # Add a small delay or button press if instant search is too slow/jarring
        # if st.button("🔍 بحث", key="search_button_sidebar", use_container_width=True):
        #    st.rerun() # Rerun to apply search only on button press

        st.markdown("---")

        # --- Additional Tools in Sidebar ---
        st.markdown("### <i class='fas fa-tools'></i> أدوات إضافية", unsafe_allow_html=True)

        # Refresh Button
        if st.button("🔄 تحديث البيانات", key="refresh_data_sidebar", use_container_width=True):
            st.toast("⏳ جارٍ تحديث البيانات...", icon="⏳")
            # Clear cache if using st.cache_data
            # st.cache_data.clear()
            time.sleep(1) # Simulate refresh time
            st.rerun()

        # Toggle Gallery Button
        gallery_button_text = "🖼️ إخفاء معرض الصور" if st.session_state.show_gallery else "🖼️ عرض معرض الصور"
        if st.button(gallery_button_text, key="toggle_gallery_sidebar", use_container_width=True):
             st.session_state.show_gallery = not st.session_state.show_gallery
             st.rerun()
        if st.button("📊 تحليل الترندات", key="trends_analysis_button"):
            st.session_state.view = "trends"
            st.rerun()



        st.markdown("---")
         # Display gallery inside the sidebar if toggled on
        if st.session_state.show_gallery:
             # Load data *once* before potentially showing gallery
             posts_df_gallery, _ = load_data() # Load only posts if needed
             if posts_df_gallery is not None:
                 render_gallery(posts_df_gallery)
             else:
                 st.warning("لا يمكن عرض المعرض، فشل تحميل بيانات المنشورات.")


        st.markdown("---")
        st.caption("© 2024 - منصة كابتن تراند")


    # --- Load Data ---
    # Load data after setting up the sidebar, before rendering main content
    posts_df, comments_df = load_data()

    # Check if data loading failed
    if posts_df is None or comments_df is None:
        st.error("🚫 فشل تحميل البيانات الأساسية. لا يمكن متابعة عرض التطبيق.")
        st.warning(
            "يرجى التحقق من وجود ملفات `facebook_posts.csv` و `facebook_comments.csv` وأنها غير فارغة وتحتوي على الأعمدة المطلوبة.")
        st.stop()  # Halt execution if data loading fails

        # --- Process Data ---
    try:
        # Apply categorization and popularity scoring
        posts_df = categorize_posts(posts_df.copy())
        posts_df = get_top_posts(posts_df.copy())
    except Exception as e:
        st.error(f" terjadi kesalahan dalam pemrosesan data: {e}")
        st.exception(e)  # Print full traceback for debugging
        st.stop()


    # --- Main Content Area ---
    # st.markdown('<div class="main-content">', unsafe_allow_html=True) # Optional wrapper

    # --- Routing based on session state ---
    if st.session_state.view == "details" and st.session_state.selected_post_id is not None:
        render_details_page(st.session_state.selected_post_id, posts_df, comments_df)
    elif st.session_state.view == "trends":  # Add this condition
        render_trends_analysis()
    else:
        # Default to main feed view
        if st.session_state.view != "main":
            st.session_state.view = "main"  # Ensure view state is correct
            st.session_state.selected_post_id = None  # Clear selected post ID
        render_main_feed(posts_df, comments_df)

    # st.markdown('</div>', unsafe_allow_html=True) # Close main-content div


# ---------------------------
# نقطة الدخول الرئيسية (مع التحقق من وجود الملفات)
# ---------------------------
if __name__ == "__main__":
     # Check for essential files before starting the app
     if not os.path.exists(POSTS_FILE) or not os.path.exists(COMMENTS_FILE):
         st.error(f"خطأ فادح: ملف البيانات '{POSTS_FILE}' أو '{COMMENTS_FILE}' غير موجود في نفس مجلد التطبيق.")
         st.info("يرجى التأكد من وجود هذين الملفين الأساسيين لتشغيل التطبيق.")
         # Display instructions on how to potentially get the files or configure paths
         st.markdown("---")
         st.markdown("### تعليمات:")
         st.markdown(f"1. تأكد من وضع ملفي `{POSTS_FILE}` و `{COMMENTS_FILE}` في نفس المجلد الذي يوجد به ملف البايثون هذا.")
         st.markdown("2. تأكد من أن الملفات ليست فارغة وأنها تحتوي على البيانات المطلوبة.")
         st.stop() # Stop the app execution if files are missing
     else:
         # Files exist, proceed to run the main application logic
         main()


