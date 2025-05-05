#!/usr/bin/env python
# coding: utf-8
# In[2]:

import streamlit as st
import pandas as pd
import os
import time # لاستخدامه مع st.toast
import streamlit.components.v1 as components # لاستخدام iframe
from analyse_sentiment import predict_sentiment, predict_emotion, labels_map_sentiment, labels_map_emotion
import matplotlib.pyplot as plt
import google.generativeai as genai
# --- إعدادات أساسية ---
POSTS_FILE = "facebook_posts.csv"
COMMENTS_FILE = "facebook_comments.csv"
LOGO_PATH = "logo.jpg" # <-- تأكد أن هذا المسار صحيح أو استخدم رابط URL

# Gemini setup - Add this after your other configuration
genai.configure(api_key="AIzaSyAhbDAMLAUTFjZh0OvKbyDp5GaELvT7fcw")  # Replace with your actual API key
model = genai.GenerativeModel("gemini-1.5-pro")


def summarize_arabic(text):
    """Summarize Arabic text using Gemini model"""
    if not text or not isinstance(text, str) or len(text.strip()) < 20:
        return "النص قصير جدًا أو غير متاح للتلخيص"

    prompt = f"قم بتلخيص هذا النص العربي بطريقة واضحة ومختصرة مع الحفاظ على المعنى الأساسي:\n\n{text}"

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"حدث خطأ أثناء التلخيص: {str(e)}"
# ---------------------------
# تحميل البيانات (بدون تغيير)
# ---------------------------
def load_data():
    # ... (نفس كود load_data من الإجابة السابقة) ...
    try:
        if not os.path.exists(POSTS_FILE): st.error(f"ملف {POSTS_FILE} غير موجود."); return None, None
        if not os.path.exists(COMMENTS_FILE): st.error(f"ملف {COMMENTS_FILE} غير موجود."); return None, None
        try: posts_df = pd.read_csv(POSTS_FILE); comments_df = pd.read_csv(COMMENTS_FILE)
        except UnicodeDecodeError:
            try: posts_df = pd.read_csv(POSTS_FILE, encoding='utf-8'); comments_df = pd.read_csv(COMMENTS_FILE, encoding='utf-8')
            except Exception as e: st.error(f"خطأ قراءة (ترميز): {e}"); return None, None
        except pd.errors.EmptyDataError: st.error("أحد ملفات البيانات فارغ."); return None, None
        required_post_cols = ['post_id', 'article_title', 'short_description', 'image', 'video_link', 'article_link', 'likes_count', 'comments_count']
        required_comment_cols = ['post_id', 'author', 'text', 'likes']
        missing_post_cols = [col for col in required_post_cols if col not in posts_df.columns]
        if missing_post_cols: st.error(f"المنشورات تفتقد: {', '.join(missing_post_cols)}"); return None, None
        missing_comment_cols = [col for col in required_comment_cols if col not in comments_df.columns]
        if missing_comment_cols:
            if 'author' in missing_comment_cols and len(missing_comment_cols) == 1:
                 required_comment_cols.remove('author')
                 missing_comment_cols = [col for col in required_comment_cols if col not in comments_df.columns]
                 if missing_comment_cols: st.error(f"التعليقات تفتقد: {', '.join(missing_comment_cols)}"); return None, None
            else: st.error(f"التعليقات تفتقد: {', '.join(missing_comment_cols)}"); return None, None
        for col in ['likes_count', 'comments_count']:
             if col in posts_df.columns: posts_df[col] = pd.to_numeric(posts_df[col], errors='coerce').fillna(0).astype(int)
        if 'likes' in comments_df.columns: comments_df['likes'] = pd.to_numeric(comments_df['likes'], errors='coerce').fillna(0).astype(int)
        else: comments_df['likes'] = 0
        for col in ['article_title', 'short_description', 'image', 'video_link', 'article_link']:
             if col in posts_df.columns: posts_df[col] = posts_df[col].fillna('').astype(str)
        for col in ['author', 'text']:
             if col in comments_df.columns: comments_df[col] = comments_df[col].fillna('').astype(str)
             elif col == 'text': st.error("عمود 'text' للتعليقات مفقود!"); return None, None
        if 'post_id' not in posts_df.columns or 'post_id' not in comments_df.columns: st.error("عمود 'post_id' للربط غير موجود."); return None, None
        posts_df['post_id'] = posts_df['post_id'].astype(str); comments_df['post_id'] = comments_df['post_id'].astype(str)
        return posts_df, comments_df
    except Exception as e: st.error(f"خطأ تحميل البيانات: {e}"); st.exception(e); return None, None


# ---------------------------
# تصنيف المنشورات (بدون تغيير)
# ---------------------------
def categorize_posts(posts_df):
    # ... (نفس كود categorize_posts) ...
    if 'article_title' not in posts_df.columns: posts_df['category'] = "أخرى"; return posts_df
    categories = {"كرة القدم": ["كرة القدم", "الدوري", "كأس", "مباراة", "فريق", "ريال مدريد", "برشلونة", "ليفربول", "مانشستر", "تشيلسي", "يوفنتوس", "ميلان", "بايرن", "هدف", "لاعب", "الملعب", "بطل", "نهائي", "تشامبيونز ليج", "يورو"], "كرة السلة": ["كرة السلة", "nba", "الدوري الاميركي", "سلة", "ليكرز", "واريورز", "ليبرون", "كيري", "جيمس", "كوبي"], "التنس": ["التنس", "ويمبلدون", "رولان غاروس", "فلاشينغ ميدوز", "أستراليا المفتوحة", "نادال", "فيدرر", "ديوكوفيتش", "مضرب"], "كرة اليد": ["كرة اليد", "بطولة العالم لكرة اليد"], "فورمولا 1": ["فورمولا 1", "فورمولا وان", "f1", "سباق سيارات", "هاميلتون", "فيرستابن", "فيراري", "مرسيدس", "ريد بول", "حلبة"]}
    def detect_category(text):
        best_match = "أخرى";
        if isinstance(text, str) and text.strip():
            text_lower = " " + text.lower() + " "; matched_cats = []
            for cat, keywords in categories.items():
                if any(f" {kw.lower()} " in text_lower for kw in keywords): matched_cats.append(cat)
            if matched_cats: return matched_cats[0]
            for cat, keywords in categories.items():
                 if any(kw.lower() in text_lower for kw in keywords): matched_cats.append(cat)
            if matched_cats: return matched_cats[0]
        return best_match
    posts_df['category'] = posts_df['article_title'].apply(detect_category)
    return posts_df

# ---------------------------
# حساب البوستات الأكثر شهرة (بدون تغيير)
# ---------------------------
def get_top_posts(posts_df):
    # ... (نفس كود get_top_posts) ...
    if 'likes_count' in posts_df.columns and 'comments_count' in posts_df.columns: posts_df['popularity_score'] = posts_df['likes_count'] * 0.7 + posts_df['comments_count'] * 0.3
    elif 'likes_count' in posts_df.columns: posts_df['popularity_score'] = posts_df['likes_count']
    else: posts_df['popularity_score'] = 0
    if not posts_df.empty: return posts_df.sort_values('popularity_score', ascending=False)
    else: return posts_df

# ---------------------------
# التحقق من صحة الرابط (بدون تغيير)
# ---------------------------
def is_valid_url(url):
    # ... (نفس كود is_valid_url) ...
    return isinstance(url, str) and url.strip().startswith(('http://', 'https://'))
def is_valid_media_link(link):
    # ... (نفس كود is_valid_media_link) ...
    if not isinstance(link, str): return False
    link_cleaned = link.strip().lower()
    if link_cleaned in ['unavailable', 'no video', '', 'nan', 'none', 'null']: return False
    if not link_cleaned.startswith(('http://', 'https://')): return False
    return True

# ---------------------------
# تطبيق CSS (تم إضافة قاعدة البحث)
# ---------------------------
def apply_css():
    main_font = "'Cairo', 'Tahoma', sans-serif"
    st.markdown(f"""
        <style>
            /* --- General Styles & Font --- */
            body, .stApp {{ direction: rtl !important; font-family: {main_font}; }}
            h1, h2, h3, h4, h5, h6, p, div, span, button, label, input, textarea, select, .stMarkdown, .stButton>button, .stExpander>summary, .stTextInput>div>div>input {{
                text-align: right !important; font-family: inherit;
            }}

             /* --- Header Styles --- */
             .app-header {{
                 display: flex; align-items: center;
                 justify-content: flex-start; /* لوغو يمين، عنوان يسار */
                 padding: 5px 0; border-bottom: 2px solid #eee; margin-bottom: 15px;
             }}
             .app-header img {{ /* اللوغو */
                 max-height: 55px; width: auto; margin-left: 15px;
             }}
             .app-header h1 {{ /* العنوان الرئيسي */
                 color: #000000; margin-bottom: 0; text-align: right !important;
                 font-family: 'Tajawal', {main_font}; font-size: 2.5em; font-weight: 800;
             }}

            /* --- Main Feed Button Styles --- */
            .stButton>button {{ width: 100%; margin-bottom: 5px; border-radius: 20px !important; border: 1px solid #4CAF50 !important; background-color: #e8f5e9 !important; color: #1B5E20 !important; font-weight: bold; transition: all 0.3s ease; padding: 8px 0; }}
            .stButton>button:hover {{ background-color: #c8e6c9 !important; border-color: #2E7D32 !important; color: #1B5E20 !important; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .stButton>button:active {{ background-color: #a5d6a7 !important; }}

            /* --- Buttons in Left Column --- */
             [data-testid="stVerticalBlock"] .stButton button:contains("تحديث البيانات") {{ border-color: #1976D2 !important; background-color: #e3f2fd !important; color: #0d47a1 !important; }}
             [data-testid="stVerticalBlock"] .stButton button:contains("تحديث البيانات"):hover {{ background-color: #bbdefb !important; border-color: #1565c0 !important; }}
             [data-testid="stVerticalBlock"] .stButton button:contains("الصور") {{ border-color: #7E57C2 !important; background-color: #ede7f6 !important; color: #4527A0 !important; }}
             [data-testid="stVerticalBlock"] .stButton button:contains("الصور"):hover {{ background-color: #d1c4e9 !important; border-color: #5E35B1 !important; }}

             /* --- ===== تنسيق خانة البحث ===== --- */
             [data-testid="stVerticalBlock"] [data-testid="stTextInput"] input {{
                 background-color: #f0f2f6; /* <-- غير لون الخلفية هنا (مثال: رمادي فاتح) */
                 border: 1px solid #bdc3c7;   /* <-- لون الحدود (مثال: رمادي أغمق) */
                 color: #2c3e50;           /* <-- لون النص المدخل (مثال: أزرق رمادي) */
                 border-radius: 5px;      /* حواف دائرية بسيطة (اختياري) */
                 padding: 8px 10px;        /* هوامش داخلية (اختياري) */
             }}
             /* (اختياري) تغيير لون النص المؤقت (placeholder) */
             [data-testid="stVerticalBlock"] [data-testid="stTextInput"] input::placeholder {{
                 color: #7f8c8d; /* مثال: رمادي للنص المؤقت */
                 opacity: 0.8;
             }}
             /* --- ============================ --- */

             /* --- Details Button --- */
            .details-button button {{ background-color: #f5f5f5 !important; color: #555 !important; border: 1px solid #ddd !important; padding: 4px 10px !important; font-size: 0.85em !important; font-weight: normal !important; border-radius: 5px !important; margin-top: 10px !important; width: auto !important; display: inline-block !important; }}
            .details-button button:hover {{ background-color: #eee !important; border-color: #ccc !important; box-shadow: none !important; }}
            /* --- Category Selection Text --- */
            .stMarkdown h4 span {{ background-color: #E8F5E9; padding: 5px 10px; border-radius: 15px; border: 1px solid #a5d6a7; display: inline-block; }}
            /* --- Post Card Styles --- */
            .post-card {{ border: 1px solid #e0e0e0; border-radius: 10px; padding: 15px 20px; margin-bottom: 20px; background-color: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.08); transition: box-shadow 0.3s ease-in-out; overflow: hidden; }}
            .post-card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
            .top-post-card {{ border: 2px solid #4CAF50; background-color: #f9fff9; }}
            .top-post-card h3 {{ color: #2E7D32; margin-bottom: 12px; font-size: 1.5em; }}
            .top-post-card p {{ line-height: 1.6; color: #37474F; margin-bottom: 15px; }}
            /* --- Video iframe container --- */
            .iframe-video-wrapper {{ width: 100%; margin-top: 15px; margin-bottom: 20px; border-radius: 8px; overflow: hidden; border: 1px solid #eee; background-color: #000; }}
            .iframe-video-wrapper iframe {{ display: block; width: 100%; border: none; }}
            /* --- Comment Card Styles --- */
            .comment-card {{ background: #f1f8e9; border-right: 5px solid #8BC34A; padding: 12px 15px; margin-bottom: 10px; border-radius: 8px; font-size: 0.95em; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
            .comment-card p {{ margin-bottom: 5px; color: #1B5E20; word-wrap: break-word; }}
            .comment-card small {{ color: #555; font-size: 0.85em; display: block; margin-top: 5px; }}
            /* --- Media (Image/Video Error) Styles --- */
            .video-error-box {{ font-size: 0.9em; background-color: #fff3e0; border: 1px solid #ffe0b2; padding: 10px; border-radius: 5px; margin-top: 10px; }}
            .video-error-box small {{ line-height: 1.5; }}
            .video-error-box li {{ margin-bottom: 5px; }}
            /* --- Read More / Stats Styles --- */
            .read-more-button a {{ text-decoration: none; }}
            .read-more-button button {{ padding: 8px 20px; background: #2E7D32; color: white !important; border: none; border-radius: 5px; cursor: pointer; transition: background-color 0.2s ease; font-weight: bold; margin-top: 10px; }}
            .read-more-button button:hover {{ background: #1B5E20; }}
            .small-read-more-button button {{ padding: 5px 10px !important; font-size: 0.8em !important; font-weight: normal !important; }}
            .stats-divider {{ margin: 0 8px; color: #ccc; }}
            .stats-container {{ margin-top: 15px; padding-top:10px; border-top: 1px solid #eee; font-size: 0.9em; color: #555; text-align: center; }}
            .stats-container span {{ display: inline-block; margin: 0 5px; color: #01579B; }}
            .stats-container .category-label {{ color: #757575; font-size: 0.9em; }}
            /* --- Gallery Styles (within left column) --- */
             [data-testid="stVerticalBlock"] .gallery-image {{ margin-bottom: 15px; border: 1px solid #ddd; padding: 8px; border-radius: 5px; background-color: #fff; text-align: center; }}
             [data-testid="stVerticalBlock"] .gallery-image img {{ border-radius: 3px; max-width: 100%; height: auto; }}
             [data-testid="stVerticalBlock"] .gallery-image p {{ font-size: 0.85em; text-align: center !important; margin-top: 5px; color: #555; word-wrap: break-word; }}
            /* --- Details Page Styles --- */
            .details-container {{ padding: 20px; background-color: #fdfdfd; border-radius: 10px; border: 1px solid #eee; max-width: 900px; margin: 20px auto; }}
            .details-title {{ color: #2E7D32; text-align: center !important; margin-bottom: 15px; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
            .stImage {{ margin-bottom: 15px !important; }}
            .details-description {{ line-height: 1.7; margin-top: 20px; margin-bottom: 25px; font-size: 1.05em; color: #333; padding: 15px; background-color: #ffffff; border-radius: 5px; border: 1px solid #f0f0f0; }}
            .details-stats {{ text-align: center !important; margin-top: 20px; padding: 15px 0; border-top: 1px solid #eee; border-bottom: 1px solid #eee; }}
            .details-comments-section h3 {{ margin-top: 30px; margin-bottom: 15px; color: #333; text-align: center !important; border-bottom: 1px solid #ddd; padding-bottom: 8px; }}
            .back-button button {{ background-color: #ffcdd2 !important; color: #c62828 !important; border: 1px solid #ef9a9a !important; font-weight: bold !important; width: auto !important; padding: 8px 20px !important; border-radius: 8px !important; display: inline-block !important; }}
            .back-button button:hover {{ background-color: #ef9a9a !important; border-color: #e57373 !important; }}
            .back-button-container {{ text-align: center; margin-bottom: 20px; }}
            .video-section-title {{ font-size: 1.2em; font-weight: bold; color: #444; margin-top: 20px !important; margin-bottom: 10px !important; text-align: center !important; padding-bottom: 5px; border-bottom: 1px solid #eee; }}
        </style>
    """, unsafe_allow_html=True)

# --- (بقية الدوال render_gallery, render_main_feed, render_details_page, main بدون تغيير) ---
# ...
# ---------------------------
# دالة عرض الغاليري (بدون تغيير)
# ---------------------------
def render_gallery(posts_df):
    # ... (نفس كود render_gallery) ...
    st.markdown("#### ✨ الصور")
    posts_with_images = posts_df[posts_df['image'].apply(is_valid_media_link)]
    if not posts_with_images.empty:
        image_count = 0; max_gallery_images = 15; gallery_cols = st.columns(2); col_idx = 0
        for _, post in posts_with_images.iterrows():
            if image_count < max_gallery_images:
                with gallery_cols[col_idx % 2]: img_src = post['image']; title_alt = post.get('article_title', 'صورة')[:30]; st.image(img_src, caption=f"{title_alt}...", use_container_width=True)
                image_count += 1; col_idx += 1
            else: break
        if len(posts_with_images) > image_count: st.caption(f"عرض أول {image_count} صورة...")
    else: st.info("لا توجد صور متاحة.")

# ---------------------------
# دالة عرض القائمة الرئيسية (مع البحث)
# ---------------------------
def render_main_feed(posts_df, comments_df):
    # ... (نفس كود render_main_feed من الإجابة السابقة) ...
    st.markdown("## 🏷️ اخبار الرياضة")
    unique_categories = sorted([cat for cat in posts_df['category'].unique() if cat != "أخرى"])
    if "أخرى" in posts_df['category'].unique(): unique_categories.append("أخرى")
    categories = ["كل الفئات"] + unique_categories
    cols_per_row = 6; num_cat_buttons = len(categories)
    num_rows = (num_cat_buttons + cols_per_row - 1) // cols_per_row
    cat_button_cols = []
    for _ in range(num_rows): cat_button_cols.append(st.columns(cols_per_row))
    button_idx = 0
    for i in range(num_rows):
        for j in range(cols_per_row):
            if button_idx < num_cat_buttons:
                cat = categories[button_idx]
                with cat_button_cols[i][j]:
                    if st.button(cat, key=f"cat_btn_{cat.replace(' ', '_').replace('1','_1')}", use_container_width=True):
                        st.session_state.selected_category = cat; st.session_state.show_gallery = False; st.rerun()
                button_idx += 1
    st.markdown(f"#### <span> تعرض الآن منشورات قسم: {st.session_state.selected_category}</span>", unsafe_allow_html=True)
    st.divider()
    left_col, center_col, right_col = st.columns([1, 3, 1.5])
    with left_col:
        st.markdown("### 🧭 أدوات إضافية"); st.markdown("---")
        if st.button("🔄 تحديث البيانات", key="refresh_data", use_container_width=True): st.toast("🔄 جارٍ التحديث...", icon="⏳"); time.sleep(1); st.rerun()
        st.markdown("---"); gallery_button_text = "🔙 إخفاء الصور" if st.session_state.show_gallery else "📸 عرض الصور"
        if st.button(gallery_button_text, key="toggle_gallery", use_container_width=True): st.session_state.show_gallery = not st.session_state.show_gallery; st.rerun()
        if st.session_state.show_gallery: st.markdown("---"); render_gallery(posts_df)
        st.markdown("---"); st.markdown("### 🔍 بحث")
        if 'search_term' not in st.session_state: st.session_state.search_term = ""
        st.session_state.search_term = st.text_input("ابحث في العناوين والوصف:", value=st.session_state.search_term, key="search_input_main", help="اكتب كلمة واضغط Enter")
    if st.session_state.selected_category == "كل الفئات": filtered_posts = posts_df.copy()
    else: filtered_posts = posts_df[posts_df['category'] == st.session_state.selected_category].copy()
    search_term = st.session_state.search_term
    if search_term:
        search_term_lower = search_term.lower()
        search_condition = pd.Series([False] * len(filtered_posts), index=filtered_posts.index)
        if 'article_title' in filtered_posts.columns: search_condition = search_condition | filtered_posts['article_title'].str.lower().str.contains(search_term_lower, na=False)
        if 'short_description' in filtered_posts.columns: search_condition = search_condition | filtered_posts['short_description'].str.lower().str.contains(search_term_lower, na=False)
        filtered_posts = filtered_posts[search_condition]
    if not filtered_posts.empty:
        filtered_posts = filtered_posts.sort_values('popularity_score', ascending=False)
        top_cat_post = filtered_posts.head(1); other_posts = filtered_posts.iloc[1:]
    else: top_cat_post = pd.DataFrame(); other_posts = pd.DataFrame()
    with center_col:
        if search_term: st.info(f"🔎 نتائج البحث عن: '{search_term}' ضمن قسم '{st.session_state.selected_category}'")
        if filtered_posts.empty:
            if search_term: st.warning(f"لم يتم العثور على نتائج تطابق بحثك '{search_term}'.")
            else: st.warning(f"لا توجد منشورات لعرضها في قسم '{st.session_state.selected_category}'.")
        else:
            if not top_cat_post.empty:
                st.markdown("### 🏆 المنشور الأبرز"); post = top_cat_post.iloc[0]
                with st.container():
                    st.markdown(f"<div class='post-card top-post-card'>", unsafe_allow_html=True)
                    st.markdown(f"<h3>{post.get('article_title', 'N/A')}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p>{post.get('short_description', '')[:200]}{'...' if len(post.get('short_description', '')) > 200 else ''}</p>", unsafe_allow_html=True)
                    image_url = post.get('image'); video_url = post.get('video_link')
                    if is_valid_media_link(image_url): st.markdown(f'<img src="{image_url}" alt="صورة" onerror="this.style.display=\'none\'">', unsafe_allow_html=True)
                    if is_valid_media_link(video_url): st.markdown("<small><i>(يحتوي على فيديو)</i></small>", unsafe_allow_html=True)
                    details_button_html = f"""<span class="details-button">""";
                    if st.button("عرض التفاصيل 📄", key=f"details_{post.get('post_id')}", help="عرض تفاصيل هذا المنشور"): st.session_state.view = 'details'; st.session_state.selected_post_id = post.get('post_id'); st.rerun()
                    details_button_html += "</span>"; article_link = post.get('article_link'); read_more_button_html = ""
                    if is_valid_url(article_link): read_more_button_html = f"""<span class="read-more-button" style="margin-right: 10px;"><a href="{article_link}" target="_blank"><button>المقال الأصلي</button></a></span>"""
                    stats_html = f"""<div class="stats-container"><span>👍 {post.get('likes_count', 0)}</span><span class="stats-divider">|</span><span>💬 {post.get('comments_count', 0)}</span><span class="stats-divider">|</span><span class="category-label">الفئة: {post.get('category', 'N/A')}</span></div>"""
                    st.markdown(stats_html + details_button_html + read_more_button_html, unsafe_allow_html=True); st.markdown("</div>", unsafe_allow_html=True)
            if not other_posts.empty:
                st.markdown("---"); st.markdown("### 📚 منشورات أخرى"); num_other_posts = 10
                for _, post in other_posts.head(num_other_posts).iterrows():
                     with st.container():
                        st.markdown("<div class='post-card'>", unsafe_allow_html=True)
                        st.markdown(f"<strong>{post.get('article_title', 'N/A')}</strong>", unsafe_allow_html=True)
                        desc = post.get('short_description', ''); st.markdown(f"<p style='font-size: 0.95em; color: #444;'>{desc[:150]}{'...' if len(desc) > 150 else ''}</p>", unsafe_allow_html=True)
                        image_url_other = post.get('image');
                        if is_valid_media_link(image_url_other): st.markdown(f'<img src="{image_url_other}" alt="صورة" style="max-height: 150px; width:auto; margin: 5px auto; border-radius: 5px;">', unsafe_allow_html=True)
                        stats_other_html = f"""<div style="font-size: 0.9em; color: #555; border-top: 1px solid #f0f0f0; padding-top: 8px; margin-top: 10px; text-align: center;"><span>👍 {post.get('likes_count', 0)}</span><span class="stats-divider">|</span><span>💬 {post.get('comments_count', 0)}</span></div>"""
                        details_button_other_html = f"""<span class="details-button">""";
                        if st.button("التفاصيل", key=f"details_{post.get('post_id')}", help="عرض التفاصيل"): st.session_state.view = 'details'; st.session_state.selected_post_id = post.get('post_id'); st.rerun()
                        details_button_other_html += "</span>"; read_more_other_button_html = ""; article_link_other = post.get('article_link')
                        if is_valid_url(article_link_other): read_more_other_button_html = f"""<span class="read-more-button small-read-more-button" style="margin-right: 10px;"><a href="{article_link_other}" target="_blank"><button>المزيد</button></a></span>"""
                        st.markdown(stats_other_html + details_button_other_html + read_more_other_button_html , unsafe_allow_html=True); st.markdown("</div>", unsafe_allow_html=True)
                if len(other_posts) > num_other_posts: st.caption(f"عرض أول {num_other_posts} منشور...")
    with right_col:
        st.markdown("### 💬 أبرز التعليقات")
        if not top_cat_post.empty:
            st.markdown("<small>(على المنشور الأبرز المعروض)</small>", unsafe_allow_html=True)
            top_post_id_filtered = top_cat_post.iloc[0].get('post_id')
            if top_post_id_filtered and all(col in comments_df.columns for col in ['post_id', 'text', 'likes']):
                post_comments = comments_df[comments_df['post_id'] == top_post_id_filtered].sort_values('likes', ascending=False).head(7)
                if not post_comments.empty:
                    st.markdown("---")
                    for _, comment in post_comments.iterrows(): text = comment.get('text', ''); likes = comment.get('likes', 0); st.markdown(f"""<div class="comment-card"><p>{text}</p><small>👍 {likes}</small></div>""", unsafe_allow_html=True)
                else: st.info("لا توجد تعليقات لهذا المنشور.")
        else: st.info("لا يوجد منشور أبرز لعرض تعليقاته (قد يكون بسبب البحث).")

# ---------------------------
# دالة عرض صفحة التفاصيل (بدون تغيير)
# ---------------------------
# Add this at the top of your imports



# Modify the render_details_page function to include analysis
def render_details_page(post_id, posts_df, comments_df):
    st.markdown('<div class="back-button-container"><span class="back-button">', unsafe_allow_html=True)
    if st.button("⬅️ العودة للقائمة", key="back_to_main"):
        st.session_state.view = 'main'
        st.session_state.selected_post_id = None
        st.rerun()
    st.markdown('</span></div>', unsafe_allow_html=True)

    post_details = posts_df[posts_df['post_id'] == post_id]
    if post_details.empty:
        st.error("لم يتم العثور على تفاصيل المنشور.")
        return

    post = post_details.iloc[0]

    with st.container():
        st.markdown('<div class="details-container">', unsafe_allow_html=True)
        st.markdown(f"<h2 class='details-title'>{post.get('article_title', 'N/A')}</h2>", unsafe_allow_html=True)

        # --- Text and Summary Section ---
        description = post.get('long_description', post.get('short_description', 'لا يوجد وصف.'))
        st.markdown("### 📝 الوصف")
        st.markdown(f"<div class='details-description'>{description}</div>", unsafe_allow_html=True)

        if len(description.split()) > 50:
            st.markdown("### 📑 ملخص النص (باستخدام الذكاء الاصطناعي)")
            if st.button("إنشاء ملخص", key="generate_summary"):
                with st.spinner("جارٍ إنشاء الملخص..."):
                    from analyse_sentiment import summarize_arabic
                    summary = summarize_arabic(description)
                    st.markdown(
                        f"<div style='background-color: #f5f5f5; padding: 15px; border-radius: 5px; border-left: 4px solid #4CAF50;'>{summary}</div>",
                        unsafe_allow_html=True)

        # --- Sentiment/Emotion Analysis Section ---
        st.markdown("### 📊 تحليل المشاعر والتعليقات")
        analysis_type = st.radio("اختر نوع التحليل:", ["تحليل المشاعر", "تحليل العواطف"], horizontal=True)

        if st.button("تشغيل تحليل التعليقات", key="run_analysis"):
            with st.spinner("جارٍ تحليل التعليقات..."):
                comments = comments_df[comments_df['post_id'] == post_id]['text'].tolist()
                if comments:
                    from analyse_sentiment import predict_sentiment, predict_emotion, labels_map_sentiment, \
                        labels_map_emotion

                    if analysis_type == "تحليل المشاعر":
                        predictions = predict_sentiment(comments)
                        df_results = pd.DataFrame({
                            "التعليق": comments,
                            "التصنيف": [labels_map_sentiment[p.item()] for p in predictions]
                        })
                    else:
                        predictions = predict_emotion(comments)
                        df_results = pd.DataFrame({
                            "التعليق": comments,
                            "التصنيف": [labels_map_emotion[p.item()] for p in predictions]
                        })

                    # Show results
                    st.dataframe(df_results)

                    # Show statistics
                    st.subheader("إحصائيات التحليل")
                    stats = df_results['التصنيف'].value_counts().reset_index()
                    stats.columns = ['التصنيف', 'العدد']

                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(stats)

                    with col2:
                        fig, ax = plt.subplots()
                        stats.plot(kind='bar', x='التصنيف', y='العدد', ax=ax, legend=False)
                        plt.title("توزيع التصنيفات")
                        st.pyplot(fig)
                else:
                    st.warning("لا توجد تعليقات لتحليلها.")

        # --- Topic Analysis Section ---
        st.markdown("### 🗂 تحليل مواضيع التعليقات")
        post_comments = comments_df[comments_df['post_id'] == post_id]['text'].dropna().tolist()

        if len(post_comments) >= 5:
            if st.button("تحليل المواضيع", key="analyze_topics"):
                with st.spinner("جارٍ تحليل المواضيع في التعليقات..."):
                    from topic import analyze_comments_topics

                    result, error = analyze_comments_topics(post_comments)

                    if error:
                        st.error(error)
                    elif result:
                        # Show topic info
                        st.markdown("#### المواضيع الرئيسية")
                        st.dataframe(result["topics"])

                        # Show visualization
                        if "visualization" in result:
                            st.markdown("#### 📊 توزيع المواضيع")
                            st.plotly_chart(result["visualization"])

                        # Show generated hashtags
                        if result.get("hashtags"):
                            st.markdown("#### 🏷 وسوم مقترحة")
                            st.write(" ".join(result["hashtags"]))
        else:
            st.warning(f"لا توجد تعليقات كافية لتحليل المواضيع (يحتاج إلى 5 على الأقل، يوجد {len(post_comments)})")

        # --- Media Display Section ---
        image_url = post.get('image')
        if is_valid_media_link(image_url):
            st.image(image_url, caption=post.get('article_title', ''), use_column_width=True)

        video_url = post.get('video_link')
        article_link_for_video = post.get("article_link", "#")
        article_link_text = '<a href="' + article_link_for_video + '" target="_blank" rel="noopener noreferrer">رابط المقال</a>' if is_valid_url(
            article_link_for_video) else "غير متاح"

        if is_valid_media_link(video_url):
            st.markdown("<h4 class='video-section-title'>🎬 الفيديو</h4>", unsafe_allow_html=True)
            if 'dailymotion.com/player' in video_url:
                try:
                    iframe_height = 450
                    st.markdown('<div class="iframe-video-wrapper">', unsafe_allow_html=True)
                    components.iframe(video_url, height=iframe_height, scrolling=False)
                    st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"خطأ تضمين الفيديو: {e}")
                    st.markdown(
                        f'''<div class="video-error-box"><small><li>يمكنك محاولة مشاهدة الفيديو في المقال الأصلي: {article_link_text}</li></small></div>''',
                        unsafe_allow_html=True)
            else:
                try:
                    st.video(video_url)
                except Exception as video_error:
                    st.warning(f"⚠️ تعذر عرض الفيديو المباشر.")
                    st.markdown(
                        f'''<div class="video-error-box"><small><li>حاول فتح رابط الفيديو مباشرة: <a href="{video_url}" target="_blank" rel="noopener noreferrer">اضغط هنا</a></li><li>أو قم بزيارة المقال الأصلي: {article_link_text}</li></small></div>''',
                        unsafe_allow_html=True)
        elif video_url and video_url not in ['unavailable', 'no video', '', 'nan', 'none', 'null']:
            st.markdown("<h4 class='video-section-title'>🎬 الفيديو</h4>", unsafe_allow_html=True)
            st.info(f"ℹ️ تم العثور على إشارة لوجود فيديو، ولكن لا يمكن عرضه مباشرة.")
            st.markdown(
                f'''<div class="video-error-box" style="background-color: #e3f2fd; border-color: #bbdefb;"><small><li>أفضل طريقة لمشاهدة الفيديو هي عبر زيارة المقال الأصلي: {article_link_text}</li></small></div>''',
                unsafe_allow_html=True)

        # --- Statistics and Links Section ---
        st.markdown("### 📊 إحصائيات وروابط")
        stats_html = f"""<div class="details-stats"><span>👍 {post.get('likes_count', 0)} إعجاب</span><span class="stats-divider">|</span><span>💬 {post.get('comments_count', 0)} تعليق</span><span class="stats-divider">|</span><span class="category-label">الفئة: {post.get('category', 'N/A')}</span></div>"""
        st.markdown(stats_html, unsafe_allow_html=True)

        article_link = post.get('article_link')
        if is_valid_url(article_link):
            st.markdown(
                f"""<div class="read-more-button" style="text-align: center; margin-top: 15px;"><a href="{article_link}" target="_blank"><button>🔗 زيارة المقال الأصلي</button></a></div>""",
                unsafe_allow_html=True)

        # --- Comments Section ---
        st.markdown('<div class="details-comments-section">', unsafe_allow_html=True)
        st.markdown("<h3>💬 التعليقات</h3>", unsafe_allow_html=True)

        if all(col in comments_df.columns for col in ['post_id', 'text', 'likes']):
            post_comments = comments_df[comments_df['post_id'] == post_id].sort_values('likes', ascending=False)
            if not post_comments.empty:
                num_comments_to_show = 10
                for _, comment in post_comments.head(num_comments_to_show).iterrows():
                    text = comment.get('text', '')
                    likes = comment.get('likes', 0)
                    st.markdown(f"""<div class="comment-card"><p>{text}</p><small>👍 {likes}</small></div>""",
                                unsafe_allow_html=True)

                if len(post_comments) > num_comments_to_show:
                    st.caption(f"عرض أفضل {num_comments_to_show} تعليقات...")
            else:
                st.info("لا توجد تعليقات لهذا المنشور.")
        else:
            st.warning("ملف التعليقات يفتقد أعمدة.")

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# الدالة الرئيسية للتطبيق (بدون تغيير)
# ---------------------------
def main():
    # ... (نفس كود main) ...
    page_title = "سبور  "
    page_icon = LOGO_PATH
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")
    apply_css()
    col1, col2 = st.columns([1, 6])
    with col1:
        try: st.image(LOGO_PATH, width=100)
        except FileNotFoundError: st.warning(f"اللوغو ({LOGO_PATH}) غير موجود")
        except Exception as e: st.warning(f"خطأ عرض اللوغو: {e}")
    with col2: st.markdown(f"<div class='app-header'><h1>{page_title}</h1></div>", unsafe_allow_html=True)
    st.markdown("---")
    if "view" not in st.session_state: st.session_state.view = "main"
    if "selected_post_id" not in st.session_state: st.session_state.selected_post_id = None
    if "selected_category" not in st.session_state: st.session_state.selected_category = "كل الفئات"
    if "show_gallery" not in st.session_state: st.session_state.show_gallery = False
    if 'search_term' not in st.session_state: st.session_state.search_term = ""
    posts_df, comments_df = load_data()
    if posts_df is None or comments_df is None: st.error("فشل تحميل البيانات."); st.stop()
    try: posts_df = categorize_posts(posts_df.copy()); posts_df = get_top_posts(posts_df.copy())
    except Exception as e: st.error(f"خطأ معالجة: {e}"); st.exception(e); st.stop()
    if st.session_state.view == "details" and st.session_state.selected_post_id is not None:
        render_details_page(st.session_state.selected_post_id, posts_df, comments_df)
    else:
        st.session_state.view = "main"; st.session_state.selected_post_id = None
        render_main_feed(posts_df, comments_df)

# ---------------------------
# نقطة الدخول الرئيسية (بدون تغيير)
# ---------------------------
if __name__ == "__main__":
     if not os.path.exists(POSTS_FILE) or not os.path.exists(COMMENTS_FILE):
         st.error(f"خطأ: ملف '{POSTS_FILE}' أو '{COMMENTS_FILE}' غير موجود."); st.info("تأكد من وجود الملفات."); st.stop()
     else: main()


# In[ ]:




