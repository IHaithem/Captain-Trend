import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import requests
import urllib.parse
import time
from collections import defaultdict
from datetime import datetime

# Handle automatic installation of statsmodels if needed
try:
    import statsmodels
except ImportError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels"])
    import statsmodels

# Configuration
st.set_page_config(layout="wide", page_title="Facebook Trends Analyzer Pro")

# Constants
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
GOOGLE_SEARCH_DELAY = 2  # seconds between requests


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
        """Calculate comprehensive trend scores from Facebook data"""
        trend_data = defaultdict(lambda: {
            'posts': 0, 'likes': 0, 'comments': 0, 'shares': 0, 'examples': []
        })

        # Process all posts
        for _, post in posts_df.iterrows():
            for tag in self.extract_hashtags(post['text']):
                data = trend_data[tag]
                data['posts'] += 1
                data['likes'] += post.get('likes_count', 0)
                data['comments'] += post.get('comments_count', 0)
                data['shares'] += post.get('shares_count', 0)
                data['examples'].append(post['text'][:50] + "...")

        if not trend_data:
            return []

        # Normalize scores (0-100 scale)
        max_engagement = max(
            (d['likes'] * 0.5 + d['comments'] * 0.3 + d['shares'] * 0.2)
            for d in trend_data.values()
        )

        # Prepare final output
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
                'google_mentions': 0  # Will be updated later
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

            # Multiple patterns to catch result count
            patterns = [
                r"([\d,\.]+)\s+results",
                r"About\s([\d,\.]+)\s+results",
                r"([\d,\.]+)\s+نتيجة"
            ]

            for pattern in patterns:
                match = re.search(pattern, response.text)
                if match:
                    count = match.group(1).replace(",", "").replace(".", "")
                    return int(count) if count.isdigit() else 0
            return 0

        except requests.exceptions.RequestException as e:
            if retry < MAX_RETRIES:
                time.sleep(2 ** retry)
                return self.safe_google_search(query, retry + 1)
            st.warning(f"Could not fetch Google data for '{query}'")
            return 0

    def calculate_veracity_scores(self, trends):
        """Calculate comprehensive veracity scores"""
        results = []
        progress_bar = st.progress(0)

        for i, trend in enumerate(trends):
            query = trend['hashtag'].lstrip('#')
            mentions = self.safe_google_search(query)

            # Calculate veracity score (60% internal, 40% external)
            internal = trend['score']
            external = min(mentions / 1000, 100)  # Normalized
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

            # Update progress
            progress_bar.progress((i + 1) / len(trends))
            if i < len(trends) - 1:
                time.sleep(GOOGLE_SEARCH_DELAY)

        progress_bar.empty()
        return pd.DataFrame(results).sort_values('Veracity Score', ascending=False)

    def display_dashboard(self, trends, veracity_df):
        """Interactive dashboard with visualizations"""
        st.title("📊 Advanced Facebook Trends Analysis")

        # Key Metrics
        st.subheader("Performance Metrics")
        cols = st.columns(4)
        cols[0].metric("Total Trends", len(trends))
        cols[1].metric("Avg Veracity", f"{veracity_df['Veracity Score'].mean():.1f}")
        cols[2].metric("High Confidence", len(veracity_df[veracity_df['Veracity Score'] > 70]))

        # Top Trends Table - Without Google Mentions column
        st.subheader("Top Performing Trends")
        st.dataframe(
            veracity_df.head(20)[[
                'Trend', 'Veracity Score', 'Internal Score',
                'Post Count'  # Removed Google Mentions from display
            ]],
            height=600
        )

        # Visualizations
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
                # Removed trendline to avoid statsmodels dependency
            )
            st.plotly_chart(fig, use_container_width=True)


def main():
    analyzer = FacebookTrendAnalyzer()

    st.sidebar.title("Configuration")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Facebook Data (CSV)",
        type=["csv"],
        help="Required columns: text, likes_count, comments_count, shares_count"
    )

    if uploaded_file:
        try:
            # Load and validate data
            posts_df = pd.read_csv(uploaded_file)
            required_cols = ['text', 'likes_count', 'comments_count', 'shares_count']

            if not all(col in posts_df.columns for col in required_cols):
                st.error(f"Missing required columns: {', '.join(required_cols)}")
                return

            if st.sidebar.button("Analyze Trends"):
                with st.spinner("Processing Facebook data..."):
                    trends = analyzer.calculate_trends(posts_df)

                if not trends:
                    st.warning("No trends found in the data")
                    return

                with st.spinner("Calculating veracity scores..."):
                    veracity_df = analyzer.calculate_veracity_scores(trends)

                analyzer.display_dashboard(trends, veracity_df)

                # Export options
                st.sidebar.download_button(
                    "Download Full Report",
                    data=veracity_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"facebook_trends_report_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime='text/csv'
                )

        except Exception as e:
            st.error(f"Processing error: {str(e)}")
    else:
        st.sidebar.info("Please upload a CSV file to begin analysis")


if __name__ == "__main__":
    main()