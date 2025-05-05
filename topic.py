import json
import requests
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential


def analyze_comments_topics(comments):
    """Analyze topics from comments with improved outlier handling"""
    if len(comments) < 10:
        return None, "Not enough comments for analysis (minimum 10 required)"

    try:
        # Arabic-specific stopwords
        custom_stopwords = [
            "في", "على", "من", "عن", "مع", "ما", "هو", "هي", "إلى", "و", "لا",
            "تم", "ليس", "الله", "هذا", "هذه", "ذلك", "تلك", "لكن", "أو", "أي", "بعض",
            "لأن", "إن", "كان", "قد", "حتى", "بعد", "قبل", "هناك", "يكون", "يمكن"
        ]

        # Improved vectorizer for Arabic
        vectorizer_model = CountVectorizer(
            stop_words=custom_stopwords,
            ngram_range=(1, 3),
            token_pattern=r"(?u)\b[ء-ي]{3,}\b",
            min_df=2
        )

        # Arabic-specific embedding model
        embedding_model = SentenceTransformer("UBC-NLP/MARBERTv2")

        # Improved UMAP configuration
        umap_model = UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine'
        )

        # Corrected BERTopic configuration
        topic_model = BERTopic(
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            umap_model=umap_model,
            min_topic_size=5,
            nr_topics="auto",
            calculate_probabilities=True,  # This is crucial for outlier reduction
            verbose=True
        )

        # Comment cleaning
        cleaned_comments = [
            comment[:500].strip() for comment in comments
            if len(str(comment).strip()) > 20
        ]

        if len(cleaned_comments) < 5:
            return None, "Not enough valid comments after cleaning"

        # Model training - get both topics and probabilities
        topics, probabilities = topic_model.fit_transform(cleaned_comments)

        # Outlier reduction with probabilities
        if probabilities is not None:
            new_topics = topic_model.reduce_outliers(
                cleaned_comments,
                topics,
                probabilities=probabilities,  # Pass the probabilities
                strategy="probabilities"
            )
            topic_model.update_topics(cleaned_comments, topics=new_topics)
        else:
            st.warning("Could not calculate probabilities for outlier reduction")

        # Results analysis
        topic_info = topic_model.get_topic_info()
        valid_topics = topic_info[topic_info['Topic'] != -1]

        if len(valid_topics) <= 1:
            return None, "No meaningful topics found. Try with more comments."

        # Hashtag generation
        hashtags = generate_hashtags(topic_model, valid_topics)

        # Visualization
        try:
            fig = topic_model.visualize_barchart(
                top_n_topics=min(5, len(valid_topics)),
                width=300,
                height=500
            )
        except Exception as e:
            st.warning(f"Could not generate visualization: {str(e)}")
            fig = None

        return {
            "topics": valid_topics[['Topic', 'Count', 'Name', 'Representation']],
            "visualization": fig,
            "hashtags": hashtags
        }, None

    except Exception as e:
        return None, f"Analysis error: {str(e)}"


def generate_hashtags(topic_model, topic_info):
    """Generate hashtags with API fallback"""
    hashtags = []

    for topic in topic_info['Topic'].values:
        if topic == -1:
            continue

        keywords = [word for word, _ in topic_model.get_topic(topic)[:5]]
        for kw in keywords:
            clean_kw = kw.replace(" ", "_")
            if 3 <= len(clean_kw) <= 20:
                hashtags.append(f"#{clean_kw}")

    return list(dict.fromkeys(hashtags))[:5]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def query_openrouter_api(prompt: str) -> str:
    """Optimized OpenRouter API query function"""
    API_KEY = "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"  # Replace with your actual key
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://yourdomain.com",
        "X-Title": "Sports Analysis"
    }

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "system",
                "content": "You are a football expert. Generate relevant Arabic hashtags."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 150,
        "timeout": 15
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except:
        return "[]"