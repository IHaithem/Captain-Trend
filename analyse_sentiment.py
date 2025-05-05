import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from functools import lru_cache
import google.generativeai as genai

# Configuration Gemini
genai.configure(api_key="AIzaSyAhbDAMLAUTFjZh0OvKbyDp5GaELvT7fcw")
gemini_model = genai.GenerativeModel("gemini-1.5-pro")


# Chargement des modèles
@lru_cache(maxsize=2)
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model


tokenizer_sentiment, model_sentiment = load_model("model")
tokenizer_emotion, model_emotion = load_model("model_2")

model_sentiment.eval()
model_emotion.eval()

# Dictionnaires de labels
labels_map_sentiment = {
    0: "negatif",
    1: "neutre",
    2: "positif",
    3: "no_relation"
}

labels_map_emotion = {
    0: "colère",
    1: "joie",
    2: "tristesse",
    3: "provocation"
}


def predict_sentiment(texts):
    """Analyse de sentiment uniquement"""
    inputs = tokenizer_sentiment(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model_sentiment(**inputs)
    return torch.argmax(outputs.logits, dim=1)


def predict_emotion(texts):
    """Analyse d'émotion uniquement pour textes non-neutres"""
    inputs = tokenizer_emotion(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model_emotion(**inputs)
    return torch.argmax(outputs.logits, dim=1)


def analyze_comments(comments):
    """Fonction principale qui combine les deux analyses"""
    if not isinstance(comments, list):
        comments = [comments]

    # 1. Analyse de sentiment
    sentiment_results = predict_sentiment(comments)

    # 2. Préparation des textes pour l'analyse d'émotion
    texts_to_analyze = []
    emotion_indices = []  # Pour garder la trace des positions originales

    for idx, (text, sentiment_idx) in enumerate(zip(comments, sentiment_results)):
        sentiment_label = labels_map_sentiment.get(sentiment_idx.item())

        # On ne garde que les textes non-neutres et avec relation
        if sentiment_label not in ["neutre", "no_relation"]:
            texts_to_analyze.append(text)
            emotion_indices.append(idx)

    # 3. Analyse d'émotion seulement sur les textes sélectionnés
    emotion_results = []
    if texts_to_analyze:
        emotion_predictions = predict_emotion(texts_to_analyze)
        emotion_results = [(idx, pred) for idx, pred in zip(emotion_indices, emotion_predictions)]

    # 4. Construction des résultats finaux
    final_results = []
    for idx, text in enumerate(comments):
        sentiment_idx = sentiment_results[idx].item()
        sentiment = labels_map_sentiment.get(sentiment_idx, "inconnu")

        # Valeur par défaut pour l'émotion
        emotion = "non_applicable"

        # On cherche si ce texte a été analysé pour l'émotion
        for emo_idx, emo_pred in emotion_results:
            if idx == emo_idx:
                emotion_idx = emo_pred.item()
                emotion = labels_map_emotion.get(emotion_idx, "inconnu")
                break

        final_results.append({
            "text": text,
            "sentiment": sentiment,
            "emotion": emotion
        })

    return final_results


def summarize_arabic(text):
    """Fonction de résumé inchangée"""
    if not text or len(text.strip()) < 20:
        return "النص قصير جدًا أو غير متاح للتلخيص"

    try:
        response = gemini_model.generate_content(f"لخص هذا النص العربي:\n\n{text}")
        return response.text
    except Exception as e:
        return f"خطأ في التلخيص: {str(e)}"