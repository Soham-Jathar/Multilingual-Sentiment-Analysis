"""A reproducible local NLP baseline used when a transformer model is unavailable."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocess import clean_text


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "multilingual_sentiment.csv"


def build_baseline(data_path: Path = DATA_PATH) -> Pipeline:
    """Fit a character-aware TF-IDF logistic-regression classifier from labelled data."""
    frame = pd.read_csv(data_path)
    texts = frame["text"].map(clean_text)
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    )
    return model.fit(texts, frame["label"])


def predict_baseline(model: Pipeline, text: str) -> tuple[str, float]:
    """Return the model's most likely label and its probability."""
    cleaned = clean_text(text)
    probabilities = model.predict_proba([cleaned])[0]
    index = int(probabilities.argmax())
    return str(model.classes_[index]), float(probabilities[index])
