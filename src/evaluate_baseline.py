"""Evaluate the transparent local NLP baseline and write report artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "multilingual_sentiment.csv"
REPORTS = ROOT / "reports"


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    frame = pd.read_csv(DATA_PATH)
    x_train, x_test, y_train, y_test = train_test_split(
        frame["text"].map(clean_text), frame["label"], test_size=0.25, random_state=42, stratify=frame["label"]
    )
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    classifier.fit(x_train_vectorized, y_train)
    predictions = classifier.predict(x_test_vectorized)

    report = classification_report(y_test, predictions, digits=3)
    (REPORTS / "baseline_classification_report.txt").write_text(report, encoding="utf-8")
    print(report)

    labels = ["negative", "neutral", "positive"]
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(cmap="Blues", colorbar=False)
    plt.title("Local NLP Baseline: Confusion Matrix")
    plt.tight_layout()
    plt.savefig(REPORTS / "baseline_confusion_matrix.png", dpi=160)
    print(f"Saved reports to: {REPORTS}")


if __name__ == "__main__":
    main()
