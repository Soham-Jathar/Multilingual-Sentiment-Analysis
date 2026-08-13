"""Download AI4Bharat IndicSentiment and export a balanced training CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "ai4bharat/IndicSentiment"
ENGLISH_DATASET_ID = "goosmanlei/amazon_reviews_multi"
INDIC_DATA_URL = "https://huggingface.co/datasets/ai4bharat/IndicSentiment/resolve/main/data/{split}/{language}.json"
LANGUAGE_NAMES = {
    "as": "Assamese", "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "or": "Odia",
    "pa": "Punjabi", "ta": "Tamil", "te": "Telugu",
}
VALID_LABELS = {"positive", "negative", "neutral"}


def get_value(row: dict, *names: str) -> str | None:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def iter_indic_records(language: str):
    """Read the published IndicSentiment JSONL files without its legacy dataset script."""
    for split in ("validation", "test"):
        url = INDIC_DATA_URL.format(language=language, split=split)
        request = Request(url, headers={"User-Agent": "MultilingualSentimentProject/1.0"})
        with urlopen(request) as response:
            for line in response:
                if line.strip():
                    yield json.loads(line.decode("utf-8"))


def append_english_reviews(rows: list[dict[str, str]], per_label: int) -> None:
    """Add balanced English Amazon reviews; 1-2 stars negative, 3 neutral, 4-5 positive."""
    dataset = load_dataset(ENGLISH_DATASET_ID, "en")
    collected = {label: 0 for label in VALID_LABELS}
    for record in dataset["train"]:
        stars = record.get("stars")
        text = get_value(record, "review_body", "review_title")
        if not isinstance(stars, int) or text is None:
            continue
        label = "negative" if stars <= 2 else "neutral" if stars == 3 else "positive"
        limit_reached = per_label > 0 and collected[label] >= per_label
        if limit_reached:
            continue
        rows.append({"text": text, "label": label, "language": "English"})
        collected[label] += 1
        if per_label > 0 and all(count >= per_label for count in collected.values()):
            break
    print(f"English: {collected}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a balanced IndicSentiment CSV for XLM-RoBERTa training.")
    parser.add_argument("--languages", nargs="+", choices=sorted(LANGUAGE_NAMES), default=["hi", "mr", "ta"])
    parser.add_argument("--include-english", action="store_true", help="Add English Amazon review data with ratings mapped to three sentiment labels.")
    parser.add_argument("--per-label", type=int, default=1500, help="Maximum examples for each label in each language; use 0 for all available examples.")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "indic_sentiment_large.csv")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for language in args.languages:
        collected = {label: 0 for label in VALID_LABELS}
        for record in iter_indic_records(language):
            label = str(get_value(record, "LABEL", "label") or "").lower()
            text = get_value(record, "INDIC REVIEW", "indic_review", "text", "review")
            if label not in VALID_LABELS or text is None:
                continue
            limit_reached = args.per_label > 0 and collected[label] >= args.per_label
            if limit_reached:
                continue
            rows.append({"text": text, "label": label, "language": LANGUAGE_NAMES[language]})
            collected[label] += 1
        print(f"{LANGUAGE_NAMES[language]}: {collected}")

    if args.include_english:
        append_english_reviews(rows, args.per_label)

    frame = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    if frame.empty or set(frame["label"]) != VALID_LABELS:
        raise ValueError("The prepared data does not contain all three sentiment labels. Check dataset access/config names.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Saved {len(frame)} rows to: {args.output}")
    print(frame.groupby(["language", "label"]).size())


if __name__ == "__main__":
    main()
