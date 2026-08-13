"""NLP utilities for cleaning multilingual text and detecting its language."""

from __future__ import annotations

import re

try:
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0
except ImportError:  # Keeps this module importable before optional installation.
    detect = None


LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "unknown": "Unknown",
}


def clean_text(text: str) -> str:
    """Remove URLs and excess whitespace without stripping non-Latin scripts/emojis."""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_language(text: str) -> tuple[str, str]:
    """Return language quickly for common scripts, then fall back to langdetect."""
    cleaned = clean_text(text)
    if len(cleaned) < 3:
        return "unknown", LANGUAGE_NAMES["unknown"]
    script_ranges = {
        "hi": r"[\u0900-\u097F]",  # Hindi/Marathi share Devanagari.
        "ta": r"[\u0B80-\u0BFF]",
        "te": r"[\u0C00-\u0C7F]",
        "bn": r"[\u0980-\u09FF]",
    }
    for code, pattern in script_ranges.items():
        if re.search(pattern, cleaned):
            if code != "hi":
                return code, LANGUAGE_NAMES[code]
            break
    else:
        if cleaned.isascii():
            return "en", LANGUAGE_NAMES["en"]
    if detect is None:
        return "unknown", LANGUAGE_NAMES["unknown"]
    try:
        code = detect(cleaned)
    except Exception:
        code = "unknown"
    return code, LANGUAGE_NAMES.get(code, code.upper())
