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
    "as": "Assamese",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "or": "Odia",
    "pa": "Punjabi",
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
    # These Unicode blocks identify scripts reliably. Assamese and Bengali share
    # a script, so distinctive Assamese letters are checked separately below.
    script_blocks = {
        "hi": (0x0900, 0x097F),  # Hindi/Marathi share Devanagari.
        "pa": (0x0A00, 0x0A7F),
        "gu": (0x0A80, 0x0AFF),
        "or": (0x0B00, 0x0B7F),
        "ta": (0x0B80, 0x0BFF),
        "te": (0x0C00, 0x0C7F),
        "kn": (0x0C80, 0x0CFF),
        "ml": (0x0D00, 0x0D7F),
    }
    for code, (start, end) in script_blocks.items():
        if any(start <= ord(character) <= end for character in cleaned):
            if code != "hi":
                return code, LANGUAGE_NAMES[code]
            break
    else:
        if re.search(r"[\u09F0\u09F1]", cleaned):  # ৰ / ৱ, Assamese letters
            return "as", LANGUAGE_NAMES["as"]
        if re.search(r"[\u0980-\u09FF]", cleaned):
            return "bn", LANGUAGE_NAMES["bn"]
        if cleaned.isascii():
            return "en", LANGUAGE_NAMES["en"]
    if detect is None:
        return "unknown", LANGUAGE_NAMES["unknown"]
    try:
        code = detect(cleaned)
    except Exception:
        code = "unknown"
    return code, LANGUAGE_NAMES.get(code, code.upper())
