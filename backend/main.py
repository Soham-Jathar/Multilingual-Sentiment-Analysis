"""FastAPI backend for the Multilingual Sentiment Analysis web application."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.baseline import build_baseline, predict_baseline
from src.preprocess import clean_text, detect_language

ROOT = Path(__file__).resolve().parents[1]
FINETUNED_MODEL = ROOT / "models" / "xlm-roberta-sentiment"

app = FastAPI(title="Multilingual Sentiment Analysis API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

baseline_model = None
transformer_model = None
transformer_name = None
transformer_tokenizer = None
translation_model = None
translation_tokenizer = None

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_LANGUAGE_CODES = {
    "en": "eng_Latn", "as": "asm_Beng", "bn": "ben_Beng", "gu": "guj_Gujr",
    "hi": "hin_Deva", "kn": "kan_Knda", "ml": "mal_Mlym", "mr": "mar_Deva",
    "or": "ory_Orya", "pa": "pan_Guru", "ta": "tam_Taml", "te": "tel_Telu",
}
TRANSLATION_LANGUAGE_NAMES = {
    "en": "English", "as": "Assamese", "bn": "Bengali", "gu": "Gujarati",
    "hi": "Hindi", "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi",
    "or": "Odia", "pa": "Punjabi", "ta": "Tamil", "te": "Telugu",
}


class AnalysisRequest(BaseModel):
    text: str = Field(min_length=3, max_length=2000)
    mode: Literal["fast", "transformer"] = "transformer"
    translation_enabled: bool = False
    translation_target: str = "en"


class AnalysisResponse(BaseModel):
    text: str
    cleaned_text: str
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    language_code: str
    language: str
    model: str
    decision_source: str = "model"
    token_count: int = 0
    preprocessing_steps: list[str] = Field(default_factory=list)
    translation: str | None = None
    translation_target: str | None = None
    translation_model: str | None = None


def get_baseline():
    global baseline_model
    if baseline_model is None:
        baseline_model = build_baseline()
    return baseline_model


def get_transformer():
    global transformer_model, transformer_name, transformer_tokenizer
    if transformer_model is None:
        from transformers import AutoTokenizer, pipeline

        if FINETUNED_MODEL.exists():
            transformer_model = pipeline("text-classification", model=str(FINETUNED_MODEL), tokenizer=str(FINETUNED_MODEL))
            transformer_tokenizer = AutoTokenizer.from_pretrained(str(FINETUNED_MODEL))
            transformer_name = "Fine-tuned XLM-RoBERTa"
        else:
            transformer_model = pipeline("text-classification", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")
            transformer_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
            transformer_name = "Pretrained XLM-RoBERTa"
    return transformer_model, transformer_name, transformer_tokenizer


def normalize_label(label: str) -> str:
    normalized = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}.get(label, label.lower())
    if normalized not in {"positive", "negative", "neutral"}:
        raise ValueError(f"Unsupported model label: {label}")
    return normalized


@lru_cache(maxsize=256)
def translate_text(text: str, language_code: str, target_code: str) -> tuple[str | None, str | None]:
    """Translate short reviews with a cached, GPU-optimised NLLB model."""
    global translation_model, translation_tokenizer
    source_language = NLLB_LANGUAGE_CODES.get(language_code)
    target_language = NLLB_LANGUAGE_CODES.get(target_code)
    if source_language is None or target_language is None:
        return None, None
    if language_code == target_code:
        return text, "Original text"
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        use_cuda = torch.cuda.is_available()
        if translation_model is None:
            translation_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL, src_lang=source_language)
            if use_cuda:
                # A 4070 runs this distilled model faster and with less VRAM in FP16.
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                translation_model = AutoModelForSeq2SeqLM.from_pretrained(
                    NLLB_MODEL,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                ).to("cuda")
            else:
                translation_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
            translation_model.eval()
        translation_tokenizer.src_lang = source_language
        inputs = translation_tokenizer(text, return_tensors="pt", truncation=True, max_length=192)
        if use_cuda:
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.inference_mode():
            output = translation_model.generate(
                **inputs,
                forced_bos_token_id=translation_tokenizer.convert_tokens_to_ids(target_language),
                max_new_tokens=96,
                num_beams=1,
                do_sample=False,
                use_cache=True,
            )
        return translation_tokenizer.batch_decode(output, skip_special_tokens=True)[0], "NLLB-200"
    except Exception:
        # Translation is supplementary; it should never prevent sentiment analysis.
        return None, None


def is_factual_question(text: str, language_code: str) -> bool:
    """Identify short English factual questions with no explicit sentiment term.

    This is deliberately narrow: opinion questions still go to XLM-RoBERTa.
    """
    if language_code != "en" or not text.rstrip().endswith("?"):
        return False
    factual_starts = r"^(who|what|when|where|which|how many|how much|is there|are there)\b"
    sentiment_words = r"\b(good|great|excellent|bad|poor|terrible|awful|love|hate|happy|angry|better|worse|helpful|rude)\b"
    return bool(re.match(factual_starts, text.strip(), flags=re.IGNORECASE)) and not bool(re.search(sentiment_words, text, flags=re.IGNORECASE))


def analyze_text(text: str, mode: str) -> tuple[str, float, str, int, list[str]]:
    if mode == "transformer":
        try:
            model, name, tokenizer = get_transformer()
            result = model(text, truncation=True)[0]
            token_count = len(tokenizer(text, truncation=True)["input_ids"])
            return normalize_label(result["label"]), float(result["score"]), name, token_count, [
                "Text cleaning", "Language identification", "XLM-RoBERTa subword tokenization", "Transformer sentiment classification"
            ]
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"XLM-RoBERTa is unavailable: {type(error).__name__}. Download or fine-tune the model, then restart the backend.")
    label, confidence = predict_baseline(get_baseline(), text)
    return label, confidence, "Local TF-IDF + Logistic Regression", 0, [
        "Text cleaning", "Language identification", "TF-IDF character n-gram vectorization", "Logistic-regression classification"
    ]


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/api/analyze", response_model=AnalysisResponse)
def analyze(request: AnalysisRequest) -> AnalysisResponse:
    cleaned = clean_text(request.text)
    if len(cleaned) < 3:
        raise HTTPException(status_code=422, detail="Enter at least three meaningful characters.")
    language_code, language = detect_language(cleaned)
    target_code = request.translation_target.lower()
    translation, translation_model_name = (None, None)
    if request.translation_enabled:
        translation, translation_model_name = translate_text(cleaned, language_code, target_code)
    if is_factual_question(cleaned, language_code):
        sentiment, confidence, model, token_count = "neutral", 1.0, "NLP factual-question rule", 0
        preprocessing_steps = ["Text cleaning", "Language identification", "Factual question detection", "No sentiment detected: classified as neutral"]
        decision_source = "nlp_rule"
    else:
        sentiment, confidence, model, token_count, preprocessing_steps = analyze_text(cleaned, request.mode)
        decision_source = "model"
    return AnalysisResponse(
        text=request.text.strip(),
        cleaned_text=cleaned,
        sentiment=sentiment,
        confidence=confidence,
        language_code=language_code,
        language=language,
        model=model,
        token_count=token_count,
        preprocessing_steps=preprocessing_steps,
        decision_source=decision_source,
        translation=translation,
        translation_target=TRANSLATION_LANGUAGE_NAMES.get(target_code) if request.translation_enabled else None,
        translation_model=translation_model_name,
    )
