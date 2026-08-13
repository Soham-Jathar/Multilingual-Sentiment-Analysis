# SentimentScope — Multilingual Sentiment Analysis

A full-stack multilingual sentiment-analysis application. The React interface sends text to a FastAPI service that cleans the text, identifies the language, predicts sentiment, and returns the result without storing the review.

## What is included

- **React + Vite frontend:** responsive analysis workspace, result dashboard, and model-mode switch.
- **FastAPI backend:** validated REST API, language detection, and model orchestration.
- **NLP layer:** text cleaning, language identification, multilingual tokenization support, and a local TF-IDF baseline.
- **Translation layer:** supported Indic-language reviews are also translated to English with NLLB-200, so a user can understand feedback written in another state's language.
- **Deep-learning layer:** XLM-RoBERTa integration and a fine-tuning script for the supplied labelled dataset.
- **No manual label/language fields:** the user enters only feedback; language and sentiment are inferred by the application.

## Architecture

```text
React + Vite UI  ->  FastAPI API  ->  NLP preprocessing
                                   ->  Local baseline or XLM-RoBERTa
```

## Run locally

Use two terminals from this project folder.

### 1. Start the backend

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
uvicorn backend.main:app --reload
```

The API starts at `http://127.0.0.1:8000`. Reviews are processed only for the current result and are not saved.

### 2. Start the frontend

```powershell
cd frontend
npm install
npm run dev
```

Open the local address shown by Vite (normally `http://localhost:5173`).

## Analysis modes

- **XLM-RoBERTa (default):** the app uses a fine-tuned XLM-RoBERTa model when available; otherwise it retrieves a pretrained multilingual XLM-RoBERTa sentiment model on first use. The initial download/loading step is slower; later analyses use the cached model.
- **Baseline comparison:** cached local TF-IDF + Logistic Regression. It is retained only for comparing a conventional NLP baseline with the transformer in the project evaluation.

## Cross-language translation

Users can select a translation target from the project's supported languages: English, Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, and Telugu. The sentiment classifier still receives the original text, not the translated text. The translation model (`facebook/nllb-200-distilled-600M`) downloads only on its first translation request and is cached afterwards.

## Fine-tune XLM-RoBERTa

Prepare a larger English + Indic dataset. This example includes English and every Indic language available in the project source:

```powershell
python -m src.prepare_indic_dataset --include-english --languages as bn gu hi kn ml mr or pa ta te --per-label 0
```

Use `--per-label 0` to include every available labelled example for the selected languages. This requires more RAM, disk space, and training time.

Then train XLM-RoBERTa on the exported data:

```powershell
python -m src.train_transformer --data data\indic_sentiment_large.csv --epochs 3 --batch-size 8
```

This saves the trained model in `models/xlm-roberta-sentiment/`. Restart the backend afterwards; selecting **Use deep-learning model** in the interface then uses the saved model.

## API routes

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/health` | Confirms the service is ready |
| `POST` | `/api/analyze` | Creates an analysis from `{ text, mode }` |
| `GET` | `/api/stats` | Returns saved-analysis totals by sentiment |

## Dataset note

`data/multilingual_sentiment.csv` contains a small, balanced demonstration dataset in six languages. It is enough to verify the full flow, but expand it with a larger, balanced, labelled dataset before reporting real-world performance metrics.
