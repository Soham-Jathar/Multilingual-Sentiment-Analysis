"""Fine-tune XLM-RoBERTa for multilingual three-class sentiment classification."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from src.preprocess import clean_text

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "multilingual_sentiment.csv"
MODEL_PATH = ROOT / "models" / "xlm-roberta-sentiment"
LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def compute_metrics(prediction):
    logits, labels = prediction
    predicted = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average="weighted", zero_division=0)
    return {"accuracy": accuracy_score(labels, predicted), "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune XLM-RoBERTa on multilingual sentiment data.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="CSV file with text,label,language columns.")
    parser.add_argument("--output", type=Path, default=MODEL_PATH, help="Directory for the trained model.")
    parser.add_argument("--model-name", default="xlm-roberta-base")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    frame = pd.read_csv(args.data)
    required_columns = {"text", "label"}
    if not required_columns.issubset(frame.columns):
        raise ValueError(f"{args.data} must contain columns: {', '.join(sorted(required_columns))}")
    frame["text"] = frame["text"].map(clean_text)
    frame["label"] = frame["label"].map(LABEL_TO_ID)
    train_frame, validation_frame = train_test_split(frame, test_size=0.25, random_state=42, stratify=frame["label"])
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_data = Dataset.from_pandas(train_frame[["text", "label"]], preserve_index=False).map(tokenize, batched=True)
    validation_data = Dataset.from_pandas(validation_frame[["text", "label"]], preserve_index=False).map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=3, id2label=ID_TO_LABEL, label2id=LABEL_TO_ID)
    settings = TrainingArguments(output_dir=str(ROOT / "training_outputs"), learning_rate=2e-5, per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size, num_train_epochs=args.epochs, weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="f1", logging_strategy="epoch", report_to="none")
    trainer = Trainer(model=model, args=settings, train_dataset=train_data, eval_dataset=validation_data, processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer=tokenizer), compute_metrics=compute_metrics)
    trainer.train()
    metrics = trainer.evaluate()
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"Saved fine-tuned model to: {args.output}")
    print(metrics)


if __name__ == "__main__":
    main()
