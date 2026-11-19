"""Prediction using a locally fine-tuned BERT checkpoint. No logging of user text."""
from __future__ import annotations

from pathlib import Path


def load_model(model_dir: str = "artifacts/banking_bert/model"):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not (Path(model_dir) / "config.json").exists():
        raise FileNotFoundError(f"No trained checkpoint in {model_dir}. Run 'train' first.")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return tokenizer, model, device


def predict(text: str, tokenizer, model, device, threshold: float = 0.55) -> dict:
    import torch
    if not text or not text.strip():
        raise ValueError("text cannot be empty")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=96)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.inference_mode():
        logits = model(**batch).logits
        scores = torch.softmax(logits, dim=-1)[0]
        index = int(scores.argmax().item())
        score = float(scores[index].item())
    name = model.config.id2label[index]
    return {"label": name, "score": round(score, 4), "review_required": score < threshold,
            "note": "Score is a model softmax value, not a calibrated probability of correctness."}
