"""Optional local inference API. No claims about production readiness."""
from __future__ import annotations

from functools import lru_cache
from os import getenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .inference import load_model, predict

app = FastAPI(title="Banking BERT Lab", version="0.1.0")


class PredictRequest(BaseModel):
    text: str = Field(min_length=3, max_length=2000)


@lru_cache(maxsize=1)
def _engine():
    return load_model(getenv("BANKING_MODEL_DIR", "artifacts/banking_bert/model"))


@app.get("/health")
def health():
    try:
        _engine()
        return {"status": "ready"}
    except (FileNotFoundError, OSError):
        return {"status": "model_not_loaded"}


@app.post("/predict")
def classify(request: PredictRequest):
    try:
        tokenizer, model, device = _engine()
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=503, detail="Train a model first") from exc
    return predict(request.text, tokenizer, model, device)
