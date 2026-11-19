"""Reproducible fine-tuning and baseline experiment.

An actual run requires PyTorch, Hugging Face Transformers and BANKING77 downloads.
Never invent results when downloads/training haven't happened.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .baseline import train_baseline
from .config import DATASET_ID, DEFAULT_LABELS, MODEL_ID, SEED
from .data import load_banking77
from .metrics import class_distribution, classification_metrics, collect_errors


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_error_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "actual", "predicted"])
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    output_dir: str = "artifacts/banking_bert",
    model_id: str = MODEL_ID,
    all_labels: bool = False,
    epochs: int = 3,
    train_per_class: int = 70,
    test_per_class: int = 25,
    batch_size: int = 16,
    lr: float = 3e-5,
    max_length: int = 96,
    seed: int = SEED,
) -> dict:
    if epochs < 1 or batch_size < 1 or lr <= 0 or train_per_class < 5 or test_per_class < 1:
        raise ValueError("Check epochs, batch_size, lr and per_class limits")
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
    from .train_loop import TokenizedBankDataset, predict_labels, train_one_epoch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = None
    if all_labels:
        from datasets import load_dataset
        ds = load_dataset(DATASET_ID)
        names = list(ds["train"].features["label"].names)
        from .data import prepare_splits
        splits = prepare_splits(ds["train"], ds["test"], names, names,
                                train_per_class, test_per_class, seed)
    else:
        splits = load_banking77(list(DEFAULT_LABELS), train_per_class,
                                test_per_class, seed)
    names = splits.labels
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Valid comparison: identical train/validation/test partitions for both methods.
    baseline = train_baseline(splits.train, splits.val, splits.test, names)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(names),
        id2label={i: name for i, name in enumerate(names)},
        label2id={name: i for i, name in enumerate(names)},
    ).to(device)

    def loader(rows: list[dict], shuffle: bool):
        dataset = TokenizedBankDataset(rows, tokenizer, max_length=max_length)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = loader(splits.train, shuffle=True)
    val_loader = loader(splits.val, shuffle=False)
    test_loader = loader(splits.test, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, int(n_steps * 0.1)), num_training_steps=n_steps
    )

    best_score = -1.0
    best_epoch = 0
    best_state = None
    history: list[dict] = []
    true_val = [r["label"] for r in splits.val]
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_pred = predict_labels(model, val_loader, device)
        val_metrics = classification_metrics(true_val, val_pred, names)
        history.append({"epoch": epoch, "train_loss": round(loss, 6),
                        "val_macro_f1": val_metrics["macro_f1"],
                        "val_accuracy": val_metrics["accuracy"]})
        print(f"Epoch {epoch}/{epochs}: loss={loss:.4f}, "
              f"val_macro_f1={val_metrics['macro_f1']:.4f}", flush=True)
        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            best_epoch = epoch
            # CPU copy is safe and small for our 2-layer BERT.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model.save_pretrained(out / "model", safe_serialization=True)
    tokenizer.save_pretrained(out / "model")

    # Evaluate on untouched official TEST only after model selection on VAL.
    b_val_pred = predict_labels(model, val_loader, device)
    b_test_pred = predict_labels(model, test_loader, device)
    bert_val = classification_metrics(true_val, b_val_pred, names)
    bert_test = classification_metrics([r["label"] for r in splits.test], b_test_pred, names)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_ID,
        "model": model_id,
        "device": str(device),
        "num_classes": len(names),
        "class_names": names,
        "sizes": {"train": len(splits.train), "val": len(splits.val), "test": len(splits.test)},
        "class_distribution": {
            "train": class_distribution(splits.train, names),
            "val": class_distribution(splits.val, names),
            "test": class_distribution(splits.test, names),
        },
        "params": {"seed": seed, "epochs": epochs, "train_per_class": train_per_class,
                   "test_per_class": test_per_class, "batch_size": batch_size,
                   "learning_rate": lr, "max_length": max_length},
        "selection": "best BERT epoch by validation macro-F1, test evaluated once afterward",
        "baseline": {"validation": baseline["validation"], "test": baseline["test"]},
        "bert": {"validation": bert_val, "test": bert_test,
                 "best_epoch": best_epoch, "history": history},
    }
    _write_json(out / "metrics.json", report)
    _write_error_csv(out / "bert_errors.csv",
                     collect_errors(splits.test, b_test_pred, names, limit=100))
    _write_error_csv(out / "baseline_errors.csv",
                     collect_errors(splits.test, baseline["test_predictions"], names, limit=100))
    print(f"Results saved in: {out.resolve()}")
    print(f"Baseline macro-F1 (test): {baseline['test']['macro_f1']:.4f}")
    print(f"BERT macro-F1 (test): {bert_test['macro_f1']:.4f}")
    return report
