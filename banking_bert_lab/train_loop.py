"""Manual PyTorch training loop; no Hugging Face Trainer.

Compatible with Hugging Face sequence classifiers and simple offline test models
that return an object exposing `.loss` and `.logits`.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class TokenizedBankDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int = 96):
        if max_length < 4:
            raise ValueError("max_length must be at least 4")
        self.labels = [int(row["label"]) for row in rows]
        encoding = tokenizer(
            [row["text"] for row in rows],
            padding="max_length", truncation=True, max_length=max_length,
            return_tensors="pt",
        )
        self.inputs = {k: v for k, v in encoding.items() if k in ("input_ids", "attention_mask", "token_type_ids")}

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        sample = {k: v[index] for k, v in self.inputs.items()}
        sample["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return sample


def train_one_epoch(model, loader, optimizer, scheduler, device: torch.device) -> float:
    """forward -> loss -> backward -> gradient clipping -> optimizer/scheduler step."""
    model.train()
    total, batches = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        result = model(**batch)
        loss = result.loss
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += float(loss.detach().item())
        batches += 1
    if batches == 0:
        raise ValueError("Training loader is empty")
    return total / batches


@torch.no_grad()
def predict_labels(model, loader, device: torch.device) -> list[int]:
    model.eval()
    output = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        output.extend(logits.argmax(dim=-1).cpu().tolist())
    return output
