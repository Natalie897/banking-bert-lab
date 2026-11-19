"""Small real PyTorch smoke tests, optional when torch is not installed in fast CI."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.utils.data import DataLoader, Dataset
from banking_bert_lab.train_loop import train_one_epoch, predict_labels, TokenizedBankDataset


class ToyDataset(Dataset):
    def __init__(self):
        self.rows = [
            {"input_ids": torch.tensor([i % 2, (i + 1) % 2], dtype=torch.float32),
             "labels": torch.tensor(i % 2, dtype=torch.long)}
            for i in range(12)
        ]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


class ToyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
    def forward(self, input_ids, labels=None):
        logits = self.linear(input_ids)
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return SimpleNamespace(loss=loss, logits=logits)


def test_manual_torch_loop_updates_weights():
    torch.manual_seed(42)
    model = ToyClassifier()
    before = model.linear.weight.detach().clone()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = DataLoader(ToyDataset(), batch_size=4)
    loss = train_one_epoch(model, loader, opt, scheduler=None, device=torch.device("cpu"))
    assert loss > 0
    assert not torch.equal(before, model.linear.weight)


def test_prediction_loop_returns_class_ids():
    model = ToyClassifier()
    loader = DataLoader(ToyDataset(), batch_size=5)
    predictions = predict_labels(model, loader, device=torch.device("cpu"))
    assert len(predictions) == 12
    assert set(predictions) <= {0, 1}


def test_train_empty_loader_raises():
    class Empty(Dataset):
        def __len__(self): return 0
        def __getitem__(self, i): raise IndexError
    model = ToyClassifier()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="empty"):
        train_one_epoch(model, DataLoader(Empty()), opt, None, torch.device("cpu"))


def test_tokenized_dataset_construction():
    def tokenizer(texts, **kwargs):
        return {"input_ids": torch.ones((len(texts), kwargs["max_length"]), dtype=torch.long),
                "attention_mask": torch.ones((len(texts), kwargs["max_length"]), dtype=torch.long)}
    ds = TokenizedBankDataset([{"text": "bank card", "label": 1}], tokenizer, max_length=8)
    assert len(ds) == 1
    assert ds[0]["input_ids"].size(0) == 8
    assert int(ds[0]["labels"].item()) == 1


def test_tokenized_dataset_bad_length():
    with pytest.raises(ValueError):
        TokenizedBankDataset([], lambda *args, **kwargs: {}, max_length=1)
