"""Banking77 loader, balanced class selection, stratified train/validation split.

The official test split is never used for training or selecting hyperparameters.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from random import Random
from typing import Any, Iterable


@dataclass
class Splits:
    train: list[dict[str, Any]]
    val: list[dict[str, Any]]
    test: list[dict[str, Any]]
    labels: list[str]


def select_rows_per_class(
    rows: Iterable[dict[str, Any]],
    selected_labels: list[int],
    max_per_class: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Keep up to N seeded examples/class; retain label IDs from the source."""
    if max_per_class is not None and max_per_class < 1:
        raise ValueError("max_per_class must be >= 1 (or None)")
    if len(selected_labels) != len(set(selected_labels)) or not selected_labels:
        raise ValueError("selected_labels must be nonempty and unique")
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    ids = set(selected_labels)
    for row in rows:
        if int(row["label"]) in ids:
            groups[int(row["label"])].append({"text": str(row["text"]), "label": int(row["label"])})
    missing = [x for x in selected_labels if not groups[x]]
    if missing:
        raise ValueError(f"Missing labels in source: {missing}")
    rng = Random(seed)
    selected = []
    for label in selected_labels:
        group = groups[label][:]
        rng.shuffle(group)
        selected.extend(group if max_per_class is None else group[:max_per_class])
    rng.shuffle(selected)
    return selected


def split_train_validation(rows: list[dict[str, Any]], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Stratify after limiting classes, to keep each class in both partitions."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must lie strictly between 0 and 1")
    labels = [int(r["label"]) for r in rows]
    counts = Counter(labels)
    if not counts or any(n < 2 for n in counts.values()):
        raise ValueError("Need at least 2 training samples for every selected class")
    from sklearn.model_selection import train_test_split

    train, val = train_test_split(
        rows, test_size=val_fraction, stratify=labels, random_state=seed
    )
    return list(train), list(val)


def prepare_splits(
    train_source: Iterable[dict],
    test_source: Iterable[dict],
    all_labels: list[str],
    wanted_names: list[str],
    train_per_class: int = 70,
    test_per_class: int = 25,
    seed: int = 42,
    val_fraction: float = 0.2,
) -> Splits:
    if len(set(wanted_names)) != len(wanted_names) or not wanted_names:
        raise ValueError("wanted_names must be unique and nonempty")
    unknown = [x for x in wanted_names if x not in all_labels]
    if unknown:
        raise ValueError(f"Unknown label names: {unknown}")
    source_ids = [all_labels.index(name) for name in wanted_names]
    local_id = {source_id: i for i, source_id in enumerate(source_ids)}
    selected_train = select_rows_per_class(train_source, source_ids, train_per_class, seed)
    selected_test = select_rows_per_class(test_source, source_ids, test_per_class, seed + 1)
    train, val = split_train_validation(selected_train, val_fraction, seed)

    def remap(rows: list[dict]) -> list[dict]:
        return [{"text": r["text"], "label": local_id[r["label"]]} for r in rows]

    return Splits(train=remap(train), val=remap(val), test=remap(selected_test), labels=wanted_names)


def load_banking77(
    wanted_names: list[str], train_per_class: int = 70,
    test_per_class: int = 25, seed: int = 42,
) -> Splits:
    """Download official BANKING77 splits via Hugging Face (internet required)."""
    from datasets import load_dataset
    ds = load_dataset("PolyAI/banking77")
    label_names = list(ds["train"].features["label"].names)
    return prepare_splits(
        ds["train"], ds["test"], label_names, wanted_names,
        train_per_class=train_per_class, test_per_class=test_per_class, seed=seed,
    )
