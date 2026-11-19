import pytest

from banking_bert_lab.data import prepare_splits, select_rows_per_class, split_train_validation


def rows(n=14, labels=3):
    return [
        {"text": f"banking example {label}-{i}", "label": label}
        for label in range(labels)
        for i in range(n)
    ]


def test_balanced_selection():
    r = select_rows_per_class(rows(), [0, 1, 2], 7, 42)
    assert len(r) == 21
    assert {k: sum(x["label"] == k for x in r) for k in range(3)} == {0: 7, 1: 7, 2: 7}


def test_selection_reproducible():
    assert select_rows_per_class(rows(), [0, 2], 6, 8) == select_rows_per_class(rows(), [0, 2], 6, 8)


def test_selection_different_seeds():
    assert select_rows_per_class(rows(), [0, 2], 6, 8) != select_rows_per_class(rows(), [0, 2], 6, 9)


def test_selection_all_when_limit_none():
    assert len(select_rows_per_class(rows(), [1], None, 42)) == 14


def test_selection_missing_label():
    with pytest.raises(ValueError, match="Missing"):
        select_rows_per_class(rows(), [12], 2, 42)


def test_selection_duplicate_label_ids():
    with pytest.raises(ValueError):
        select_rows_per_class(rows(), [1, 1], 2, 42)


def test_selection_bad_limit():
    with pytest.raises(ValueError):
        select_rows_per_class(rows(), [1], 0, 42)


def test_split_stratified_no_overlap():
    train, val = split_train_validation(rows(), 0.25, 42)
    assert len(train) + len(val) == len(rows())
    assert set(x["text"] for x in train).isdisjoint(x["text"] for x in val)
    assert {x["label"] for x in val} == {0, 1, 2}


def test_split_bad_val_fraction():
    with pytest.raises(ValueError):
        split_train_validation(rows(), 1.0, 42)


def test_split_bad_too_few_per_class():
    with pytest.raises(ValueError):
        split_train_validation(rows(1), 0.25, 42)


def test_prepare_splits_label_mapping():
    # Only classes "second" and "fourth", remapped to zero and one.
    names = ["first", "second", "third", "fourth"]
    source_train = rows(n=20, labels=4)
    source_test = [{"text": f"official test {l}-{i}", "label": l}
                   for l in range(4) for i in range(8)]
    obj = prepare_splits(source_train, source_test, names, ["fourth", "second"],
                         train_per_class=10, test_per_class=5, seed=42)
    assert obj.labels == ["fourth", "second"]
    assert len(obj.train) + len(obj.val) == 20
    assert len(obj.test) == 10
    assert {row["label"] for row in obj.train + obj.val + obj.test} == {0, 1}
    assert {row["text"] for row in obj.train + obj.val}.isdisjoint(row["text"] for row in obj.test)


def test_prepare_splits_invalid_class():
    with pytest.raises(ValueError, match="Unknown"):
        prepare_splits(rows(), rows(), ["one", "two", "three"], ["unknown"])


def test_prepare_splits_duplicate_names():
    with pytest.raises(ValueError):
        prepare_splits(rows(), rows(), ["one", "two", "three"], ["two", "two"])
