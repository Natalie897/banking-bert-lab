import pytest

from banking_bert_lab.metrics import class_distribution, classification_metrics, collect_errors


def test_perfect_scores():
    m = classification_metrics([0, 1, 2], [0, 1, 2], ["a", "b", "c"])
    assert m["accuracy"] == m["macro_f1"] == 1
    assert m["classes"]["a"]["support"] == 1


def test_some_errors_reduce_macro_f1():
    m = classification_metrics([0, 1, 0, 1], [0, 0, 0, 1], ["a", "b"])
    assert m["accuracy"] == 0.75
    assert 0 < m["macro_f1"] < 1


def test_missing_class_counts_in_macro():
    m = classification_metrics([0, 0], [0, 0], ["a", "b"])
    assert m["macro_f1"] == 0.5
    assert m["classes"]["b"]["support"] == 0


def test_empty_metrics_rejected():
    with pytest.raises(ValueError):
        classification_metrics([], [], ["a"])


def test_unknown_id_rejected():
    with pytest.raises(ValueError):
        classification_metrics([5], [0], ["a"])


def test_collect_errors_only_mistakes():
    rows = [{"text": "one", "label": 0}, {"text": "two", "label": 1}]
    e = collect_errors(rows, [0, 0], ["a", "b"])
    assert e == [{"text": "two", "actual": "b", "predicted": "a"}]


def test_collect_errors_limit():
    rows = [{"text": str(i), "label": 0} for i in range(5)]
    assert len(collect_errors(rows, [1]*5, ["a", "b"], 2)) == 2


def test_collect_errors_bad_length():
    with pytest.raises(ValueError):
        collect_errors([{"text":"x","label":0}], [], ["a"])


def test_class_distribution_all_labels():
    assert class_distribution([{"label": 0}, {"label": 0}], ["a", "b"]) == {"a": 2, "b": 0}
