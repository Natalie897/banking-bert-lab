from banking_bert_lab.baseline import train_baseline


def test_baseline_predicts_banking_like_examples():
    names = ["card", "transfer"]
    train = ([{"text": f"card payment issue {i}", "label": 0} for i in range(12)]
             + [{"text": f"bank transfer failed {i}", "label": 1} for i in range(12)])
    val = [{"text": "card payment issue new", "label": 0}, {"text": "bank transfer failed new", "label": 1}]
    test = [{"text": "card payment issue again", "label": 0}, {"text": "bank transfer failed again", "label": 1}]
    result = train_baseline(train, val, test, names)
    assert result["name"] == "tfidf_logistic_regression"
    assert 0 <= result["test"]["macro_f1"] <= 1
    assert len(result["test_predictions"]) == 2
