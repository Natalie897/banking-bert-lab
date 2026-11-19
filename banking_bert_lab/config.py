"""Defaults intentionally keep the first BERT experiment small enough for Colab/CPU."""

MODEL_ID = "google/bert_uncased_L-2_H-128_A-2"
DATASET_ID = "PolyAI/banking77"
DEFAULT_LABELS = (
    "card_arrival",
    "card_not_working",
    "card_payment_fee_charged",
    "cash_withdrawal_charge",
    "declined_card_payment",
    "pending_card_payment",
    "transfer_not_received_by_recipient",
    "wrong_exchange_rate_for_cash_withdrawal",
)
SEED = 42
