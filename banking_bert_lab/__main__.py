"""Command line interface: train and predict."""
from __future__ import annotations

import argparse
import json

from .config import MODEL_ID, SEED


def main() -> None:
    parser = argparse.ArgumentParser(description="Banking BERT Lab — educational PyTorch / Transformers fine-tuning")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Compare TF-IDF baseline with a BERT custom train loop")
    train.add_argument("--output-dir", default="artifacts/banking_bert")
    train.add_argument("--model-id", default=MODEL_ID)
    train.add_argument("--all-labels", action="store_true", help="Use 77 instead of 8 labels (slower)")
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--train-per-class", type=int, default=70)
    train.add_argument("--test-per-class", type=int, default=25)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--lr", type=float, default=3e-5)
    train.add_argument("--max-length", type=int, default=96)
    train.add_argument("--seed", type=int, default=SEED)

    pred = sub.add_parser("predict", help="Predict the intent using a previously trained model")
    pred.add_argument("--model-dir", default="artifacts/banking_bert/model")
    pred.add_argument("--text", required=True)
    pred.add_argument("--threshold", type=float, default=0.55)

    args = parser.parse_args()
    if args.command == "train":
        from .experiment import run_experiment
        run_experiment(output_dir=args.output_dir, model_id=args.model_id,
                       all_labels=args.all_labels, epochs=args.epochs,
                       train_per_class=args.train_per_class, test_per_class=args.test_per_class,
                       batch_size=args.batch_size, lr=args.lr,
                       max_length=args.max_length, seed=args.seed)
    elif args.command == "predict":
        from .inference import load_model, predict
        tokenizer, model, device = load_model(args.model_dir)
        print(json.dumps(predict(args.text, tokenizer, model, device, args.threshold),
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
