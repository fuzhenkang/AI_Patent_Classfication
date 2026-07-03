"""Retrain the selected model on the full training set using Optuna best parameters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain final model with Optuna best parameters.")
    parser.add_argument("--model-type", required=True, choices=["word2vec_cnn", "word2vec_textcnn", "bert_cnn"])
    parser.add_argument("--best-params", required=True, help="Path to Optuna best_params.json.")
    parser.add_argument("--train-csv", required=True, help="80% training CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for final retrained model.")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--bert-model", default="hfl/chinese-roberta-wwm-ext")
    return parser.parse_args()


def load_best_params(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload.get("best_params", payload)


def build_args(args: argparse.Namespace, best_params: dict[str, object]) -> SimpleNamespace:
    params = {
        "data_csv": None,
        "train_csv": args.train_csv,
        "valid_csv": None,
        "output_dir": args.output_dir,
        "text_col": args.text_col,
        "label_col": args.label_col,
        "fold_col": "cv_fold",
        "cv_folds": 10,
        "encoding": args.encoding,
        "seed": args.seed,
        "device": args.device,
        "max_len": 256,
        "batch_size": 64,
        "epochs": 10,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "dropout": 0.5,
        "num_filters": 128,
    }
    params.update(best_params)

    if args.model_type in {"word2vec_cnn", "word2vec_textcnn"}:
        params.setdefault("min_freq", 1)
        params.setdefault("max_vocab_size", 50000)
        params.setdefault("embedding_dim", 200)
        params.setdefault("window", 5)
        params.setdefault("word2vec_epochs", 10)
        if args.model_type == "word2vec_cnn":
            params.setdefault("kernel_size", 3)
        else:
            params.setdefault("kernel_sizes", "3,4,5")

    if args.model_type == "bert_cnn":
        params.setdefault("bert_model", args.bert_model)
        params.setdefault("kernel_sizes", "3,4,5")
        params.setdefault("warmup_ratio", 0.1)
        params.setdefault("batch_size", 16)
        params.setdefault("epochs", 3)
        params.setdefault("lr", 2e-5)
        params.setdefault("weight_decay", 0.01)
        params.setdefault("dropout", 0.3)

    return SimpleNamespace(**params)


def main() -> int:
    args = parse_args()
    best_params = load_best_params(args.best_params)
    final_args = build_args(args, best_params)

    if args.model_type == "word2vec_cnn":
        from Models.word2vec_cnn import train
    elif args.model_type == "word2vec_textcnn":
        from Models.word2vec_textcnn import train
    else:
        from Models.bert_cnn import train

    train(final_args)
    print(f"Final model saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
