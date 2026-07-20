"""Train a Word2Vec + TextCNN classifier for cleaned patent text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.common import (  # noqa: E402
    PAD_TOKEN,
    TextDataset,
    average_metrics,
    build_vocab,
    classification_metrics,
    fit_label_encoder,
    get_device,
    read_text_label_csv,
    save_label_encoder,
    save_vocab,
    set_seed,
    stratified_kfold_indices,
    texts_to_tokens,
    write_metrics,
)


class Word2VecTextCNN(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, num_classes: int, num_filters: int, kernel_sizes: list[int], dropout: float):
        super().__init__()
        embeddings = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embeddings.shape[1], num_filters, kernel_size) for kernel_size in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            features = torch.relu(conv(embedded))
            pooled.append(torch.max(features, dim=2).values)
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Word2Vec + TextCNN classifier.")
    parser.add_argument("--data-csv", help="Full cleaned CSV for stratified k-fold cross-validation.")
    parser.add_argument("--train-csv", help="Optional explicit training CSV.")
    parser.add_argument("--valid-csv", help="Optional explicit validation CSV.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--fold-col", default="cv_fold")
    parser.add_argument("--cv-folds", type=int, default=10)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=50000)
    parser.add_argument("--embedding-dim", type=int, default=200)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--word2vec-epochs", type=int, default=10)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--kernel-sizes", default="3,4,5")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def parse_kernel_sizes(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_embedding_matrix(vocab: dict[str, int], w2v: Word2Vec, embedding_dim: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    matrix = rng.normal(0, 0.05, size=(len(vocab), embedding_dim)).astype("float32")
    matrix[vocab[PAD_TOKEN]] = 0.0
    for token, idx in vocab.items():
        if token in w2v.wv:
            matrix[idx] = w2v.wv[token]
    return matrix


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, label_names: list[str]) -> dict[str, object]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            logits = model(input_ids)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
    return classification_metrics(y_true, y_pred, label_names)


def train_once(args: argparse.Namespace, train_df, valid_df, output_dir: Path, seed: int) -> dict[str, object]:
    set_seed(seed)
    device = get_device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    kernel_sizes = parse_kernel_sizes(args.kernel_sizes)

    encoder = fit_label_encoder(train_df[args.label_col], valid_df[args.label_col])
    y_train = encoder.transform(train_df[args.label_col])
    y_valid = encoder.transform(valid_df[args.label_col])

    tokenized_train = texts_to_tokens(train_df[args.text_col])
    vocab = build_vocab(tokenized_train, min_freq=args.min_freq, max_vocab_size=args.max_vocab_size)
    w2v = Word2Vec(
        sentences=tokenized_train,
        vector_size=args.embedding_dim,
        window=args.window,
        min_count=args.min_freq,
        workers=4,
        sg=1,
        epochs=args.word2vec_epochs,
        seed=seed,
    )
    embedding_matrix = build_embedding_matrix(vocab, w2v, args.embedding_dim)

    train_loader = DataLoader(TextDataset(train_df[args.text_col], y_train, vocab, args.max_len), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(TextDataset(valid_df[args.text_col], y_valid, vocab, args.max_len), batch_size=args.batch_size)

    model = Word2VecTextCNN(embedding_matrix, len(encoder.classes_), args.num_filters, kernel_sizes, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1.0
    best_metrics: dict[str, object] = {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(input_ids), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        metrics = evaluate(model, valid_loader, device, list(encoder.classes_))
        print(f"epoch={epoch} loss={total_loss / max(1, len(train_loader)):.4f} valid_f1_macro={metrics['f1_macro']:.4f}")
        if metrics["f1_macro"] > best_f1:
            best_f1 = float(metrics["f1_macro"])
            best_metrics = metrics
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    save_vocab(vocab, output_dir)
    save_label_encoder(encoder, output_dir)
    config = vars(args).copy()
    config.update({"model_type": "word2vec_textcnn", "num_classes": len(encoder.classes_)})
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
    write_metrics(best_metrics, output_dir / "valid_metrics.json")
    return best_metrics


def train_final(args: argparse.Namespace, train_df, output_dir: Path, seed: int) -> dict[str, object]:
    set_seed(seed)
    device = get_device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    kernel_sizes = parse_kernel_sizes(args.kernel_sizes)

    encoder = fit_label_encoder(train_df[args.label_col])
    y_train = encoder.transform(train_df[args.label_col])
    tokenized_train = texts_to_tokens(train_df[args.text_col])
    vocab = build_vocab(tokenized_train, min_freq=args.min_freq, max_vocab_size=args.max_vocab_size)
    w2v = Word2Vec(
        sentences=tokenized_train,
        vector_size=args.embedding_dim,
        window=args.window,
        min_count=args.min_freq,
        workers=4,
        sg=1,
        epochs=args.word2vec_epochs,
        seed=seed,
    )
    embedding_matrix = build_embedding_matrix(vocab, w2v, args.embedding_dim)
    train_loader = DataLoader(
        TextDataset(train_df[args.text_col], y_train, vocab, args.max_len),
        batch_size=args.batch_size,
        shuffle=True,
    )
    model = Word2VecTextCNN(embedding_matrix, len(encoder.classes_), args.num_filters, kernel_sizes, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(input_ids), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch={epoch} train_loss={total_loss / max(1, len(train_loader)):.4f}")

    torch.save(model.state_dict(), output_dir / "best_model.pt")
    save_vocab(vocab, output_dir)
    save_label_encoder(encoder, output_dir)
    config = vars(args).copy()
    config.update({"model_type": "word2vec_textcnn", "num_classes": len(encoder.classes_), "training_mode": "final_train"})
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
    return {"train_rows": len(train_df)}


def cross_validate(args: argparse.Namespace) -> dict[str, object]:
    data_df = read_text_label_csv(args.data_csv, args.text_col, args.label_col, args.encoding)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.fold_col in data_df.columns:
        fold_ids = sorted(data_df[args.fold_col].dropna().unique().tolist())
        folds = [data_df.index[data_df[args.fold_col] == fold_id].to_numpy() for fold_id in fold_ids]
    else:
        folds = stratified_kfold_indices(data_df[args.label_col], n_splits=args.cv_folds, seed=args.seed)

    fold_metrics = []
    for fold_idx, valid_indices in enumerate(folds):
        valid_set = set(valid_indices.tolist())
        train_indices = [idx for idx in data_df.index if idx not in valid_set]
        train_df = data_df.loc[train_indices].reset_index(drop=True)
        valid_df = data_df.loc[valid_indices].reset_index(drop=True)
        print(f"fold={fold_idx + 1}/{len(folds)} train={len(train_df)} valid={len(valid_df)}")
        metrics = train_once(args, train_df, valid_df, output_dir / f"fold_{fold_idx:02d}", args.seed + fold_idx)
        fold_metrics.append(metrics)

    cv_metrics = average_metrics(fold_metrics)
    write_metrics(cv_metrics, output_dir / "cv_metrics.json")
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args) | {"model_type": "word2vec_textcnn", "cv_folds_actual": len(folds)}, file, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in cv_metrics.items() if k != "fold_metrics"}, ensure_ascii=False, indent=2))
    return cv_metrics


def train(args: argparse.Namespace) -> dict[str, object]:
    if args.data_csv:
        return cross_validate(args)
    if not args.train_csv:
        raise ValueError("Use --data-csv for k-fold cross-validation, or provide --train-csv for final training.")
    train_df = read_text_label_csv(args.train_csv, args.text_col, args.label_col, args.encoding)
    if not args.valid_csv:
        return train_final(args, train_df, Path(args.output_dir), args.seed)
    valid_df = read_text_label_csv(args.valid_csv, args.text_col, args.label_col, args.encoding)
    return train_once(args, train_df, valid_df, Path(args.output_dir), args.seed)


def main() -> int:
    train(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
