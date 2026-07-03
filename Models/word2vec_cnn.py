"""Train a Word2Vec + CNN classifier for cleaned patent text."""

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
from Models.common import (  # noqa: E402
    PAD_TOKEN,
    TextDataset,
    build_vocab,
    classification_metrics,
    fit_label_encoder,
    get_device,
    read_text_label_csv,
    save_label_encoder,
    save_vocab,
    set_seed,
    texts_to_tokens,
    write_metrics,
)


class Word2VecCNN(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, num_classes: int, num_filters: int, kernel_size: int, dropout: float):
        super().__init__()
        embeddings = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=0)
        self.conv = nn.Conv1d(embeddings.shape[1], num_filters, kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids).transpose(1, 2)
        features = torch.relu(self.conv(embedded))
        pooled = torch.max(features, dim=2).values
        return self.classifier(self.dropout(pooled))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Word2Vec + CNN classifier.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--valid-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=50000)
    parser.add_argument("--embedding-dim", type=int, default=200)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--word2vec-epochs", type=int, default=10)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


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


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = read_text_label_csv(args.train_csv, args.text_col, args.label_col, args.encoding)
    valid_df = read_text_label_csv(args.valid_csv, args.text_col, args.label_col, args.encoding)
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
        seed=args.seed,
    )
    embedding_matrix = build_embedding_matrix(vocab, w2v, args.embedding_dim)

    train_loader = DataLoader(TextDataset(train_df[args.text_col], y_train, vocab, args.max_len), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(TextDataset(valid_df[args.text_col], y_valid, vocab, args.max_len), batch_size=args.batch_size)

    model = Word2VecCNN(embedding_matrix, len(encoder.classes_), args.num_filters, args.kernel_size, args.dropout).to(device)
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
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args) | {"model_type": "word2vec_cnn", "num_classes": len(encoder.classes_)}, file, ensure_ascii=False, indent=2)
    write_metrics(best_metrics, output_dir / "valid_metrics.json")
    return best_metrics


def main() -> int:
    train(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
