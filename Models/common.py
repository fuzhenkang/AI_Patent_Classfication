"""Shared utilities for AI patent text classification models."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def ensure_project_root_on_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_text_label_csv(path: str | Path, text_col: str, label_col: str, encoding: str = "utf-8-sig") -> pd.DataFrame:
    df = pd.read_csv(path, encoding=encoding)
    missing = [col for col in [text_col, label_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {', '.join(missing)}")
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""].reset_index(drop=True)
    return df


def tokenize(text: str) -> list[str]:
    text = str(text).strip()
    if not text:
        return []
    tokens = text.split()
    if len(tokens) > 1:
        return tokens
    return list(text)


def texts_to_tokens(texts: Sequence[str]) -> list[list[str]]:
    return [tokenize(text) for text in texts]


def build_vocab(tokenized_texts: Sequence[Sequence[str]], min_freq: int = 1, max_vocab_size: int | None = None) -> dict[str, int]:
    freq: dict[str, int] = {}
    for tokens in tokenized_texts:
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1

    items = [(token, count) for token, count in freq.items() if count >= min_freq]
    items.sort(key=lambda item: (-item[1], item[0]))
    if max_vocab_size:
        items = items[: max(0, max_vocab_size - 2)]

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _ in items:
        vocab[token] = len(vocab)
    return vocab


def encode_tokens(tokens: Sequence[str], vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens[:max_len]]
    if len(ids) < max_len:
        ids.extend([vocab[PAD_TOKEN]] * (max_len - len(ids)))
    return ids


class SimpleLabelEncoder:
    def __init__(self) -> None:
        self.classes_: np.ndarray = np.array([])
        self._mapping: dict[object, int] = {}

    def fit(self, labels: Sequence[object]) -> "SimpleLabelEncoder":
        self.classes_ = np.array(sorted(set(labels), key=lambda value: str(value)))
        self._mapping = {label: idx for idx, label in enumerate(self.classes_)}
        return self

    def transform(self, labels: Sequence[object]) -> np.ndarray:
        return np.array([self._mapping[label] for label in labels], dtype=np.int64)

    def inverse_transform(self, labels: Sequence[int]) -> np.ndarray:
        return np.array([self.classes_[int(label)] for label in labels])


class TextDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], vocab: dict[str, int], max_len: int):
        self.inputs = [encode_tokens(tokenize(text), vocab, max_len) for text in texts]
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.inputs[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class BertCnnDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], tokenizer, max_len: int):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


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


def fit_label_encoder(*label_sequences: Sequence[object]) -> SimpleLabelEncoder:
    labels: list[object] = []
    for seq in label_sequences:
        labels.extend(list(seq))
    return SimpleLabelEncoder().fit(labels)


def save_label_encoder(encoder: SimpleLabelEncoder, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "label_classes.json").open("w", encoding="utf-8") as file:
        json.dump(list(encoder.classes_), file, ensure_ascii=False, indent=2)


def load_label_encoder(model_dir: str | Path) -> SimpleLabelEncoder:
    with (Path(model_dir) / "label_classes.json").open("r", encoding="utf-8") as file:
        classes = json.load(file)
    encoder = SimpleLabelEncoder()
    encoder.classes_ = np.array(classes)
    encoder._mapping = {label: idx for idx, label in enumerate(encoder.classes_)}
    return encoder


def save_vocab(vocab: dict[str, int], output_dir: str | Path) -> None:
    with (Path(output_dir) / "vocab.json").open("w", encoding="utf-8") as file:
        json.dump(vocab, file, ensure_ascii=False, indent=2)


def load_vocab(model_dir: str | Path) -> dict[str, int]:
    with (Path(model_dir) / "vocab.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[str]) -> dict[str, object]:
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    class_metrics: dict[str, dict[str, float]] = {}
    f1_values = []
    weighted_f1_values = []
    precision_values = []
    recall_values = []

    for idx, label in enumerate(labels):
        tp = float(np.sum((y_true_arr == idx) & (y_pred_arr == idx)))
        fp = float(np.sum((y_true_arr != idx) & (y_pred_arr == idx)))
        fn = float(np.sum((y_true_arr == idx) & (y_pred_arr != idx)))
        support = float(np.sum(y_true_arr == idx))
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        class_metrics[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
        }
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        weighted_f1_values.append(f1 * support)

    total = float(len(y_true_arr))
    accuracy = _safe_divide(float(np.sum(y_true_arr == y_pred_arr)), total)
    return {
        "accuracy": accuracy,
        "precision_macro": float(np.mean(precision_values)) if precision_values else 0.0,
        "recall_macro": float(np.mean(recall_values)) if recall_values else 0.0,
        "f1_macro": float(np.mean(f1_values)) if f1_values else 0.0,
        "f1_weighted": _safe_divide(float(np.sum(weighted_f1_values)), total),
        "report": class_metrics,
    }


def write_metrics(metrics: dict[str, object], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
