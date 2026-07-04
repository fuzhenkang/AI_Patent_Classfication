"""Evaluate a trained patent text classification model on a held-out CSV file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Models.common import (  # noqa: E402
    BertCnnDataset,
    TextDataset,
    Word2VecCNN,
    Word2VecTextCNN,
    classification_metrics,
    get_device,
    load_label_encoder,
    load_vocab,
    read_text_label_csv,
    write_metrics,
)


def parse_kernel_sizes(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier.")
    parser.add_argument("--model-dir", required=True, help="Directory containing best_model.pt and config.json.")
    parser.add_argument("--test-csv", required=True, help="Test CSV file.")
    parser.add_argument("--output-dir", required=True, help="Directory for metrics and predictions.")
    parser.add_argument("--text-col", default=None, help="Override text column name.")
    parser.add_argument("--label-col", default=None, help="Override label column name.")
    parser.add_argument("--encoding", default=None, help="Override CSV encoding.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_config(model_dir: Path) -> dict[str, object]:
    with (model_dir / "config.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_word2vec(config: dict[str, object], model_dir: Path, test_df: pd.DataFrame, labels: np.ndarray, output_dir: Path, device: torch.device, batch_size: int) -> dict[str, object]:
    vocab = load_vocab(model_dir)
    label_encoder = load_label_encoder(model_dir)
    embedding_dim = int(config["embedding_dim"])
    embedding_matrix = np.zeros((len(vocab), embedding_dim), dtype="float32")
    model_type = str(config["model_type"])

    if model_type == "word2vec_cnn":
        model = Word2VecCNN(
            embedding_matrix,
            len(label_encoder.classes_),
            int(config["num_filters"]),
            int(config["kernel_size"]),
            float(config["dropout"]),
        )
    elif model_type == "word2vec_textcnn":
        model = Word2VecTextCNN(
            embedding_matrix,
            len(label_encoder.classes_),
            int(config["num_filters"]),
            parse_kernel_sizes(str(config["kernel_sizes"])),
            float(config["dropout"]),
        )
    else:
        raise ValueError(f"Unsupported Word2Vec model type: {model_type}")

    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    model.to(device)
    model.eval()

    dataset = TextDataset(test_df[str(config["text_col"])], labels, vocab, int(config["max_len"]))
    loader = DataLoader(dataset, batch_size=batch_size)
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for input_ids, batch_labels in loader:
            logits = model(input_ids.to(device))
            y_true.extend(batch_labels.numpy().tolist())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

    metrics = classification_metrics(y_true, y_pred, list(label_encoder.classes_))
    predictions = test_df.copy()
    predictions["pred_label"] = label_encoder.inverse_transform(y_pred)
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    return metrics


def evaluate_bert(config: dict[str, object], model_dir: Path, test_df: pd.DataFrame, labels: np.ndarray, output_dir: Path, device: torch.device, batch_size: int) -> dict[str, object]:
    from transformers import AutoTokenizer

    from Models.bert_cnn import BertCNN

    label_encoder = load_label_encoder(model_dir)
    tokenizer_path = model_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path if tokenizer_path.exists() else str(config["bert_model"]))
    model = BertCNN(
        str(config["bert_model"]),
        len(label_encoder.classes_),
        int(config["num_filters"]),
        parse_kernel_sizes(str(config["kernel_sizes"])),
        float(config["dropout"]),
    )
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    model.to(device)
    model.eval()

    dataset = BertCnnDataset(test_df[str(config["text_col"])], labels, tokenizer, int(config["max_len"]))
    loader = DataLoader(dataset, batch_size=batch_size)
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            y_true.extend(batch["labels"].numpy().tolist())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

    metrics = classification_metrics(y_true, y_pred, list(label_encoder.classes_))
    predictions = test_df.copy()
    predictions["pred_label"] = label_encoder.inverse_transform(y_pred)
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    return metrics


def evaluate_lora_sequence_classifier(config: dict[str, object], model_dir: Path, test_df: pd.DataFrame, labels: np.ndarray, output_dir: Path, device: torch.device, batch_size: int) -> dict[str, object]:
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from LLM.llm_lora_classifier import SequenceClassificationDataset

    label_encoder = load_label_encoder(model_dir)
    tokenizer_path = model_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path if tokenizer_path.exists() else str(config["base_model"]))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    label_names = [str(label) for label in label_encoder.classes_]
    id2label = {idx: label for idx, label in enumerate(label_names)}
    label2id = {label: idx for idx, label in enumerate(label_names)}
    model_kwargs = {
        "num_labels": len(label_names),
        "id2label": id2label,
        "label2id": label2id,
        "trust_remote_code": bool(config.get("trust_remote_code", False)),
    }
    if config.get("load_in_4bit") or config.get("load_in_8bit"):
        from transformers import BitsAndBytesConfig

        compute_dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if config.get("load_in_4bit"):
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=str(config.get("bnb_4bit_quant_type", "nf4")),
                bnb_4bit_compute_dtype=compute_dtype_map[str(config.get("bnb_4bit_compute_dtype", "float16"))],
                bnb_4bit_use_double_quant=bool(config.get("bnb_4bit_use_double_quant", False)),
            )
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["device_map"] = "auto"

    base_model = AutoModelForSequenceClassification.from_pretrained(
        str(config["base_model"]),
        **model_kwargs,
    )
    if getattr(base_model.config, "pad_token_id", None) is None and getattr(base_model.config, "eos_token_id", None) is not None:
        base_model.config.pad_token_id = base_model.config.eos_token_id
    model = PeftModel.from_pretrained(base_model, model_dir / "adapter")
    model.to(device)
    model.eval()

    dataset = SequenceClassificationDataset(test_df[str(config["text_col"])], labels, tokenizer, int(config["max_len"]))
    loader = DataLoader(dataset, batch_size=batch_size)
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for batch in loader:
            batch_labels = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            y_true.extend(batch_labels.numpy().tolist())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

    metrics = classification_metrics(y_true, y_pred, label_names)
    predictions = test_df.copy()
    predictions["pred_label"] = label_encoder.inverse_transform(y_pred)
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    return metrics


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(model_dir)

    text_col = args.text_col or str(config["text_col"])
    label_col = args.label_col or str(config["label_col"])
    encoding = args.encoding or str(config.get("encoding", "utf-8-sig"))
    batch_size = args.batch_size or int(config.get("batch_size", 32))
    device = get_device(args.device)

    test_df = read_text_label_csv(args.test_csv, text_col, label_col, encoding)
    label_encoder = load_label_encoder(model_dir)
    labels = label_encoder.transform(test_df[label_col])
    config["text_col"] = text_col
    config["label_col"] = label_col

    if str(config["model_type"]) in {"word2vec_cnn", "word2vec_textcnn"}:
        metrics = evaluate_word2vec(config, model_dir, test_df, labels, output_dir, device, batch_size)
    elif str(config["model_type"]) == "bert_cnn":
        metrics = evaluate_bert(config, model_dir, test_df, labels, output_dir, device, batch_size)
    elif str(config["model_type"]) == "llm_lora_sequence_classification":
        metrics = evaluate_lora_sequence_classifier(config, model_dir, test_df, labels, output_dir, device, batch_size)
    else:
        raise ValueError(f"Unsupported model type: {config['model_type']}")

    write_metrics(metrics, output_dir / "test_metrics.json")
    print(json.dumps({k: v for k, v in metrics.items() if k != "report"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
