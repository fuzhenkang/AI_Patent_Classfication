"""Fine-tune AutoModelForSequenceClassification with LoRA for patent text classification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from LLM.llm_registry import MODEL_CONFIGS, get_llm_config  # noqa: E402
from Models.common import (  # noqa: E402
    average_metrics,
    classification_metrics,
    fit_label_encoder,
    get_device,
    read_text_label_csv,
    save_label_encoder,
    set_seed,
    stratified_kfold_indices,
    write_metrics,
)


class SequenceClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int):
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
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AutoModelForSequenceClassification with LoRA.")
    parser.add_argument("--model-key", default="chinese_roberta", choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--data-csv", help="Training CSV with optional cv_fold column for k-fold CV.")
    parser.add_argument("--train-csv", help="Training CSV for final training.")
    parser.add_argument("--valid-csv", help="Optional explicit validation CSV.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--fold-col", default="cv_fold")
    parser.add_argument("--cv-folds", type=int, default=10)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora-target-modules",
        default=None,
        help="Comma-separated LoRA target modules. For LLaMA/Qwen-like models try q_proj,v_proj.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--torch-dtype", default=None, choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    return apply_model_defaults(args)


def apply_model_defaults(args: argparse.Namespace) -> argparse.Namespace:
    config = get_llm_config(args.model_key)
    if args.base_model is None:
        args.base_model = config.base_model
    if args.lora_target_modules is None:
        args.lora_target_modules = config.lora_target_modules
    if args.max_len is None:
        args.max_len = config.max_len
    if args.batch_size is None:
        args.batch_size = config.batch_size
    if args.lr is None:
        args.lr = config.lr
    if args.torch_dtype is None:
        args.torch_dtype = config.torch_dtype
    args.trust_remote_code = bool(args.trust_remote_code or config.trust_remote_code)
    return args


def parse_target_modules(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_model(args: argparse.Namespace, label_names: list[str]):
    from peft import LoraConfig, TaskType, get_peft_model

    id2label = {idx: str(label) for idx, label in enumerate(label_names)}
    label2id = {str(label): idx for idx, label in enumerate(label_names)}
    model_kwargs = {
        "num_labels": len(label_names),
        "id2label": id2label,
        "label2id": label2id,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.torch_dtype != "auto":
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        model_kwargs["torch_dtype"] = dtype_map[args.torch_dtype]
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        **model_kwargs,
    )
    if getattr(model.config, "pad_token_id", None) is None and getattr(model.config, "eos_token_id", None) is not None:
        model.config.pad_token_id = model.config.eos_token_id
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=parse_target_modules(args.lora_target_modules),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def evaluate(model, loader: DataLoader, device: torch.device, label_names: list[str]) -> dict[str, object]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
    return classification_metrics(y_true, y_pred, label_names)


def save_run_artifacts(args: argparse.Namespace, output_dir: Path, tokenizer, encoder, model, extra_config: dict[str, object]) -> None:
    tokenizer.save_pretrained(output_dir / "tokenizer")
    model.save_pretrained(output_dir / "adapter")
    save_label_encoder(encoder, output_dir)
    config = vars(args).copy()
    config.update(extra_config)
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)


def train_once(args: argparse.Namespace, train_df, valid_df, output_dir: Path, seed: int) -> dict[str, object]:
    set_seed(seed)
    device = get_device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = fit_label_encoder(train_df[args.label_col], valid_df[args.label_col])
    y_train = encoder.transform(train_df[args.label_col])
    y_valid = encoder.transform(valid_df[args.label_col])
    label_names = [str(label) for label in encoder.classes_]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader = DataLoader(
        SequenceClassificationDataset(train_df[args.text_col], y_train, tokenizer, args.max_len),
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        SequenceClassificationDataset(valid_df[args.text_col], y_valid, tokenizer, args.max_len),
        batch_size=args.batch_size,
    )

    model = build_model(args, label_names).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_f1 = -1.0
    best_metrics: dict[str, object] = {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())

        metrics = evaluate(model, valid_loader, device, label_names)
        print(f"epoch={epoch} loss={total_loss / max(1, len(train_loader)):.4f} valid_f1_macro={metrics['f1_macro']:.4f}")
        if metrics["f1_macro"] > best_f1:
            best_f1 = float(metrics["f1_macro"])
            best_metrics = metrics
            save_run_artifacts(
                args,
                output_dir,
                tokenizer,
                encoder,
                model,
                {"model_type": "llm_lora_sequence_classification", "num_classes": len(encoder.classes_)},
            )

    write_metrics(best_metrics, output_dir / "valid_metrics.json")
    return best_metrics


def train_final(args: argparse.Namespace, train_df, output_dir: Path, seed: int) -> dict[str, object]:
    set_seed(seed)
    device = get_device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = fit_label_encoder(train_df[args.label_col])
    y_train = encoder.transform(train_df[args.label_col])
    label_names = [str(label) for label in encoder.classes_]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_loader = DataLoader(
        SequenceClassificationDataset(train_df[args.text_col], y_train, tokenizer, args.max_len),
        batch_size=args.batch_size,
        shuffle=True,
    )
    model = build_model(args, label_names).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())
        print(f"epoch={epoch} train_loss={total_loss / max(1, len(train_loader)):.4f}")

    save_run_artifacts(
        args,
        output_dir,
        tokenizer,
        encoder,
        model,
        {"model_type": "llm_lora_sequence_classification", "num_classes": len(encoder.classes_), "training_mode": "final_train"},
    )
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
        json.dump(vars(args) | {"model_type": "llm_lora_sequence_classification", "cv_folds_actual": len(folds)}, file, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in cv_metrics.items() if k != "fold_metrics"}, ensure_ascii=False, indent=2))
    return cv_metrics


def train(args: argparse.Namespace) -> dict[str, object]:
    args = apply_model_defaults(args)
    if args.data_csv:
        return cross_validate(args)
    if not args.train_csv:
        raise ValueError("Use --data-csv for k-fold cross-validation, or provide --train-csv for final training.")
    train_df = read_text_label_csv(args.train_csv, args.text_col, args.label_col, args.encoding)
    if args.valid_csv:
        valid_df = read_text_label_csv(args.valid_csv, args.text_col, args.label_col, args.encoding)
        return train_once(args, train_df, valid_df, Path(args.output_dir), args.seed)
    return train_final(args, train_df, Path(args.output_dir), args.seed)


def main() -> int:
    train(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
