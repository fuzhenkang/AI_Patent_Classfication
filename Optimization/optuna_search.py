"""Run Optuna hyperparameter search for patent text classification models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from LLM.llm_classifier import train as train_llm_lora  # noqa: E402
from LLM.llm_registry import MODEL_CONFIGS  # noqa: E402
from Models.bert_cnn import train as train_bert_cnn  # noqa: E402
from Models.word2vec_cnn import train as train_word2vec_cnn  # noqa: E402
from Models.word2vec_textcnn import train as train_word2vec_textcnn  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search.")
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["word2vec_cnn", "word2vec_textcnn", "bert_cnn", "llm_lora", "llama", "qwen", "glm", "mistral", "baichuan"],
    )
    parser.add_argument("--data-csv", required=True, help="Full cleaned CSV for stratified k-fold cross-validation.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--fold-col", default="cv_fold")
    parser.add_argument("--cv-folds", type=int, default=10)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None, help="Optuna timeout in seconds.")
    parser.add_argument("--metric", default="f1_macro", help="Validation metric to maximize.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--bert-model", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--llm-model-key", default="qwen", choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--llm-base-model", default=None)
    parser.add_argument("--lora-target-modules", default=None)
    return parser.parse_args()


def suggest_word2vec_common(trial: optuna.Trial, args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    return {
        "data_csv": args.data_csv,
        "train_csv": None,
        "valid_csv": None,
        "output_dir": str(output_dir),
        "text_col": args.text_col,
        "label_col": args.label_col,
        "fold_col": args.fold_col,
        "cv_folds": args.cv_folds,
        "encoding": args.encoding,
        "max_len": trial.suggest_categorical("max_len", [128, 256, 384]),
        "min_freq": trial.suggest_categorical("min_freq", [1, 2, 3]),
        "max_vocab_size": trial.suggest_categorical("max_vocab_size", [30000, 50000, 80000]),
        "embedding_dim": trial.suggest_categorical("embedding_dim", [100, 200, 300]),
        "window": trial.suggest_categorical("window", [3, 5, 7]),
        "word2vec_epochs": trial.suggest_int("word2vec_epochs", 5, 20),
        "num_filters": trial.suggest_categorical("num_filters", [64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.2, 0.6),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "epochs": trial.suggest_int("epochs", 5, 20),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "seed": args.seed,
        "device": args.device,
    }


def build_trial_args(trial: optuna.Trial, args: argparse.Namespace, output_dir: Path) -> SimpleNamespace:
    if args.model_type == "word2vec_cnn":
        params = suggest_word2vec_common(trial, args, output_dir)
        params["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 5, 7])
        return SimpleNamespace(**params)

    if args.model_type == "word2vec_textcnn":
        params = suggest_word2vec_common(trial, args, output_dir)
        params["kernel_sizes"] = trial.suggest_categorical("kernel_sizes", ["2,3,4", "3,4,5", "3,5,7"])
        return SimpleNamespace(**params)

    if args.model_type == "bert_cnn":
        return SimpleNamespace(
            data_csv=args.data_csv,
            train_csv=None,
            valid_csv=None,
            output_dir=str(output_dir),
            text_col=args.text_col,
            label_col=args.label_col,
            fold_col=args.fold_col,
            cv_folds=args.cv_folds,
            encoding=args.encoding,
            bert_model=args.bert_model,
            max_len=trial.suggest_categorical("max_len", [128, 256]),
            num_filters=trial.suggest_categorical("num_filters", [64, 128, 256]),
            kernel_sizes=trial.suggest_categorical("kernel_sizes", ["2,3,4", "3,4,5", "3,5,7"]),
            dropout=trial.suggest_float("dropout", 0.1, 0.5),
            batch_size=trial.suggest_categorical("batch_size", [8, 16, 32]),
            epochs=trial.suggest_int("epochs", 2, 5),
            lr=trial.suggest_float("lr", 1e-5, 5e-5, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            warmup_ratio=trial.suggest_float("warmup_ratio", 0.0, 0.2),
            seed=args.seed,
            device=args.device,
        )

    if args.model_type in {"llm_lora", "llama", "qwen", "glm", "mistral", "baichuan"}:
        model_key = args.llm_model_key if args.model_type == "llm_lora" else args.model_type
        return SimpleNamespace(
            model_key=model_key,
            data_csv=args.data_csv,
            train_csv=None,
            valid_csv=None,
            output_dir=str(output_dir),
            text_col=args.text_col,
            label_col=args.label_col,
            fold_col=args.fold_col,
            cv_folds=args.cv_folds,
            encoding=args.encoding,
            base_model=args.llm_base_model,
            max_len=trial.suggest_categorical("max_len", [128, 256]),
            batch_size=trial.suggest_categorical("batch_size", [2, 4, 8]),
            epochs=trial.suggest_int("epochs", 2, 5),
            lr=trial.suggest_float("lr", 1e-5, 5e-5, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            warmup_ratio=trial.suggest_float("warmup_ratio", 0.0, 0.2),
            lora_r=trial.suggest_categorical("lora_r", [4, 8, 16]),
            lora_alpha=trial.suggest_categorical("lora_alpha", [8, 16, 32]),
            lora_dropout=trial.suggest_float("lora_dropout", 0.0, 0.2),
            lora_target_modules=args.lora_target_modules,
            trust_remote_code=False,
            torch_dtype=None,
            seed=args.seed,
            device=args.device,
        )

    raise ValueError(f"Unsupported model type: {args.model_type}")


def train_one_model(model_type: str, trial_args: SimpleNamespace) -> dict[str, object]:
    if model_type == "word2vec_cnn":
        return train_word2vec_cnn(trial_args)
    if model_type == "word2vec_textcnn":
        return train_word2vec_textcnn(trial_args)
    if model_type == "bert_cnn":
        return train_bert_cnn(trial_args)
    if model_type in {"llm_lora", "llama", "qwen", "glm", "mistral", "baichuan"}:
        return train_llm_lora(trial_args)
    raise ValueError(f"Unsupported model type: {model_type}")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        trial_dir = output_dir / f"trial_{trial.number:04d}"
        trial_args = build_trial_args(trial, args, trial_dir)
        metrics = train_one_model(args.model_type, trial_args)
        value = float(metrics[args.metric])
        trial.set_user_attr("output_dir", str(trial_dir))
        return value

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / "optuna_trials.csv", index=False, encoding="utf-8-sig")
    best = {
        "model_type": args.model_type,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
        "best_trial_output_dir": study.best_trial.user_attrs.get("output_dir"),
    }
    with (output_dir / "best_params.json").open("w", encoding="utf-8") as file:
        json.dump(best, file, ensure_ascii=False, indent=2)
    print(json.dumps(best, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

