"""Split cleaned patent text data into stratified train/valid/test CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split data into stratified train/valid/test sets.")
    parser.add_argument("--input", required=True, help="Cleaned CSV file.")
    parser.add_argument("--output-dir", required=True, help="Directory for train.csv, valid.csv, and test.csv.")
    parser.add_argument("--label-col", default="label", help="Label column name.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training set ratio.")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Validation set ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio.")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV encoding.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def split_train_valid_test(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    ratios = [args.train_ratio, args.valid_ratio, args.test_ratio]
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("--train-ratio, --valid-ratio, and --test-ratio must all be positive.")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must sum to 1.0.")

    df = pd.read_csv(args.input, encoding=args.encoding)
    if args.label_col not in df.columns:
        raise ValueError(f"Missing label column: {args.label_col}")

    df = df.dropna(subset=[args.label_col]).reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    train_parts = []
    valid_parts = []
    test_parts = []

    for _, group in df.groupby(args.label_col, sort=False):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        n_items = len(indices)
        train_size = int(round(n_items * args.train_ratio))
        valid_size = int(round(n_items * args.valid_ratio))

        if n_items >= 3:
            train_size = min(max(train_size, 1), n_items - 2)
            valid_size = min(max(valid_size, 1), n_items - train_size - 1)
        elif n_items == 2:
            train_size = 1
            valid_size = 0
        else:
            train_size = 1
            valid_size = 0

        valid_end = train_size + valid_size
        train_parts.append(df.loc[indices[:train_size]])
        if valid_size:
            valid_parts.append(df.loc[indices[train_size:valid_end]])
        if valid_end < n_items:
            test_parts.append(df.loc[indices[valid_end:]])

    train_df = pd.concat(train_parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    valid_df = pd.concat(valid_parts).sample(frac=1, random_state=args.seed).reset_index(drop=True) if valid_parts else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=args.seed).reset_index(drop=True) if test_parts else pd.DataFrame(columns=df.columns)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    valid_path = output_dir / "valid.csv"
    test_path = output_dir / "test.csv"
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    valid_df.to_csv(valid_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {"split": "train", "rows": len(train_df), "ratio": len(train_df) / len(df)},
            {"split": "valid", "rows": len(valid_df), "ratio": len(valid_df) / len(df)},
            {"split": "test", "rows": len(test_df), "ratio": len(test_df) / len(df)},
        ]
    )
    summary.to_csv(output_dir / "split_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    return train_path, valid_path, test_path


def main() -> int:
    split_train_valid_test(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
