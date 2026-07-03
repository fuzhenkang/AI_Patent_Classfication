"""Split cleaned patent text data into stratified train/test CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split data into stratified train/test sets.")
    parser.add_argument("--input", required=True, help="Cleaned CSV file.")
    parser.add_argument("--output-dir", required=True, help="Directory for train.csv and test.csv.")
    parser.add_argument("--label-col", default="label", help="Label column name.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training set ratio.")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV encoding.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def split_train_test(args: argparse.Namespace) -> tuple[Path, Path]:
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1.")

    df = pd.read_csv(args.input, encoding=args.encoding)
    if args.label_col not in df.columns:
        raise ValueError(f"Missing label column: {args.label_col}")

    df = df.dropna(subset=[args.label_col]).reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    train_parts = []
    test_parts = []

    for _, group in df.groupby(args.label_col, sort=False):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        train_size = int(round(len(indices) * args.train_ratio))
        train_size = min(max(train_size, 1), len(indices) - 1) if len(indices) > 1 else len(indices)
        train_parts.append(df.loc[indices[:train_size]])
        test_parts.append(df.loc[indices[train_size:]])

    train_df = pd.concat(train_parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {"split": "train", "rows": len(train_df), "ratio": len(train_df) / len(df)},
            {"split": "test", "rows": len(test_df), "ratio": len(test_df) / len(df)},
        ]
    )
    summary.to_csv(output_dir / "train_test_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    return train_path, test_path


def main() -> int:
    split_train_test(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
