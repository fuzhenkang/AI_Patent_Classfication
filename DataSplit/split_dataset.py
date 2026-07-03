"""Split cleaned patent text data into train/validation/test CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split dataset into 8:1:1 train/valid/test sets.")
    parser.add_argument("--input", required=True, help="Cleaned CSV file.")
    parser.add_argument("--output-dir", required=True, help="Directory for train.csv, valid.csv, test.csv.")
    parser.add_argument("--text-col", default="text", help="Text column name.")
    parser.add_argument("--label-col", default="label", help="Label column name.")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV encoding.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no-stratify", action="store_true", help="Disable stratified split by label.")
    return parser.parse_args()


def split_dataset(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    df = pd.read_csv(args.input, encoding=args.encoding)
    missing = [col for col in [args.text_col, args.label_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = df.dropna(subset=[args.text_col, args.label_col]).reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    if args.no_stratify:
        shuffled = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        train_end = int(len(shuffled) * 0.8)
        valid_end = int(len(shuffled) * 0.9)
        train_df = shuffled.iloc[:train_end]
        valid_df = shuffled.iloc[train_end:valid_end]
        test_df = shuffled.iloc[valid_end:]
    else:
        train_parts = []
        valid_parts = []
        test_parts = []
        for _, group in df.groupby(args.label_col, sort=False):
            indices = group.index.to_numpy().copy()
            rng.shuffle(indices)
            train_end = int(len(indices) * 0.8)
            valid_end = int(len(indices) * 0.9)
            train_parts.append(df.loc[indices[:train_end]])
            valid_parts.append(df.loc[indices[train_end:valid_end]])
            test_parts.append(df.loc[indices[valid_end:]])

        train_df = pd.concat(train_parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)
        valid_df = pd.concat(valid_parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)
        test_df = pd.concat(test_parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)

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
    args = parse_args()
    split_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
