"""Create stratified k-fold assignments for cleaned patent text data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified k-fold assignments.")
    parser.add_argument("--input", required=True, help="Cleaned CSV file.")
    parser.add_argument("--output", required=True, help="CSV file with cv_fold column.")
    parser.add_argument("--label-col", default="label", help="Label column name.")
    parser.add_argument("--fold-col", default="cv_fold", help="Output fold column name.")
    parser.add_argument("--n-splits", type=int, default=10, help="Number of cross-validation folds.")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV encoding.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def stratified_kfold_indices(labels, n_splits: int = 10, seed: int = 42) -> list[np.ndarray]:
    labels_arr = np.array(list(labels), dtype=object)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    rng = np.random.default_rng(seed)

    for label in sorted(set(labels_arr), key=lambda value: str(value)):
        label_indices = np.where(labels_arr == label)[0].copy()
        rng.shuffle(label_indices)
        for position, index in enumerate(label_indices):
            folds[position % n_splits].append(int(index))

    return [np.array(sorted(fold), dtype=np.int64) for fold in folds]


def create_cv_folds(args: argparse.Namespace) -> Path:
    df = pd.read_csv(args.input, encoding=args.encoding)
    if args.label_col not in df.columns:
        raise ValueError(f"Missing label column: {args.label_col}")

    df = df.dropna(subset=[args.label_col]).reset_index(drop=True)
    df[args.fold_col] = -1
    folds = stratified_kfold_indices(df[args.label_col], n_splits=args.n_splits, seed=args.seed)
    for fold_id, valid_indices in enumerate(folds):
        df.loc[valid_indices, args.fold_col] = fold_id

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = df.groupby([args.fold_col, args.label_col]).size().reset_index(name="rows")
    print(summary.to_string(index=False))
    return output_path


def main() -> int:
    create_cv_folds(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
