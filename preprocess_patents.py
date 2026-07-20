"""Preprocess patent title and abstract text for AI patent classification."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = ["PN", "title", "abstract"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HIT_STOPWORDS_PATH = PROJECT_ROOT / "stopwords" / "hit_stopwords.txt"
DEFAULT_USER_STOPWORDS_PATH = PROJECT_ROOT / "stopwords" / "user_stopwords.txt"
DEFAULT_PATENT_PHRASES_PATH = PROJECT_ROOT / "stopwords" / "patent_phrases.txt"


FALLBACK_STOPWORDS = {
    "\u7684",
    "\u4e86",
    "\u548c",
    "\u4e0e",
    "\u53ca",
    "\u6216",
    "\u5e76",
    "\u4e14",
    "\u5728",
    "\u5bf9",
    "\u5c06",
    "\u4e3a",
    "\u662f",
    "\u7531",
    "\u4e8e",
    "\u4e2d",
    "\u4e0a",
    "\u4e0b",
    "\u5185",
    "\u5916",
    "\u8be5",
    "\u5176",
    "\u8fd9",
    "\u6b64",
    "\u4ece",
    "\u5230",
    "\u4ee5",
    "\u88ab",
    "\u628a",
    "\u800c",
    "\u4f46",
    "\u7b49",
}


INLINE_PARTICLE_STOPWORDS = {
    "\u7684",
    "\u4e86",
    "\u8be5",
    "\u5176",
    "\u6b64",
}


FALLBACK_PATENT_PHRASES = {
    "\u672c\u53d1\u660e",
    "\u672c\u5b9e\u7528\u65b0\u578b",
    "\u672c\u5916\u89c2\u8bbe\u8ba1",
    "\u672c\u7533\u8bf7",
    "\u672c\u516c\u5f00",
    "\u672c\u62ab\u9732",
    "\u672c\u5b9e\u65bd\u4f8b",
    "\u5b9e\u65bd\u4f8b",
    "\u5177\u4f53\u5b9e\u65bd\u65b9\u5f0f",
    "\u6280\u672f\u9886\u57df",
    "\u80cc\u666f\u6280\u672f",
    "\u53d1\u660e\u5185\u5bb9",
    "\u6743\u5229\u8981\u6c42",
    "\u8bf4\u660e\u4e66",
    "\u9644\u56fe\u8bf4\u660e",
    "\u4f18\u9009\u5730",
    "\u8fdb\u4e00\u6b65\u5730",
    "\u8fdb\u4e00\u6b65\u7684",
    "\u5177\u4f53\u5730",
    "\u5177\u4f53\u7684",
    "\u5176\u4e2d",
    "\u4e00\u79cd",
    "\u4e00\u7c7b",
    "\u4e00\u4e2a",
    "\u591a\u4e2a",
    "\u82e5\u5e72",
    "\u6240\u8ff0",
    "\u4e0a\u8ff0",
    "\u4e0b\u8ff0",
    "\u524d\u8ff0",
    "\u8be5\u65b9\u6cd5",
    "\u8be5\u88c5\u7f6e",
    "\u8be5\u7cfb\u7edf",
    "\u8be5\u8bbe\u5907",
}


REPLACEMENTS = str.maketrans(
    {
        "\u3000": " ",
        "\xa0": " ",
        "\t": " ",
        "\r": " ",
        "\n": " ",
        "\uff0c": " ",
        "\u3002": " ",
        "\u3001": " ",
        "\uff1b": " ",
        "\uff1a": " ",
        "\uff01": " ",
        "\uff1f": " ",
        "\uff08": " ",
        "\uff09": " ",
        "\u3010": " ",
        "\u3011": " ",
        "\u300a": " ",
        "\u300b": " ",
        "\u201c": " ",
        "\u201d": " ",
        "\u2018": " ",
        "\u2019": " ",
    }
)


HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
LATEX_RE = re.compile(r"\\[a-zA-Z]+(?:\{[^{}]*\})*")
FORMULA_RE = re.compile(
    r"(?<![A-Za-z0-9\u4e00-\u9fff])"
    r"(?:[A-Za-z]\s*[=<>]\s*[-+]?\d+(?:\.\d+)?|"
    r"[-+]?\d+(?:\.\d+)?\s*[+\-*/^=<>]\s*[-+]?\d+(?:\.\d+)?)"
    r"(?![A-Za-z0-9\u4e00-\u9fff])"
)
UNRECOGNIZED_SYMBOL_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9\s]")
SPACE_RE = re.compile(r"\s+")


def get_pandas():
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pandas is required to run preprocessing. Install dependencies with: "
            "pip install pandas"
        ) from exc
    return pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean patent title/abstract CSV files and export text length statistics."
    )
    parser.add_argument("--input", required=True, help="Path to source CSV file.")
    parser.add_argument("--output", required=True, help="Path to cleaned CSV file.")
    parser.add_argument("--stats-output", required=True, help="Path to descriptive statistics CSV.")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV encoding.")
    parser.add_argument("--no-header", action="store_true", help="Treat first three columns as PN,title,abstract.")
    parser.add_argument("--hit-stopwords-file", default=str(DEFAULT_HIT_STOPWORDS_PATH))
    parser.add_argument("--user-stopwords-file", default=str(DEFAULT_USER_STOPWORDS_PATH))
    parser.add_argument("--stopwords-file", action="append", default=[])
    parser.add_argument("--patent-phrases-file", default=str(DEFAULT_PATENT_PHRASES_PATH))
    parser.add_argument("--no-default-stopwords", action="store_true")
    parser.add_argument("--keep-original-columns", action="store_true")
    return parser.parse_args()


def load_terms(path: str | Path | None, *, required: bool = False) -> set[str]:
    if not path:
        return set()

    term_path = Path(path)
    if not term_path.exists():
        if required:
            raise FileNotFoundError(f"Dictionary file not found: {term_path}")
        return set()

    with term_path.open("r", encoding="utf-8-sig") as file:
        return {
            line.strip()
            for line in file
            if line.strip() and not line.lstrip().startswith("#")
        }


def load_stopwords(args: argparse.Namespace) -> set[str]:
    if args.no_default_stopwords:
        stopwords: set[str] = set()
    else:
        stopwords = set(FALLBACK_STOPWORDS)
        stopwords |= load_terms(args.hit_stopwords_file)
        stopwords |= load_terms(args.user_stopwords_file)

    for path in args.stopwords_file:
        stopwords |= load_terms(path, required=True)
    return stopwords


def load_patent_phrases(args: argparse.Namespace) -> set[str]:
    phrases = set(FALLBACK_PATENT_PHRASES)
    phrases |= load_terms(args.patent_phrases_file)
    if not args.no_default_stopwords:
        phrases |= load_terms(args.user_stopwords_file)
    return phrases


def read_patent_csv(path: str, encoding: str, no_header: bool):
    pd = get_pandas()
    read_kwargs = {"encoding": encoding}
    if no_header:
        read_kwargs["header"] = None

    try:
        df = pd.read_csv(path, **read_kwargs)
    except UnicodeDecodeError:
        if encoding.lower().replace("-", "") == "utf8sig":
            df = pd.read_csv(path, encoding="gb18030", header=None if no_header else "infer")
        else:
            raise

    if no_header:
        if df.shape[1] < 3:
            raise ValueError("CSV without header must contain at least three columns.")
        df = df.rename(columns={0: "PN", 1: "title", 2: "abstract"})

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return df


def remove_terms(text: str, terms: Iterable[str]) -> str:
    cleaned = text
    for term in sorted(terms, key=len, reverse=True):
        if term:
            cleaned = cleaned.replace(term, " ")
    return cleaned


def remove_stopwords(text: str, stopwords: set[str]) -> str:
    cleaned_tokens = []
    for token in text.split():
        if token in stopwords:
            continue
        for word in INLINE_PARTICLE_STOPWORDS & stopwords:
            token = token.replace(word, "")
        if token:
            cleaned_tokens.append(token)
    return " ".join(cleaned_tokens)


def is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def clean_text(value: object, stopwords: set[str], patent_phrases: set[str]) -> str:
    if is_missing(value):
        return ""

    text = str(value)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = CONTROL_RE.sub(" ", text)
    text = LATEX_RE.sub(" ", text)
    text = FORMULA_RE.sub(" ", text)
    text = text.translate(REPLACEMENTS)
    text = UNRECOGNIZED_SYMBOL_RE.sub(" ", text)
    text = remove_terms(text, patent_phrases)
    text = SPACE_RE.sub(" ", text).strip()
    text = remove_stopwords(text, stopwords)
    return SPACE_RE.sub(" ", text).strip()


def build_length_stats(length_series):
    stats = length_series.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    stats = stats.rename(
        index={
            "count": "\u6837\u672c\u6570",
            "mean": "\u5747\u503c",
            "std": "\u6807\u51c6\u5dee",
            "min": "\u6700\u5c0f\u503c",
            "25%": "25\u5206\u4f4d\u6570",
            "50%": "\u4e2d\u4f4d\u6570",
            "75%": "75\u5206\u4f4d\u6570",
            "90%": "90\u5206\u4f4d\u6570",
            "95%": "95\u5206\u4f4d\u6570",
            "99%": "99\u5206\u4f4d\u6570",
            "max": "\u6700\u5927\u503c",
        }
    )
    return stats.reset_index().rename(
        columns={"index": "\u7edf\u8ba1\u91cf", length_series.name: "\u6587\u672c\u957f\u5ea6"}
    )


def preprocess(args: argparse.Namespace):
    stopwords = load_stopwords(args)
    patent_phrases = load_patent_phrases(args)

    df = read_patent_csv(args.input, args.encoding, args.no_header)
    result = df.copy() if args.keep_original_columns else df[REQUIRED_COLUMNS].copy()

    result["clean_title"] = df["title"].map(lambda value: clean_text(value, stopwords, patent_phrases))
    result["clean_abstract"] = df["abstract"].map(lambda value: clean_text(value, stopwords, patent_phrases))
    result["text"] = (result["clean_title"].fillna("") + " " + result["clean_abstract"].fillna("")).map(
        lambda value: SPACE_RE.sub(" ", value).strip()
    )
    result["text_len_chars"] = result["text"].str.len()
    result["text_len_no_space"] = result["text"].str.replace(r"\s+", "", regex=True).str.len()

    stats = build_length_stats(result["text_len_no_space"])
    output_path = Path(args.output)
    stats_path = Path(args.stats_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    stats.to_csv(stats_path, index=False, encoding="utf-8-sig")
    return output_path, stats_path, stats


def main() -> int:
    args = parse_args()
    try:
        output_path, stats_path, stats = preprocess(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Cleaned CSV written to: {output_path}")
    print(f"Length statistics written to: {stats_path}")
    print(stats.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
