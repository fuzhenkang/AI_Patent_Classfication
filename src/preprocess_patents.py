"""Preprocess patent title and abstract text for AI patent classification."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = ["PN", "title", "abstract"]


DEFAULT_STOPWORDS = {
    "的",
    "了",
    "和",
    "与",
    "及",
    "或",
    "并",
    "且",
    "在",
    "对",
    "将",
    "为",
    "是",
    "由",
    "于",
    "中",
    "上",
    "下",
    "内",
    "外",
    "该",
    "其",
    "它",
    "他",
    "她",
    "这",
    "此",
    "从",
    "到",
    "以",
    "被",
    "把",
    "而",
    "但",
    "等",
    "进行",
    "通过",
    "根据",
    "用于",
    "包括",
    "包含",
    "具有",
    "得到",
    "获得",
    "实现",
    "提供",
}


INLINE_PARTICLE_STOPWORDS = {
    "的",
    "了",
    "该",
    "其",
    "此",
}


DEFAULT_PATENT_PHRASES = {
    "本发明",
    "本实用新型",
    "本外观设计",
    "本申请",
    "本公开",
    "本披露",
    "本实施例",
    "实施例",
    "具体实施方式",
    "技术领域",
    "背景技术",
    "发明内容",
    "权利要求",
    "说明书",
    "附图说明",
    "优选地",
    "进一步地",
    "进一步的",
    "具体地",
    "具体的",
    "其中",
    "一种",
    "一类",
    "一个",
    "多个",
    "若干",
    "所述",
    "上述",
    "下述",
    "前述",
    "该方法",
    "该装置",
    "该系统",
    "该设备",
}


REPLACEMENTS = str.maketrans(
    {
        "\u3000": " ",
        "\xa0": " ",
        "\t": " ",
        "\r": " ",
        "\n": " ",
        "，": " ",
        "。": " ",
        "、": " ",
        "；": " ",
        "：": " ",
        "！": " ",
        "？": " ",
        "（": " ",
        "）": " ",
        "【": " ",
        "】": " ",
        "《": " ",
        "》": " ",
        "“": " ",
        "”": " ",
        "‘": " ",
        "’": " ",
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
            "pip install -r requirements.txt"
        ) from exc
    return pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean patent title/abstract CSV files and export text length statistics."
    )
    parser.add_argument("--input", required=True, help="Path to source CSV file.")
    parser.add_argument("--output", required=True, help="Path to cleaned CSV file.")
    parser.add_argument(
        "--stats-output",
        required=True,
        help="Path to descriptive statistics CSV for concatenated text length.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV encoding. Defaults to utf-8-sig.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Treat the first three columns as PN,title,abstract when the CSV has no header.",
    )
    parser.add_argument(
        "--stopwords-file",
        help="Optional UTF-8 text file with one extra stopword per line.",
    )
    parser.add_argument(
        "--patent-phrases-file",
        help="Optional UTF-8 text file with one extra patent boilerplate phrase per line.",
    )
    parser.add_argument(
        "--keep-original-columns",
        action="store_true",
        help="Keep all original columns instead of only PN,title,abstract plus processed fields.",
    )
    return parser.parse_args()


def load_terms(path: str | None) -> set[str]:
    if not path:
        return set()

    term_path = Path(path)
    with term_path.open("r", encoding="utf-8-sig") as file:
        return {
            line.strip()
            for line in file
            if line.strip() and not line.lstrip().startswith("#")
        }


def read_patent_csv(path: str, encoding: str, no_header: bool) -> pd.DataFrame:
    pd = get_pandas()
    csv_path = Path(path)
    read_kwargs = {"encoding": encoding}
    if no_header:
        read_kwargs.update({"header": None})

    try:
        df = pd.read_csv(csv_path, **read_kwargs)
    except UnicodeDecodeError:
        if encoding.lower().replace("-", "") == "utf8sig":
            df = pd.read_csv(csv_path, encoding="gb18030", header=None if no_header else "infer")
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
    tokens = text.split()
    if len(tokens) > 1:
        cleaned_tokens = []
        for token in tokens:
            if token in stopwords:
                continue
            for word in INLINE_PARTICLE_STOPWORDS & stopwords:
                token = token.replace(word, "")
            if token:
                cleaned_tokens.append(token)
        return " ".join(cleaned_tokens)

    cleaned = text
    for word in sorted(stopwords, key=len, reverse=True):
        if word:
            cleaned = cleaned.replace(word, " ")
    return cleaned


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


def build_length_stats(length_series: pd.Series) -> pd.DataFrame:
    stats = length_series.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    stats = stats.rename(
        index={
            "count": "样本数",
            "mean": "均值",
            "std": "标准差",
            "min": "最小值",
            "25%": "25分位数",
            "50%": "中位数",
            "75%": "75分位数",
            "90%": "90分位数",
            "95%": "95分位数",
            "99%": "99分位数",
            "max": "最大值",
        }
    )
    return stats.reset_index().rename(columns={"index": "统计量", length_series.name: "文本长度"})


def preprocess(args: argparse.Namespace) -> tuple[Path, Path, pd.DataFrame]:
    stopwords = DEFAULT_STOPWORDS | load_terms(args.stopwords_file)
    patent_phrases = DEFAULT_PATENT_PHRASES | load_terms(args.patent_phrases_file)

    df = read_patent_csv(args.input, args.encoding, args.no_header)
    result = df.copy() if args.keep_original_columns else df[REQUIRED_COLUMNS].copy()

    result["clean_title"] = df["title"].map(
        lambda value: clean_text(value, stopwords, patent_phrases)
    )
    result["clean_abstract"] = df["abstract"].map(
        lambda value: clean_text(value, stopwords, patent_phrases)
    )
    result["text"] = (
        result["clean_title"].fillna("") + " " + result["clean_abstract"].fillna("")
    ).map(lambda value: SPACE_RE.sub(" ", value).strip())
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
