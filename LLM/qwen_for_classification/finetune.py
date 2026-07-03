"""Entry point for Qwen LoRA patent classification."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from LLM.llm_lora_classifier import parse_args, train  # noqa: E402


if __name__ == "__main__":
    if "--model-key" not in sys.argv:
        sys.argv.extend(["--model-key", "qwen"])
    args = parse_args()
    train(args)
