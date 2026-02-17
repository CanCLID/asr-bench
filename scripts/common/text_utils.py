from __future__ import annotations

import re
from pathlib import Path

EN_DIGIT_SPACE_RE = re.compile(r"[a-zA-Z0-9\s]+")
PUNCT_RE = re.compile(r"[，。！？：；“”\"'（）【】《》、,\.!?:;\"'\(\)\[\]\{\}]")
TAG_RE = re.compile(r"<\|[^>]+?\|>")


def clean_asr_text(text: str) -> str:
    """Remove common ASR control tags while keeping transcript content."""
    return TAG_RE.sub("", text).strip()


def preprocess_chinese_text(text: str, include_punctuation: bool = False) -> str:
    """Normalize Chinese text for CER by filtering non-target symbols."""
    text = EN_DIGIT_SPACE_RE.sub("", text)
    if not include_punctuation:
        text = PUNCT_RE.sub("", text)
    return " ".join(list(text))


def parse_srt_text(path: Path) -> str:
    """Concatenate subtitle payload lines from an SRT file."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    payload: list[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.isdigit() or "-->" in line:
            continue
        payload.append(line)
    return "".join(payload)

