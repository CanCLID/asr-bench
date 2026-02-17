from __future__ import annotations

from typing import Sequence

from .text_utils import preprocess_chinese_text


def sequence_for_cer(text: str, include_punctuation: bool) -> list[str]:
    preprocessed = preprocess_chinese_text(
        text, include_punctuation=include_punctuation
    )
    return preprocessed.split() if preprocessed else []


def levenshtein_distance(a: Sequence[str], b: Sequence[str]) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]


def compute_cer(
    ref_text: str,
    hyp_text: str,
    include_punctuation: bool,
) -> tuple[int, int, int, float]:
    ref_seq = sequence_for_cer(ref_text, include_punctuation=include_punctuation)
    hyp_seq = sequence_for_cer(hyp_text, include_punctuation=include_punctuation)
    dist = levenshtein_distance(ref_seq, hyp_seq)
    cer = float("nan") if not ref_seq else dist / len(ref_seq)
    return len(ref_seq), len(hyp_seq), dist, cer
