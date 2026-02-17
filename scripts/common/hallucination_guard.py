from __future__ import annotations

import re
from collections import Counter

MAX_LOOP_CHECK_CHARS = 12_000


def normalize_text_for_loop_check(text: str) -> str:
    text = re.sub(r"[a-zA-Z0-9\s]+", "", text)
    text = re.sub(
        r'[，。！？：；""\'（）【】《》、,\.!?:;"\'\(\)\[\]\{\}]',
        "",
        text,
    )
    return text.strip()


def has_consecutive_repeat(
    text: str,
    unit_len: int,
    repeat_count: int,
) -> bool:
    if unit_len <= 0 or repeat_count <= 1:
        return False
    span = unit_len * repeat_count
    if len(text) < span:
        return False

    max_i = len(text) - span
    for i in range(max_i + 1):
        unit = text[i : i + unit_len]
        matched = True
        for rep_idx in range(1, repeat_count):
            seg_start = i + rep_idx * unit_len
            if text[seg_start : seg_start + unit_len] != unit:
                matched = False
                break
        if matched:
            return True
    return False


def detect_repetition_loop(
    text: str,
    min_chars: int = 120,
) -> tuple[bool, str]:
    norm = normalize_text_for_loop_check(text)
    if len(norm) > MAX_LOOP_CHECK_CHARS:
        norm = norm[:MAX_LOOP_CHECK_CHARS]
    if len(norm) < max(1, min_chars):
        return False, ""

    # Fast path: repeated units appearing consecutively.
    for unit_len in (6, 8, 10, 12, 16, 20, 24, 32, 40, 48):
        if has_consecutive_repeat(norm, unit_len=unit_len, repeat_count=4):
            return True, f"repeat4_{unit_len}"
    for unit_len in (20, 24, 30, 40, 50, 64, 80, 100, 120, 160, 200):
        if has_consecutive_repeat(norm, unit_len=unit_len, repeat_count=3):
            return True, f"repeat3_{unit_len}"
    for unit_len in (40, 60, 80, 100, 120, 160, 200, 240, 320, 400):
        if has_consecutive_repeat(norm, unit_len=unit_len, repeat_count=2):
            return True, f"repeat2_{unit_len}"

    # N-gram dominance: one phrase repeatedly occupies a large text portion.
    for n in (8, 10, 12, 16):
        if len(norm) < n * 4:
            continue
        grams = [norm[i : i + n] for i in range(len(norm) - n + 1)]
        if not grams:
            continue
        top_count = Counter(grams).most_common(1)[0][1]
        covered_ratio = (top_count * n) / len(norm)
        if top_count >= 5 and covered_ratio >= 0.20:
            return True, f"ngram_{n}"

    return False, ""


def split_time_range_evenly(
    start_ms: int,
    end_ms: int,
    max_chunk_ms: int,
    min_chunk_ms: int,
) -> list[tuple[int, int]]:
    duration_ms = end_ms - start_ms
    if duration_ms <= 0:
        return []
    if max_chunk_ms <= 0 or duration_ms <= max_chunk_ms:
        return [(start_ms, end_ms)]

    chunk_count = max(2, (duration_ms + max_chunk_ms - 1) // max_chunk_ms)
    while chunk_count > 1:
        base_len = duration_ms / chunk_count
        if base_len >= min_chunk_ms:
            break
        chunk_count -= 1
    chunk_count = max(chunk_count, 1)

    bounds = [
        start_ms + int(round(i * duration_ms / chunk_count))
        for i in range(chunk_count + 1)
    ]
    bounds[0] = start_ms
    bounds[-1] = end_ms

    chunks: list[tuple[int, int]] = []
    for idx in range(chunk_count):
        s = bounds[idx]
        e = bounds[idx + 1]
        if e > s:
            chunks.append((s, e))
    return chunks
