from __future__ import annotations

import math
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Protocol


class EvalReportResult(Protocol):
    cer: float
    cer_no_punc: float
    audio_duration_sec: float
    runtime_sec: float
    asr_runtime_sec: float
    rtf: float
    asr_rtf: float
    segment_count: int
    reference_chars_no_punc: int
    hypothesis_chars_no_punc: int
    edit_distance_no_punc: int
    substitution_count: int
    deletion_count: int
    insertion_count: int


def clip_text(text: str, max_len: int = 80) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def analyze_char_errors(
    ref_seq: list[str], hyp_seq: list[str], max_examples: int = 50
) -> tuple[
    Counter[tuple[str, str]],
    Counter[str],
    Counter[str],
    int,
    list[tuple[str, str, str, str, str]],
]:
    substitutions: Counter[tuple[str, str]] = Counter()
    deletions: Counter[str] = Counter()
    insertions: Counter[str] = Counter()
    equal_count = 0
    examples: list[tuple[str, str, str, str, str]] = []

    matcher = SequenceMatcher(a=ref_seq, b=hyp_seq, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            equal_count += i2 - i1
            continue

        ref_chunk = ref_seq[i1:i2]
        hyp_chunk = hyp_seq[j1:j2]

        if tag == "replace":
            common_len = min(len(ref_chunk), len(hyp_chunk))
            for k in range(common_len):
                substitutions[(ref_chunk[k], hyp_chunk[k])] += 1
            for ch in ref_chunk[common_len:]:
                deletions[ch] += 1
            for ch in hyp_chunk[common_len:]:
                insertions[ch] += 1
        elif tag == "delete":
            for ch in ref_chunk:
                deletions[ch] += 1
        elif tag == "insert":
            for ch in hyp_chunk:
                insertions[ch] += 1

        if len(examples) < max_examples:
            examples.append(
                (
                    tag,
                    f"{i1}:{i2}",
                    f"{j1}:{j2}",
                    clip_text("".join(ref_chunk) if ref_chunk else "∅"),
                    clip_text("".join(hyp_chunk) if hyp_chunk else "∅"),
                )
            )

    return substitutions, deletions, insertions, equal_count, examples


def build_counter_table(title: str, rows: list[tuple[str, int]], label: str) -> str:
    lines = [f"## {title}", "", f"| Rank | {label} | Count |", "| --- | --- | ---: |"]
    if not rows:
        lines.append("| 1 | (none) | 0 |")
        lines.append("")
        return "\n".join(lines)
    for idx, (item, count) in enumerate(rows, start=1):
        lines.append(f"| {idx} | `{_md_escape(item)}` | {count} |")
    lines.append("")
    return "\n".join(lines)


def build_file_analysis_markdown(
    *,
    audio_path: Path,
    reference_path: Path,
    output_srt_path: Path,
    result: EvalReportResult,
    ref_seq_no_punc: list[str],
    hyp_seq_no_punc: list[str],
    substitutions: Counter[tuple[str, str]],
    deletions: Counter[str],
    insertions: Counter[str],
    equal_count: int,
    examples: list[tuple[str, str, str, str, str]],
) -> str:
    sub_rows = [
        ((f"{src} -> {dst}"), cnt)
        for (src, dst), cnt in substitutions.most_common(20)
    ]
    del_rows = [(ch, cnt) for ch, cnt in deletions.most_common(20)]
    ins_rows = [(ch, cnt) for ch, cnt in insertions.most_common(20)]

    total_errors = result.edit_distance_no_punc
    total_ref = result.reference_chars_no_punc
    char_accuracy = float("nan") if total_ref == 0 else 1.0 - (total_errors / total_ref)
    ref_preview = clip_text("".join(ref_seq_no_punc), max_len=300)
    hyp_preview = clip_text("".join(hyp_seq_no_punc), max_len=300)

    lines = [
        "# ASR Error Analysis",
        "",
        "## File",
        f"- Audio: `{audio_path}`",
        f"- Reference: `{reference_path}`",
        f"- Prediction SRT: `{output_srt_path}`",
        "",
        "## Metrics",
        f"- CER (with punctuation): {result.cer:.6f}"
        if not math.isnan(result.cer)
        else "- CER (with punctuation): NaN",
        f"- CER (without punctuation): {result.cer_no_punc:.6f}"
        if not math.isnan(result.cer_no_punc)
        else "- CER (without punctuation): NaN",
        f"- Audio duration (s): {result.audio_duration_sec:.3f}",
        f"- Runtime (s): {result.runtime_sec:.3f}",
        f"- ASR runtime only (s): {result.asr_runtime_sec:.3f}",
        f"- End-to-end RTF: {result.rtf:.6f}"
        if not math.isnan(result.rtf)
        else "- End-to-end RTF: NaN",
        f"- ASR-only RTF: {result.asr_rtf:.6f}"
        if not math.isnan(result.asr_rtf)
        else "- ASR-only RTF: NaN",
        f"- Segments used: {result.segment_count}",
        f"- Reference chars (no punctuation): {result.reference_chars_no_punc}",
        f"- Hypothesis chars (no punctuation): {result.hypothesis_chars_no_punc}",
        f"- Edit distance (no punctuation): {result.edit_distance_no_punc}",
        f"- Character accuracy (no punctuation): {char_accuracy:.6f}"
        if not math.isnan(char_accuracy)
        else "- Character accuracy (no punctuation): NaN",
        "",
        "## Error Breakdown (No Punctuation)",
        f"- Correct characters: {equal_count}",
        f"- Substitutions: {result.substitution_count}",
        f"- Deletions: {result.deletion_count}",
        f"- Insertions: {result.insertion_count}",
        "",
    ]

    lines.append(build_counter_table("Top Substitution Patterns", sub_rows, "Ref -> Hyp"))
    lines.append(build_counter_table("Top Deleted Characters", del_rows, "Reference Char"))
    lines.append(
        build_counter_table("Top Inserted Characters", ins_rows, "Hypothesis Char")
    )

    lines.extend(
        [
            "## Mismatch Examples (No Punctuation)",
            "",
            "| # | Op | Ref Span | Hyp Span | Reference | Hypothesis |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    if examples:
        for idx, (op, ref_span, hyp_span, ref_text, hyp_text) in enumerate(
            examples, start=1
        ):
            lines.append(
                f"| {idx} | `{op}` | `{ref_span}` | `{hyp_span}` | `{_md_escape(ref_text)}` | `{_md_escape(hyp_text)}` |"
            )
    else:
        lines.append("| 1 | `none` | `-` | `-` | `-` | `-` |")
    lines.extend(
        [
            "",
            "## Normalized Text Preview (No Punctuation)",
            "",
            "```text",
            f"REF: {ref_preview}",
            f"HYP: {hyp_preview}",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
