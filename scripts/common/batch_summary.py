from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Protocol


class BatchSummaryResult(Protocol):
    audio_path: Path
    analysis_report_path: Path
    cer: float
    cer_no_punc: float
    runtime_sec: float
    asr_runtime_sec: float
    rtf: float
    asr_rtf: float
    audio_duration_sec: float
    substitution_count: int
    deletion_count: int
    insertion_count: int
    substitution_counter: Counter[tuple[str, str]]
    deletion_counter: Counter[str]
    insertion_counter: Counter[str]


def _md_escape(text: str) -> str:
    return str(text).replace("|", "\\|")


def _build_counter_table(
    title: str,
    rows: list[tuple[str, int]],
    label: str,
) -> str:
    lines = [f"## {title}", "", f"| Rank | {label} | Count |", "| --- | --- | ---: |"]
    if not rows:
        lines.append("| 1 | (none) | 0 |")
        lines.append("")
        return "\n".join(lines)
    for idx, (item, count) in enumerate(rows, start=1):
        lines.append(f"| {idx} | `{_md_escape(item)}` | {count} |")
    lines.append("")
    return "\n".join(lines)


def write_batch_analysis_summary(
    output_path: Path,
    results: list[BatchSummaryResult],
    micro_cer: float,
    micro_cer_no_punc: float,
    macro_cer: float,
    macro_cer_no_punc: float,
) -> None:
    sub_all: Counter[tuple[str, str]] = Counter()
    del_all: Counter[str] = Counter()
    ins_all: Counter[str] = Counter()
    total_sub = 0
    total_del = 0
    total_ins = 0
    total_audio_sec = 0.0
    total_runtime_sec = 0.0
    total_asr_runtime_sec = 0.0
    for r in results:
        sub_all.update(r.substitution_counter)
        del_all.update(r.deletion_counter)
        ins_all.update(r.insertion_counter)
        total_sub += r.substitution_count
        total_del += r.deletion_count
        total_ins += r.insertion_count
        total_audio_sec += r.audio_duration_sec
        total_runtime_sec += r.runtime_sec
        total_asr_runtime_sec += r.asr_runtime_sec

    avg_runtime_sec = float("nan") if not results else total_runtime_sec / len(results)
    avg_audio_sec = float("nan") if not results else total_audio_sec / len(results)
    overall_rtf = (
        float("nan")
        if total_audio_sec <= 0
        else total_runtime_sec / total_audio_sec
    )
    overall_asr_rtf = (
        float("nan")
        if total_audio_sec <= 0
        else total_asr_runtime_sec / total_audio_sec
    )

    lines = [
        "# ASR Batch Analysis Summary",
        "",
        "## Metrics",
        f"- Files evaluated: {len(results)}",
        f"- Micro CER (with punctuation): {micro_cer:.6f}",
        f"- Micro CER (without punctuation): {micro_cer_no_punc:.6f}",
        f"- Macro CER (with punctuation): {macro_cer:.6f}" if not math.isnan(macro_cer) else "- Macro CER (with punctuation): NaN",
        f"- Macro CER (without punctuation): {macro_cer_no_punc:.6f}" if not math.isnan(macro_cer_no_punc) else "- Macro CER (without punctuation): NaN",
        f"- Total audio duration (s): {total_audio_sec:.3f}",
        f"- Total runtime (s): {total_runtime_sec:.3f}",
        f"- Total ASR runtime only (s): {total_asr_runtime_sec:.3f}",
        f"- Average runtime per file (s): {avg_runtime_sec:.3f}" if not math.isnan(avg_runtime_sec) else "- Average runtime per file (s): NaN",
        f"- Average audio duration per file (s): {avg_audio_sec:.3f}" if not math.isnan(avg_audio_sec) else "- Average audio duration per file (s): NaN",
        f"- End-to-end RTF (batch): {overall_rtf:.6f}" if not math.isnan(overall_rtf) else "- End-to-end RTF (batch): NaN",
        f"- ASR-only RTF (batch): {overall_asr_rtf:.6f}" if not math.isnan(overall_asr_rtf) else "- ASR-only RTF (batch): NaN",
        "",
        "## Error Totals (No Punctuation)",
        f"- Substitutions: {total_sub}",
        f"- Deletions: {total_del}",
        f"- Insertions: {total_ins}",
        "",
        "## Per-file Metrics",
        "",
        "| File | CER | CER (No Punc) | Runtime (s) | ASR Runtime (s) | RTF | ASR RTF | Analysis Report |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in results:
        lines.append(
            f"| `{r.audio_path.name}` | {r.cer:.6f} | {r.cer_no_punc:.6f} | {r.runtime_sec:.3f} | {r.asr_runtime_sec:.3f} | {r.rtf:.6f} | {r.asr_rtf:.6f} | `{r.analysis_report_path.name}` |"
        )
    lines.append("")

    sub_rows = [((f"{src} -> {dst}"), cnt) for (src, dst), cnt in sub_all.most_common(30)]
    del_rows = [(ch, cnt) for ch, cnt in del_all.most_common(30)]
    ins_rows = [(ch, cnt) for ch, cnt in ins_all.most_common(30)]
    lines.append(_build_counter_table("Top Substitution Patterns (All Files)", sub_rows, "Ref -> Hyp"))
    lines.append(_build_counter_table("Top Deleted Characters (All Files)", del_rows, "Reference Char"))
    lines.append(_build_counter_table("Top Inserted Characters (All Files)", ins_rows, "Hypothesis Char"))

    output_path.write_text("\n".join(lines), encoding="utf-8")

