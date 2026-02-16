#!/usr/bin/env python3
"""Run Qwen3-ASR-1.7B, export SRT, and compute CER (single file or batch)."""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import opencc
import torch
from funasr import AutoModel
from huggingface_hub import snapshot_download
from transformers import AutoModel as HFAutoModel
from transformers import AutoProcessor


@dataclass
class EvalResult:
    audio_path: Path
    reference_path: Path
    output_srt_path: Path
    audio_duration_sec: float
    segment_count: int
    runtime_sec: float
    asr_runtime_sec: float
    rtf: float
    asr_rtf: float
    reference_chars: int
    hypothesis_chars: int
    edit_distance: int
    cer: float
    reference_chars_no_punc: int
    hypothesis_chars_no_punc: int
    edit_distance_no_punc: int
    cer_no_punc: float
    analysis_report_path: Path
    substitution_count: int
    deletion_count: int
    insertion_count: int
    substitution_counter: Counter[tuple[str, str]]
    deletion_counter: Counter[str]
    insertion_counter: Counter[str]


class CantonesePostProcessor:
    def __init__(self) -> None:
        self.converter = opencc.OpenCC("s2hk")
        self.regular_errors: list[tuple[re.Pattern[str], str]] = [
            (re.compile(r"俾(?!(?:路支|斯麥|益))"), r"畀"),
            (re.compile(r"(?<!(?:聯))[系繫](?!(?:統))"), r"係"),
            (re.compile(r"噶"), r"㗎"),
            (re.compile(r"咁(?=[我你佢就樣就話係啊呀嘅，。])"), r"噉"),
            (re.compile(r"(?<![曝晾])曬(?:[衣太衫褲被命嘢相])"), r"晒"),
            (re.compile(r"(?<=[好])翻(?=[去到嚟])"), r"返"),
            (re.compile(r"<\|\w+\|>"), r""),
        ]

    def apply(self, text: str) -> str:
        text = self.converter.convert(text)
        for pattern, replacement in self.regular_errors:
            text = pattern.sub(replacement, text)
        return text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR-1.7B transcription -> SRT + CER vs golden SRT"
    )
    parser.add_argument("--audio", help="Input audio path (single-file mode)")
    parser.add_argument(
        "--golden-srt", help="Golden reference SRT path for CER (single-file mode)"
    )
    parser.add_argument(
        "--output-srt", help="Output path for generated SRT (single-file mode)"
    )
    parser.add_argument(
        "--input-dir",
        default="input",
        help='Input audio directory (batch mode). Default: "input". Reference is matched by stem, e.g. 001.opus -> reference/001.srt',
    )
    parser.add_argument(
        "--reference-dir",
        default="reference",
        help='Reference SRT directory (batch mode). Default: "reference"',
    )
    parser.add_argument(
        "--output-dir",
        default="predicted/qwen3asr_1_7b",
        help='Output SRT directory (batch mode). Default: "predicted/qwen3asr_1_7b"',
    )
    parser.add_argument(
        "--audio-extensions",
        default=".opus,.wav,.mp3,.m4a,.flac,.ogg,.aac",
        help='Comma-separated audio extensions for batch mode. Default: ".opus,.wav,.mp3,.m4a,.flac,.ogg,.aac"',
    )
    parser.add_argument(
        "--strict-missing-reference",
        action="store_true",
        help="Fail if any audio in batch mode has no matching reference .srt",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Inference device. Use "auto" (default) to prefer CUDA, else "cpu"/"cuda:0"',
    )
    parser.add_argument(
        "--min-segment-ms",
        type=int,
        default=300,
        help="Skip VAD segments shorter than this (ms)",
    )
    parser.add_argument(
        "--vad-max-segment-ms",
        type=int,
        default=10000,
        help="Maximum VAD segment duration in ms. Default: 10000 (10s).",
    )
    parser.add_argument(
        "--vad-backend",
        choices=["fsmn", "firered"],
        default="fsmn",
        help='VAD backend. "fsmn" (default) or FireRed VAD ("firered").',
    )
    parser.add_argument(
        "--segment-batch-size",
        type=int,
        default=0,
        help="ASR batch size for VAD segments. 0 = auto (GPU:128, CPU:4).",
    )
    parser.add_argument(
        "--qwen-src-dir",
        default=".cache/Qwen3-ASR-src",
        help='Path to Qwen3-ASR source code. Auto-cloned if missing. Default: ".cache/Qwen3-ASR-src".',
    )
    parser.add_argument(
        "--qwen-repo-url",
        default="https://github.com/QwenLM/Qwen3-ASR",
        help="Git URL used when auto-cloning Qwen3-ASR source.",
    )
    parser.add_argument(
        "--qwen-model",
        default="Qwen/Qwen3-ASR-1.7B",
        help='Hugging Face model id or local path. Default: "Qwen/Qwen3-ASR-1.7B".',
    )
    parser.add_argument(
        "--qwen-use-forced-aligner",
        action="store_true",
        help="Enable Qwen forced aligner during transcription (for timestamp alignment).",
    )
    parser.add_argument(
        "--qwen-forced-aligner-model",
        default="Qwen/Qwen3-ForcedAligner-0.6B",
        help='Forced aligner model id or local path. Default: "Qwen/Qwen3-ForcedAligner-0.6B".',
    )
    parser.add_argument(
        "--qwen-language",
        default="Cantonese",
        help='Forced language name for Qwen3-ASR. Default: "Cantonese".',
    )
    parser.add_argument(
        "--qwen-dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help='Model dtype. "auto" uses bfloat16 on CUDA and float32 on CPU.',
    )
    parser.add_argument(
        "--qwen-max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per segment batch.",
    )
    parser.add_argument(
        "--firered-src-dir",
        default=".cache/FireRedASR2S-src",
        help='Path to FireRedASR2S source code. Auto-cloned if missing. Default: ".cache/FireRedASR2S-src".',
    )
    parser.add_argument(
        "--firered-repo-url",
        default="https://github.com/FireRedTeam/FireRedASR2S",
        help="Git URL used when auto-cloning FireRedASR2S source.",
    )
    parser.add_argument(
        "--firered-vad-model-dir",
        default="models/FireRedVAD",
        help='Local model directory for FireRedVAD. Auto-downloaded if missing. Default: "models/FireRedVAD".',
    )
    parser.add_argument(
        "--firered-vad-model-repo",
        default="FireRedTeam/FireRedVAD",
        help='Hugging Face repo id for FireRedVAD. Default: "FireRedTeam/FireRedVAD".',
    )
    parser.add_argument(
        "--summary-dir",
        default="summary",
        help='Batch summary output directory. Default: "summary".',
    )
    parser.add_argument(
        "--summary-name",
        default="qwen3_asr_1_7b",
        help='Batch summary filename stem (or .md name). Default: "qwen3_asr_1_7b".',
    )
    return parser.parse_args()


def is_single_mode(args: argparse.Namespace) -> bool:
    return bool(args.audio and args.golden_srt and args.output_srt)


def has_any_single_arg(args: argparse.Namespace) -> bool:
    return bool(args.audio or args.golden_srt or args.output_srt)


def validate_args(args: argparse.Namespace) -> None:
    if has_any_single_arg(args) and not is_single_mode(args):
        raise ValueError(
            "Single-file mode requires all args: --audio --golden-srt --output-srt"
        )


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def resolve_segment_batch_size(device: str, batch_size_arg: int) -> int:
    if batch_size_arg > 0:
        return batch_size_arg
    return 128 if device.startswith("cuda") else 4


def resolve_summary_path(summary_dir: Path, summary_name: str) -> Path:
    name = summary_name.strip() or "qwen3_asr_1_7b"
    if not name.lower().endswith(".md"):
        name = f"{name}.md"
    summary_dir.mkdir(parents=True, exist_ok=True)
    return summary_dir / name


def ensure_qwen_source(src_dir: Path, repo_url: str) -> Path:
    if (src_dir / "qwen_asr").is_dir():
        return src_dir
    src_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(src_dir)],
        check=True,
    )
    return src_dir


def ensure_firered_source(src_dir: Path, repo_url: str) -> Path:
    if (src_dir / "fireredasr2s").is_dir():
        return src_dir
    src_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(src_dir)],
        check=True,
    )
    return src_dir


def ensure_firered_vad_model(model_dir: Path, model_repo: str) -> Path:
    direct_files = ["cmvn.ark", "model.pth.tar"]
    nested_files = [("VAD", "cmvn.ark"), ("VAD", "model.pth.tar")]
    has_direct = model_dir.is_dir() and all((model_dir / name).is_file() for name in direct_files)
    has_nested = model_dir.is_dir() and all((model_dir / Path(*parts)).is_file() for parts in nested_files)
    if has_direct or has_nested:
        return model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=model_repo,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
    except TypeError:
        snapshot_download(
            repo_id=model_repo,
            local_dir=str(model_dir),
        )
    return model_dir


def resolve_firered_vad_dir(model_root_dir: Path) -> Path:
    direct_dir = model_root_dir
    nested_dir = model_root_dir / "VAD"
    direct_ok = (direct_dir / "cmvn.ark").is_file() and (direct_dir / "model.pth.tar").is_file()
    nested_ok = (nested_dir / "cmvn.ark").is_file() and (nested_dir / "model.pth.tar").is_file()
    if direct_ok:
        return direct_dir
    if nested_ok:
        return nested_dir
    raise FileNotFoundError(
        f"FireRedVAD model files not found under {model_root_dir}. "
        "Expected cmvn.ark/model.pth.tar either in model root or in model_root/VAD."
    )


def resolve_qwen_dtype(dtype_arg: str, device: str) -> torch.dtype:
    if dtype_arg == "auto":
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_arg}")


def ensure_qwen_transformers_compat() -> None:
    try:
        import transformers.utils.generic as generic
        from transformers import modeling_rope_utils
    except Exception:
        return

    if hasattr(generic, "check_model_inputs"):
        pass
    else:
        def check_model_inputs(*_args: Any, **_kwargs: Any) -> Any:
            def decorator(func: Any) -> Any:
                return func

            return decorator

        generic.check_model_inputs = check_model_inputs  # type: ignore[attr-defined]

    if "default" in modeling_rope_utils.ROPE_INIT_FUNCTIONS:
        return

    # Compatibility shim for Qwen3-ASR source code that expects a "default"
    # RoPE type entry in older Transformers versions.
    def qwen_default_rope_parameters(
        config: Any = None,
        device: Any = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple[torch.Tensor, float]:
        del seq_len
        if config is None:
            raise ValueError("config is required for default rope parameters")

        base = float(getattr(config, "rope_theta", 10000.0))
        partial_rotary_factor = 1.0

        if layer_type is not None and hasattr(config, "rope_parameters"):
            rope_parameters = getattr(config, "rope_parameters")
            if isinstance(rope_parameters, dict):
                layer_params = rope_parameters.get(layer_type, {})
                if isinstance(layer_params, dict):
                    base = float(layer_params.get("rope_theta", base))
                    partial_rotary_factor = float(
                        layer_params.get("partial_rotary_factor", partial_rotary_factor)
                    )

        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = int(config.hidden_size) // int(config.num_attention_heads)

        dim = int(head_dim * partial_rotary_factor)
        if dim < 2:
            dim = 2
        if dim % 2 == 1:
            dim -= 1
            if dim < 2:
                dim = 2

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.float32, device=device)
                / float(dim)
            )
        )
        return inv_freq, 1.0

    modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = qwen_default_rope_parameters


def ensure_qwen_config_compat() -> None:
    try:
        from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
            Qwen3ASRThinkerConfig,
        )
    except Exception:
        return

    if not hasattr(Qwen3ASRThinkerConfig, "pad_token_id"):
        Qwen3ASRThinkerConfig.pad_token_id = -1  # type: ignore[attr-defined]


def ensure_qwen_modeling_compat() -> None:
    try:
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRThinkerTextRotaryEmbedding,
        )
    except Exception:
        return

    if hasattr(Qwen3ASRThinkerTextRotaryEmbedding, "compute_default_rope_parameters"):
        return

    def compute_default_rope_parameters(
        self: Any,
        config: Any,
        device: Any = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple[torch.Tensor, float]:
        del seq_len, layer_type
        rope_init_fn = getattr(self, "rope_init_fn", None)
        if rope_init_fn is None:
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

            rope_init_fn = ROPE_INIT_FUNCTIONS["default"]
        return rope_init_fn(config, device)

    Qwen3ASRThinkerTextRotaryEmbedding.compute_default_rope_parameters = (  # type: ignore[attr-defined]
        compute_default_rope_parameters
    )


def move_qwen_model_to_device(qwen_model: Any, device: str) -> None:
    if not device.startswith("cuda"):
        return
    qwen_model.model = qwen_model.model.to(device)
    qwen_model.device = next(qwen_model.model.parameters()).device
    qwen_model.dtype = qwen_model.model.dtype


def ms_to_srt_time(ms: int) -> str:
    ms = max(0, int(ms))
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def clean_asr_text(text: str) -> str:
    # Remove residual tags like <|yue|> while keeping transcript text.
    text = re.sub(r"<\|[^>]+?\|>", "", text)
    return text.strip()


def preprocess_chinese_text(text: str, include_punctuation: bool = False) -> str:
    # Remove English letters, digits, and extra spaces.
    text = re.sub(r"[a-zA-Z0-9\s]+", "", text)
    if not include_punctuation:
        text = re.sub(
            r"[，。！？：；“”\"'（）【】《》、,\.!?:;\"'\(\)\[\]\{\}]",
            "",
            text,
        )
    return " ".join(list(text))


def parse_srt_text(path: Path) -> str:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    payload = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.isdigit():
            continue
        if "-->" in line:
            continue
        payload.append(line)
    return "".join(payload)


def levenshtein_distance(a: str, b: str) -> int:
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


def sequence_for_cer(text: str, include_punctuation: bool) -> list[str]:
    preprocessed = preprocess_chinese_text(
        text, include_punctuation=include_punctuation
    )
    return preprocessed.split() if preprocessed else []


def compute_cer(ref_text: str, hyp_text: str, include_punctuation: bool) -> tuple[int, int, int, float]:
    ref_seq = sequence_for_cer(ref_text, include_punctuation=include_punctuation)
    hyp_seq = sequence_for_cer(hyp_text, include_punctuation=include_punctuation)
    dist = levenshtein_distance(ref_seq, hyp_seq)
    cer = float("nan") if not ref_seq else dist / len(ref_seq)
    return len(ref_seq), len(hyp_seq), dist, cer


def clip_text(text: str, max_len: int = 80) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def md_escape(text: str) -> str:
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


def build_counter_table(
    title: str, rows: list[tuple[str, int]], label: str
) -> str:
    lines = [f"## {title}", "", f"| Rank | {label} | Count |", "| --- | --- | ---: |"]
    if not rows:
        lines.append("| 1 | (none) | 0 |")
        lines.append("")
        return "\n".join(lines)
    for idx, (item, count) in enumerate(rows, start=1):
        lines.append(f"| {idx} | `{md_escape(item)}` | {count} |")
    lines.append("")
    return "\n".join(lines)


def build_file_analysis_markdown(
    *,
    audio_path: Path,
    reference_path: Path,
    output_srt_path: Path,
    result: EvalResult,
    ref_seq_no_punc: list[str],
    hyp_seq_no_punc: list[str],
    substitutions: Counter[tuple[str, str]],
    deletions: Counter[str],
    insertions: Counter[str],
    equal_count: int,
    examples: list[tuple[str, str, str, str, str]],
) -> str:
    sub_rows = [((f"{src} -> {dst}"), cnt) for (src, dst), cnt in substitutions.most_common(20)]
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
        f"- CER (with punctuation): {result.cer:.6f}" if not math.isnan(result.cer) else "- CER (with punctuation): NaN",
        f"- CER (without punctuation): {result.cer_no_punc:.6f}" if not math.isnan(result.cer_no_punc) else "- CER (without punctuation): NaN",
        f"- Audio duration (s): {result.audio_duration_sec:.3f}",
        f"- Runtime (s): {result.runtime_sec:.3f}",
        f"- ASR runtime only (s): {result.asr_runtime_sec:.3f}",
        f"- End-to-end RTF: {result.rtf:.6f}" if not math.isnan(result.rtf) else "- End-to-end RTF: NaN",
        f"- ASR-only RTF: {result.asr_rtf:.6f}" if not math.isnan(result.asr_rtf) else "- ASR-only RTF: NaN",
        f"- Segments used: {result.segment_count}",
        f"- Reference chars (no punctuation): {result.reference_chars_no_punc}",
        f"- Hypothesis chars (no punctuation): {result.hypothesis_chars_no_punc}",
        f"- Edit distance (no punctuation): {result.edit_distance_no_punc}",
        f"- Character accuracy (no punctuation): {char_accuracy:.6f}" if not math.isnan(char_accuracy) else "- Character accuracy (no punctuation): NaN",
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
    lines.append(build_counter_table("Top Inserted Characters", ins_rows, "Hypothesis Char"))

    lines.extend(
        [
            "## Mismatch Examples (No Punctuation)",
            "",
            "| # | Op | Ref Span | Hyp Span | Reference | Hypothesis |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    if examples:
        for idx, (op, ref_span, hyp_span, ref_text, hyp_text) in enumerate(examples, start=1):
            lines.append(
                f"| {idx} | `{op}` | `{ref_span}` | `{hyp_span}` | `{md_escape(ref_text)}` | `{md_escape(hyp_text)}` |"
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


def write_batch_analysis_summary(
    output_path: Path,
    results: list[EvalResult],
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

    avg_runtime_sec = (
        float("nan") if not results else total_runtime_sec / len(results)
    )
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
    lines.append(build_counter_table("Top Substitution Patterns (All Files)", sub_rows, "Ref -> Hyp"))
    lines.append(build_counter_table("Top Deleted Characters (All Files)", del_rows, "Reference Char"))
    lines.append(build_counter_table("Top Inserted Characters (All Files)", ins_rows, "Hypothesis Char"))

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_srt(path: Path, entries: list[tuple[int, int, str]]) -> None:
    chunks = []
    for idx, (start_ms, end_ms, text) in enumerate(entries, start=1):
        chunks.append(
            f"{idx}\n{ms_to_srt_time(start_ms)} --> {ms_to_srt_time(end_ms)}\n{text}\n"
        )
    path.write_text("\n".join(chunks), encoding="utf-8")


def parse_extensions(exts: str) -> set[str]:
    return {
        ext.strip().lower() if ext.strip().startswith(".") else f".{ext.strip().lower()}"
        for ext in exts.split(",")
        if ext.strip()
    }


def find_audio_files(input_dir: Path, extensions: set[str]) -> list[Path]:
    files = [
        p
        for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in extensions
    ]
    return files


def get_audio_duration_sec(audio_path: Path) -> float:
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        return max(0.0, float(out))
    except Exception:
        return float("nan")


def transcode_audio_to_16k_wav(input_path: Path, output_wav_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_wav_path),
    ]
    subprocess.run(cmd, check=True)


def split_long_segments(
    segments_ms: list[tuple[int, int]],
    max_segment_ms: int,
) -> list[tuple[int, int]]:
    if max_segment_ms <= 0:
        return segments_ms
    out: list[tuple[int, int]] = []
    for start_ms, end_ms in segments_ms:
        if end_ms <= start_ms:
            continue
        cur = start_ms
        while cur < end_ms:
            chunk_end = min(cur + max_segment_ms, end_ms)
            if chunk_end > cur:
                out.append((cur, chunk_end))
            cur = chunk_end
    return out


def get_vad_segments_ms(
    audio_path: Path,
    vad_backend: str,
    fsmn_vad_model: AutoModel | None,
    firered_vad_model: Any | None,
    min_segment_ms: int,
    max_segment_ms: int,
) -> tuple[list[tuple[int, int]], int]:
    raw_segments_ms: list[tuple[int, int]] = []

    if vad_backend == "fsmn":
        if fsmn_vad_model is None:
            raise RuntimeError("FSMN VAD backend selected, but model is not loaded.")
        vad_res = fsmn_vad_model.generate(input=str(audio_path))
        if not vad_res or "value" not in vad_res[0]:
            raise RuntimeError(f"Unexpected VAD output for {audio_path}: {vad_res}")
        raw_segments_ms = [
            (int(start_ms), int(end_ms))
            for start_ms, end_ms in vad_res[0]["value"]
        ]
    elif vad_backend == "firered":
        if firered_vad_model is None:
            raise RuntimeError("FireRed VAD backend selected, but model is not loaded.")
        with tempfile.NamedTemporaryFile(
            prefix="firered_vad_input_",
            suffix=".wav",
            delete=False,
        ) as tmp_wav_file:
            tmp_wav_path = Path(tmp_wav_file.name)
        try:
            transcode_audio_to_16k_wav(audio_path, tmp_wav_path)
            vad_res, _ = firered_vad_model.detect(str(tmp_wav_path))
        finally:
            if tmp_wav_path.exists():
                tmp_wav_path.unlink()
        if not isinstance(vad_res, dict) or "timestamps" not in vad_res:
            raise RuntimeError(f"Unexpected FireRedVAD output for {audio_path}: {vad_res}")
        raw_segments_ms = [
            (int(round(float(start_s) * 1000.0)), int(round(float(end_s) * 1000.0)))
            for start_s, end_s in vad_res["timestamps"]
        ]
    else:
        raise ValueError(f"Unsupported VAD backend: {vad_backend}")

    split_segments = split_long_segments(raw_segments_ms, max_segment_ms=max_segment_ms)
    segments = [
        (start_ms, end_ms)
        for start_ms, end_ms in split_segments
        if end_ms - start_ms >= min_segment_ms
    ]
    return segments, len(raw_segments_ms)


def run_one_file(
    audio_path: Path,
    reference_srt_path: Path,
    output_srt_path: Path,
    asr_model: Any,
    vad_backend: str,
    fsmn_vad_model: AutoModel | None,
    firered_vad_model: Any | None,
    postprocessor: CantonesePostProcessor,
    min_segment_ms: int,
    vad_max_segment_ms: int,
    segment_batch_size: int,
    qwen_language: str,
    use_forced_aligner: bool,
) -> EvalResult:
    print(f"\n=== Processing: {audio_path.name} ===")
    file_start_ts = time.perf_counter()
    audio_duration_sec = get_audio_duration_sec(audio_path)
    asr_runtime_sec = 0.0

    segments, raw_segment_count = get_vad_segments_ms(
        audio_path=audio_path,
        vad_backend=vad_backend,
        fsmn_vad_model=fsmn_vad_model,
        firered_vad_model=firered_vad_model,
        min_segment_ms=min_segment_ms,
        max_segment_ms=vad_max_segment_ms,
    )
    print(f"VAD backend: {vad_backend}; segments: {raw_segment_count}; used: {len(segments)}")

    entries: list[tuple[int, int, str]] = []
    with tempfile.TemporaryDirectory(prefix="qwen3_segments_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for batch_start in range(0, len(segments), segment_batch_size):
            batch_segments = segments[batch_start : batch_start + segment_batch_size]
            batch_wavs: list[str] = []
            for seg_idx, (start_ms, end_ms) in enumerate(batch_segments, start=batch_start + 1):
                seg_wav = tmpdir_path / f"seg_{seg_idx:05d}.wav"
                extract_segment_to_wav(audio_path, start_ms, end_ms, seg_wav)
                batch_wavs.append(str(seg_wav))

            asr_start_ts = time.perf_counter()
            res = asr_model.transcribe(
                audio=batch_wavs,
                language=qwen_language,
                return_time_stamps=use_forced_aligner,
            )
            asr_runtime_sec += time.perf_counter() - asr_start_ts
            if len(res) != len(batch_segments):
                raise RuntimeError(
                    f"ASR result size mismatch: got {len(res)}, expected {len(batch_segments)}"
                )

            for item, (start_ms, end_ms) in zip(res, batch_segments):
                raw_text = str(getattr(item, "text", ""))
                text = clean_asr_text(raw_text)
                text = postprocessor.apply(text)
                if text:
                    entries.append((start_ms, end_ms, text))

            done = min(batch_start + len(batch_segments), len(segments))
            print(f"Transcribed {done}/{len(segments)} segments (batch={segment_batch_size})")

    runtime_sec = time.perf_counter() - file_start_ts
    rtf = float("nan") if audio_duration_sec <= 0 else runtime_sec / audio_duration_sec
    asr_rtf = (
        float("nan")
        if audio_duration_sec <= 0
        else asr_runtime_sec / audio_duration_sec
    )

    output_srt_path.parent.mkdir(parents=True, exist_ok=True)
    write_srt(output_srt_path, entries)

    ref_text = parse_srt_text(reference_srt_path)
    hyp_text = "".join(text for _, _, text in entries)
    ref_len, hyp_len, dist, cer = compute_cer(
        ref_text, hyp_text, include_punctuation=True
    )
    ref_len_no_punc, hyp_len_no_punc, dist_no_punc, cer_no_punc = compute_cer(
        ref_text, hyp_text, include_punctuation=False
    )
    ref_seq_no_punc = sequence_for_cer(ref_text, include_punctuation=False)
    hyp_seq_no_punc = sequence_for_cer(hyp_text, include_punctuation=False)
    substitutions, deletions, insertions, equal_count, examples = analyze_char_errors(
        ref_seq_no_punc, hyp_seq_no_punc
    )
    analysis_report_path = output_srt_path.with_suffix(".analysis.md")
    temp_result = EvalResult(
        audio_path=audio_path,
        reference_path=reference_srt_path,
        output_srt_path=output_srt_path,
        audio_duration_sec=audio_duration_sec,
        segment_count=len(segments),
        runtime_sec=runtime_sec,
        asr_runtime_sec=asr_runtime_sec,
        rtf=rtf,
        asr_rtf=asr_rtf,
        reference_chars=ref_len,
        hypothesis_chars=hyp_len,
        edit_distance=dist,
        cer=cer,
        reference_chars_no_punc=ref_len_no_punc,
        hypothesis_chars_no_punc=hyp_len_no_punc,
        edit_distance_no_punc=dist_no_punc,
        cer_no_punc=cer_no_punc,
        analysis_report_path=analysis_report_path,
        substitution_count=sum(substitutions.values()),
        deletion_count=sum(deletions.values()),
        insertion_count=sum(insertions.values()),
        substitution_counter=substitutions,
        deletion_counter=deletions,
        insertion_counter=insertions,
    )
    analysis_md = build_file_analysis_markdown(
        audio_path=audio_path,
        reference_path=reference_srt_path,
        output_srt_path=output_srt_path,
        result=temp_result,
        ref_seq_no_punc=ref_seq_no_punc,
        hyp_seq_no_punc=hyp_seq_no_punc,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        equal_count=equal_count,
        examples=examples,
    )
    analysis_report_path.write_text(analysis_md, encoding="utf-8")
    return temp_result


def extract_segment_to_wav(
    audio_path: Path, start_ms: int, end_ms: int, out_wav_path: Path
) -> None:
    duration_ms = end_ms - start_ms
    if duration_ms <= 0:
        raise ValueError(f"Invalid segment range: {start_ms} -> {end_ms}")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_ms / 1000:.3f}",
        "-t",
        f"{duration_ms / 1000:.3f}",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(out_wav_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    resolved_device = resolve_device(args.device)
    qwen_dtype = resolve_qwen_dtype(args.qwen_dtype, resolved_device)
    segment_batch_size = resolve_segment_batch_size(
        resolved_device, args.segment_batch_size
    )
    print(f"Using device: {resolved_device}")
    print(f"Qwen dtype: {qwen_dtype}")
    print(f"Segment batch size: {segment_batch_size}")
    print(f"VAD backend: {args.vad_backend}")
    print(
        "Use Qwen forced aligner: "
        f"{'yes' if args.qwen_use_forced_aligner else 'no'}"
    )

    print("Loading models...")
    qwen_src_dir = ensure_qwen_source(Path(args.qwen_src_dir), args.qwen_repo_url)
    if str(qwen_src_dir) not in sys.path:
        sys.path.insert(0, str(qwen_src_dir))
    ensure_qwen_transformers_compat()

    from qwen_asr import Qwen3ASRModel
    from qwen_asr.inference.qwen3_forced_aligner import (
        Qwen3ForceAlignProcessor,
        Qwen3ForcedAligner,
    )

    ensure_qwen_config_compat()
    ensure_qwen_modeling_compat()

    asr_model_hf = HFAutoModel.from_pretrained(
        args.qwen_model,
        dtype=qwen_dtype,
    )
    try:
        asr_processor = AutoProcessor.from_pretrained(
            args.qwen_model, fix_mistral_regex=True
        )
    except TypeError:
        asr_processor = AutoProcessor.from_pretrained(args.qwen_model)

    forced_aligner = None
    if args.qwen_use_forced_aligner:
        aligner_hf = HFAutoModel.from_pretrained(
            args.qwen_forced_aligner_model,
            dtype=qwen_dtype,
        )
        try:
            aligner_processor = AutoProcessor.from_pretrained(
                args.qwen_forced_aligner_model, fix_mistral_regex=True
            )
        except TypeError:
            aligner_processor = AutoProcessor.from_pretrained(
                args.qwen_forced_aligner_model
            )
        forced_aligner = Qwen3ForcedAligner(
            model=aligner_hf,
            processor=aligner_processor,
            aligner_processor=Qwen3ForceAlignProcessor(),
        )
        if resolved_device.startswith("cuda"):
            forced_aligner.model = forced_aligner.model.to(resolved_device)
            forced_aligner.device = next(forced_aligner.model.parameters()).device

    asr_model = Qwen3ASRModel(
        backend="transformers",
        model=asr_model_hf,
        processor=asr_processor,
        sampling_params=None,
        forced_aligner=forced_aligner,
        max_inference_batch_size=segment_batch_size,
        max_new_tokens=args.qwen_max_new_tokens,
    )
    move_qwen_model_to_device(asr_model, resolved_device)
    if hasattr(asr_model.model, "generation_config"):
        eos_token_id = getattr(asr_model.model.generation_config, "eos_token_id", None)
        if isinstance(eos_token_id, list) and eos_token_id:
            asr_model.model.generation_config.pad_token_id = eos_token_id[-1]
    fsmn_vad_model: AutoModel | None = None
    firered_vad_model: Any | None = None
    if args.vad_backend == "fsmn":
        fsmn_vad_model = AutoModel(
            model="fsmn-vad",
            hub="ms",
            device=resolved_device,
            disable_update=True,
            max_single_segment_time=args.vad_max_segment_ms,
        )
    elif args.vad_backend == "firered":
        firered_src_dir = ensure_firered_source(
            Path(args.firered_src_dir),
            args.firered_repo_url,
        )
        if str(firered_src_dir) not in sys.path:
            sys.path.insert(0, str(firered_src_dir))
        from fireredasr2s.fireredvad import FireRedVad, FireRedVadConfig

        firered_vad_model_root = ensure_firered_vad_model(
            Path(args.firered_vad_model_dir),
            args.firered_vad_model_repo,
        )
        firered_vad_model_dir = resolve_firered_vad_dir(firered_vad_model_root)
        max_speech_frame = max(1, args.vad_max_segment_ms // 10)
        vad_cfg = FireRedVadConfig(
            use_gpu=resolved_device.startswith("cuda"),
            max_speech_frame=max_speech_frame,
        )
        firered_vad_model = FireRedVad.from_pretrained(
            str(firered_vad_model_dir),
            vad_cfg,
        )
    else:
        raise ValueError(f"Unsupported VAD backend: {args.vad_backend}")
    postprocessor = CantonesePostProcessor()

    if is_single_mode(args):
        audio_path = Path(args.audio)
        golden_srt_path = Path(args.golden_srt)
        output_srt_path = Path(args.output_srt)

        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not golden_srt_path.is_file():
            raise FileNotFoundError(f"Golden SRT file not found: {golden_srt_path}")

        result = run_one_file(
            audio_path=audio_path,
            reference_srt_path=golden_srt_path,
            output_srt_path=output_srt_path,
            asr_model=asr_model,
            vad_backend=args.vad_backend,
            fsmn_vad_model=fsmn_vad_model,
            firered_vad_model=firered_vad_model,
            postprocessor=postprocessor,
            min_segment_ms=args.min_segment_ms,
            vad_max_segment_ms=args.vad_max_segment_ms,
            segment_batch_size=segment_batch_size,
            qwen_language=args.qwen_language,
            use_forced_aligner=args.qwen_use_forced_aligner,
        )

        print("\nDone.")
        print(f"Output SRT: {result.output_srt_path}")
        print(f"Reference chars: {result.reference_chars}")
        print(f"Hypothesis chars: {result.hypothesis_chars}")
        print(f"Edit distance: {result.edit_distance}")
        print(
            f"CER (with punctuation): {result.cer:.6f}"
            if not math.isnan(result.cer)
            else "CER (with punctuation): NaN (empty reference)"
        )
        print(f"Reference chars (no punctuation): {result.reference_chars_no_punc}")
        print(f"Hypothesis chars (no punctuation): {result.hypothesis_chars_no_punc}")
        print(f"Edit distance (no punctuation): {result.edit_distance_no_punc}")
        print(
            f"CER (without punctuation): {result.cer_no_punc:.6f}"
            if not math.isnan(result.cer_no_punc)
            else "CER (without punctuation): NaN (empty reference)"
        )
        print(f"Audio duration (s): {result.audio_duration_sec:.3f}")
        print(f"Runtime (s): {result.runtime_sec:.3f}")
        print(f"ASR runtime only (s): {result.asr_runtime_sec:.3f}")
        print(
            f"End-to-end RTF: {result.rtf:.6f}"
            if not math.isnan(result.rtf)
            else "End-to-end RTF: NaN"
        )
        print(
            f"ASR-only RTF: {result.asr_rtf:.6f}"
            if not math.isnan(result.asr_rtf)
            else "ASR-only RTF: NaN"
        )
        print(f"Analysis report: {result.analysis_report_path}")
        return

    input_dir = Path(args.input_dir)
    reference_dir = Path(args.reference_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    if not reference_dir.is_dir():
        raise NotADirectoryError(f"Reference directory not found: {reference_dir}")

    extensions = parse_extensions(args.audio_extensions)
    audio_files = find_audio_files(input_dir, extensions)
    if not audio_files:
        raise RuntimeError(
            f"No audio files found in {input_dir} with extensions: {sorted(extensions)}"
        )

    print(f"\nBatch mode: found {len(audio_files)} audio files")
    results: list[EvalResult] = []
    missing_refs: list[Path] = []

    for audio_path in audio_files:
        reference_path = reference_dir / f"{audio_path.stem}.srt"
        if not reference_path.is_file():
            missing_refs.append(reference_path)
            print(f"Missing reference, skipped: {reference_path}")
            continue
        output_srt_path = output_dir / f"{audio_path.stem}.qwen3_asr_1_7b.srt"
        result = run_one_file(
            audio_path=audio_path,
            reference_srt_path=reference_path,
            output_srt_path=output_srt_path,
            asr_model=asr_model,
            vad_backend=args.vad_backend,
            fsmn_vad_model=fsmn_vad_model,
            firered_vad_model=firered_vad_model,
            postprocessor=postprocessor,
            min_segment_ms=args.min_segment_ms,
            vad_max_segment_ms=args.vad_max_segment_ms,
            segment_batch_size=segment_batch_size,
            qwen_language=args.qwen_language,
            use_forced_aligner=args.qwen_use_forced_aligner,
        )
        results.append(result)
        print(
            f"Result {audio_path.name}: CER={result.cer:.6f}, CER_no_punc={result.cer_no_punc:.6f} "
            f"(edit={result.edit_distance}, edit_no_punc={result.edit_distance_no_punc}, "
            f"runtime={result.runtime_sec:.3f}s, rtf={result.rtf:.6f}, "
            f"analysis={result.analysis_report_path.name})"
        )

    if missing_refs and args.strict_missing_reference:
        missing_str = "\n".join(str(p) for p in missing_refs)
        raise RuntimeError(f"Missing references:\n{missing_str}")
    if not results:
        raise RuntimeError("No files were evaluated (all references missing?)")

    total_ref_chars = sum(r.reference_chars for r in results)
    total_edit = sum(r.edit_distance for r in results)
    micro_cer = float("nan") if total_ref_chars == 0 else total_edit / total_ref_chars
    total_ref_chars_no_punc = sum(r.reference_chars_no_punc for r in results)
    total_edit_no_punc = sum(r.edit_distance_no_punc for r in results)
    micro_cer_no_punc = (
        float("nan")
        if total_ref_chars_no_punc == 0
        else total_edit_no_punc / total_ref_chars_no_punc
    )
    valid_cers = [r.cer for r in results if not math.isnan(r.cer)]
    macro_cer = float("nan") if not valid_cers else sum(valid_cers) / len(valid_cers)
    valid_cers_no_punc = [r.cer_no_punc for r in results if not math.isnan(r.cer_no_punc)]
    macro_cer_no_punc = (
        float("nan")
        if not valid_cers_no_punc
        else sum(valid_cers_no_punc) / len(valid_cers_no_punc)
    )
    total_audio_sec = sum(r.audio_duration_sec for r in results)
    total_runtime_sec = sum(r.runtime_sec for r in results)
    total_asr_runtime_sec = sum(r.asr_runtime_sec for r in results)
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

    print("\nBatch summary")
    for r in results:
        print(
            f"- {r.audio_path.name}: CER={r.cer:.6f}, CER_no_punc={r.cer_no_punc:.6f}, "
            f"runtime={r.runtime_sec:.3f}s, asr_runtime={r.asr_runtime_sec:.3f}s, "
            f"rtf={r.rtf:.6f}, asr_rtf={r.asr_rtf:.6f}, output={r.output_srt_path}"
        )
    print(f"Files evaluated: {len(results)}")
    print(f"Micro CER (with punctuation): {micro_cer:.6f}")
    print(f"Micro CER (without punctuation): {micro_cer_no_punc:.6f}")
    print(
        f"Macro CER (with punctuation): {macro_cer:.6f}"
        if not math.isnan(macro_cer)
        else "Macro CER (with punctuation): NaN"
    )
    print(
        f"Macro CER (without punctuation): {macro_cer_no_punc:.6f}"
        if not math.isnan(macro_cer_no_punc)
        else "Macro CER (without punctuation): NaN"
    )
    print(f"Total audio duration (s): {total_audio_sec:.3f}")
    print(f"Total runtime (s): {total_runtime_sec:.3f}")
    print(f"Total ASR runtime only (s): {total_asr_runtime_sec:.3f}")
    print(
        f"End-to-end RTF (batch): {overall_rtf:.6f}"
        if not math.isnan(overall_rtf)
        else "End-to-end RTF (batch): NaN"
    )
    print(
        f"ASR-only RTF (batch): {overall_asr_rtf:.6f}"
        if not math.isnan(overall_asr_rtf)
        else "ASR-only RTF (batch): NaN"
    )
    batch_analysis_path = resolve_summary_path(
        summary_dir=Path(args.summary_dir),
        summary_name=args.summary_name,
    )
    write_batch_analysis_summary(
        output_path=batch_analysis_path,
        results=results,
        micro_cer=micro_cer,
        micro_cer_no_punc=micro_cer_no_punc,
        macro_cer=macro_cer,
        macro_cer_no_punc=macro_cer_no_punc,
    )
    print(f"Output dir: {output_dir}")
    print(f"Batch summary report: {batch_analysis_path}")


if __name__ == "__main__":
    main()
