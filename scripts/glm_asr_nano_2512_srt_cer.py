#!/usr/bin/env python3
"""Run GLM-ASR-Nano-2512, export SRT, and compute CER (single file or batch)."""

from __future__ import annotations

import argparse
import math
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from funasr import AutoModel
from transformers import AutoModel as HfAutoModel
from transformers import AutoProcessor

from common.batch_summary import write_batch_analysis_summary
from common.cantonese_postprocess import CantonesePostProcessor
from common.cer_utils import compute_cer, sequence_for_cer
from common.error_analysis import analyze_char_errors, build_file_analysis_markdown
from common.io_utils import (
    extract_segment_to_wav,
    find_audio_files,
    get_audio_duration_sec,
    parse_extensions,
    write_srt,
)
from common.text_utils import clean_asr_text, parse_srt_text


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GLM-ASR-Nano-2512 transcription -> SRT + CER vs golden SRT"
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
        default="predicted",
        help='Output SRT directory (batch mode). Default: "predicted"',
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
        "--segment-batch-size",
        type=int,
        default=0,
        help="ASR batch size for VAD segments. 0 = auto (GPU:128, CPU:4).",
    )
    parser.add_argument(
        "--glm-model",
        default="zai-org/GLM-ASR-Nano-2512",
        help='Hugging Face model id or local path. Default: "zai-org/GLM-ASR-Nano-2512".',
    )
    parser.add_argument(
        "--glm-prompt",
        default="Please transcribe this audio into Cantonese Chinese text.",
        help='Prompt passed to GLM-ASR transcription API. Default asks for Cantonese transcription.',
    )
    parser.add_argument(
        "--glm-dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help='Model dtype. "auto" uses bfloat16 on CUDA and float32 on CPU.',
    )
    parser.add_argument(
        "--glm-max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per segment batch.",
    )
    parser.add_argument(
        "--summary-dir",
        default="summary",
        help='Batch summary output directory. Default: "summary".',
    )
    parser.add_argument(
        "--summary-name",
        default="glm_asr_nano_2512",
        help='Batch summary filename stem (or .md name). Default: "glm_asr_nano_2512".',
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
    name = summary_name.strip() or "glm_asr_nano_2512"
    if not name.lower().endswith(".md"):
        name = f"{name}.md"
    summary_dir.mkdir(parents=True, exist_ok=True)
    return summary_dir / name


def resolve_glm_dtype(dtype_arg: str, device: str) -> torch.dtype:
    if dtype_arg == "auto":
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_arg}")


class GlmAsrWrapper:
    def __init__(
        self,
        model_id_or_path: str,
        device: str,
        dtype: torch.dtype,
        max_new_tokens: int,
    ) -> None:
        self.processor = AutoProcessor.from_pretrained(
            model_id_or_path, trust_remote_code=True
        )
        self.model = HfAutoModel.from_pretrained(
            model_id_or_path,
            dtype=dtype,
            trust_remote_code=True,
        )
        self.model = self.model.to(device)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.dtype = self.model.dtype
        self.max_new_tokens = max_new_tokens
        if hasattr(self.model, "generation_config"):
            eos_token_id = getattr(self.model.generation_config, "eos_token_id", None)
            if isinstance(eos_token_id, list) and eos_token_id:
                self.model.generation_config.pad_token_id = eos_token_id[0]

    def transcribe(self, audio: Sequence[str], prompt: str) -> list[str]:
        prompts = [prompt] * len(audio)
        inputs = self.processor.apply_transcription_request(audio=list(audio), prompt=prompts)
        inputs = inputs.to(self.device)
        inputs = inputs.to(dtype=self.dtype)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        decoded = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        return [str(text).strip() for text in decoded]


def run_one_file(
    audio_path: Path,
    reference_srt_path: Path,
    output_srt_path: Path,
    asr_model: GlmAsrWrapper,
    vad_model: AutoModel,
    postprocessor: CantonesePostProcessor,
    min_segment_ms: int,
    segment_batch_size: int,
    glm_prompt: str,
) -> EvalResult:
    print(f"\n=== Processing: {audio_path.name} ===")
    file_start_ts = time.perf_counter()
    audio_duration_sec = get_audio_duration_sec(audio_path)
    asr_runtime_sec = 0.0

    vad_res = vad_model.generate(input=str(audio_path))
    if not vad_res or "value" not in vad_res[0]:
        raise RuntimeError(f"Unexpected VAD output for {audio_path}: {vad_res}")
    raw_segments = vad_res[0]["value"]
    segments = [
        (int(start), int(end))
        for start, end in raw_segments
        if int(end) - int(start) >= min_segment_ms
    ]
    print(f"VAD segments: {len(raw_segments)}; used: {len(segments)}")

    entries: list[tuple[int, int, str]] = []
    with tempfile.TemporaryDirectory(prefix="glm_asr_segments_") as tmpdir:
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
                prompt=glm_prompt,
            )
            asr_runtime_sec += time.perf_counter() - asr_start_ts
            if len(res) != len(batch_segments):
                raise RuntimeError(
                    f"ASR result size mismatch: got {len(res)}, expected {len(batch_segments)}"
                )

            for raw_text, (start_ms, end_ms) in zip(res, batch_segments):
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


def main() -> None:
    args = parse_args()
    validate_args(args)
    resolved_device = resolve_device(args.device)
    glm_dtype = resolve_glm_dtype(args.glm_dtype, resolved_device)
    segment_batch_size = resolve_segment_batch_size(
        resolved_device, args.segment_batch_size
    )
    print(f"Using device: {resolved_device}")
    print(f"GLM dtype: {glm_dtype}")
    print(f"Segment batch size: {segment_batch_size}")

    print("Loading models...")
    asr_model = GlmAsrWrapper(
        model_id_or_path=args.glm_model,
        device=resolved_device,
        dtype=glm_dtype,
        max_new_tokens=args.glm_max_new_tokens,
    )
    vad_model = AutoModel(
        model="fsmn-vad",
        hub="ms",
        device=resolved_device,
        disable_update=True,
        max_single_segment_time=args.vad_max_segment_ms,
    )
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
            vad_model=vad_model,
            postprocessor=postprocessor,
            min_segment_ms=args.min_segment_ms,
            segment_batch_size=segment_batch_size,
            glm_prompt=args.glm_prompt,
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
        output_srt_path = output_dir / f"{audio_path.stem}.glm_asr_nano_2512.srt"
        result = run_one_file(
            audio_path=audio_path,
            reference_srt_path=reference_path,
            output_srt_path=output_srt_path,
            asr_model=asr_model,
            vad_model=vad_model,
            postprocessor=postprocessor,
            min_segment_ms=args.min_segment_ms,
            segment_batch_size=segment_batch_size,
            glm_prompt=args.glm_prompt,
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
