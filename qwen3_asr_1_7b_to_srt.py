#!/usr/bin/env python3
"""Minimal Qwen3-ASR-1.7B script: audio input -> Cantonese post-processed SRT."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import opencc
import torch
from funasr import AutoModel
from transformers import AutoModel as HFAutoModel
from transformers import AutoProcessor


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
        description="Transcribe one audio file with Qwen3-ASR-1.7B and output SRT."
    )
    parser.add_argument("--audio", required=True, help="Input audio path.")
    parser.add_argument(
        "--output-srt",
        default="",
        help="Output SRT path. Default: same stem as input with .qwen3_asr_1_7b.srt",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Inference device: "auto" (default), "cpu", or "cuda:0".',
    )
    parser.add_argument(
        "--segment-batch-size",
        type=int,
        default=0,
        help="ASR batch size for VAD segments. 0 = auto (GPU:128, CPU:4).",
    )
    parser.add_argument(
        "--min-segment-ms",
        type=int,
        default=300,
        help="Skip VAD segments shorter than this (ms).",
    )
    parser.add_argument(
        "--vad-max-segment-ms",
        type=int,
        default=10000,
        help="Maximum VAD segment duration in ms. Default: 10000 (10s).",
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
        "--qwen-language",
        default="Cantonese",
        help='Forced language for Qwen ASR. Default: "Cantonese".',
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
    return parser.parse_args()


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


def ensure_qwen_source(src_dir: Path, repo_url: str) -> Path:
    if (src_dir / "qwen_asr").is_dir():
        return src_dir
    src_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(src_dir)],
        check=True,
    )
    return src_dir


def ensure_qwen_transformers_compat() -> None:
    try:
        import transformers.utils.generic as generic
        from transformers import modeling_rope_utils
    except Exception:
        return

    if not hasattr(generic, "check_model_inputs"):

        def check_model_inputs(*_args: Any, **_kwargs: Any) -> Any:
            def decorator(func: Any) -> Any:
                return func

            return decorator

        generic.check_model_inputs = check_model_inputs  # type: ignore[attr-defined]

    if "default" in modeling_rope_utils.ROPE_INIT_FUNCTIONS:
        return

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


def clean_asr_text(text: str) -> str:
    text = re.sub(r"<\|[^>]+?\|>", "", text)
    return text.strip()


def ms_to_srt_time(ms: int) -> str:
    ms = max(0, int(ms))
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def write_srt(path: Path, entries: list[tuple[int, int, str]]) -> None:
    chunks = []
    for idx, (start_ms, end_ms, text) in enumerate(entries, start=1):
        chunks.append(
            f"{idx}\n{ms_to_srt_time(start_ms)} --> {ms_to_srt_time(end_ms)}\n{text}\n"
        )
    path.write_text("\n".join(chunks), encoding="utf-8")


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
    audio_path = Path(args.audio)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_srt_path = (
        Path(args.output_srt)
        if args.output_srt
        else audio_path.with_suffix(".qwen3_asr_1_7b.srt")
    )

    resolved_device = resolve_device(args.device)
    qwen_dtype = resolve_qwen_dtype(args.qwen_dtype, resolved_device)
    segment_batch_size = resolve_segment_batch_size(
        resolved_device, args.segment_batch_size
    )

    print(f"Using device: {resolved_device}")
    print(f"Qwen dtype: {qwen_dtype}")
    print(f"Segment batch size: {segment_batch_size}")

    qwen_src_dir = ensure_qwen_source(Path(args.qwen_src_dir), args.qwen_repo_url)
    if str(qwen_src_dir) not in sys.path:
        sys.path.insert(0, str(qwen_src_dir))
    ensure_qwen_transformers_compat()

    from qwen_asr import Qwen3ASRModel
    ensure_qwen_config_compat()
    ensure_qwen_modeling_compat()

    model = HFAutoModel.from_pretrained(
        args.qwen_model,
        dtype=qwen_dtype,
    )
    try:
        processor = AutoProcessor.from_pretrained(
            args.qwen_model, fix_mistral_regex=True
        )
    except TypeError:
        processor = AutoProcessor.from_pretrained(args.qwen_model)

    asr_model = Qwen3ASRModel(
        backend="transformers",
        model=model,
        processor=processor,
        sampling_params=None,
        forced_aligner=None,
        max_inference_batch_size=segment_batch_size,
        max_new_tokens=args.qwen_max_new_tokens,
    )
    move_qwen_model_to_device(asr_model, resolved_device)
    if hasattr(asr_model.model, "generation_config"):
        eos_token_id = getattr(asr_model.model.generation_config, "eos_token_id", None)
        if isinstance(eos_token_id, list) and eos_token_id:
            asr_model.model.generation_config.pad_token_id = eos_token_id[-1]

    vad_model = AutoModel(
        model="fsmn-vad",
        hub="ms",
        device=resolved_device,
        disable_update=True,
        max_single_segment_time=args.vad_max_segment_ms,
    )
    postprocessor = CantonesePostProcessor()

    vad_res = vad_model.generate(input=str(audio_path))
    if not vad_res or "value" not in vad_res[0]:
        raise RuntimeError(f"Unexpected VAD output for {audio_path}: {vad_res}")
    raw_segments = vad_res[0]["value"]
    segments = [
        (int(start), int(end))
        for start, end in raw_segments
        if int(end) - int(start) >= args.min_segment_ms
    ]
    print(f"VAD segments: {len(raw_segments)}; used: {len(segments)}")

    entries: list[tuple[int, int, str]] = []
    with tempfile.TemporaryDirectory(prefix="qwen3_segments_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for batch_start in range(0, len(segments), segment_batch_size):
            batch_segments = segments[batch_start : batch_start + segment_batch_size]
            batch_wavs: list[str] = []
            for seg_idx, (start_ms, end_ms) in enumerate(
                batch_segments, start=batch_start + 1
            ):
                seg_wav = tmpdir_path / f"seg_{seg_idx:05d}.wav"
                extract_segment_to_wav(audio_path, start_ms, end_ms, seg_wav)
                batch_wavs.append(str(seg_wav))

            results = asr_model.transcribe(audio=batch_wavs, language=args.qwen_language)
            if len(results) != len(batch_segments):
                raise RuntimeError(
                    f"ASR result size mismatch: got {len(results)}, expected {len(batch_segments)}"
                )

            for item, (start_ms, end_ms) in zip(results, batch_segments):
                raw_text = str(getattr(item, "text", ""))
                text = clean_asr_text(raw_text)
                text = postprocessor.apply(text)
                if text:
                    entries.append((start_ms, end_ms, text))

            done = min(batch_start + len(batch_segments), len(segments))
            print(f"Transcribed {done}/{len(segments)} segments")

    output_srt_path.parent.mkdir(parents=True, exist_ok=True)
    write_srt(output_srt_path, entries)
    print(f"Wrote SRT: {output_srt_path}")


if __name__ == "__main__":
    main()
