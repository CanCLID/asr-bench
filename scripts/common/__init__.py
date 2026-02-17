"""Shared helpers for benchmark scripts."""

from .batch_summary import write_batch_analysis_summary
from .cantonese_postprocess import CantonesePostProcessor
from .cer_utils import compute_cer, levenshtein_distance, sequence_for_cer
from .error_analysis import (
    analyze_char_errors,
    build_counter_table,
    build_file_analysis_markdown,
    clip_text,
)
from .hallucination_guard import detect_repetition_loop, split_time_range_evenly
from .io_utils import (
    extract_segment_to_wav,
    find_audio_files,
    get_audio_duration_sec,
    ms_to_srt_time,
    parse_extensions,
    split_long_segments,
    transcode_audio_to_16k_wav,
    write_srt,
)
from .text_utils import clean_asr_text, parse_srt_text, preprocess_chinese_text

__all__ = [
    "CantonesePostProcessor",
    "analyze_char_errors",
    "build_counter_table",
    "build_file_analysis_markdown",
    "clean_asr_text",
    "clip_text",
    "compute_cer",
    "detect_repetition_loop",
    "extract_segment_to_wav",
    "find_audio_files",
    "get_audio_duration_sec",
    "levenshtein_distance",
    "ms_to_srt_time",
    "parse_srt_text",
    "parse_extensions",
    "preprocess_chinese_text",
    "sequence_for_cer",
    "split_long_segments",
    "split_time_range_evenly",
    "transcode_audio_to_16k_wav",
    "write_srt",
    "write_batch_analysis_summary",
]
