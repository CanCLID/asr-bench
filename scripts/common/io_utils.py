from __future__ import annotations

import subprocess
from pathlib import Path


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


def parse_extensions(exts: str) -> set[str]:
    return {
        ext.strip().lower() if ext.strip().startswith(".") else f".{ext.strip().lower()}"
        for ext in exts.split(",")
        if ext.strip()
    }


def find_audio_files(input_dir: Path, extensions: set[str]) -> list[Path]:
    return [
        p
        for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in extensions
    ]


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
