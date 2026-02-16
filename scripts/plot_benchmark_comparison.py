#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATASET_SIZE_RE = re.compile(r"Latest run .*?,\s*(\d+)\s+files\):")
TABLE_ROW_RE = re.compile(
    r"^\|\s*`(?P<model>[^`]+)`\s*\|\s*`(?P<micro_cer>[0-9.]+)`\s*\|\s*`(?P<micro_cer_no_punc>[0-9.]+)`\s*\|\s*`(?P<macro_cer>[0-9.]+)`\s*\|\s*`(?P<macro_cer_no_punc>[0-9.]+)`\s*\|\s*`(?P<runtime_s>[0-9.]+)`\s*\|\s*`(?P<rtf>[0-9.]+)`\s*\|",
    re.MULTILINE,
)


def parse_readme_metrics(readme_path: Path) -> tuple[int, list[dict[str, float | str]]]:
    text = readme_path.read_text(encoding="utf-8")

    start = text.find("## Benchmark results")
    end = text.find("## Current optimizations")
    section = text[start:end] if start != -1 and end != -1 else text

    dataset_match = DATASET_SIZE_RE.search(section)
    if dataset_match is None:
        raise ValueError(f"Could not parse dataset size from {readme_path}")
    dataset_size = int(dataset_match.group(1))

    rows: list[dict[str, float | str]] = []
    for match in TABLE_ROW_RE.finditer(section):
        row = {
            "model": match.group("model"),
            "micro_cer": float(match.group("micro_cer")),
            "micro_cer_no_punc": float(match.group("micro_cer_no_punc")),
            "runtime_s": float(match.group("runtime_s")),
            "rtf": float(match.group("rtf")),
        }
        rows.append(row)

    if not rows:
        raise ValueError(f"Could not parse benchmark table rows from {readme_path}")

    return dataset_size, rows


def compact_model_name(full_name: str) -> str:
    return full_name.split("/")[-1]


def create_chart(output_path: Path, readme_path: Path) -> None:
    dataset_size, rows = parse_readme_metrics(readme_path)

    labels: list[str] = []
    micro_cer: list[float] = []
    micro_cer_no_punc: list[float] = []
    runtime_s: list[float] = []
    rtf: list[float] = []

    for row in rows:
        labels.append(compact_model_name(str(row["model"])))
        micro_cer.append(float(row["micro_cer"]))
        micro_cer_no_punc.append(float(row["micro_cer_no_punc"]))
        runtime_s.append(float(row["runtime_s"]))
        rtf.append(float(row["rtf"]))

    x = np.arange(len(labels))
    width = 0.38

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=200)

    bars_a = ax1.bar(
        x - width / 2,
        micro_cer,
        width,
        label="Micro CER (with punctuation)",
        color="#d1495b",
    )
    bars_b = ax1.bar(
        x + width / 2,
        micro_cer_no_punc,
        width,
        label="Micro CER (without punctuation)",
        color="#457b9d",
    )
    ax1.set_ylabel("CER (lower is better)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=10, ha="right")
    ax1.set_ylim(0, max(micro_cer) * 1.25)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.legend()

    for bars in (bars_a, bars_b):
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.005,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    bars_runtime = ax2.bar(x, runtime_s, width=0.6, color="#2a9d8f")
    ax2.set_ylabel("Total runtime (s, lower is faster)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=10, ha="right")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    for index, bar in enumerate(bars_runtime):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(runtime_s) * 0.015,
            f"{height:.1f}s\nRTF {rtf[index]:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle(f"Cantonese ASR Benchmark Comparison ({dataset_size} files)")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ASR benchmark comparison from README benchmark table."
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path("README.md"),
        help="Path to README file with benchmark table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/benchmark_comparison_20files.png"),
        help="Path to output image file.",
    )
    args = parser.parse_args()

    create_chart(args.output, args.readme)
    print(f"Saved benchmark chart to {args.output}")


if __name__ == "__main__":
    main()
