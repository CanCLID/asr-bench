from __future__ import annotations

import re

import opencc


class CantonesePostProcessor:
    """Shared Cantonese post-processing for ASR outputs."""

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

