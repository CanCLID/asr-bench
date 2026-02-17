# ASR Batch Analysis Summary

## Metrics
- Files evaluated: 20
- Micro CER (with punctuation): 0.210085
- Micro CER (without punctuation): 0.110692
- Macro CER (with punctuation): 0.206295
- Macro CER (without punctuation): 0.107077
- Total audio duration (s): 30831.959
- Total runtime (s): 1429.858
- Total ASR runtime only (s): 1329.653
- Average runtime per file (s): 71.493
- Average audio duration per file (s): 1541.598
- End-to-end RTF (batch): 0.046376
- ASR-only RTF (batch): 0.043126

## Error Totals (No Punctuation)
- Substitutions: 9069
- Deletions: 1235
- Insertions: 395

## Per-file Metrics

| File | CER | CER (No Punc) | Runtime (s) | ASR Runtime (s) | RTF | ASR RTF | Analysis Report |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ldg001.opus` | 0.223055 | 0.123384 | 40.317 | 36.039 | 0.028763 | 0.025710 | `ldg001.fireredasr2_aed.analysis.md` |
| `ldg002.opus` | 0.202758 | 0.114039 | 61.787 | 57.644 | 0.043931 | 0.040986 | `ldg002.fireredasr2_aed.analysis.md` |
| `ldg003.opus` | 0.215296 | 0.124973 | 61.762 | 57.600 | 0.043854 | 0.040899 | `ldg003.fireredasr2_aed.analysis.md` |
| `ldg004.opus` | 0.201336 | 0.099289 | 51.977 | 47.717 | 0.037019 | 0.033985 | `ldg004.fireredasr2_aed.analysis.md` |
| `ldg005.opus` | 0.227966 | 0.136510 | 81.667 | 77.308 | 0.057626 | 0.054550 | `ldg005.fireredasr2_aed.analysis.md` |
| `mzd001.opus` | 0.154425 | 0.055120 | 25.222 | 21.195 | 0.020405 | 0.017147 | `mzd001.fireredasr2_aed.analysis.md` |
| `mzd002.opus` | 0.172934 | 0.069124 | 41.010 | 37.187 | 0.033891 | 0.030732 | `mzd002.fireredasr2_aed.analysis.md` |
| `mzd003.opus` | 0.169455 | 0.068938 | 40.433 | 36.576 | 0.033722 | 0.030505 | `mzd003.fireredasr2_aed.analysis.md` |
| `mzd004.opus` | 0.166231 | 0.069743 | 27.512 | 23.522 | 0.022606 | 0.019328 | `mzd004.fireredasr2_aed.analysis.md` |
| `mzd005.opus` | 0.159927 | 0.061271 | 43.627 | 39.637 | 0.035267 | 0.032042 | `mzd005.fireredasr2_aed.analysis.md` |
| `sgjj001.opus` | 0.213143 | 0.111784 | 88.327 | 82.216 | 0.047744 | 0.044440 | `sgjj001.fireredasr2_aed.analysis.md` |
| `sgjj002.opus` | 0.213522 | 0.122076 | 86.037 | 80.094 | 0.048745 | 0.045378 | `sgjj002.fireredasr2_aed.analysis.md` |
| `sgjj003.opus` | 0.223594 | 0.120686 | 112.245 | 106.172 | 0.063956 | 0.060496 | `sgjj003.fireredasr2_aed.analysis.md` |
| `sgjj004.opus` | 0.215940 | 0.114754 | 100.592 | 94.383 | 0.056511 | 0.053023 | `sgjj004.fireredasr2_aed.analysis.md` |
| `sgjj005.opus` | 0.213020 | 0.102986 | 72.816 | 66.880 | 0.040907 | 0.037572 | `sgjj005.fireredasr2_aed.analysis.md` |
| `swz001.opus` | 0.214222 | 0.112457 | 81.461 | 76.062 | 0.048136 | 0.044946 | `swz001.fireredasr2_aed.analysis.md` |
| `swz002.opus` | 0.228053 | 0.128266 | 189.786 | 184.003 | 0.107205 | 0.103938 | `swz002.fireredasr2_aed.analysis.md` |
| `swz003.opus` | 0.221547 | 0.118354 | 64.796 | 59.072 | 0.037077 | 0.033802 | `swz003.fireredasr2_aed.analysis.md` |
| `swz004.opus` | 0.245631 | 0.146328 | 75.145 | 69.011 | 0.041714 | 0.038309 | `swz004.fireredasr2_aed.analysis.md` |
| `swz005.opus` | 0.243842 | 0.141464 | 83.342 | 77.335 | 0.047535 | 0.044109 | `swz005.fireredasr2_aed.analysis.md` |

## Top Substitution Patterns (All Files)

| Rank | Ref -> Hyp | Count |
| --- | --- | ---: |
| 1 | `噉 -> 咁` | 450 |
| 2 | `喇 -> 啦` | 333 |
| 3 | `便 -> 邊` | 328 |
| 4 | `嘞 -> 啦` | 324 |
| 5 | `進 -> 俊` | 228 |
| 6 | `返 -> 翻` | 223 |
| 7 | `呀 -> 啊` | 148 |
| 8 | `晒 -> 曬` | 108 |
| 9 | `個 -> 嘅` | 88 |
| 10 | `吖 -> 啊` | 86 |
| 11 | `璜 -> 王` | 76 |
| 12 | `伊 -> 醫` | 73 |
| 13 | `就 -> 啊` | 70 |
| 14 | `留 -> 劉` | 66 |
| 15 | `王 -> 黃` | 62 |
| 16 | `魯 -> 老` | 59 |
| 17 | `搵 -> 穩` | 56 |
| 18 | `喎 -> 噃` | 51 |
| 19 | `嘅 -> 㗎` | 50 |
| 20 | `群 -> 羣` | 42 |
| 21 | `處 -> 樹` | 37 |
| 22 | `說 -> 説` | 34 |
| 23 | `城 -> 成` | 32 |
| 24 | `角 -> 國` | 30 |
| 25 | `玄 -> 元` | 29 |
| 26 | `羲 -> 熙` | 28 |
| 27 | `吓 -> 嚇` | 27 |
| 28 | `唧 -> 啫` | 26 |
| 29 | `隻 -> 只` | 26 |
| 30 | `兒 -> 衣` | 26 |

## Top Deleted Characters (All Files)

| Rank | Reference Char | Count |
| --- | --- | ---: |
| 1 | `嗰` | 426 |
| 2 | `哈` | 79 |
| 3 | `嘞` | 72 |
| 4 | `嘢` | 45 |
| 5 | `喎` | 42 |
| 6 | `嘿` | 39 |
| 7 | `係` | 34 |
| 8 | `一` | 32 |
| 9 | `啊` | 29 |
| 10 | `誒` | 26 |
| 11 | `去` | 18 |
| 12 | `噉` | 17 |
| 13 | `就` | 14 |
| 14 | `嘅` | 14 |
| 15 | `個` | 14 |
| 16 | `嚱` | 13 |
| 17 | `呢` | 11 |
| 18 | `便` | 11 |
| 19 | `喇` | 9 |
| 20 | `哎` | 9 |
| 21 | `噃` | 8 |
| 22 | `—` | 8 |
| 23 | `…` | 8 |
| 24 | `喺` | 7 |
| 25 | `起` | 7 |
| 26 | `咗` | 7 |
| 27 | `處` | 7 |
| 28 | `嘻` | 7 |
| 29 | `吖` | 6 |
| 30 | `好` | 5 |

## Top Inserted Characters (All Files)

| Rank | Hypothesis Char | Count |
| --- | --- | ---: |
| 1 | `啊` | 174 |
| 2 | `就` | 27 |
| 3 | `嘅` | 23 |
| 4 | `係` | 21 |
| 5 | `啦` | 19 |
| 6 | `又` | 6 |
| 7 | `一` | 6 |
| 8 | `呢` | 5 |
| 9 | `佢` | 4 |
| 10 | `好` | 3 |
| 11 | `㗎` | 3 |
| 12 | `個` | 3 |
| 13 | `人` | 3 |
| 14 | `哈` | 3 |
| 15 | `嚟` | 3 |
| 16 | `咁` | 3 |
| 17 | `嘢` | 3 |
| 18 | `去` | 3 |
| 19 | `嗯` | 3 |
| 20 | `都` | 2 |
| 21 | `以` | 2 |
| 22 | `嘞` | 2 |
| 23 | `正` | 2 |
| 24 | `袁` | 2 |
| 25 | `將` | 2 |
| 26 | `噃` | 2 |
| 27 | `虎` | 2 |
| 28 | `尊` | 1 |
| 29 | `時` | 1 |
| 30 | `而` | 1 |
