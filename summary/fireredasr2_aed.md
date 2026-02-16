# ASR Batch Analysis Summary

## Metrics
- Files evaluated: 20
- Micro CER (with punctuation): 0.211381
- Micro CER (without punctuation): 0.112433
- Macro CER (with punctuation): 0.207529
- Macro CER (without punctuation): 0.108727
- Total audio duration (s): 30831.959
- Total runtime (s): 760.629
- Total ASR runtime only (s): 660.890
- Average runtime per file (s): 38.031
- Average audio duration per file (s): 1541.598
- End-to-end RTF (batch): 0.024670
- ASR-only RTF (batch): 0.021435

## Error Totals (No Punctuation)
- Substitutions: 9161
- Deletions: 1244
- Insertions: 463

## Per-file Metrics

| File | CER | CER (No Punc) | Runtime (s) | ASR Runtime (s) | RTF | ASR RTF | Analysis Report |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ldg001.opus` | 0.223679 | 0.124089 | 35.893 | 31.712 | 0.025606 | 0.022624 | `ldg001.fireredasr2_aed.analysis.md` |
| `ldg002.opus` | 0.205865 | 0.118143 | 32.223 | 28.115 | 0.022911 | 0.019990 | `ldg002.fireredasr2_aed.analysis.md` |
| `ldg003.opus` | 0.217591 | 0.127533 | 33.680 | 29.548 | 0.023915 | 0.020981 | `ldg003.fireredasr2_aed.analysis.md` |
| `ldg004.opus` | 0.202863 | 0.101012 | 32.840 | 28.755 | 0.023389 | 0.020480 | `ldg004.fireredasr2_aed.analysis.md` |
| `ldg005.opus` | 0.229223 | 0.138721 | 35.981 | 31.847 | 0.025389 | 0.022472 | `ldg005.fireredasr2_aed.analysis.md` |
| `mzd001.opus` | 0.154425 | 0.055120 | 25.719 | 21.702 | 0.020808 | 0.017558 | `mzd001.fireredasr2_aed.analysis.md` |
| `mzd002.opus` | 0.173702 | 0.069988 | 25.031 | 21.210 | 0.020686 | 0.017529 | `mzd002.fireredasr2_aed.analysis.md` |
| `mzd003.opus` | 0.170411 | 0.070279 | 27.419 | 23.558 | 0.022867 | 0.019647 | `mzd003.fireredasr2_aed.analysis.md` |
| `mzd004.opus` | 0.166231 | 0.069743 | 28.847 | 24.908 | 0.023703 | 0.020466 | `mzd004.fireredasr2_aed.analysis.md` |
| `mzd005.opus` | 0.160155 | 0.061527 | 28.127 | 24.135 | 0.022737 | 0.019510 | `mzd005.fireredasr2_aed.analysis.md` |
| `sgjj001.opus` | 0.215339 | 0.114632 | 40.855 | 34.587 | 0.022084 | 0.018695 | `sgjj001.fireredasr2_aed.analysis.md` |
| `sgjj002.opus` | 0.216509 | 0.125770 | 38.692 | 32.666 | 0.021922 | 0.018507 | `sgjj002.fireredasr2_aed.analysis.md` |
| `sgjj003.opus` | 0.223269 | 0.120317 | 43.637 | 37.510 | 0.024864 | 0.021373 | `sgjj003.fireredasr2_aed.analysis.md` |
| `sgjj004.opus` | 0.218302 | 0.118140 | 39.480 | 33.218 | 0.022179 | 0.018661 | `sgjj004.fireredasr2_aed.analysis.md` |
| `sgjj005.opus` | 0.211529 | 0.101474 | 41.743 | 35.826 | 0.023451 | 0.020127 | `sgjj005.fireredasr2_aed.analysis.md` |
| `swz001.opus` | 0.216902 | 0.116293 | 34.197 | 28.712 | 0.020207 | 0.016966 | `swz001.fireredasr2_aed.analysis.md` |
| `swz002.opus` | 0.229819 | 0.130987 | 93.130 | 87.303 | 0.052607 | 0.049315 | `swz002.fireredasr2_aed.analysis.md` |
| `swz003.opus` | 0.222950 | 0.120120 | 39.360 | 33.594 | 0.022523 | 0.019223 | `swz003.fireredasr2_aed.analysis.md` |
| `swz004.opus` | 0.246266 | 0.147228 | 44.621 | 38.632 | 0.024770 | 0.021445 | `swz004.fireredasr2_aed.analysis.md` |
| `swz005.opus` | 0.245557 | 0.143414 | 39.153 | 33.351 | 0.022332 | 0.019022 | `swz005.fireredasr2_aed.analysis.md` |

## Top Substitution Patterns (All Files)

| Rank | Ref -> Hyp | Count |
| --- | --- | ---: |
| 1 | `噉 -> 咁` | 447 |
| 2 | `喇 -> 啦` | 333 |
| 3 | `便 -> 邊` | 328 |
| 4 | `嘞 -> 啦` | 318 |
| 5 | `返 -> 翻` | 223 |
| 6 | `進 -> 俊` | 223 |
| 7 | `呀 -> 啊` | 147 |
| 8 | `晒 -> 曬` | 108 |
| 9 | `個 -> 嘅` | 87 |
| 10 | `吖 -> 啊` | 85 |
| 11 | `璜 -> 王` | 76 |
| 12 | `伊 -> 醫` | 73 |
| 13 | `就 -> 啊` | 71 |
| 14 | `留 -> 劉` | 66 |
| 15 | `王 -> 黃` | 62 |
| 16 | `魯 -> 老` | 60 |
| 17 | `搵 -> 穩` | 57 |
| 18 | `嘅 -> 㗎` | 50 |
| 19 | `喎 -> 噃` | 49 |
| 20 | `群 -> 羣` | 42 |
| 21 | `處 -> 樹` | 36 |
| 22 | `說 -> 説` | 34 |
| 23 | `城 -> 成` | 31 |
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
| 2 | `嘞` | 73 |
| 3 | `哈` | 68 |
| 4 | `嘢` | 45 |
| 5 | `喎` | 43 |
| 6 | `嘿` | 37 |
| 7 | `一` | 32 |
| 8 | `係` | 32 |
| 9 | `啊` | 27 |
| 10 | `誒` | 26 |
| 11 | `去` | 18 |
| 12 | `噉` | 18 |
| 13 | `嘅` | 17 |
| 14 | `就` | 14 |
| 15 | `個` | 14 |
| 16 | `嚱` | 13 |
| 17 | `呢` | 11 |
| 18 | `便` | 11 |
| 19 | `喇` | 11 |
| 20 | `哎` | 9 |
| 21 | `噃` | 8 |
| 22 | `處` | 8 |
| 23 | `—` | 8 |
| 24 | `…` | 8 |
| 25 | `喺` | 7 |
| 26 | `起` | 7 |
| 27 | `咗` | 7 |
| 28 | `嘻` | 7 |
| 29 | `吖` | 6 |
| 30 | `好` | 5 |

## Top Inserted Characters (All Files)

| Rank | Hypothesis Char | Count |
| --- | --- | ---: |
| 1 | `啊` | 171 |
| 2 | `就` | 30 |
| 3 | `係` | 23 |
| 4 | `嘅` | 23 |
| 5 | `啦` | 19 |
| 6 | `一` | 6 |
| 7 | `佢` | 6 |
| 8 | `咁` | 6 |
| 9 | `又` | 5 |
| 10 | `都` | 4 |
| 11 | `人` | 4 |
| 12 | `呢` | 4 |
| 13 | `嘢` | 4 |
| 14 | `好` | 3 |
| 15 | `㗎` | 3 |
| 16 | `了` | 3 |
| 17 | `個` | 3 |
| 18 | `哈` | 3 |
| 19 | `嚟` | 3 |
| 20 | `去` | 3 |
| 21 | `嗯` | 3 |
| 22 | `而` | 2 |
| 23 | `誒` | 2 |
| 24 | `為` | 2 |
| 25 | `以` | 2 |
| 26 | `飛` | 2 |
| 27 | `嘞` | 2 |
| 28 | `正` | 2 |
| 29 | `要` | 2 |
| 30 | `中` | 2 |
