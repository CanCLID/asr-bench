# ASR Batch Analysis Summary

## Metrics
- Files evaluated: 20
- Micro CER (with punctuation): 0.211372
- Micro CER (without punctuation): 0.112464
- Macro CER (with punctuation): 0.207522
- Macro CER (without punctuation): 0.108761
- Total audio duration (s): 30831.959
- Total runtime (s): 761.631
- Total ASR runtime only (s): 661.174
- Average runtime per file (s): 38.082
- Average audio duration per file (s): 1541.598
- End-to-end RTF (batch): 0.024703
- ASR-only RTF (batch): 0.021444

## Error Totals (No Punctuation)
- Substitutions: 9156
- Deletions: 1238
- Insertions: 477

## Per-file Metrics

| File | CER | CER (No Punc) | Runtime (s) | ASR Runtime (s) | RTF | ASR RTF | Analysis Report |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ldg001.opus` | 0.223471 | 0.124089 | 35.947 | 31.706 | 0.025645 | 0.022619 | `ldg001.fireredasr2_aed.analysis.md` |
| `ldg002.opus` | 0.205671 | 0.117927 | 32.276 | 28.105 | 0.022949 | 0.019983 | `ldg002.fireredasr2_aed.analysis.md` |
| `ldg003.opus` | 0.217591 | 0.127533 | 33.745 | 29.545 | 0.023960 | 0.020979 | `ldg003.fireredasr2_aed.analysis.md` |
| `ldg004.opus` | 0.202863 | 0.101228 | 32.872 | 28.750 | 0.023412 | 0.020476 | `ldg004.fireredasr2_aed.analysis.md` |
| `ldg005.opus` | 0.229223 | 0.138721 | 35.986 | 31.837 | 0.025392 | 0.022465 | `ldg005.fireredasr2_aed.analysis.md` |
| `mzd001.opus` | 0.154425 | 0.055120 | 25.689 | 21.689 | 0.020784 | 0.017547 | `mzd001.fireredasr2_aed.analysis.md` |
| `mzd002.opus` | 0.173702 | 0.069988 | 25.062 | 21.201 | 0.020712 | 0.017521 | `mzd002.fireredasr2_aed.analysis.md` |
| `mzd003.opus` | 0.170650 | 0.070547 | 27.644 | 23.789 | 0.023056 | 0.019841 | `mzd003.fireredasr2_aed.analysis.md` |
| `mzd004.opus` | 0.166231 | 0.069743 | 28.890 | 24.901 | 0.023738 | 0.020461 | `mzd004.fireredasr2_aed.analysis.md` |
| `mzd005.opus` | 0.160155 | 0.061527 | 28.106 | 24.130 | 0.022720 | 0.019506 | `mzd005.fireredasr2_aed.analysis.md` |
| `sgjj001.opus` | 0.215025 | 0.114276 | 40.913 | 34.609 | 0.022115 | 0.018707 | `sgjj001.fireredasr2_aed.analysis.md` |
| `sgjj002.opus` | 0.216509 | 0.125770 | 38.713 | 32.677 | 0.021933 | 0.018514 | `sgjj002.fireredasr2_aed.analysis.md` |
| `sgjj003.opus` | 0.223594 | 0.120686 | 43.670 | 37.523 | 0.024883 | 0.021380 | `sgjj003.fireredasr2_aed.analysis.md` |
| `sgjj004.opus` | 0.218302 | 0.118140 | 39.532 | 33.222 | 0.022208 | 0.018664 | `sgjj004.fireredasr2_aed.analysis.md` |
| `sgjj005.opus` | 0.211860 | 0.102041 | 41.882 | 35.835 | 0.023529 | 0.020131 | `sgjj005.fireredasr2_aed.analysis.md` |
| `swz001.opus` | 0.216902 | 0.116495 | 34.236 | 28.720 | 0.020231 | 0.016971 | `swz001.fireredasr2_aed.analysis.md` |
| `swz002.opus` | 0.229658 | 0.130806 | 93.171 | 87.325 | 0.052630 | 0.049328 | `swz002.fireredasr2_aed.analysis.md` |
| `swz003.opus` | 0.223106 | 0.120297 | 39.352 | 33.600 | 0.022518 | 0.019227 | `swz003.fireredasr2_aed.analysis.md` |
| `swz004.opus` | 0.246266 | 0.147228 | 44.642 | 38.652 | 0.024781 | 0.021456 | `swz004.fireredasr2_aed.analysis.md` |
| `swz005.opus` | 0.245245 | 0.143060 | 39.305 | 33.357 | 0.022418 | 0.019026 | `swz005.fireredasr2_aed.analysis.md` |

## Top Substitution Patterns (All Files)

| Rank | Ref -> Hyp | Count |
| --- | --- | ---: |
| 1 | `噉 -> 咁` | 447 |
| 2 | `喇 -> 啦` | 334 |
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
| 13 | `嘅` | 16 |
| 14 | `就` | 14 |
| 15 | `個` | 14 |
| 16 | `嚱` | 13 |
| 17 | `呢` | 11 |
| 18 | `便` | 11 |
| 19 | `喇` | 10 |
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
| 3 | `嘅` | 24 |
| 4 | `係` | 23 |
| 5 | `啦` | 21 |
| 6 | `一` | 6 |
| 7 | `佢` | 6 |
| 8 | `咁` | 6 |
| 9 | `都` | 5 |
| 10 | `又` | 5 |
| 11 | `好` | 4 |
| 12 | `人` | 4 |
| 13 | `呢` | 4 |
| 14 | `嘢` | 4 |
| 15 | `去` | 4 |
| 16 | `㗎` | 3 |
| 17 | `了` | 3 |
| 18 | `個` | 3 |
| 19 | `哈` | 3 |
| 20 | `嚟` | 3 |
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
