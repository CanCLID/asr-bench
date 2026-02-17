# ASR Batch Analysis Summary

## Metrics
- Files evaluated: 20
- Micro CER (with punctuation): 0.191024
- Micro CER (without punctuation): 0.130386
- Macro CER (with punctuation): 0.186819
- Macro CER (without punctuation): 0.126799
- Total audio duration (s): 30831.959
- Total runtime (s): 166.841
- Total ASR runtime only (s): 67.638
- Average runtime per file (s): 8.342
- Average audio duration per file (s): 1541.598
- End-to-end RTF (batch): 0.005411
- ASR-only RTF (batch): 0.002194

## Error Totals (No Punctuation)
- Substitutions: 11195
- Deletions: 938
- Insertions: 501

## Per-file Metrics

| File | CER | CER (No Punc) | Runtime (s) | ASR Runtime (s) | RTF | ASR RTF | Analysis Report |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ldg001.opus` | 0.198710 | 0.137015 | 8.539 | 4.144 | 0.006092 | 0.002957 | `ldg001.sensevoice.analysis.md` |
| `ldg002.opus` | 0.192659 | 0.140605 | 7.211 | 3.025 | 0.005127 | 0.002151 | `ldg002.sensevoice.analysis.md` |
| `ldg003.opus` | 0.204589 | 0.152271 | 7.417 | 3.217 | 0.005266 | 0.002284 | `ldg003.sensevoice.analysis.md` |
| `ldg004.opus` | 0.187023 | 0.124704 | 7.183 | 3.111 | 0.005116 | 0.002216 | `ldg004.sensevoice.analysis.md` |
| `ldg005.opus` | 0.205170 | 0.148573 | 7.571 | 3.463 | 0.005342 | 0.002444 | `ldg005.sensevoice.analysis.md` |
| `mzd001.opus` | 0.129769 | 0.071657 | 6.399 | 2.423 | 0.005177 | 0.001960 | `mzd001.sensevoice.analysis.md` |
| `mzd002.opus` | 0.141980 | 0.083525 | 6.432 | 2.558 | 0.005315 | 0.002114 | `mzd002.sensevoice.analysis.md` |
| `mzd003.opus` | 0.139340 | 0.084496 | 6.298 | 2.403 | 0.005253 | 0.002004 | `mzd003.sensevoice.analysis.md` |
| `mzd004.opus` | 0.151033 | 0.100239 | 6.454 | 2.430 | 0.005303 | 0.001997 | `mzd004.sensevoice.analysis.md` |
| `mzd005.opus` | 0.133882 | 0.080929 | 6.477 | 2.517 | 0.005236 | 0.002035 | `mzd005.sensevoice.analysis.md` |
| `sgjj001.opus` | 0.193381 | 0.127625 | 9.788 | 3.719 | 0.005291 | 0.002010 | `sgjj001.sensevoice.analysis.md` |
| `sgjj002.opus` | 0.186478 | 0.132454 | 9.456 | 3.573 | 0.005357 | 0.002024 | `sgjj002.sensevoice.analysis.md` |
| `sgjj003.opus` | 0.201495 | 0.138033 | 10.697 | 4.464 | 0.006095 | 0.002544 | `sgjj003.sensevoice.analysis.md` |
| `sgjj004.opus` | 0.197196 | 0.138632 | 10.252 | 3.979 | 0.005759 | 0.002235 | `sgjj004.sensevoice.analysis.md` |
| `sgjj005.opus` | 0.197449 | 0.129819 | 10.030 | 4.101 | 0.005635 | 0.002304 | `sgjj005.sensevoice.analysis.md` |
| `swz001.opus` | 0.201179 | 0.133253 | 8.661 | 3.364 | 0.005118 | 0.001988 | `swz001.sensevoice.analysis.md` |
| `swz002.opus` | 0.214733 | 0.149492 | 9.393 | 3.737 | 0.005306 | 0.002111 | `swz002.sensevoice.analysis.md` |
| `swz003.opus` | 0.207047 | 0.141848 | 9.199 | 3.551 | 0.005264 | 0.002032 | `swz003.sensevoice.analysis.md` |
| `swz004.opus` | 0.227359 | 0.161267 | 10.102 | 4.203 | 0.005608 | 0.002333 | `swz004.sensevoice.analysis.md` |
| `swz005.opus` | 0.225912 | 0.159546 | 9.282 | 3.655 | 0.005294 | 0.002084 | `swz005.sensevoice.analysis.md` |

## Top Substitution Patterns (All Files)

| Rank | Ref -> Hyp | Count |
| --- | --- | ---: |
| 1 | `噉 -> 咁` | 382 |
| 2 | `便 -> 邊` | 340 |
| 3 | `喇 -> 啦` | 322 |
| 4 | `嘞 -> 啦` | 301 |
| 5 | `返 -> 翻` | 220 |
| 6 | `進 -> 俊` | 200 |
| 7 | `呀 -> 啊` | 145 |
| 8 | `晒 -> 曬` | 106 |
| 9 | `個 -> 嘅` | 84 |
| 10 | `吖 -> 啊` | 82 |
| 11 | `就 -> 啊` | 75 |
| 12 | `王 -> 黃` | 65 |
| 13 | `魯 -> 老` | 62 |
| 14 | `伊 -> 醫` | 59 |
| 15 | `搵 -> 穩` | 58 |
| 16 | `文 -> 民` | 57 |
| 17 | `璜 -> 王` | 53 |
| 18 | `嘅 -> 㗎` | 51 |
| 19 | `喎 -> 噃` | 45 |
| 20 | `群 -> 羣` | 44 |
| 21 | `留 -> 劉` | 43 |
| 22 | `處 -> 樹` | 38 |
| 23 | `說 -> 説` | 33 |
| 24 | `城 -> 成` | 30 |
| 25 | `角 -> 國` | 30 |
| 26 | `之 -> 知` | 29 |
| 27 | `玄 -> 元` | 29 |
| 28 | `王 -> 皇` | 27 |
| 29 | `唧 -> 啫` | 26 |
| 30 | `隻 -> 只` | 26 |

## Top Deleted Characters (All Files)

| Rank | Reference Char | Count |
| --- | --- | ---: |
| 1 | `哈` | 84 |
| 2 | `嘞` | 79 |
| 3 | `喎` | 45 |
| 4 | `嘢` | 44 |
| 5 | `係` | 42 |
| 6 | `一` | 39 |
| 7 | `嘿` | 36 |
| 8 | `啊` | 26 |
| 9 | `誒` | 24 |
| 10 | `噉` | 24 |
| 11 | `去` | 19 |
| 12 | `個` | 19 |
| 13 | `就` | 18 |
| 14 | `喇` | 16 |
| 15 | `嘅` | 14 |
| 16 | `嗰` | 13 |
| 17 | `呢` | 13 |
| 18 | `嚱` | 12 |
| 19 | `哎` | 11 |
| 20 | `喺` | 9 |
| 21 | `噃` | 8 |
| 22 | `…` | 8 |
| 23 | `咗` | 7 |
| 24 | `佢` | 7 |
| 25 | `—` | 7 |
| 26 | `嘻` | 7 |
| 27 | `唥` | 6 |
| 28 | `唔` | 6 |
| 29 | `吖` | 6 |
| 30 | `都` | 5 |

## Top Inserted Characters (All Files)

| Rank | Hypothesis Char | Count |
| --- | --- | ---: |
| 1 | `啊` | 178 |
| 2 | `就` | 31 |
| 3 | `係` | 26 |
| 4 | `啦` | 25 |
| 5 | `嘅` | 24 |
| 6 | `一` | 13 |
| 7 | `嗯` | 9 |
| 8 | `呢` | 8 |
| 9 | `好` | 6 |
| 10 | `咁` | 6 |
| 11 | `又` | 6 |
| 12 | `家` | 5 |
| 13 | `唔` | 5 |
| 14 | `佢` | 4 |
| 15 | `都` | 3 |
| 16 | `個` | 3 |
| 17 | `要` | 3 |
| 18 | `以` | 3 |
| 19 | `而` | 3 |
| 20 | `嚟` | 3 |
| 21 | `嗰` | 3 |
| 22 | `有` | 3 |
| 23 | `嘞` | 3 |
| 24 | `唉` | 3 |
| 25 | `去` | 3 |
| 26 | `面` | 2 |
| 27 | `為` | 2 |
| 28 | `呀` | 2 |
| 29 | `哈` | 2 |
| 30 | `人` | 2 |
