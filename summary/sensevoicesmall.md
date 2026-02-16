# ASR Batch Analysis Summary

## Metrics
- Files evaluated: 20
- Micro CER (with punctuation): 0.195603
- Micro CER (without punctuation): 0.132843
- Macro CER (with punctuation): 0.191163
- Macro CER (without punctuation): 0.129152
- Total audio duration (s): 30831.959
- Total runtime (s): 156.283
- Total ASR runtime only (s): 56.854
- Average runtime per file (s): 7.814
- Average audio duration per file (s): 1541.598
- End-to-end RTF (batch): 0.005069
- ASR-only RTF (batch): 0.001844

## Error Totals (No Punctuation)
- Substitutions: 11294
- Deletions: 1008
- Insertions: 571

## Per-file Metrics

| File | CER | CER (No Punc) | Runtime (s) | ASR Runtime (s) | RTF | ASR RTF | Analysis Report |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ldg001.opus` | 0.200999 | 0.138425 | 7.334 | 2.975 | 0.005232 | 0.002122 | `ldg001.sensevoice.analysis.md` |
| `ldg002.opus` | 0.194407 | 0.141037 | 6.773 | 2.670 | 0.004816 | 0.001898 | `ldg002.sensevoice.analysis.md` |
| `ldg003.opus` | 0.210707 | 0.156323 | 6.780 | 2.656 | 0.004814 | 0.001886 | `ldg003.sensevoice.analysis.md` |
| `ldg004.opus` | 0.191412 | 0.127288 | 6.760 | 2.661 | 0.004815 | 0.001895 | `ldg004.sensevoice.analysis.md` |
| `ldg005.opus` | 0.209837 | 0.151588 | 6.817 | 2.670 | 0.004810 | 0.001884 | `ldg005.sensevoice.analysis.md` |
| `mzd001.opus` | 0.129250 | 0.071076 | 6.369 | 2.386 | 0.005153 | 0.001931 | `mzd001.sensevoice.analysis.md` |
| `mzd002.opus` | 0.145817 | 0.086406 | 5.966 | 2.148 | 0.004930 | 0.001775 | `mzd002.sensevoice.analysis.md` |
| `mzd003.opus` | 0.143881 | 0.087983 | 5.980 | 2.132 | 0.004988 | 0.001778 | `mzd003.sensevoice.analysis.md` |
| `mzd004.opus` | 0.152220 | 0.100769 | 6.258 | 2.310 | 0.005142 | 0.001898 | `mzd004.sensevoice.analysis.md` |
| `mzd005.opus` | 0.135252 | 0.080929 | 6.288 | 2.240 | 0.005083 | 0.001811 | `mzd005.sensevoice.analysis.md` |
| `sgjj001.opus` | 0.199498 | 0.130295 | 9.492 | 3.200 | 0.005131 | 0.001730 | `sgjj001.sensevoice.analysis.md` |
| `sgjj002.opus` | 0.193239 | 0.135972 | 8.941 | 2.976 | 0.005065 | 0.001686 | `sgjj002.sensevoice.analysis.md` |
| `sgjj003.opus` | 0.204257 | 0.138586 | 9.444 | 3.382 | 0.005381 | 0.001927 | `sgjj003.sensevoice.analysis.md` |
| `sgjj004.opus` | 0.204442 | 0.142195 | 9.367 | 3.152 | 0.005262 | 0.001771 | `sgjj004.sensevoice.analysis.md` |
| `sgjj005.opus` | 0.203412 | 0.133598 | 9.308 | 3.413 | 0.005229 | 0.001917 | `sgjj005.sensevoice.analysis.md` |
| `swz001.opus` | 0.210827 | 0.137291 | 8.310 | 2.857 | 0.004910 | 0.001688 | `swz001.sensevoice.analysis.md` |
| `swz002.opus` | 0.218584 | 0.153483 | 8.976 | 3.208 | 0.005070 | 0.001812 | `swz002.sensevoice.analysis.md` |
| `swz003.opus` | 0.212192 | 0.144497 | 8.789 | 3.137 | 0.005029 | 0.001795 | `swz003.sensevoice.analysis.md` |
| `swz004.opus` | 0.231967 | 0.164327 | 9.400 | 3.481 | 0.005218 | 0.001932 | `swz004.sensevoice.analysis.md` |
| `swz005.opus` | 0.231057 | 0.160964 | 8.932 | 3.198 | 0.005095 | 0.001824 | `swz005.sensevoice.analysis.md` |

## Top Substitution Patterns (All Files)

| Rank | Ref -> Hyp | Count |
| --- | --- | ---: |
| 1 | `噉 -> 咁` | 379 |
| 2 | `便 -> 邊` | 340 |
| 3 | `喇 -> 啦` | 320 |
| 4 | `嘞 -> 啦` | 300 |
| 5 | `返 -> 翻` | 218 |
| 6 | `進 -> 俊` | 195 |
| 7 | `呀 -> 啊` | 145 |
| 8 | `晒 -> 曬` | 105 |
| 9 | `個 -> 嘅` | 84 |
| 10 | `吖 -> 啊` | 79 |
| 11 | `就 -> 啊` | 70 |
| 12 | `王 -> 黃` | 63 |
| 13 | `魯 -> 老` | 62 |
| 14 | `伊 -> 醫` | 59 |
| 15 | `搵 -> 穩` | 59 |
| 16 | `文 -> 民` | 57 |
| 17 | `嘅 -> 㗎` | 52 |
| 18 | `璜 -> 王` | 52 |
| 19 | `喎 -> 噃` | 45 |
| 20 | `群 -> 羣` | 44 |
| 21 | `留 -> 劉` | 44 |
| 22 | `處 -> 樹` | 38 |
| 23 | `說 -> 説` | 33 |
| 24 | `城 -> 成` | 32 |
| 25 | `角 -> 國` | 30 |
| 26 | `玄 -> 元` | 29 |
| 27 | `之 -> 知` | 28 |
| 28 | `王 -> 皇` | 27 |
| 29 | `隻 -> 只` | 26 |
| 30 | `武 -> 母` | 26 |

## Top Deleted Characters (All Files)

| Rank | Reference Char | Count |
| --- | --- | ---: |
| 1 | `嘞` | 79 |
| 2 | `哈` | 75 |
| 3 | `喎` | 45 |
| 4 | `嘢` | 44 |
| 5 | `係` | 43 |
| 6 | `一` | 40 |
| 7 | `嘿` | 36 |
| 8 | `啊` | 26 |
| 9 | `噉` | 24 |
| 10 | `誒` | 21 |
| 11 | `去` | 20 |
| 12 | `個` | 20 |
| 13 | `就` | 18 |
| 14 | `喇` | 16 |
| 15 | `嘅` | 15 |
| 16 | `嗰` | 13 |
| 17 | `呢` | 13 |
| 18 | `嚱` | 13 |
| 19 | `哎` | 10 |
| 20 | `喺` | 9 |
| 21 | `噃` | 9 |
| 22 | `咗` | 8 |
| 23 | `佢` | 8 |
| 24 | `唔` | 8 |
| 25 | `吖` | 8 |
| 26 | `—` | 7 |
| 27 | `嘻` | 7 |
| 28 | `…` | 7 |
| 29 | `唥` | 5 |
| 30 | `哋` | 5 |

## Top Inserted Characters (All Files)

| Rank | Hypothesis Char | Count |
| --- | --- | ---: |
| 1 | `啊` | 185 |
| 2 | `係` | 29 |
| 3 | `就` | 28 |
| 4 | `啦` | 28 |
| 5 | `嘅` | 27 |
| 6 | `嗯` | 15 |
| 7 | `一` | 11 |
| 8 | `呢` | 8 |
| 9 | `咁` | 7 |
| 10 | `唔` | 7 |
| 11 | `好` | 6 |
| 12 | `家` | 6 |
| 13 | `又` | 5 |
| 14 | `佢` | 5 |
| 15 | `誒` | 4 |
| 16 | `而` | 4 |
| 17 | `嚟` | 4 |
| 18 | `啲` | 4 |
| 19 | `嘢` | 4 |
| 20 | `去` | 4 |
| 21 | `都` | 3 |
| 22 | `個` | 3 |
| 23 | `要` | 3 |
| 24 | `以` | 3 |
| 25 | `為` | 3 |
| 26 | `嗰` | 3 |
| 27 | `你` | 3 |
| 28 | `有` | 3 |
| 29 | `咧` | 3 |
| 30 | `演` | 2 |
