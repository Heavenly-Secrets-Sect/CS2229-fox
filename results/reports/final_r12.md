# Fox reproduction -- results

All cells are `value [95% CI]` in percentage points. `†` marks a gap against the baseline that the sample size cannot resolve at 95% confidence: the two methods are indistinguishable on this evidence, whatever the point estimates say.

Prompt arm: `--system-prompt llava_v1`, `--wrapper released`, `chair_v1`, `pope_v0_raw`.

The tables below are **seed 42 alone** (of 11 present), because a single run is what their within-run intervals describe. The claim this report actually makes is in **Across seeds**, further down.

### Table 1 -- POPE (nucleus sampling, p=0.9, T=1.0)

| method | random Acc | random F1 | random yes-ratio | random n | popular Acc | popular F1 | popular yes-ratio | popular n | adversarial Acc | adversarial F1 | adversarial yes-ratio | adversarial n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `causalmm` | - | - | - | - | - | - | - | - | 77.70 [76.18, 79.15] † | 80.01 [78.51, 81.51] | 61.6 | 3000 |
| `fox` | 89.10 [87.93, 90.17] | 88.52 [87.28, 89.72] | 45.0 | 3000 | 86.93 [85.68, 88.09] | 86.53 [85.18, 87.80] | 47.0 | 3000 | 81.40 [79.97, 82.75] | 81.85 [80.34, 83.30] | 52.5 | 3000 |
| `icd` | - | - | - | - | - | - | - | - | 77.37 [75.83, 78.83] † | 79.44 [77.91, 80.94] | 60.1 | 3000 |
| `sampling` | 85.07 [83.75, 86.30] | 85.39 [84.05, 86.69] | 52.2 | 3000 | 82.10 [80.69, 83.43] | 82.93 [81.47, 84.32] | 54.8 | 3000 | 77.17 [75.63, 78.63] | 79.41 [77.85, 80.87] | 60.9 | 3000 |
| `sid` | - | - | - | - | - | - | - | - | 79.63 [78.15, 81.04] | 80.60 [79.05, 82.10] | 55.0 | 3000 |
| `vcd` | - | - | - | - | - | - | - | - | 75.03 [73.45, 76.55] | 77.86 [76.30, 79.41] | 62.8 | 3000 |
| `vcd_fox_fusion` | - | - | - | - | - | - | - | - | 78.83 [77.34, 80.26] † | 79.02 [77.39, 80.56] | 50.9 | 3000 |

Paper-reported accuracy (LLaVA-1.5), for reference:

| method | random | popular | adversarial |
| --- | --- | --- | --- |
| `sampling` | 85.13 | 82.53 | 76.77 |
| `vcd` | 86.84 | 80.87 | 75.33 |
| `icd` | 86.27 | 83.57 | 77.90 |
| `opera` | 89.27 | 86.80 | 81.13 |
| `causalmm` | 89.15 | 86.63 | 81.27 |
| `sid` | 88.67 | 85.37 | 81.10 |
| `fox` | 89.33 | 86.70 | 81.93 |

**Gaps against `sampling`** († = not resolvable at 95%):

- POPE/adversarial: causalmm vs sampling: gap +0.53 pts, 95% CI [-0.74, +1.80] -- NOT distinguishable at 95%  _[exact McNemar (discordant=382)]_
- POPE/random: fox vs sampling: gap +4.03 pts, 95% CI [+2.86, +5.12] -- distinguishable  _[exact McNemar (discordant=343)]_
- POPE/popular: fox vs sampling: gap +4.83 pts, 95% CI [+3.60, +5.97] -- distinguishable  _[exact McNemar (discordant=385)]_
- POPE/adversarial: fox vs sampling: gap +4.23 pts, 95% CI [+2.92, +5.47] -- distinguishable  _[exact McNemar (discordant=423)]_
- POPE/adversarial: icd vs sampling: gap +0.20 pts, 95% CI [-1.21, +1.61] -- NOT distinguishable at 95%  _[exact McNemar (discordant=468)]_
- POPE/adversarial: sid vs sampling: gap +2.47 pts, 95% CI [+1.06, +3.83] -- distinguishable  _[exact McNemar (discordant=466)]_
- POPE/adversarial: vcd vs sampling: gap -2.13 pts, 95% CI [-3.64, -0.60] -- distinguishable  _[exact McNemar (discordant=552)]_
- POPE/adversarial: vcd_fox_fusion vs sampling: gap +1.67 pts, 95% CI [+0.02, +3.30] -- NOT distinguishable at 95%  _[exact McNemar (discordant=638)]_


### Table 2 -- CHAIR (500 COCO val2014 images)

| method | CHAIR_S (lower better) | CHAIR_S n (captions) | CHAIR_I (lower better) | CHAIR_I n (mentions) | recall | F1 | mean words | correct objects |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `causalmm` | 50.20 [45.83, 54.56] | 500 | 14.90 [13.26, 16.68] | 3623 | 79.36 | 76.47 | 100.5 | 1238 |
| `fox` | 49.60 [45.24, 53.97] | 500 | 14.08 [12.49, 15.71] | 3829 | 76.35 | 75.31 | 96.3 | 1191 |
| `icd` | 52.00 [47.62, 56.35] † | 500 | 15.31 [13.79, 16.89] | 3605 | 77.05 | 75.01 | 102.3 | 1202 |
| `sampling` | 56.80 [52.42, 61.07] | 500 | 18.14 [16.56, 19.78] | 3599 | 76.09 | 72.40 | 102.5 | 1187 |
| `sid` | 53.00 [48.62, 57.34] † | 500 | 15.20 [13.73, 16.69] | 3578 | 77.12 | 74.63 | 98.8 | 1203 |
| `vcd` | 54.40 [50.02, 58.72] † | 500 | 16.80 [15.22, 18.41] † | 3626 | 77.56 | 74.05 | 100.7 | 1210 |
| `vcd_fox_fusion` | 41.60 [37.36, 45.97] | 500 | 14.84 [13.02, 16.67] | 3106 | 66.35 ⚠ | 70.99 | 95.3 | 1035 |

Paper-reported CHAIR (LLaVA-1.5), for reference:

| method | CHAIR_S | CHAIR_I |
| --- | --- | --- |
| `sampling` | 57.60 | 17.18 |
| `vcd` | 54.00 | 16.02 |
| `icd` | 53.40 | 15.67 |
| `opera` | 49.00 | 13.52 |
| `causalmm` | 47.60 | 13.49 |
| `sid` | 53.40 | 15.40 |
| `fox` | 46.40 | 12.90 |

_The paper states neither CHAIR_I's denominator (total mentioned object instances) nor any interval, so the reference column cannot be given error bars._

**Gaps against `sampling`** († = not resolvable at 95%):

- CHAIR_S: causalmm vs sampling: gap -6.60 pts, 95% CI [-11.20, -2.00] -- distinguishable  _[paired percentile bootstrap (B=10000, seed=20260101)]_
- CHAIR_I: causalmm vs sampling: gap -3.24 pts, 95% CI [-4.89, -1.56] -- distinguishable  _[paired cluster bootstrap on ratio of sums (B=10000, seed=20260101)]_
- CHAIR_S: fox vs sampling: gap -7.20 pts, 95% CI [-12.00, -2.40] -- distinguishable  _[paired percentile bootstrap (B=10000, seed=20260101)]_
- CHAIR_I: fox vs sampling: gap -4.07 pts, 95% CI [-5.79, -2.32] -- distinguishable  _[paired cluster bootstrap on ratio of sums (B=10000, seed=20260101)]_
- CHAIR_S: icd vs sampling: gap -4.80 pts, 95% CI [-9.60, +0.00] -- NOT distinguishable at 95%  _[paired percentile bootstrap (B=10000, seed=20260101)]_
- CHAIR_I: icd vs sampling: gap -2.83 pts, 95% CI [-4.44, -1.24] -- distinguishable  _[paired cluster bootstrap on ratio of sums (B=10000, seed=20260101)]_
- CHAIR_S: sid vs sampling: gap -3.80 pts, 95% CI [-8.60, +1.00] -- NOT distinguishable at 95%  _[paired percentile bootstrap (B=10000, seed=20260101)]_
- CHAIR_I: sid vs sampling: gap -2.94 pts, 95% CI [-4.58, -1.29] -- distinguishable  _[paired cluster bootstrap on ratio of sums (B=10000, seed=20260101)]_
- CHAIR_S: vcd vs sampling: gap -2.40 pts, 95% CI [-7.40, +2.60] -- NOT distinguishable at 95%  _[paired percentile bootstrap (B=10000, seed=20260101)]_
- CHAIR_I: vcd vs sampling: gap -1.35 pts, 95% CI [-3.03, +0.30] -- NOT distinguishable at 95%  _[paired cluster bootstrap on ratio of sums (B=10000, seed=20260101)]_
- CHAIR_S: vcd_fox_fusion vs sampling: gap -15.20 pts, 95% CI [-20.40, -10.00] -- distinguishable  _[paired percentile bootstrap (B=10000, seed=20260101)]_
- CHAIR_I: vcd_fox_fusion vs sampling: gap -3.30 pts, 95% CI [-5.26, -1.38] -- distinguishable  _[paired cluster bootstrap on ratio of sums (B=10000, seed=20260101)]_
- ⚠ `vcd_fox_fusion`: recall 66.35 against the baseline's 76.09, and 1035 objects named correctly against 1187. Its CHAIR scores are **not** evidence of better grounding: a caption that names fewer objects hallucinates fewer.


### Table 9a -- POPE under greedy decoding (control)

| method | random Acc | random F1 | random yes-ratio | random n | popular Acc | popular F1 | popular yes-ratio | popular n | adversarial Acc | adversarial F1 | adversarial yes-ratio | adversarial n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `causalmm` | - | - | - | - | - | - | - | - | 76.67 [75.12, 78.15] | 79.50 [77.99, 80.97] | 63.8 | 3000 |
| `fox` | 89.53 [88.39, 90.58] † | 88.96 [87.71, 90.14] | 44.8 | 3000 | 86.90 [85.65, 88.06] | 86.51 [85.15, 87.80] | 47.1 | 3000 | 82.37 [80.96, 83.69] | 82.65 [81.15, 84.10] | 51.6 | 3000 |
| `icd` | - | - | - | - | - | - | - | - | 78.60 [77.10, 80.03] † | 80.41 [78.88, 81.88] | 59.3 | 3000 |
| `opera` | 89.37 [88.21, 90.42] † | 89.32 [88.13, 90.46] | 49.6 | 3000 | 85.93 [84.64, 87.13] † | 86.33 [85.00, 87.59] | 52.9 | 3000 | 79.10 [77.61, 80.52] † | 80.95 [79.48, 82.39] | 59.7 | 3000 |
| `sampling` | 89.33 [88.18, 90.39] | 89.28 [88.09, 90.42] | 49.5 | 3000 | 85.90 [84.61, 87.10] | 86.29 [84.97, 87.55] | 52.8 | 3000 | 79.03 [77.54, 80.45] | 80.89 [79.41, 82.34] | 59.7 | 3000 |
| `sid` | - | - | - | - | - | - | - | - | 80.23 [78.77, 81.62] | 81.29 [79.78, 82.75] | 55.6 | 3000 |
| `vcd` | - | - | - | - | - | - | - | - | 76.93 [75.39, 78.41] | 79.08 [77.53, 80.59] | 60.3 | 3000 |

Paper-reported accuracy (LLaVA-1.5), for reference:

| method | random | popular | adversarial |
| --- | --- | --- | --- |
| `sampling` | 89.33 | 85.93 | 79.10 |
| `vcd` | 87.97 | 85.33 | 78.43 |
| `opera` | 89.27 | 86.80 | 81.13 |
| `sid` | 89.33 | 86.47 | 81.77 |
| `fox` | 89.43 | 87.07 | 82.43 |

**Gaps against `sampling`** († = not resolvable at 95%):

- POPE/adversarial: causalmm vs sampling: gap -2.37 pts, 95% CI [-2.88, -1.71] -- distinguishable  _[exact McNemar (discordant=123)]_
- POPE/random: fox vs sampling: gap +0.20 pts, 95% CI [-0.57, +0.96] -- NOT distinguishable at 95%  _[exact McNemar (discordant=142)]_
- POPE/popular: fox vs sampling: gap +1.00 pts, 95% CI [+0.14, +1.82] -- distinguishable  _[exact McNemar (discordant=174)]_
- POPE/adversarial: fox vs sampling: gap +3.33 pts, 95% CI [+2.36, +4.21] -- distinguishable  _[exact McNemar (discordant=244)]_
- POPE/adversarial: icd vs sampling: gap -0.43 pts, 95% CI [-1.10, +0.26] -- NOT distinguishable at 95%  _[exact McNemar (discordant=113)]_
- POPE/random: opera vs sampling: gap +0.03 pts, 95% CI [-0.02, +0.03] -- NOT distinguishable at 95%  _[exact McNemar (discordant=1)]_
- POPE/popular: opera vs sampling: gap +0.03 pts, 95% CI [-0.06, +0.09] -- NOT distinguishable at 95%  _[exact McNemar (discordant=3)]_
- POPE/adversarial: opera vs sampling: gap +0.07 pts, 95% CI [-0.05, +0.12] -- NOT distinguishable at 95%  _[exact McNemar (discordant=4)]_
- POPE/adversarial: sid vs sampling: gap +1.20 pts, 95% CI [+0.44, +1.90] -- distinguishable  _[exact McNemar (discordant=138)]_
- POPE/adversarial: vcd vs sampling: gap -2.10 pts, 95% CI [-2.94, -1.18] -- distinguishable  _[exact McNemar (discordant=203)]_


### Table 9b -- CHAIR under greedy decoding (control)

| method | CHAIR_S (lower better) | CHAIR_S n (captions) | CHAIR_I (lower better) | CHAIR_I n (mentions) | recall | F1 | mean words | correct objects |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fox` | 46.20 [41.88, 50.58] † | 500 | 13.45 [11.81, 15.21] † | 3896 | 77.82 | 76.62 | 97.3 | 1214 |
| `icd` | 49.00 [44.64, 53.37] † | 500 | 13.48 [11.98, 14.97] † | 3747 | 79.17 | 77.09 | 102.3 | 1235 |
| `sampling` | 49.20 [44.84, 53.57] | 500 | 13.82 [12.28, 15.39] | 3762 | 79.23 | 76.60 | 100.5 | 1236 |
| `sid` | 49.00 [44.64, 53.37] † | 500 | 12.83 [11.47, 14.29] † | 3686 | 80.96 | 78.47 | 97.2 | 1263 |
| `vcd` | 50.40 [46.03, 54.76] † | 500 | 14.84 [13.33, 16.35] † | 3700 | 78.46 | 75.39 | 100.5 | 1224 |
| `fox` | 41.40 [37.16, 45.77] | 500 | 11.91 [10.45, 13.38] † | 3914 | 76.76 | 76.91 | 95.8 | 1209 |
| `sampling` | 46.40 [42.07, 50.78] | 500 | 12.96 [11.53, 14.40] | 3782 | 78.86 | 76.98 | 98.7 | 1242 |
| `fox` | 44.60 [40.30, 48.98] | 500 | 13.15 [11.57, 14.73] † | 3849 | 77.64 | 76.80 | 97.0 | 1177 |
| `sampling` | 49.40 [45.04, 53.77] | 500 | 14.39 [12.86, 15.91] | 3711 | 79.42 | 76.74 | 99.2 | 1204 |
| `fox` | 49.40 [45.04, 53.77] † | 500 | 14.81 [13.18, 16.54] † | 4017 | 77.69 | 75.92 | 97.0 | 1247 |
| `sampling` | 49.80 [45.44, 54.17] | 500 | 14.81 [13.23, 16.48] | 3775 | 78.63 | 76.35 | 100.3 | 1262 |
| `fox` | 44.00 [39.71, 48.38] † | 500 | 12.85 [11.29, 14.43] † | 3884 | 76.51 | 76.72 | 94.6 | 1251 |
| `sampling` | 47.60 [43.26, 51.98] | 500 | 13.16 [11.71, 14.61] | 3799 | 78.78 | 77.66 | 99.5 | 1288 |

Paper-reported CHAIR (LLaVA-1.5), for reference:

| method | CHAIR_S | CHAIR_I |
| --- | --- | --- |
| `sampling` | 50.90 | 15.40 |
| `vcd` | 54.60 | 16.60 |
| `opera` | 49.00 | 13.52 |
| `sid` | 48.00 | 13.60 |
| `fox` | 42.60 | 12.16 |

_The paper states neither CHAIR_I's denominator (total mentioned object instances) nor any interval, so the reference column cannot be given error bars._

**Gaps against `sampling`** († = not resolvable at 95%):

- CHAIR_S: fox vs sampling: gap -3.00 pts, 95% CI [-7.40, +1.40] -- NOT distinguishable at 95%  _[paired percentile bootstrap (B=10000, seed=20260101)]_
- CHAIR_I: fox vs sampling: gap -0.37 pts, 95% CI [-1.74, +0.99] -- NOT distinguishable at 95%  _[paired cluster bootstrap on ratio of sums (B=10000, seed=20260101)]_
- CHAIR_S: icd vs sampling: gap -0.20 pts, 95% CI [-3.60, +3.20] -- NOT distinguishable at 95%  _[paired percentile bootstrap (B=10000, seed=20260101)]_
- CHAIR_I: icd vs sampling: gap -0.34 pts, 95% CI [-1.37, +0.71] -- NOT distinguishable at 95%  _[paired cluster bootstrap on ratio of sums (B=10000, seed=20260101)]_
- CHAIR_S: sid vs sampling: gap -0.20 pts, 95% CI [-4.80, +4.60] -- NOT distinguishable at 95%  _[paired percentile bootstrap (B=10000, seed=20260101)]_
- CHAIR_I: sid vs sampling: gap -0.99 pts, 95% CI [-2.35, +0.40] -- NOT distinguishable at 95%  _[paired cluster bootstrap on ratio of sums (B=10000, seed=20260101)]_
- CHAIR_S: vcd vs sampling: gap +1.20 pts, 95% CI [-3.00, +5.40] -- NOT distinguishable at 95%  _[paired percentile bootstrap (B=10000, seed=20260101)]_
- CHAIR_I: vcd vs sampling: gap +1.02 pts, 95% CI [-0.33, +2.37] -- NOT distinguishable at 95%  _[paired cluster bootstrap on ratio of sums (B=10000, seed=20260101)]_
- CHAIR_S: fox vs sampling: gap -5.00 pts, 95% CI [-9.40, -0.60] -- distinguishable  _[paired percentile bootstrap (B=10000, seed=20260101)]_
- CHAIR_I: fox vs sampling: gap -1.05 pts, 95% CI [-2.37, +0.28] -- NOT distinguishable at 95%  _[paired cluster bootstrap on ratio of sums (B=10000, seed=20260101)]_
- CHAIR_S: fox vs sampling: gap -4.80 pts, 95% CI [-9.20, -0.40] -- distinguishable  _[paired percentile bootstrap (B=10000, seed=20260101)]_
- CHAIR_I: fox vs sampling: gap -1.24 pts, 95% CI [-2.71, +0.18] -- NOT distinguishable at 95%  _[paired cluster bootstrap on ratio of sums (B=10000, seed=20260101)]_
- CHAIR_S: fox vs sampling: gap -0.40 pts, 95% CI [-4.40, +3.60] -- NOT distinguishable at 95%  _[paired percentile bootstrap (B=10000, seed=20260101)]_
- CHAIR_I: fox vs sampling: gap +0.00 pts, 95% CI [-1.48, +1.47] -- NOT distinguishable at 95%  _[paired cluster bootstrap on ratio of sums (B=10000, seed=20260101)]_
- CHAIR_S: fox vs sampling: gap -3.60 pts, 95% CI [-8.00, +0.80] -- NOT distinguishable at 95%  _[paired percentile bootstrap (B=10000, seed=20260101)]_
- CHAIR_I: fox vs sampling: gap -0.31 pts, 95% CI [-1.58, +0.95] -- NOT distinguishable at 95%  _[paired cluster bootstrap on ratio of sums (B=10000, seed=20260101)]_


### Across seeds

`mean gap` pairs the two methods seed by seed and puts a Student-t interval on the mean of those paired differences. That is the only interval here that sees the generation draw; every interval in the tables above is computed from a single run and quantifies item difficulty alone.

`best gap` is the best seed of each method differenced. It is reported because the paper's aggregation is ambiguous -- C.2 says "the best results averaged over 10 independent runs" without saying what "best" ranges over. If it ranges over runs, `best gap` is the comparable figure and `mean gap` is not.

| artefact | method | metric | seeds | sampling mean+-sd | method mean+-sd | mean gap [95% CI] | p | p (Holm) | best gap | paper gap | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chair/val2014_500 (greedy) | `fox` | CHAIR_S | 1 | 49.20+-0.00 | 46.20+-0.00 | n/a -- deterministic | - | - | -3.00 | -8.30 | see table above |
| chair/val2014_500 (greedy) | `fox` | CHAIR_I | 1 | 13.82+-0.00 | 13.45+-0.00 | n/a -- deterministic | - | - | -0.37 | -3.24 | see table above |
| chair/val2014_500 (greedy) | `fox L=2-13` | CHAIR_S | 1 | 49.20+-0.00 | 42.80+-0.00 | n/a -- deterministic | - | - | -6.40 | - | see table above |
| chair/val2014_500 (greedy) | `fox L=2-13` | CHAIR_I | 1 | 13.82+-0.00 | 12.83+-0.00 | n/a -- deterministic | - | - | -0.99 | - | see table above |
| chair/val2014_500 (greedy) | `fox as_described` | CHAIR_S | 1 | 49.20+-0.00 | 50.60+-0.00 | n/a -- deterministic | - | - | +1.40 | - | see table above |
| chair/val2014_500 (greedy) | `fox as_described` | CHAIR_I | 1 | 13.82+-0.00 | 14.37+-0.00 | n/a -- deterministic | - | - | +0.54 | - | see table above |
| chair/val2014_500 (greedy) | `fox boost_visual g=4` | CHAIR_S | 1 | 49.20+-0.00 | 42.80+-0.00 | n/a -- deterministic | - | - | -6.40 | - | see table above |
| chair/val2014_500 (greedy) | `fox boost_visual g=4` | CHAIR_I | 1 | 13.82+-0.00 | 11.52+-0.00 | n/a -- deterministic | - | - | -2.30 | - | see table above |
| chair/val2014_500 (greedy) | `fox g=100` | CHAIR_S | 1 | 49.20+-0.00 | 49.40+-0.00 | n/a -- deterministic | - | - | +0.20 | - | see table above |
| chair/val2014_500 (greedy) | `fox g=100` | CHAIR_I | 1 | 13.82+-0.00 | 13.66+-0.00 | n/a -- deterministic | - | - | -0.17 | - | see table above |
| chair/val2014_500 (greedy) | `fox g=10000` | CHAIR_S | 1 | 49.20+-0.00 | 46.20+-0.00 | n/a -- deterministic | - | - | -3.00 | - | see table above |
| chair/val2014_500 (greedy) | `fox g=10000` | CHAIR_I | 1 | 13.82+-0.00 | 13.45+-0.00 | n/a -- deterministic | - | - | -0.37 | - | see table above |
| chair/val2014_500 (greedy) | `fox ld=bf16` | CHAIR_S | 1 | 49.20+-0.00 | 25.60+-0.00 | n/a -- deterministic | - | - | -23.60 | - | see table above |
| chair/val2014_500 (greedy) | `fox ld=bf16` | CHAIR_I | 1 | 13.82+-0.00 | 10.62+-0.00 | n/a -- deterministic | - | - | -3.20 | - | see table above |
| chair/val2014_500 (greedy) | `fox ld=fp32` | CHAIR_S | 1 | 49.20+-0.00 | 49.00+-0.00 | n/a -- deterministic | - | - | -0.20 | - | see table above |
| chair/val2014_500 (greedy) | `fox ld=fp32` | CHAIR_I | 1 | 13.82+-0.00 | 13.80+-0.00 | n/a -- deterministic | - | - | -0.02 | - | see table above |
| chair/val2014_500 (greedy) | `fox noise_control` | CHAIR_S | 1 | 49.20+-0.00 | 44.60+-0.00 | n/a -- deterministic | - | - | -4.60 | - | see table above |
| chair/val2014_500 (greedy) | `fox noise_control` | CHAIR_I | 1 | 13.82+-0.00 | 11.88+-0.00 | n/a -- deterministic | - | - | -1.95 | - | see table above |
| chair/val2014_500 (greedy) | `icd` | CHAIR_S | 1 | 49.20+-0.00 | 49.00+-0.00 | n/a -- deterministic | - | - | -0.20 | - | see table above |
| chair/val2014_500 (greedy) | `icd` | CHAIR_I | 1 | 13.82+-0.00 | 13.48+-0.00 | n/a -- deterministic | - | - | -0.34 | - | see table above |
| chair/val2014_500 (greedy) | `sid` | CHAIR_S | 1 | 49.20+-0.00 | 49.00+-0.00 | n/a -- deterministic | - | - | -0.20 | -2.90 | see table above |
| chair/val2014_500 (greedy) | `sid` | CHAIR_I | 1 | 13.82+-0.00 | 12.83+-0.00 | n/a -- deterministic | - | - | -0.99 | -1.80 | see table above |
| chair/val2014_500 (greedy) | `vcd` | CHAIR_S | 1 | 49.20+-0.00 | 50.40+-0.00 | n/a -- deterministic | - | - | +1.20 | +3.70 | see table above |
| chair/val2014_500 (greedy) | `vcd` | CHAIR_I | 1 | 13.82+-0.00 | 14.84+-0.00 | n/a -- deterministic | - | - | +1.02 | +1.20 | see table above |
| chair/val2014_500 (sampling) | `causalmm` | CHAIR_S | 3 | 54.40+-2.40 | 51.33+-1.03 | -3.07 [-10.99, +4.86] | 0.24 | 1 **ns** | -1.80 | -10.00 | not resolved after Holm |
| chair/val2014_500 (sampling) | `causalmm` | CHAIR_I | 3 | 16.90+-1.09 | 14.67+-0.22 | -2.23 [-4.49, +0.02] | 0.051 | 0.87 **ns** | -1.66 | -3.69 | not resolved after Holm |
| chair/val2014_500 (sampling) | `fox` | CHAIR_S | 10 | 53.72+-2.39 | 46.74+-2.12 | -6.98 [-9.25, -4.71] | 6.6e-05 | 0.0022 | -5.80 | -11.20 | **excludes paper** |
| chair/val2014_500 (sampling) | `fox` | CHAIR_I | 10 | 16.39+-0.86 | 13.72+-0.54 | -2.67 [-3.37, -1.96] | 1.3e-05 | 0.00044 | -2.02 | -4.28 | **excludes paper** |
| chair/val2014_500 (sampling) | `fox boost_visual g=4` | CHAIR_S | 10 | 53.72+-2.39 | 40.50+-0.96 | -13.22 [-14.82, -11.62] | 1.7e-08 | 6.7e-07 | -10.40 | - | resolved |
| chair/val2014_500 (sampling) | `fox boost_visual g=4` | CHAIR_I | 10 | 16.39+-0.86 | 11.58+-0.44 | -4.81 [-5.33, -4.29] | 6.4e-09 | 2.7e-07 | -3.98 | - | resolved |
| chair/val2014_500 (sampling) | `fox off` | CHAIR_S | 3 | 54.40+-2.40 | 49.20+-0.20 | -5.20 [-11.43, +1.03] | 0.069 | 0.97 **ns** | -3.00 | - | not resolved after Holm |
| chair/val2014_500 (sampling) | `fox off` | CHAIR_I | 3 | 16.90+-1.09 | 14.05+-0.25 | -2.85 [-5.15, -0.56] | 0.033 | 0.7 **ns** | -2.36 | - | not resolved after Holm |
| chair/val2014_500 (sampling) | `fox random_heads` | CHAIR_S | 3 | 54.40+-2.40 | 42.27+-2.04 | -12.13 [-19.37, -4.89] | 0.019 | 0.43 **ns** | -11.20 | - | not resolved after Holm |
| chair/val2014_500 (sampling) | `fox random_heads` | CHAIR_I | 3 | 16.90+-1.09 | 12.11+-0.96 | -4.79 [-8.67, -0.90] | 0.034 | 0.7 **ns** | -4.76 | - | not resolved after Holm |
| chair/val2014_500 (sampling) | `fox ulp_control` | CHAIR_S | 3 | 54.40+-2.40 | 45.53+-1.14 | -8.87 [-17.07, -0.67] | 0.043 | 0.82 **ns** | -7.40 | - | not resolved after Holm |
| chair/val2014_500 (sampling) | `fox ulp_control` | CHAIR_I | 3 | 16.90+-1.09 | 13.34+-0.62 | -3.56 [-7.45, +0.33] | 0.059 | 0.93 **ns** | -3.16 | - | not resolved after Holm |
| chair/val2014_500 (sampling) | `icd` | CHAIR_S | 3 | 54.40+-2.40 | 52.80+-0.72 | -1.60 [-9.31, +6.11] | 0.47 | 1 **ns** | +0.00 | -4.20 | not resolved after Holm |
| chair/val2014_500 (sampling) | `icd` | CHAIR_I | 3 | 16.90+-1.09 | 15.21+-0.12 | -1.69 [-4.22, +0.85] | 0.1 | 1 **ns** | -1.04 | -1.51 | not resolved after Holm |
| chair/val2014_500 (sampling) | `sid` | CHAIR_S | 3 | 54.40+-2.40 | 52.13+-1.17 | -2.27 [-8.44, +3.91] | 0.25 | 1 **ns** | -1.20 | -4.20 | not resolved after Holm |
| chair/val2014_500 (sampling) | `sid` | CHAIR_I | 3 | 16.90+-1.09 | 15.05+-0.77 | -1.85 [-5.11, +1.41] | 0.13 | 1 **ns** | -1.90 | -1.78 | not resolved after Holm |
| chair/val2014_500 (sampling) | `vcd` | CHAIR_S | 3 | 54.40+-2.40 | 55.13+-2.19 | +0.73 [-6.37, +7.84] | 0.7 | 1 **ns** | +1.40 | -3.60 | not resolved after Holm |
| chair/val2014_500 (sampling) | `vcd` | CHAIR_I | 3 | 16.90+-1.09 | 16.78+-0.26 | -0.12 [-2.77, +2.53] | 0.86 | 1 **ns** | +0.39 | -1.16 | not resolved after Holm |
| chair/val2014_500 (sampling) | `vcd_fox_fusion` | CHAIR_S | 3 | 54.40+-2.40 | 40.53+-1.36 | -13.87 [-20.04, -7.69] | 0.011 | 0.27 **ns** | -13.00 | - | not resolved after Holm |
| chair/val2014_500 (sampling) | `vcd_fox_fusion` | CHAIR_I | 3 | 16.90+-1.09 | 14.56+-0.28 | -2.34 [-4.54, -0.14] | 0.045 | 0.82 **ns** | -1.83 | - | not resolved after Holm |
| chair/val2014_500_d1 (greedy) | `fox` | CHAIR_S | 1 | 46.40+-0.00 | 41.40+-0.00 | n/a -- deterministic | - | - | -5.00 | -8.30 | see table above |
| chair/val2014_500_d1 (greedy) | `fox` | CHAIR_I | 1 | 12.96+-0.00 | 11.91+-0.00 | n/a -- deterministic | - | - | -1.05 | -3.24 | see table above |
| chair/val2014_500_d2 (greedy) | `fox` | CHAIR_S | 1 | 49.40+-0.00 | 44.60+-0.00 | n/a -- deterministic | - | - | -4.80 | -8.30 | see table above |
| chair/val2014_500_d2 (greedy) | `fox` | CHAIR_I | 1 | 14.39+-0.00 | 13.15+-0.00 | n/a -- deterministic | - | - | -1.24 | -3.24 | see table above |
| chair/val2014_500_d3 (greedy) | `fox` | CHAIR_S | 1 | 49.80+-0.00 | 49.40+-0.00 | n/a -- deterministic | - | - | -0.40 | -8.30 | see table above |
| chair/val2014_500_d3 (greedy) | `fox` | CHAIR_I | 1 | 14.81+-0.00 | 14.81+-0.00 | n/a -- deterministic | - | - | +0.00 | -3.24 | see table above |
| chair/val2014_500_d4 (greedy) | `fox` | CHAIR_S | 1 | 47.60+-0.00 | 44.00+-0.00 | n/a -- deterministic | - | - | -3.60 | -8.30 | see table above |
| chair/val2014_500_d4 (greedy) | `fox` | CHAIR_I | 1 | 13.16+-0.00 | 12.85+-0.00 | n/a -- deterministic | - | - | -0.31 | -3.24 | see table above |
| pope/adversarial (greedy) | `causalmm` | accuracy | 1 | 79.03+-0.00 | 76.67+-0.00 | n/a -- deterministic | - | - | -2.37 | - | see table above |
| pope/adversarial (greedy) | `causalmm` | F1 | 1 | 80.89+-0.00 | 79.50+-0.00 | n/a -- deterministic | - | - | -1.39 | - | see table above |
| pope/adversarial (greedy) | `causalmm alpha=0` | accuracy | 1 | 79.03+-0.00 | 79.40+-0.00 | n/a -- deterministic | - | - | +0.37 | - | see table above |
| pope/adversarial (greedy) | `causalmm alpha=0` | F1 | 1 | 80.89+-0.00 | 81.31+-0.00 | n/a -- deterministic | - | - | +0.42 | - | see table above |
| pope/adversarial (greedy) | `causalmm control=cf_blind` | accuracy | 1 | 79.03+-0.00 | 77.53+-0.00 | n/a -- deterministic | - | - | -1.50 | - | see table above |
| pope/adversarial (greedy) | `causalmm control=cf_blind` | F1 | 1 | 80.89+-0.00 | 79.99+-0.00 | n/a -- deterministic | - | - | -0.90 | - | see table above |
| pope/adversarial (greedy) | `causalmm control=cf_constant` | accuracy | 1 | 79.03+-0.00 | 76.47+-0.00 | n/a -- deterministic | - | - | -2.57 | - | see table above |
| pope/adversarial (greedy) | `causalmm control=cf_constant` | F1 | 1 | 80.89+-0.00 | 79.39+-0.00 | n/a -- deterministic | - | - | -1.49 | - | see table above |
| pope/adversarial (greedy) | `fox` | accuracy | 1 | 79.03+-0.00 | 82.37+-0.00 | n/a -- deterministic | - | - | +3.33 | +3.33 | see table above |
| pope/adversarial (greedy) | `fox` | F1 | 1 | 80.89+-0.00 | 82.65+-0.00 | n/a -- deterministic | - | - | +1.76 | +1.69 | see table above |
| pope/adversarial (greedy) | `fox L=2-13` | accuracy | 1 | 79.03+-0.00 | 82.20+-0.00 | n/a -- deterministic | - | - | +3.17 | - | see table above |
| pope/adversarial (greedy) | `fox L=2-13` | F1 | 1 | 80.89+-0.00 | 82.73+-0.00 | n/a -- deterministic | - | - | +1.84 | - | see table above |
| pope/adversarial (greedy) | `fox alpha=0` | accuracy | 1 | 79.03+-0.00 | 79.10+-0.00 | n/a -- deterministic | - | - | +0.07 | - | see table above |
| pope/adversarial (greedy) | `fox alpha=0` | F1 | 1 | 80.89+-0.00 | 80.95+-0.00 | n/a -- deterministic | - | - | +0.06 | - | see table above |
| pope/adversarial (greedy) | `fox alpha=1` | accuracy | 1 | 79.03+-0.00 | 81.83+-0.00 | n/a -- deterministic | - | - | +2.80 | - | see table above |
| pope/adversarial (greedy) | `fox alpha=1` | F1 | 1 | 80.89+-0.00 | 82.44+-0.00 | n/a -- deterministic | - | - | +1.55 | - | see table above |
| pope/adversarial (greedy) | `fox alpha=3` | accuracy | 1 | 79.03+-0.00 | 82.37+-0.00 | n/a -- deterministic | - | - | +3.33 | - | see table above |
| pope/adversarial (greedy) | `fox alpha=3` | F1 | 1 | 80.89+-0.00 | 82.51+-0.00 | n/a -- deterministic | - | - | +1.63 | - | see table above |
| pope/adversarial (greedy) | `fox alpha=4` | accuracy | 1 | 79.03+-0.00 | 82.37+-0.00 | n/a -- deterministic | - | - | +3.33 | - | see table above |
| pope/adversarial (greedy) | `fox alpha=4` | F1 | 1 | 80.89+-0.00 | 82.44+-0.00 | n/a -- deterministic | - | - | +1.56 | - | see table above |
| pope/adversarial (greedy) | `fox as_described` | accuracy | 1 | 79.03+-0.00 | 79.43+-0.00 | n/a -- deterministic | - | - | +0.40 | - | see table above |
| pope/adversarial (greedy) | `fox as_described` | F1 | 1 | 80.89+-0.00 | 81.13+-0.00 | n/a -- deterministic | - | - | +0.24 | - | see table above |
| pope/adversarial (greedy) | `fox beta=0.05` | accuracy | 1 | 79.03+-0.00 | 82.60+-0.00 | n/a -- deterministic | - | - | +3.57 | - | see table above |
| pope/adversarial (greedy) | `fox beta=0.05` | F1 | 1 | 80.89+-0.00 | 82.97+-0.00 | n/a -- deterministic | - | - | +2.09 | - | see table above |
| pope/adversarial (greedy) | `fox beta=0.2` | accuracy | 1 | 79.03+-0.00 | 82.60+-0.00 | n/a -- deterministic | - | - | +3.57 | - | see table above |
| pope/adversarial (greedy) | `fox beta=0.2` | F1 | 1 | 80.89+-0.00 | 82.97+-0.00 | n/a -- deterministic | - | - | +2.09 | - | see table above |
| pope/adversarial (greedy) | `fox boost_visual g=4` | accuracy | 1 | 79.03+-0.00 | 80.70+-0.00 | n/a -- deterministic | - | - | +1.67 | - | see table above |
| pope/adversarial (greedy) | `fox boost_visual g=4` | F1 | 1 | 80.89+-0.00 | 81.80+-0.00 | n/a -- deterministic | - | - | +0.91 | - | see table above |
| pope/adversarial (greedy) | `fox k=0.25` | accuracy | 1 | 79.03+-0.00 | 81.60+-0.00 | n/a -- deterministic | - | - | +2.57 | - | see table above |
| pope/adversarial (greedy) | `fox k=0.25` | F1 | 1 | 80.89+-0.00 | 82.17+-0.00 | n/a -- deterministic | - | - | +1.28 | - | see table above |
| pope/adversarial (greedy) | `fox k=0.45` | accuracy | 1 | 79.03+-0.00 | 81.80+-0.00 | n/a -- deterministic | - | - | +2.77 | - | see table above |
| pope/adversarial (greedy) | `fox k=0.45` | F1 | 1 | 80.89+-0.00 | 82.05+-0.00 | n/a -- deterministic | - | - | +1.16 | - | see table above |
| pope/adversarial (greedy) | `fox ld=bf16` | accuracy | 1 | 79.03+-0.00 | 81.97+-0.00 | n/a -- deterministic | - | - | +2.93 | - | see table above |
| pope/adversarial (greedy) | `fox ld=bf16` | F1 | 1 | 80.89+-0.00 | 81.91+-0.00 | n/a -- deterministic | - | - | +1.03 | - | see table above |
| pope/adversarial (greedy) | `fox ld=fp32` | accuracy | 1 | 79.03+-0.00 | 79.07+-0.00 | n/a -- deterministic | - | - | +0.03 | - | see table above |
| pope/adversarial (greedy) | `fox ld=fp32` | F1 | 1 | 80.89+-0.00 | 80.92+-0.00 | n/a -- deterministic | - | - | +0.04 | - | see table above |
| pope/adversarial (greedy) | `fox noise_control` | accuracy | 1 | 79.03+-0.00 | 80.07+-0.00 | n/a -- deterministic | - | - | +1.03 | - | see table above |
| pope/adversarial (greedy) | `fox noise_control` | F1 | 1 | 80.89+-0.00 | 81.53+-0.00 | n/a -- deterministic | - | - | +0.64 | - | see table above |
| pope/adversarial (greedy) | `fox tau=0.1` | accuracy | 1 | 79.03+-0.00 | 82.20+-0.00 | n/a -- deterministic | - | - | +3.17 | - | see table above |
| pope/adversarial (greedy) | `fox tau=0.1` | F1 | 1 | 80.89+-0.00 | 82.65+-0.00 | n/a -- deterministic | - | - | +1.76 | - | see table above |
| pope/adversarial (greedy) | `fox tau=0.2` | accuracy | 1 | 79.03+-0.00 | 82.80+-0.00 | n/a -- deterministic | - | - | +3.77 | - | see table above |
| pope/adversarial (greedy) | `fox tau=0.2` | F1 | 1 | 80.89+-0.00 | 83.14+-0.00 | n/a -- deterministic | - | - | +2.25 | - | see table above |
| pope/adversarial (greedy) | `icd` | accuracy | 1 | 79.03+-0.00 | 78.60+-0.00 | n/a -- deterministic | - | - | -0.43 | - | see table above |
| pope/adversarial (greedy) | `icd` | F1 | 1 | 80.89+-0.00 | 80.41+-0.00 | n/a -- deterministic | - | - | -0.47 | - | see table above |
| pope/adversarial (greedy) | `icd alpha=0` | accuracy | 1 | 79.03+-0.00 | 79.20+-0.00 | n/a -- deterministic | - | - | +0.17 | - | see table above |
| pope/adversarial (greedy) | `icd alpha=0` | F1 | 1 | 80.89+-0.00 | 81.09+-0.00 | n/a -- deterministic | - | - | +0.20 | - | see table above |
| pope/adversarial (greedy) | `opera` | accuracy | 1 | 79.03+-0.00 | 79.10+-0.00 | n/a -- deterministic | - | - | +0.07 | +2.03 | see table above |
| pope/adversarial (greedy) | `opera` | F1 | 1 | 80.89+-0.00 | 80.95+-0.00 | n/a -- deterministic | - | - | +0.06 | +0.92 | see table above |
| pope/adversarial (greedy) | `opera control=opera_no_penalty` | accuracy | 1 | 79.03+-0.00 | 79.03+-0.00 | n/a -- deterministic | - | - | +0.00 | - | see table above |
| pope/adversarial (greedy) | `opera control=opera_no_penalty` | F1 | 1 | 80.89+-0.00 | 80.90+-0.00 | n/a -- deterministic | - | - | +0.01 | - | see table above |
| pope/adversarial (greedy) | `opera control=opera_no_penalty mnt=10` | accuracy | 1 | 79.03+-0.00 | 81.47+-0.00 | n/a -- deterministic | - | - | +2.43 | - | see table above |
| pope/adversarial (greedy) | `opera control=opera_no_penalty mnt=10` | F1 | 1 | 80.89+-0.00 | 82.27+-0.00 | n/a -- deterministic | - | - | +1.38 | - | see table above |
| pope/adversarial (greedy) | `opera control=opera_unrestricted` | accuracy | 1 | 79.03+-0.00 | 79.13+-0.00 | n/a -- deterministic | - | - | +0.10 | - | see table above |
| pope/adversarial (greedy) | `opera control=opera_unrestricted` | F1 | 1 | 80.89+-0.00 | 80.97+-0.00 | n/a -- deterministic | - | - | +0.09 | - | see table above |
| pope/adversarial (greedy) | `opera control=opera_unrestricted mnt=10` | accuracy | 1 | 79.03+-0.00 | 81.47+-0.00 | n/a -- deterministic | - | - | +2.43 | - | see table above |
| pope/adversarial (greedy) | `opera control=opera_unrestricted mnt=10` | F1 | 1 | 80.89+-0.00 | 81.97+-0.00 | n/a -- deterministic | - | - | +1.08 | - | see table above |
| pope/adversarial (greedy) | `opera mnt=10` | accuracy | 1 | 79.03+-0.00 | 81.13+-0.00 | n/a -- deterministic | - | - | +2.10 | - | see table above |
| pope/adversarial (greedy) | `opera mnt=10` | F1 | 1 | 80.89+-0.00 | 81.88+-0.00 | n/a -- deterministic | - | - | +0.99 | - | see table above |
| pope/adversarial (greedy) | `sid` | accuracy | 1 | 79.03+-0.00 | 80.23+-0.00 | n/a -- deterministic | - | - | +1.20 | +2.67 | see table above |
| pope/adversarial (greedy) | `sid` | F1 | 1 | 80.89+-0.00 | 81.29+-0.00 | n/a -- deterministic | - | - | +0.40 | +1.36 | see table above |
| pope/adversarial (greedy) | `sid alpha=0` | accuracy | 1 | 79.03+-0.00 | 79.20+-0.00 | n/a -- deterministic | - | - | +0.17 | - | see table above |
| pope/adversarial (greedy) | `sid alpha=0` | F1 | 1 | 80.89+-0.00 | 81.09+-0.00 | n/a -- deterministic | - | - | +0.20 | - | see table above |
| pope/adversarial (greedy) | `vcd` | accuracy | 1 | 79.03+-0.00 | 76.93+-0.00 | n/a -- deterministic | - | - | -2.10 | -0.67 | see table above |
| pope/adversarial (greedy) | `vcd` | F1 | 1 | 80.89+-0.00 | 79.08+-0.00 | n/a -- deterministic | - | - | -1.81 | -0.69 | see table above |
| pope/adversarial (greedy) | `vcd alpha=0` | accuracy | 1 | 79.03+-0.00 | 79.20+-0.00 | n/a -- deterministic | - | - | +0.17 | - | see table above |
| pope/adversarial (greedy) | `vcd alpha=0` | F1 | 1 | 80.89+-0.00 | 81.09+-0.00 | n/a -- deterministic | - | - | +0.20 | - | see table above |
| pope/adversarial (sampling) | `causalmm` | accuracy | 3 | 76.72+-0.42 | 77.86+-0.17 | +1.13 [-0.32, +2.58] | 0.078 | 1 **ns** | +0.87 | +4.50 | not resolved after Holm |
| pope/adversarial (sampling) | `causalmm` | F1 | 3 | 79.04+-0.34 | 80.13+-0.15 | +1.09 [-0.10, +2.28] | 0.058 | 0.93 **ns** | +0.89 | - | not resolved after Holm |
| pope/adversarial (sampling) | `causalmm alpha=0` | accuracy | 1 | 77.17+-0.00 | 78.20+-0.00 | **one seed -- no interval** | - | - | +1.03 | - | **not assessable** |
| pope/adversarial (sampling) | `causalmm alpha=0` | F1 | 1 | 79.41+-0.00 | 80.36+-0.00 | **one seed -- no interval** | - | - | +0.95 | - | **not assessable** |
| pope/adversarial (sampling) | `causalmm control=cf_blind` | accuracy | 1 | 77.17+-0.00 | 78.00+-0.00 | **one seed -- no interval** | - | - | +0.83 | - | **not assessable** |
| pope/adversarial (sampling) | `causalmm control=cf_blind` | F1 | 1 | 79.41+-0.00 | 80.07+-0.00 | **one seed -- no interval** | - | - | +0.66 | - | **not assessable** |
| pope/adversarial (sampling) | `causalmm control=cf_constant` | accuracy | 1 | 77.17+-0.00 | 77.80+-0.00 | **one seed -- no interval** | - | - | +0.63 | - | **not assessable** |
| pope/adversarial (sampling) | `causalmm control=cf_constant` | F1 | 1 | 79.41+-0.00 | 80.21+-0.00 | **one seed -- no interval** | - | - | +0.80 | - | **not assessable** |
| pope/adversarial (sampling) | `causalmm control=negative_only` | accuracy | 1 | 77.17+-0.00 | 51.97+-0.00 | **one seed -- no interval** | - | - | -25.20 | - | **not assessable** |
| pope/adversarial (sampling) | `causalmm control=negative_only` | F1 | 1 | 79.41+-0.00 | 48.81+-0.00 | **one seed -- no interval** | - | - | -30.60 | - | **not assessable** |
| pope/adversarial (sampling) | `fox` | accuracy | 10 | 76.72+-0.31 | 81.76+-0.33 | +5.03 [+4.68, +5.38] | 1.2e-10 | 5.7e-09 | +4.90 | +5.16 | reproduces |
| pope/adversarial (sampling) | `fox` | F1 | 10 | 78.93+-0.28 | 82.13+-0.32 | +3.21 [+2.88, +3.53] | 3.7e-09 | 1.6e-07 | +3.18 | - | resolved |
| pope/adversarial (sampling) | `fox L=2-13` | accuracy | 3 | 76.72+-0.42 | 81.53+-0.24 | +4.81 [+3.98, +5.64] | 0.0016 | 0.048 | +4.57 | - | resolved |
| pope/adversarial (sampling) | `fox L=2-13` | F1 | 3 | 79.04+-0.34 | 82.13+-0.24 | +3.09 [+2.36, +3.82] | 0.003 | 0.087 **ns** | +2.91 | - | not resolved after Holm |
| pope/adversarial (sampling) | `fox boost_visual g=4` | accuracy | 10 | 76.72+-0.31 | 80.46+-0.34 | +3.74 [+3.46, +4.02] | 2.1e-10 | 9.8e-09 | +3.73 | - | resolved |
| pope/adversarial (sampling) | `fox boost_visual g=4` | F1 | 10 | 78.93+-0.28 | 81.62+-0.30 | +2.69 [+2.42, +2.96] | 3e-09 | 1.3e-07 | +2.66 | - | resolved |
| pope/adversarial (sampling) | `fox ld=bf16` | accuracy | 10 | 76.72+-0.31 | 81.77+-0.28 | +5.05 [+4.69, +5.41] | 1.5e-10 | 7.1e-09 | +4.83 | - | resolved |
| pope/adversarial (sampling) | `fox ld=bf16` | F1 | 10 | 78.93+-0.28 | 81.62+-0.26 | +2.69 [+2.36, +3.03] | 2.1e-08 | 8.2e-07 | +2.54 | - | resolved |
| pope/adversarial (sampling) | `fox ld=fp32` | accuracy | 10 | 76.72+-0.31 | 78.92+-0.22 | +2.20 [+1.90, +2.50] | 5e-08 | 1.9e-06 | +1.90 | - | resolved |
| pope/adversarial (sampling) | `fox ld=fp32` | F1 | 10 | 78.93+-0.28 | 80.74+-0.18 | +1.82 [+1.55, +2.08] | 7.7e-08 | 2.9e-06 | +1.60 | - | resolved |
| pope/adversarial (sampling) | `fox off` | accuracy | 10 | 76.72+-0.31 | 78.93+-0.30 | +2.21 [+1.84, +2.58] | 2.8e-07 | 1e-05 | +2.23 | - | resolved |
| pope/adversarial (sampling) | `fox off` | F1 | 10 | 78.93+-0.28 | 80.75+-0.25 | +1.82 [+1.51, +2.14] | 3.6e-07 | 1.3e-05 | +1.83 | - | resolved |
| pope/adversarial (sampling) | `fox random_heads` | accuracy | 10 | 76.72+-0.31 | 81.27+-0.34 | +4.54 [+4.17, +4.91] | 4.8e-10 | 2.2e-08 | +4.63 | - | resolved |
| pope/adversarial (sampling) | `fox random_heads` | F1 | 10 | 78.93+-0.28 | 81.90+-0.34 | +2.97 [+2.62, +3.33] | 1.3e-08 | 5.4e-07 | +3.10 | - | resolved |
| pope/adversarial (sampling) | `fox ulp_control` | accuracy | 10 | 76.72+-0.31 | 81.83+-0.28 | +5.11 [+4.91, +5.30] | 5.2e-13 | 2.6e-11 | +4.97 | - | resolved |
| pope/adversarial (sampling) | `fox ulp_control` | F1 | 10 | 78.93+-0.28 | 82.19+-0.28 | +3.26 [+3.04, +3.49] | 1.2e-10 | 6e-09 | +3.15 | - | resolved |
| pope/adversarial (sampling) | `icd` | accuracy | 3 | 76.72+-0.42 | 77.20+-0.38 | +0.48 [-0.27, +1.23] | 0.11 | 1 **ns** | +0.30 | +1.13 | not resolved after Holm |
| pope/adversarial (sampling) | `icd` | F1 | 3 | 79.04+-0.34 | 79.15+-0.35 | +0.11 [-0.25, +0.48] | 0.31 | 1 **ns** | +0.03 | - | not resolved after Holm |
| pope/adversarial (sampling) | `icd alpha=0` | accuracy | 1 | 77.17+-0.00 | 77.00+-0.00 | **one seed -- no interval** | - | - | -0.17 | - | **not assessable** |
| pope/adversarial (sampling) | `icd alpha=0` | F1 | 1 | 79.41+-0.00 | 79.05+-0.00 | **one seed -- no interval** | - | - | -0.36 | - | **not assessable** |
| pope/adversarial (sampling) | `icd control=negative_only` | accuracy | 1 | 77.17+-0.00 | 73.70+-0.00 | **one seed -- no interval** | - | - | -3.47 | - | **not assessable** |
| pope/adversarial (sampling) | `icd control=negative_only` | F1 | 1 | 79.41+-0.00 | 76.71+-0.00 | **one seed -- no interval** | - | - | -2.71 | - | **not assessable** |
| pope/adversarial (sampling) | `sid` | accuracy | 3 | 76.72+-0.42 | 79.67+-0.22 | +2.94 [+1.91, +3.98] | 0.0066 | 0.19 **ns** | +2.73 | +4.33 | not resolved after Holm |
| pope/adversarial (sampling) | `sid` | F1 | 3 | 79.04+-0.34 | 80.75+-0.28 | +1.72 [+0.52, +2.91] | 0.025 | 0.55 **ns** | +1.67 | - | not resolved after Holm |
| pope/adversarial (sampling) | `sid alpha=0` | accuracy | 1 | 77.17+-0.00 | 77.20+-0.00 | **one seed -- no interval** | - | - | +0.03 | - | **not assessable** |
| pope/adversarial (sampling) | `sid alpha=0` | F1 | 1 | 79.41+-0.00 | 79.42+-0.00 | **one seed -- no interval** | - | - | +0.01 | - | **not assessable** |
| pope/adversarial (sampling) | `sid control=negative_only` | accuracy | 1 | 77.17+-0.00 | 56.43+-0.00 | **one seed -- no interval** | - | - | -20.73 | - | **not assessable** |
| pope/adversarial (sampling) | `sid control=negative_only` | F1 | 1 | 79.41+-0.00 | 63.23+-0.00 | **one seed -- no interval** | - | - | -16.18 | - | **not assessable** |
| pope/adversarial (sampling) | `sid control=sid_keep_most` | accuracy | 1 | 77.17+-0.00 | 75.47+-0.00 | **one seed -- no interval** | - | - | -1.70 | - | **not assessable** |
| pope/adversarial (sampling) | `sid control=sid_keep_most` | F1 | 1 | 79.41+-0.00 | 78.38+-0.00 | **one seed -- no interval** | - | - | -1.03 | - | **not assessable** |
| pope/adversarial (sampling) | `sid control=sid_random` | accuracy | 1 | 77.17+-0.00 | 77.50+-0.00 | **one seed -- no interval** | - | - | +0.33 | - | **not assessable** |
| pope/adversarial (sampling) | `sid control=sid_random` | F1 | 1 | 79.41+-0.00 | 79.48+-0.00 | **one seed -- no interval** | - | - | +0.07 | - | **not assessable** |
| pope/adversarial (sampling) | `vcd` | accuracy | 3 | 76.72+-0.42 | 75.80+-0.91 | -0.92 [-3.76, +1.91] | 0.3 | 1 **ns** | -0.37 | -1.44 | not resolved after Holm |
| pope/adversarial (sampling) | `vcd` | F1 | 3 | 79.04+-0.34 | 78.34+-0.64 | -0.70 [-2.76, +1.37] | 0.28 | 1 **ns** | -0.35 | - | not resolved after Holm |
| pope/adversarial (sampling) | `vcd alpha=0` | accuracy | 1 | 77.17+-0.00 | 76.40+-0.00 | **one seed -- no interval** | - | - | -0.77 | - | **not assessable** |
| pope/adversarial (sampling) | `vcd alpha=0` | F1 | 1 | 79.41+-0.00 | 78.23+-0.00 | **one seed -- no interval** | - | - | -1.18 | - | **not assessable** |
| pope/adversarial (sampling) | `vcd control=negative_only` | accuracy | 1 | 77.17+-0.00 | 76.10+-0.00 | **one seed -- no interval** | - | - | -1.07 | - | **not assessable** |
| pope/adversarial (sampling) | `vcd control=negative_only` | F1 | 1 | 79.41+-0.00 | 77.45+-0.00 | **one seed -- no interval** | - | - | -1.97 | - | **not assessable** |
| pope/adversarial (sampling) | `vcd_fox_fusion` | accuracy | 3 | 76.72+-0.42 | 78.81+-0.07 | +2.09 [+1.01, +3.17] | 0.014 | 0.34 **ns** | +1.70 | - | not resolved after Holm |
| pope/adversarial (sampling) | `vcd_fox_fusion` | F1 | 3 | 79.04+-0.34 | 79.07+-0.04 | +0.03 [-0.91, +0.98] | 0.89 | 1 **ns** | -0.32 | - | not resolved after Holm |
| pope/popular (greedy) | `fox` | accuracy | 1 | 85.90+-0.00 | 86.90+-0.00 | n/a -- deterministic | - | - | +1.00 | +1.14 | see table above |
| pope/popular (greedy) | `fox` | F1 | 1 | 86.29+-0.00 | 86.51+-0.00 | n/a -- deterministic | - | - | +0.22 | +0.27 | see table above |
| pope/popular (greedy) | `opera` | accuracy | 1 | 85.90+-0.00 | 85.93+-0.00 | n/a -- deterministic | - | - | +0.03 | +0.87 | see table above |
| pope/popular (greedy) | `opera` | F1 | 1 | 86.29+-0.00 | 86.33+-0.00 | n/a -- deterministic | - | - | +0.04 | +0.26 | see table above |
| pope/popular (sampling) | `fox` | accuracy | 3 | 82.57+-0.48 | 86.70+-0.35 | +4.13 [+2.63, +5.64] | 0.0071 | 0.19 **ns** | +3.87 | +4.17 | not resolved after Holm |
| pope/popular (sampling) | `fox` | F1 | 3 | 83.37+-0.41 | 86.31+-0.35 | +2.94 [+1.46, +4.41] | 0.013 | 0.33 **ns** | +2.79 | - | not resolved after Holm |
| pope/random (greedy) | `fox` | accuracy | 1 | 89.33+-0.00 | 89.53+-0.00 | n/a -- deterministic | - | - | +0.20 | +0.10 | see table above |
| pope/random (greedy) | `fox` | F1 | 1 | 89.28+-0.00 | 88.96+-0.00 | n/a -- deterministic | - | - | -0.32 | -0.47 | see table above |
| pope/random (greedy) | `opera` | accuracy | 1 | 89.33+-0.00 | 89.37+-0.00 | n/a -- deterministic | - | - | +0.03 | -0.06 | see table above |
| pope/random (greedy) | `opera` | F1 | 1 | 89.28+-0.00 | 89.32+-0.00 | n/a -- deterministic | - | - | +0.04 | -0.56 | see table above |
| pope/random (sampling) | `fox` | accuracy | 3 | 85.12+-0.07 | 89.28+-0.15 | +4.16 [+3.86, +4.45] | 0.00026 | 0.0085 | +4.17 | +4.20 | reproduces |
| pope/random (sampling) | `fox` | F1 | 3 | 85.48+-0.10 | 88.72+-0.17 | +3.24 [+2.97, +3.52] | 0.00039 | 0.012 | +3.24 | - | resolved |

`p (Holm)` is the Holm-Bonferroni adjustment over every row in this table; `**ns**` marks a row that does not survive it. 

Greedy rows are expected to show `sd = 0`: argmax is deterministic, so replicating a seed changes nothing and the paper's Table 9 is single-run too ("we omit averaged scores", C.7). Their intervals come from the per-run tables above, where item variance is the only variance there is.


### Equivalence to `fox`

Two one-sided tests (TOST). The null is that the arms differ by at least the margin, so **rejecting it is positive evidence that they do the same thing** -- unlike a large p from an ordinary difference test, which only means the data could not tell them apart and gets easier to obtain the noisier the measurement is.

The margin is a judgement, not a default, and is printed with every row. `smallest margin` is the tightest equivalence the data support at this alpha.

| artefact | arm | metric | seeds | diff vs fox | 90% CI | margin | TOST p | smallest margin | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chair/val2014_500 (sampling) | `causalmm` | CHAIR_S | 3 | +2.867 | [-0.847, +6.581] | +-2 | 0.7170 | +-6.58 | not shown |
| chair/val2014_500 (sampling) | `causalmm` | CHAIR_I | 3 | +0.777 | [+0.645, +0.909] | +-0.5 | 0.9872 | +-0.91 | not shown |
| chair/val2014_500 (sampling) | `fox boost_visual g=4` | CHAIR_S | 10 | -6.240 | [-7.237, -5.243] | +-2 | 1.0000 | +-7.24 | not shown |
| chair/val2014_500 (sampling) | `fox boost_visual g=4` | CHAIR_I | 10 | -2.142 | [-2.388, -1.897] | +-0.5 | 1.0000 | +-2.39 | not shown |
| chair/val2014_500 (sampling) | `fox off` | CHAIR_S | 3 | +0.733 | [-1.635, +3.102] | +-2 | 0.1294 | +-3.10 | not shown |
| chair/val2014_500 (sampling) | `fox off` | CHAIR_I | 3 | +0.158 | [-0.224, +0.541] | +-0.5 | 0.0605 | +-0.54 | not shown |
| chair/val2014_500 (sampling) | `fox random_heads` | CHAIR_S | 3 | -6.200 | [-11.467, -0.933] | +-2 | 0.9274 | +-11.47 | not shown |
| chair/val2014_500 (sampling) | `fox random_heads` | CHAIR_I | 3 | -1.778 | [-3.544, -0.012] | +-0.5 | 0.9155 | +-3.54 | not shown |
| chair/val2014_500 (sampling) | `fox ulp_control` | CHAIR_S | 3 | -2.933 | [-5.178, -0.688] | +-2 | 0.8257 | +-5.18 | not shown |
| chair/val2014_500 (sampling) | `fox ulp_control` | CHAIR_I | 3 | -0.549 | [-1.731, +0.632] | +-0.5 | 0.5429 | +-1.73 | not shown |
| chair/val2014_500 (sampling) | `icd` | CHAIR_S | 3 | +4.333 | [+1.388, +7.279] | +-2 | 0.9266 | +-7.28 | not shown |
| chair/val2014_500 (sampling) | `icd` | CHAIR_I | 3 | +1.322 | [+1.167, +1.477] | +-0.5 | 0.9979 | +-1.48 | not shown |
| chair/val2014_500 (sampling) | `sampling` | CHAIR_S | 10 | +6.980 | [+5.142, +8.818] | +-2 | 0.9996 | +-8.82 | not shown |
| chair/val2014_500 (sampling) | `sampling` | CHAIR_I | 10 | +2.668 | [+2.097, +3.240] | +-0.5 | 1.0000 | +-3.24 | not shown |
| chair/val2014_500 (sampling) | `sid` | CHAIR_S | 3 | +3.667 | [+3.152, +4.182] | +-2 | 0.9945 | +-4.18 | not shown |
| chair/val2014_500 (sampling) | `sid` | CHAIR_I | 3 | +1.159 | [-0.068, +2.386] | +-0.5 | 0.8712 | +-2.39 | not shown |
| chair/val2014_500 (sampling) | `vcd` | CHAIR_S | 3 | +6.667 | [+1.216, +12.117] | +-2 | 0.9352 | +-12.12 | not shown |
| chair/val2014_500 (sampling) | `vcd` | CHAIR_I | 3 | +2.888 | [+2.340, +3.436] | +-0.5 | 0.9969 | +-3.44 | not shown |
| chair/val2014_500 (sampling) | `vcd_fox_fusion` | CHAIR_S | 3 | -7.933 | [-8.448, -7.418] | +-2 | 0.9996 | +-8.45 | not shown |
| chair/val2014_500 (sampling) | `vcd_fox_fusion` | CHAIR_I | 3 | +0.671 | [+0.441, +0.900] | +-0.5 | 0.9189 | +-0.90 | not shown |
| pope/adversarial (sampling) | `causalmm` | accuracy | 3 | -3.700 | [-4.037, -3.363] | +-0.5 | 0.9994 | +-4.04 | not shown |
| pope/adversarial (sampling) | `causalmm` | F1 | 3 | -1.852 | [-2.298, -1.407] | +-0.5 | 0.9938 | +-2.30 | not shown |
| pope/adversarial (sampling) | `fox L=2-13` | accuracy | 3 | -0.022 | [-0.417, +0.372] | +-0.5 | 0.0358 | +-0.42 | **equivalent** |
| pope/adversarial (sampling) | `fox L=2-13` | F1 | 3 | +0.145 | [-0.178, +0.468] | +-0.5 | 0.0426 | +-0.47 | **equivalent** |
| pope/adversarial (sampling) | `fox boost_visual g=4` | accuracy | 10 | -1.293 | [-1.562, -1.025] | +-0.5 | 0.9998 | +-1.56 | not shown |
| pope/adversarial (sampling) | `fox boost_visual g=4` | F1 | 10 | -0.516 | [-0.753, -0.280] | +-0.5 | 0.5491 | +-0.75 | not shown |
| pope/adversarial (sampling) | `fox ld=bf16` | accuracy | 10 | +0.013 | [-0.244, +0.270] | +-0.5 | 0.0035 | +-0.27 | **equivalent** |
| pope/adversarial (sampling) | `fox ld=bf16` | F1 | 10 | -0.516 | [-0.754, -0.278] | +-0.5 | 0.5476 | +-0.75 | not shown |
| pope/adversarial (sampling) | `fox ld=fp32` | accuracy | 10 | -2.833 | [-3.105, -2.562] | +-0.5 | 1.0000 | +-3.11 | not shown |
| pope/adversarial (sampling) | `fox ld=fp32` | F1 | 10 | -1.390 | [-1.643, -1.138] | +-0.5 | 0.9999 | +-1.64 | not shown |
| pope/adversarial (sampling) | `fox off` | accuracy | 10 | -2.827 | [-3.044, -2.610] | +-0.5 | 1.0000 | +-3.04 | not shown |
| pope/adversarial (sampling) | `fox off` | F1 | 10 | -1.384 | [-1.588, -1.181] | +-0.5 | 1.0000 | +-1.59 | not shown |
| pope/adversarial (sampling) | `fox random_heads` | accuracy | 10 | -0.490 | [-0.816, -0.164] | +-0.5 | 0.4782 | +-0.82 | not shown |
| pope/adversarial (sampling) | `fox random_heads` | F1 | 10 | -0.233 | [-0.546, +0.081] | +-0.5 | 0.0760 | +-0.55 | not shown |
| pope/adversarial (sampling) | `fox ulp_control` | accuracy | 10 | +0.073 | [-0.232, +0.379] | +-0.5 | 0.0154 | +-0.38 | **equivalent** |
| pope/adversarial (sampling) | `fox ulp_control` | F1 | 10 | +0.057 | [-0.237, +0.350] | +-0.5 | 0.0109 | +-0.35 | **equivalent** |
| pope/adversarial (sampling) | `icd` | accuracy | 3 | -4.356 | [-4.987, -3.724] | +-0.5 | 0.9984 | +-4.99 | not shown |
| pope/adversarial (sampling) | `icd` | F1 | 3 | -2.830 | [-3.466, -2.193] | +-0.5 | 0.9957 | +-3.47 | not shown |
| pope/adversarial (sampling) | `sampling` | accuracy | 10 | -5.033 | [-5.316, -4.751] | +-0.5 | 1.0000 | +-5.32 | not shown |
| pope/adversarial (sampling) | `sampling` | F1 | 10 | -3.207 | [-3.472, -2.942] | +-0.5 | 1.0000 | +-3.47 | not shown |
| pope/adversarial (sampling) | `sid` | accuracy | 3 | -1.889 | [-2.154, -1.623] | +-0.5 | 0.9979 | +-2.15 | not shown |
| pope/adversarial (sampling) | `sid` | F1 | 3 | -1.228 | [-1.398, -1.057] | +-0.5 | 0.9968 | +-1.40 | not shown |
| pope/adversarial (sampling) | `vcd` | accuracy | 3 | -5.756 | [-7.002, -4.509] | +-0.5 | 0.9967 | +-7.00 | not shown |
| pope/adversarial (sampling) | `vcd` | F1 | 3 | -3.639 | [-4.394, -2.885] | +-0.5 | 0.9966 | +-4.39 | not shown |
| pope/adversarial (sampling) | `vcd_fox_fusion` | accuracy | 3 | -2.744 | [-3.127, -2.362] | +-0.5 | 0.9983 | +-3.13 | not shown |
| pope/adversarial (sampling) | `vcd_fox_fusion` | F1 | 3 | -2.911 | [-3.188, -2.633] | +-0.5 | 0.9992 | +-3.19 | not shown |
| pope/popular (sampling) | `sampling` | accuracy | 3 | -4.133 | [-5.156, -3.111] | +-0.5 | 0.9954 | +-5.16 | not shown |
| pope/popular (sampling) | `sampling` | F1 | 3 | -2.938 | [-3.940, -1.937] | +-0.5 | 0.9904 | +-3.94 | not shown |
| pope/random (sampling) | `sampling` | accuracy | 3 | -4.156 | [-4.353, -3.958] | +-0.5 | 0.9998 | +-4.35 | not shown |
| pope/random (sampling) | `sampling` | F1 | 3 | -3.244 | [-3.430, -3.057] | +-0.5 | 0.9997 | +-3.43 | not shown |


### Degenerate CHAIR arms

CHAIR without recall beside it is gameable, and these arms game it.

- ⚠ **`fox ld=bf16`** on chair/val2014_500 (greedy): CHAIR_S 25.60 against 49.20, which looks like a +23.60 point win -- but recall fell 79.23 -> 59.81, correct objects 1236 -> 933, caption length 100 -> 156 words. It hallucinates less because it names less. Do not read its CHAIR scores as grounding.
- ⚠ **`vcd_fox_fusion`** on chair/val2014_500 (sampling): CHAIR_S 40.53 against 54.40, which looks like a +13.87 point win -- but recall fell 75.64 -> 66.73, correct objects 1180 -> 1041, caption length 102 -> 96 words. It hallucinates less because it names less. Do not read its CHAIR scores as grounding.


### Resolution budget

Smallest gap each run can resolve (two-sided 95%, 80% power):

| benchmark / split | metric | n | p | smallest resolvable gap (pts) |
| --- | --- | --- | --- | --- |
| chair/val2014_500 | CHAIR_S | 500 | 0.5020 | 8.81 |
| chair/val2014_500 | CHAIR_I | 3623 | 0.1490 | 2.42 |
| chair/val2014_500 | CHAIR_I | 3645 | 0.1446 | 2.38 |
| chair/val2014_500 | CHAIR_I | 3606 | 0.1464 | 2.41 |
| pope/adversarial | accuracy | 3000 | 0.7753 | 2.95 |
| pope/adversarial | accuracy | 500 | 0.7940 | 6.69 |
| chair/val2014_500 | CHAIR_I | 3761 | 0.1380 | 2.30 |
| chair/val2014_500 | CHAIR_I | 5236 | 0.1062 | 1.75 |
| chair/val2014_500 | CHAIR_I | 3503 | 0.1188 | 2.25 |
| chair/val2014_500 | CHAIR_I | 3896 | 0.1345 | 2.24 |
| chair/val2014_500 | CHAIR_I | 3803 | 0.1283 | 2.23 |
| chair/val2014_500 | CHAIR_I | 3703 | 0.1437 | 2.36 |
| chair/val2014_500 | CHAIR_I | 3749 | 0.1366 | 2.30 |
| chair/val2014_500 | CHAIR_I | 3602 | 0.1152 | 2.19 |
| chair/val2014_500 | CHAIR_I | 3683 | 0.1423 | 2.36 |
| chair/val2014_500 | CHAIR_I | 3829 | 0.1408 | 2.30 |
| chair/val2014_500 | CHAIR_I | 3820 | 0.1178 | 2.15 |
| chair/val2014_500 | CHAIR_I | 3797 | 0.1296 | 2.24 |
| chair/val2014_500 | CHAIR_I | 3521 | 0.1252 | 2.29 |
| chair/val2014_500 | CHAIR_I | 3644 | 0.1416 | 2.37 |
| chair/val2014_500 | CHAIR_I | 3776 | 0.1377 | 2.30 |
| chair/val2014_500 | CHAIR_I | 3773 | 0.1320 | 2.26 |
| chair/val2014_500 | CHAIR_I | 3782 | 0.1301 | 2.24 |
| chair/val2014_500 | CHAIR_I | 3625 | 0.1137 | 2.17 |
| chair/val2014_500 | CHAIR_I | 3626 | 0.1376 | 2.34 |
| chair/val2014_500 | CHAIR_I | 3855 | 0.1383 | 2.28 |
| chair/val2014_500 | CHAIR_I | 3613 | 0.1174 | 2.20 |
| chair/val2014_500 | CHAIR_I | 3748 | 0.1307 | 2.26 |
| chair/val2014_500 | CHAIR_I | 3543 | 0.1118 | 2.18 |
| chair/val2014_500 | CHAIR_I | 3842 | 0.1476 | 2.34 |
| chair/val2014_500 | CHAIR_I | 3570 | 0.1196 | 2.24 |
| chair/val2014_500 | CHAIR_I | 3837 | 0.1306 | 2.23 |
| chair/val2014_500 | CHAIR_I | 3560 | 0.1110 | 2.17 |
| chair/val2014_500 | CHAIR_I | 3824 | 0.1352 | 2.27 |
| chair/val2014_500 | CHAIR_I | 3594 | 0.1152 | 2.19 |
| chair/val2014_500 | CHAIR_I | 3751 | 0.1330 | 2.27 |
| chair/val2014_500 | CHAIR_I | 3559 | 0.1115 | 2.18 |
| chair/val2014_500 | CHAIR_I | 3823 | 0.1355 | 2.27 |
| chair/val2014_500 | CHAIR_I | 3589 | 0.1181 | 2.22 |
| chair/val2014_500 | CHAIR_I | 3554 | 0.1142 | 2.20 |
| chair/val2014_500 | CHAIR_S | 50 | 0.5400 | 26.03 |
| chair/val2014_500 | CHAIR_I | 428 | 0.1379 | 7.25 |
| chair/val2014_500_d1 | CHAIR_S | 500 | 0.4140 | 8.82 |
| chair/val2014_500_d1 | CHAIR_I | 3914 | 0.1191 | 2.13 |
| chair/val2014_500_d2 | CHAIR_S | 500 | 0.4460 | 8.85 |
| chair/val2014_500_d2 | CHAIR_I | 3849 | 0.1315 | 2.23 |
| chair/val2014_500_d3 | CHAIR_S | 500 | 0.4940 | 8.82 |
| chair/val2014_500_d3 | CHAIR_I | 4017 | 0.1481 | 2.29 |
| chair/val2014_500_d4 | CHAIR_S | 500 | 0.4400 | 8.84 |
| chair/val2014_500_d4 | CHAIR_I | 3884 | 0.1285 | 2.20 |
| pope/adversarial | accuracy | 50 | 0.7400 | 20.24 |
| pope/popular | accuracy | 3000 | 0.8690 | 2.34 |
| pope/random | accuracy | 3000 | 0.8953 | 2.11 |
| chair/val2014_500 | CHAIR_I | 3747 | 0.1348 | 2.29 |
| chair/val2014_500 | CHAIR_I | 3605 | 0.1531 | 2.45 |
| chair/val2014_500 | CHAIR_I | 3575 | 0.1524 | 2.46 |
| pope/adversarial | accuracy | 1500 | 0.8147 | 3.81 |
| chair/val2014_500 | CHAIR_I | 3762 | 0.1382 | 2.30 |
| chair/val2014_500 | CHAIR_I | 3599 | 0.1814 | 2.61 |
| chair/val2014_500 | CHAIR_I | 3486 | 0.1644 | 2.56 |
| chair/val2014_500 | CHAIR_I | 3536 | 0.1612 | 2.52 |
| chair/val2014_500 | CHAIR_I | 3572 | 0.1632 | 2.52 |
| chair/val2014_500 | CHAIR_I | 3562 | 0.1555 | 2.48 |
| chair/val2014_500 | CHAIR_I | 3545 | 0.1614 | 2.52 |
| chair/val2014_500 | CHAIR_I | 3495 | 0.1508 | 2.48 |
| chair/val2014_500 | CHAIR_I | 3658 | 0.1613 | 2.48 |
| chair/val2014_500 | CHAIR_I | 3487 | 0.1735 | 2.61 |
| chair/val2014_500 | CHAIR_I | 3534 | 0.1661 | 2.55 |
| chair/val2014_500_d1 | CHAIR_I | 3782 | 0.1296 | 2.24 |
| chair/val2014_500_d2 | CHAIR_I | 3711 | 0.1439 | 2.36 |
| chair/val2014_500_d3 | CHAIR_I | 3775 | 0.1481 | 2.36 |
| chair/val2014_500_d4 | CHAIR_I | 3799 | 0.1316 | 2.25 |
| chair/val2014_500 | CHAIR_I | 3686 | 0.1283 | 2.26 |
| chair/val2014_500 | CHAIR_I | 3578 | 0.1520 | 2.45 |
| chair/val2014_500 | CHAIR_I | 3601 | 0.1422 | 2.38 |
| chair/val2014_500 | CHAIR_I | 3700 | 0.1484 | 2.39 |
| chair/val2014_500 | CHAIR_I | 3675 | 0.1703 | 2.53 |
| chair/val2014_500 | CHAIR_I | 3695 | 0.1651 | 2.49 |
| chair/val2014_500 | CHAIR_I | 3106 | 0.1484 | 2.62 |
| chair/val2014_500 | CHAIR_I | 3164 | 0.1429 | 2.55 |
| chair/val2014_500 | CHAIR_I | 3249 | 0.1456 | 2.54 |

Samples **per method** needed for a given gap at p = 0.82 (`paired_rho` is the correlation between two methods' per-sample outcomes; paired runs on identical inputs are strongly correlated):

| gap (pts) | rho = 0 | rho = 0.5 | rho = 0.8 |
| --- | --- | --- | --- |
| 0.5 | 91,670 | 45,835 | 18,334 |
| 1.0 | 22,663 | 11,332 | 4,533 |
| 2.0 | 5,537 | 2,769 | 1,108 |
| 5.0 | 822 | 411 | 165 |


### Provenance audit

| run | method | bench/split | dtype | gamma | k | tau_js | seed | n | model | commit | transformers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `m-causalmm__b-chair__s-val2014_500__seed-42__dt-fp16__v-sampling-5c7c06df` | `causalmm` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `ced26c0` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-causalmm__b-chair__s-val2014_500__seed-43__dt-fp16__v-sampling-5c7c06df` | `causalmm` | chair/val2014_500 | fp16 | None | None | None | 43 | 500 | `llava-hf/llava-1.5-7b-hf` | `ced26c0` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-causalmm__b-chair__s-val2014_500__seed-44__dt-fp16__v-sampling-5c7c06df` | `causalmm` | chair/val2014_500 | fp16 | None | None | None | 44 | 500 | `llava-hf/llava-1.5-7b-hf` | `ced26c0` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-causalmm__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-0d6afcd8` | `causalmm` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `['0ec984e', '3579f38']` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-causalmm__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-17d4eb8d` | `causalmm` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `['0ec984e', '3579f38']` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-causalmm__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-851cf803` | `causalmm` | pope/adversarial | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `ced26c0` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-causalmm__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-d9edc133` | `causalmm` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `['0ec984e', '3579f38']` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-causalmm__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-2d07edcb` | `causalmm` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `a22119b` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-causalmm__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-4ff6b811` | `causalmm` | pope/adversarial | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `a22119b` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-causalmm__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-6ad493ca` | `causalmm` | pope/adversarial | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `a22119b` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-causalmm__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-a0cac4b7` | `causalmm` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `a22119b` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-causalmm__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-fa5d6379` | `causalmm` | pope/adversarial | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `a22119b` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-causalmm__b-pope__s-adversarial__seed-43__dt-fp16__v-sampling-2d07edcb` | `causalmm` | pope/adversarial | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `a22119b` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-causalmm__b-pope__s-adversarial__seed-44__dt-fp16__v-sampling-2d07edcb` | `causalmm` | pope/adversarial | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `ced26c0` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-greedy-11fe4386` | `fox` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-greedy-129590cd` | `fox` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-greedy-41d5cbc1` | `fox` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-greedy-4a534adb` | `fox` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `68374d2` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-greedy-5f61b81c` | `fox` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `6d14816` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-greedy-7dc3dd43` | `fox` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-greedy-g100-9ebfabc4` | `fox` | chair/val2014_500 | fp16 | 100.0 | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-greedy-g10000-1c417645` | `fox` | chair/val2014_500 | fp16 | 10000.0 | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-greedy-g4-4f055e8b` | `fox` | chair/val2014_500 | fp16 | 4.0 | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-sampling-1bea1044` | `fox` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-sampling-5c7c06df` | `fox` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `68374d2` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-sampling-ae0d095f` | `fox` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `6d14816` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-sampling-c676d8ec` | `fox` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-sampling-g4-60acf738` | `fox` | chair/val2014_500 | fp16 | 4.0 | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-43__dt-fp16__v-sampling-1bea1044` | `fox` | chair/val2014_500 | fp16 | None | None | None | 43 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-43__dt-fp16__v-sampling-5c7c06df` | `fox` | chair/val2014_500 | fp16 | None | None | None | 43 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-43__dt-fp16__v-sampling-ae0d095f` | `fox` | chair/val2014_500 | fp16 | None | None | None | 43 | 500 | `llava-hf/llava-1.5-7b-hf` | `6d14816` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-43__dt-fp16__v-sampling-c676d8ec` | `fox` | chair/val2014_500 | fp16 | None | None | None | 43 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-43__dt-fp16__v-sampling-g4-60acf738` | `fox` | chair/val2014_500 | fp16 | 4.0 | None | None | 43 | 500 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-44__dt-fp16__v-sampling-1bea1044` | `fox` | chair/val2014_500 | fp16 | None | None | None | 44 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-44__dt-fp16__v-sampling-5c7c06df` | `fox` | chair/val2014_500 | fp16 | None | None | None | 44 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-44__dt-fp16__v-sampling-ae0d095f` | `fox` | chair/val2014_500 | fp16 | None | None | None | 44 | 500 | `llava-hf/llava-1.5-7b-hf` | `6d14816` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-44__dt-fp16__v-sampling-c676d8ec` | `fox` | chair/val2014_500 | fp16 | None | None | None | 44 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-44__dt-fp16__v-sampling-g4-60acf738` | `fox` | chair/val2014_500 | fp16 | 4.0 | None | None | 44 | 500 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-45__dt-fp16__v-sampling-5c7c06df` | `fox` | chair/val2014_500 | fp16 | None | None | None | 45 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-45__dt-fp16__v-sampling-g4-60acf738` | `fox` | chair/val2014_500 | fp16 | 4.0 | None | None | 45 | 500 | `llava-hf/llava-1.5-7b-hf` | `['434b004', '7d52f44']` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-46__dt-fp16__v-sampling-5c7c06df` | `fox` | chair/val2014_500 | fp16 | None | None | None | 46 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-46__dt-fp16__v-sampling-g4-60acf738` | `fox` | chair/val2014_500 | fp16 | 4.0 | None | None | 46 | 500 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-47__dt-fp16__v-sampling-5c7c06df` | `fox` | chair/val2014_500 | fp16 | None | None | None | 47 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-47__dt-fp16__v-sampling-g4-60acf738` | `fox` | chair/val2014_500 | fp16 | 4.0 | None | None | 47 | 500 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-48__dt-fp16__v-sampling-5c7c06df` | `fox` | chair/val2014_500 | fp16 | None | None | None | 48 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-48__dt-fp16__v-sampling-g4-60acf738` | `fox` | chair/val2014_500 | fp16 | 4.0 | None | None | 48 | 500 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-49__dt-fp16__v-sampling-5c7c06df` | `fox` | chair/val2014_500 | fp16 | None | None | None | 49 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-49__dt-fp16__v-sampling-g4-60acf738` | `fox` | chair/val2014_500 | fp16 | 4.0 | None | None | 49 | 500 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-50__dt-fp16__v-sampling-5c7c06df` | `fox` | chair/val2014_500 | fp16 | None | None | None | 50 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-50__dt-fp16__v-sampling-g4-60acf738` | `fox` | chair/val2014_500 | fp16 | 4.0 | None | None | 50 | 500 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-51__dt-fp16__v-sampling-5c7c06df` | `fox` | chair/val2014_500 | fp16 | None | None | None | 51 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-51__dt-fp16__v-sampling-g4-60acf738` | `fox` | chair/val2014_500 | fp16 | 4.0 | None | None | 51 | 500 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-997__dt-fp16__v-greedy-4a534adb` | `fox` | chair/val2014_500 | fp16 | None | None | None | 997 | 50 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500_d1__seed-42__dt-fp16__v-greedy-4a534adb` | `fox` | chair/val2014_500_d1 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500_d2__seed-42__dt-fp16__v-greedy-4a534adb` | `fox` | chair/val2014_500_d2 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500_d3__seed-42__dt-fp16__v-greedy-4a534adb` | `fox` | chair/val2014_500_d3 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500_d4__seed-42__dt-fp16__v-greedy-4a534adb` | `fox` | chair/val2014_500_d4 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-17d4eb8d` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-2849ecbb` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-30d01489` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-51104c9a` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-7d65ec90` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-851cf803` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `e598678` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-88b7237f` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `e598678` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-945b2de4` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `6d14816` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-94ffb3d6` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-d0fd6cf4` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `6d14816` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-d777f5d0` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `6d14816` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-ec5b84b4` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-g4-77f044a1` | `fox` | pope/adversarial | fp16 | 4.0 | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-k0.25-403b917a` | `fox` | pope/adversarial | fp16 | None | 0.25 | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-k0.45-a259d07c` | `fox` | pope/adversarial | fp16 | None | 0.45 | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-t0.1-4476086d` | `fox` | pope/adversarial | fp16 | None | None | 0.1 | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-t0.2-748a28e2` | `fox` | pope/adversarial | fp16 | None | None | 0.2 | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-01013bf7` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-08d18927` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `68374d2` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-5e27e634` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `6d14816` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-93866aba` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-c7d23f9e` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-dc856055` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `6d14816` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-g4-01a74670` | `fox` | pope/adversarial | fp16 | 4.0 | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-43__dt-fp16__v-sampling-01013bf7` | `fox` | pope/adversarial | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-43__dt-fp16__v-sampling-08d18927` | `fox` | pope/adversarial | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-43__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/adversarial | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-43__dt-fp16__v-sampling-5e27e634` | `fox` | pope/adversarial | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `6d14816` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-43__dt-fp16__v-sampling-93866aba` | `fox` | pope/adversarial | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-43__dt-fp16__v-sampling-c7d23f9e` | `fox` | pope/adversarial | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-43__dt-fp16__v-sampling-dc856055` | `fox` | pope/adversarial | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `6d14816` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-43__dt-fp16__v-sampling-g4-01a74670` | `fox` | pope/adversarial | fp16 | 4.0 | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-44__dt-fp16__v-sampling-01013bf7` | `fox` | pope/adversarial | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-44__dt-fp16__v-sampling-08d18927` | `fox` | pope/adversarial | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-44__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/adversarial | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-44__dt-fp16__v-sampling-5e27e634` | `fox` | pope/adversarial | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `['0ec984e', '6d14816']` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-44__dt-fp16__v-sampling-93866aba` | `fox` | pope/adversarial | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-44__dt-fp16__v-sampling-c7d23f9e` | `fox` | pope/adversarial | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-44__dt-fp16__v-sampling-dc856055` | `fox` | pope/adversarial | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `6d14816` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-44__dt-fp16__v-sampling-g4-01a74670` | `fox` | pope/adversarial | fp16 | 4.0 | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-45__dt-fp16__v-sampling-01013bf7` | `fox` | pope/adversarial | fp16 | None | None | None | 45 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-45__dt-fp16__v-sampling-08d18927` | `fox` | pope/adversarial | fp16 | None | None | None | 45 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-45__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/adversarial | fp16 | None | None | None | 45 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-45__dt-fp16__v-sampling-93866aba` | `fox` | pope/adversarial | fp16 | None | None | None | 45 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-45__dt-fp16__v-sampling-c7d23f9e` | `fox` | pope/adversarial | fp16 | None | None | None | 45 | 3000 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-45__dt-fp16__v-sampling-dc856055` | `fox` | pope/adversarial | fp16 | None | None | None | 45 | 3000 | `llava-hf/llava-1.5-7b-hf` | `6d14816` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-45__dt-fp16__v-sampling-g4-01a74670` | `fox` | pope/adversarial | fp16 | 4.0 | None | None | 45 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-46__dt-fp16__v-sampling-01013bf7` | `fox` | pope/adversarial | fp16 | None | None | None | 46 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-46__dt-fp16__v-sampling-08d18927` | `fox` | pope/adversarial | fp16 | None | None | None | 46 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-46__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/adversarial | fp16 | None | None | None | 46 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-46__dt-fp16__v-sampling-93866aba` | `fox` | pope/adversarial | fp16 | None | None | None | 46 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-46__dt-fp16__v-sampling-c7d23f9e` | `fox` | pope/adversarial | fp16 | None | None | None | 46 | 3000 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-46__dt-fp16__v-sampling-dc856055` | `fox` | pope/adversarial | fp16 | None | None | None | 46 | 3000 | `llava-hf/llava-1.5-7b-hf` | `6d14816` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-46__dt-fp16__v-sampling-g4-01a74670` | `fox` | pope/adversarial | fp16 | 4.0 | None | None | 46 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-47__dt-fp16__v-sampling-01013bf7` | `fox` | pope/adversarial | fp16 | None | None | None | 47 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-47__dt-fp16__v-sampling-08d18927` | `fox` | pope/adversarial | fp16 | None | None | None | 47 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-47__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/adversarial | fp16 | None | None | None | 47 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-47__dt-fp16__v-sampling-93866aba` | `fox` | pope/adversarial | fp16 | None | None | None | 47 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-47__dt-fp16__v-sampling-c7d23f9e` | `fox` | pope/adversarial | fp16 | None | None | None | 47 | 3000 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-47__dt-fp16__v-sampling-dc856055` | `fox` | pope/adversarial | fp16 | None | None | None | 47 | 3000 | `llava-hf/llava-1.5-7b-hf` | `6d14816` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-47__dt-fp16__v-sampling-g4-01a74670` | `fox` | pope/adversarial | fp16 | 4.0 | None | None | 47 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-48__dt-fp16__v-sampling-01013bf7` | `fox` | pope/adversarial | fp16 | None | None | None | 48 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-48__dt-fp16__v-sampling-08d18927` | `fox` | pope/adversarial | fp16 | None | None | None | 48 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-48__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/adversarial | fp16 | None | None | None | 48 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-48__dt-fp16__v-sampling-93866aba` | `fox` | pope/adversarial | fp16 | None | None | None | 48 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-48__dt-fp16__v-sampling-c7d23f9e` | `fox` | pope/adversarial | fp16 | None | None | None | 48 | 3000 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-48__dt-fp16__v-sampling-dc856055` | `fox` | pope/adversarial | fp16 | None | None | None | 48 | 3000 | `llava-hf/llava-1.5-7b-hf` | `6d14816` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-48__dt-fp16__v-sampling-g4-01a74670` | `fox` | pope/adversarial | fp16 | 4.0 | None | None | 48 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-49__dt-fp16__v-sampling-01013bf7` | `fox` | pope/adversarial | fp16 | None | None | None | 49 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-49__dt-fp16__v-sampling-08d18927` | `fox` | pope/adversarial | fp16 | None | None | None | 49 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-49__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/adversarial | fp16 | None | None | None | 49 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-49__dt-fp16__v-sampling-93866aba` | `fox` | pope/adversarial | fp16 | None | None | None | 49 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-49__dt-fp16__v-sampling-c7d23f9e` | `fox` | pope/adversarial | fp16 | None | None | None | 49 | 3000 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-49__dt-fp16__v-sampling-dc856055` | `fox` | pope/adversarial | fp16 | None | None | None | 49 | 3000 | `llava-hf/llava-1.5-7b-hf` | `6d14816` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-49__dt-fp16__v-sampling-g4-01a74670` | `fox` | pope/adversarial | fp16 | 4.0 | None | None | 49 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-50__dt-fp16__v-sampling-01013bf7` | `fox` | pope/adversarial | fp16 | None | None | None | 50 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-50__dt-fp16__v-sampling-08d18927` | `fox` | pope/adversarial | fp16 | None | None | None | 50 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-50__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/adversarial | fp16 | None | None | None | 50 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-50__dt-fp16__v-sampling-93866aba` | `fox` | pope/adversarial | fp16 | None | None | None | 50 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-50__dt-fp16__v-sampling-c7d23f9e` | `fox` | pope/adversarial | fp16 | None | None | None | 50 | 3000 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-50__dt-fp16__v-sampling-dc856055` | `fox` | pope/adversarial | fp16 | None | None | None | 50 | 3000 | `llava-hf/llava-1.5-7b-hf` | `6d14816` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-50__dt-fp16__v-sampling-g4-01a74670` | `fox` | pope/adversarial | fp16 | 4.0 | None | None | 50 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-51__dt-fp16__v-sampling-01013bf7` | `fox` | pope/adversarial | fp16 | None | None | None | 51 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-51__dt-fp16__v-sampling-08d18927` | `fox` | pope/adversarial | fp16 | None | None | None | 51 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-51__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/adversarial | fp16 | None | None | None | 51 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-51__dt-fp16__v-sampling-93866aba` | `fox` | pope/adversarial | fp16 | None | None | None | 51 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-51__dt-fp16__v-sampling-c7d23f9e` | `fox` | pope/adversarial | fp16 | None | None | None | 51 | 3000 | `llava-hf/llava-1.5-7b-hf` | `['5bbbec3', '7d52f44']` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-51__dt-fp16__v-sampling-dc856055` | `fox` | pope/adversarial | fp16 | None | None | None | 51 | 3000 | `llava-hf/llava-1.5-7b-hf` | `6d14816` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-51__dt-fp16__v-sampling-g4-01a74670` | `fox` | pope/adversarial | fp16 | 4.0 | None | None | 51 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-997__dt-fp16__v-greedy-17d4eb8d` | `fox` | pope/adversarial | fp16 | None | None | None | 997 | 50 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-popular__seed-42__dt-fp16__v-greedy-17d4eb8d` | `fox` | pope/popular | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-popular__seed-42__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/popular | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-popular__seed-43__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/popular | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-popular__seed-44__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/popular | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-random__seed-42__dt-fp16__v-greedy-17d4eb8d` | `fox` | pope/random | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-random__seed-42__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/random | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-random__seed-43__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/random | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-random__seed-44__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/random | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-icd__b-chair__s-val2014_500__seed-42__dt-fp16__v-greedy-4a534adb` | `icd` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `ced26c0` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-icd__b-chair__s-val2014_500__seed-42__dt-fp16__v-sampling-5c7c06df` | `icd` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `ced26c0` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-icd__b-chair__s-val2014_500__seed-43__dt-fp16__v-sampling-5c7c06df` | `icd` | chair/val2014_500 | fp16 | None | None | None | 43 | 500 | `llava-hf/llava-1.5-7b-hf` | `ced26c0` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-icd__b-chair__s-val2014_500__seed-44__dt-fp16__v-sampling-5c7c06df` | `icd` | chair/val2014_500 | fp16 | None | None | None | 44 | 500 | `llava-hf/llava-1.5-7b-hf` | `3990a41` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-icd__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-17d4eb8d` | `icd` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `a22119b` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-icd__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-851cf803` | `icd` | pope/adversarial | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `ced26c0` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-icd__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-2d07edcb` | `icd` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `a22119b` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-icd__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-6ad493ca` | `icd` | pope/adversarial | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `a22119b` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-icd__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-a0cac4b7` | `icd` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `a22119b` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-icd__b-pope__s-adversarial__seed-43__dt-fp16__v-sampling-2d07edcb` | `icd` | pope/adversarial | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `a22119b` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-icd__b-pope__s-adversarial__seed-44__dt-fp16__v-sampling-2d07edcb` | `icd` | pope/adversarial | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `ced26c0` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-opera__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-109f4ec7` | `opera` | pope/adversarial | fp16 | None | None | None | 42 | 1500 | `llava-hf/llava-1.5-7b-hf` | `90e4b12` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-opera__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-17d4eb8d` | `opera` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `a74eb34` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-opera__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-3992f856` | `opera` | pope/adversarial | fp16 | None | None | None | 42 | 1500 | `llava-hf/llava-1.5-7b-hf` | `90e4b12` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-opera__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-5e8ffbea` | `opera` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `0ec984e` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-opera__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-91c9bf8b` | `opera` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `a74eb34` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-opera__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-f8b39d91` | `opera` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `a74eb34` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-opera__b-pope__s-popular__seed-42__dt-fp16__v-greedy-17d4eb8d` | `opera` | pope/popular | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `a74eb34` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-opera__b-pope__s-random__seed-42__dt-fp16__v-greedy-17d4eb8d` | `opera` | pope/random | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `a74eb34` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500__seed-42__dt-fp16__v-greedy-4a534adb` | `sampling` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `68374d2` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500__seed-42__dt-fp16__v-sampling-5c7c06df` | `sampling` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `68374d2` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500__seed-43__dt-fp16__v-sampling-5c7c06df` | `sampling` | chair/val2014_500 | fp16 | None | None | None | 43 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500__seed-44__dt-fp16__v-sampling-5c7c06df` | `sampling` | chair/val2014_500 | fp16 | None | None | None | 44 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500__seed-45__dt-fp16__v-sampling-5c7c06df` | `sampling` | chair/val2014_500 | fp16 | None | None | None | 45 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500__seed-46__dt-fp16__v-sampling-5c7c06df` | `sampling` | chair/val2014_500 | fp16 | None | None | None | 46 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500__seed-47__dt-fp16__v-sampling-5c7c06df` | `sampling` | chair/val2014_500 | fp16 | None | None | None | 47 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500__seed-48__dt-fp16__v-sampling-5c7c06df` | `sampling` | chair/val2014_500 | fp16 | None | None | None | 48 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500__seed-49__dt-fp16__v-sampling-5c7c06df` | `sampling` | chair/val2014_500 | fp16 | None | None | None | 49 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500__seed-50__dt-fp16__v-sampling-5c7c06df` | `sampling` | chair/val2014_500 | fp16 | None | None | None | 50 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500__seed-51__dt-fp16__v-sampling-5c7c06df` | `sampling` | chair/val2014_500 | fp16 | None | None | None | 51 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500_d1__seed-42__dt-fp16__v-greedy-4a534adb` | `sampling` | chair/val2014_500_d1 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500_d2__seed-42__dt-fp16__v-greedy-4a534adb` | `sampling` | chair/val2014_500_d2 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500_d3__seed-42__dt-fp16__v-greedy-4a534adb` | `sampling` | chair/val2014_500_d3 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500_d4__seed-42__dt-fp16__v-greedy-4a534adb` | `sampling` | chair/val2014_500_d4 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-17d4eb8d` | `sampling` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `68374d2` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-adversarial__seed-43__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/adversarial | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-adversarial__seed-44__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/adversarial | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-adversarial__seed-45__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/adversarial | fp16 | None | None | None | 45 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-adversarial__seed-46__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/adversarial | fp16 | None | None | None | 46 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-adversarial__seed-47__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/adversarial | fp16 | None | None | None | 47 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-adversarial__seed-48__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/adversarial | fp16 | None | None | None | 48 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-adversarial__seed-49__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/adversarial | fp16 | None | None | None | 49 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-adversarial__seed-50__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/adversarial | fp16 | None | None | None | 50 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-adversarial__seed-51__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/adversarial | fp16 | None | None | None | 51 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-popular__seed-42__dt-fp16__v-greedy-17d4eb8d` | `sampling` | pope/popular | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-popular__seed-42__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/popular | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-popular__seed-43__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/popular | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-popular__seed-44__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/popular | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-random__seed-42__dt-fp16__v-greedy-17d4eb8d` | `sampling` | pope/random | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-random__seed-42__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/random | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-random__seed-43__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/random | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-random__seed-44__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/random | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sid__b-chair__s-val2014_500__seed-42__dt-fp16__v-greedy-4a534adb` | `sid` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `ced26c0` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sid__b-chair__s-val2014_500__seed-42__dt-fp16__v-sampling-5c7c06df` | `sid` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `ced26c0` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sid__b-chair__s-val2014_500__seed-43__dt-fp16__v-sampling-5c7c06df` | `sid` | chair/val2014_500 | fp16 | None | None | None | 43 | 500 | `llava-hf/llava-1.5-7b-hf` | `ced26c0` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sid__b-chair__s-val2014_500__seed-44__dt-fp16__v-sampling-5c7c06df` | `sid` | chair/val2014_500 | fp16 | None | None | None | 44 | 500 | `llava-hf/llava-1.5-7b-hf` | `ced26c0` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sid__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-17d4eb8d` | `sid` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `a22119b` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sid__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-851cf803` | `sid` | pope/adversarial | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `ced26c0` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sid__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-2d07edcb` | `sid` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `a22119b` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sid__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-6234e7b0` | `sid` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `a22119b` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sid__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-6ad493ca` | `sid` | pope/adversarial | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `a22119b` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sid__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-a0cac4b7` | `sid` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `a22119b` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sid__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-d31f34b3` | `sid` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `a22119b` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sid__b-pope__s-adversarial__seed-43__dt-fp16__v-sampling-2d07edcb` | `sid` | pope/adversarial | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `a22119b` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sid__b-pope__s-adversarial__seed-44__dt-fp16__v-sampling-2d07edcb` | `sid` | pope/adversarial | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `ced26c0` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-vcd__b-chair__s-val2014_500__seed-42__dt-fp16__v-greedy-4a534adb` | `vcd` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `ced26c0` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-vcd__b-chair__s-val2014_500__seed-42__dt-fp16__v-sampling-5c7c06df` | `vcd` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `ced26c0` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-vcd__b-chair__s-val2014_500__seed-43__dt-fp16__v-sampling-5c7c06df` | `vcd` | chair/val2014_500 | fp16 | None | None | None | 43 | 500 | `llava-hf/llava-1.5-7b-hf` | `ced26c0` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-vcd__b-chair__s-val2014_500__seed-44__dt-fp16__v-sampling-5c7c06df` | `vcd` | chair/val2014_500 | fp16 | None | None | None | 44 | 500 | `llava-hf/llava-1.5-7b-hf` | `ced26c0` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-vcd__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-17d4eb8d` | `vcd` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `a22119b` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-vcd__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-851cf803` | `vcd` | pope/adversarial | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `ced26c0` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-vcd__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-2d07edcb` | `vcd` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `a22119b` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-vcd__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-6ad493ca` | `vcd` | pope/adversarial | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `a22119b` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-vcd__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-a0cac4b7` | `vcd` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `a22119b` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-vcd__b-pope__s-adversarial__seed-43__dt-fp16__v-sampling-2d07edcb` | `vcd` | pope/adversarial | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `a22119b` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-vcd__b-pope__s-adversarial__seed-44__dt-fp16__v-sampling-2d07edcb` | `vcd` | pope/adversarial | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `ced26c0` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-vcd_fox_fusion__b-chair__s-val2014_500__seed-42__dt-fp16__v-sampling-5c7c06df` | `vcd_fox_fusion` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-vcd_fox_fusion__b-chair__s-val2014_500__seed-43__dt-fp16__v-sampling-5c7c06df` | `vcd_fox_fusion` | chair/val2014_500 | fp16 | None | None | None | 43 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-vcd_fox_fusion__b-chair__s-val2014_500__seed-44__dt-fp16__v-sampling-5c7c06df` | `vcd_fox_fusion` | chair/val2014_500 | fp16 | None | None | None | 44 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-vcd_fox_fusion__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-2d07edcb` | `vcd_fox_fusion` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-vcd_fox_fusion__b-pope__s-adversarial__seed-43__dt-fp16__v-sampling-2d07edcb` | `vcd_fox_fusion` | pope/adversarial | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-vcd_fox_fusion__b-pope__s-adversarial__seed-44__dt-fp16__v-sampling-2d07edcb` | `vcd_fox_fusion` | pope/adversarial | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |

CHAIR scorer: synonyms `chair_synonyms.txt` sha256 `a4c231100e8a...` (transcription-of-rohrbach-synonyms-txt), tokenizer `regex`, gt_source `both`

**Warnings**

- `fox` run with **gamma = None**: the intervention constant was not recorded.
