# Figures 12 and 14 re-measured from the r13 diagnostic data

Regenerate: `python3 -I scripts/analyse_figures_12_14.py --data-dir <extracted r13 zip> --out-dir results/reports`.

Samples: 3000 POPE-adversarial questions, 459 hallucinated (greedy false positives, 15.3%). Labels from `m-sampling__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-0b099584.jsonl`. The paper's 1,000 + 1,000 cannot be formed from this data; see the script docstring.

## Figure 12 - global visual attention mass m_V

Paper: tail=32 faithful 0.1653 +- 0.0086, hallucinated 0.1666 +- 0.0072 (diff +0.0013); tail=1 0.1049 +- 0.0071 vs 0.1067 +- 0.0063 (diff +0.0018).

| tail | faithful pool | n_F | mean F | mean H | H-F | sd F / sd H | Cohen d | AUC (H pos) [95% CI] | KS D (p) | overlap |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | all_rest | 2541 | 0.2264 | 0.2268 | +0.0005 | 0.0099 / 0.0097 | +0.05 | 0.503 [0.477, 0.531] | 0.047 (3.4e-01) | 0.91 |
| 32 | correct | 2372 | 0.2264 | 0.2268 | +0.0004 | 0.0099 / 0.0097 | +0.05 | 0.503 [0.473, 0.528] | 0.048 (3.2e-01) | 0.90 |
| 32 | tn | 1041 | 0.2246 | 0.2268 | +0.0023 | 0.0099 / 0.0097 | +0.23 | 0.555 [0.529, 0.582] | 0.093 (7.6e-03) | 0.87 |
| 32 | tp | 1331 | 0.2278 | 0.2268 | -0.0010 | 0.0097 / 0.0097 | -0.10 | 0.463 [0.433, 0.492] | 0.075 (4.2e-02) | 0.89 |
| 1 | all_rest | 2541 | 0.1053 | 0.1076 | +0.0022 | 0.0069 / 0.0066 | +0.32 | 0.592 [0.562, 0.617] | 0.139 (5.1e-07) | 0.84 |
| 1 | correct | 2372 | 0.1053 | 0.1076 | +0.0022 | 0.0069 / 0.0066 | +0.32 | 0.593 [0.563, 0.618] | 0.144 (1.9e-07) | 0.84 |
| 1 | tn | 1041 | 0.1032 | 0.1076 | +0.0043 | 0.0066 / 0.0066 | +0.65 | 0.679 [0.651, 0.707] | 0.260 (2.1e-19) | 0.74 |
| 1 | tp | 1331 | 0.1070 | 0.1076 | +0.0006 | 0.0067 / 0.0066 | +0.09 | 0.526 [0.498, 0.555] | 0.055 (2.5e-01) | 0.90 |

Class-balanced draws (as many faithful as hallucinated, 200 draws): tail=32 all_rest: AUC 0.505 +- 0.013, KS 0.055; tail=32 correct: AUC 0.504 +- 0.012, KS 0.056; tail=32 tn: AUC 0.556 +- 0.009, KS 0.101; tail=32 tp: AUC 0.463 +- 0.011, KS 0.083; tail=1 all_rest: AUC 0.593 +- 0.013, KS 0.148; tail=1 correct: AUC 0.593 +- 0.013, KS 0.153; tail=1 tn: AUC 0.679 +- 0.010, KS 0.265; tail=1 tp: AUC 0.527 +- 0.011, KS 0.065.

## Figure 14 - Top-K projected joint risk S = m_sys * H_vis (tail=1)

Paper: Top-32 ROC AUC 0.8180; all heads 0.7852 (global aggregation dilutes). Labels: 459 hallucinated of 3000 (prevalence 0.153).

| K | in-sample AUC | held-out AUC (10 half-splits) |
|---|---|---|
| 1 | 0.7836 | 0.7815 +- 0.0081 |
| 2 | 0.8114 | 0.8100 +- 0.0116 |
| 4 | 0.8452 | 0.8417 +- 0.0075 |
| 8 | 0.8474 | 0.8461 +- 0.0057 |
| 16 | 0.8590 | 0.8549 +- 0.0054 |
| 32 | 0.8681 | 0.8657 +- 0.0064 |
| 64 | 0.8728 | 0.8660 +- 0.0075 |
| 128 | 0.8654 | 0.8612 +- 0.0101 |
| 256 | 0.8438 | 0.8360 +- 0.0188 |
| 512 | 0.8213 | 0.8083 +- 0.0299 |
| 1024 | 0.8034 | 0.7918 +- 0.0399 |

- Top-32 AP 0.4766 in-sample, 0.4708 held out (chance = prevalence 0.1530).
- All 1024 heads, unweighted mean z: AUC 0.5224; all heads with the dAUC weights (Eq. 19 with H_K = everything): 0.8034.
- Permutation null (labels shuffled, Top-32 re-selected on the shuffled labels): mean AUC 0.5785, max 0.5977; all-heads weighted 0.5622. This is the in-sample selection bias.
- Top-32 heads carry 13.6% of sum|dAUC| over 1024 heads (uniform would be 3.1%); 14/32 have dAUC > 0; per layer: {'11': 1, '12': 1, '14': 2, '15': 1, '16': 2, '17': 2, '18': 1, '19': 2, '20': 2, '21': 2, '22': 3, '23': 1, '24': 2, '25': 2, '26': 1, '27': 1, '29': 3, '30': 1, '31': 2}.
- Matched control, hallucinated vs right-'no' (TN): n=1500 (prevalence 0.306); Top-32 AUC 0.9885 in-sample, 0.9869 held out; AP 0.9712 held out; K=1 0.9506, K=4 0.9773, all-heads weighted 0.9646.
- Matched control, hallucinated vs right-'yes' (TP): n=1790 (prevalence 0.256); Top-32 AUC 0.8818 in-sample, 0.8779 held out; AP 0.6604 held out; K=1 0.8480, K=4 0.8683, all-heads weighted 0.8693.
- Top-32 heads (layer.head, dAUC): 25.28 (-0.284), 24.7 (-0.276), 27.16 (-0.265), 30.12 (-0.252), 22.8 (-0.249), 17.8 (+0.244), 31.27 (-0.240), 19.18 (-0.224), 19.4 (-0.217), 14.23 (+0.217), 20.31 (-0.215), 21.25 (-0.213), 21.14 (-0.213), 25.4 (-0.212), 29.30 (+0.211), 22.20 (+0.207), 31.24 (+0.207), 23.17 (-0.207), 26.19 (-0.204), 29.15 (-0.203), 15.19 (+0.201), 29.10 (-0.197), 17.5 (+0.194), 16.5 (+0.188), 14.28 (+0.187), 22.29 (+0.186), 24.18 (+0.183), 16.3 (+0.182), 18.15 (-0.180), 12.27 (+0.177), 20.11 (+0.176), 11.7 (-0.175).
