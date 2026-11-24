# GPT-judge substitute for Figure 6 / Table 5 (gpt-4.1-2025-04-14)

Judge `gpt-4.1-2025-04-14` (substitute for deprecated GPT-4V). 50 COCO val2014 images (first by sorted image_id of the CHAIR set), LLaVA-1.5 only, seed 42, sampling decoding. Calls 100, unparsed 0.

Scores are 1-10. Difference is Fox minus Sampling; interval is a paired bootstrap 90% CI over images.

| criterion | reading | n | Sampling | Fox | Fox - Sampling | 90% CI | Fox/tie/Sampling |
|---|---|---|---|---|---|---|---|
| accuracy | paper-style (Assistant 1 = Sampling) | 50 | 5.88 | 5.68 | -0.20 | [-0.62, +0.22] | 17/6/27 |
| accuracy | swapped (Assistant 1 = Fox) | 50 | 5.22 | 7.14 | +1.92 | [+1.50, +2.32] | 45/0/5 |
| accuracy | order-averaged (our addition) | 50 | 5.55 | 6.41 | +0.86 | [+0.47, +1.25] | 33/8/9 |
| detailedness | paper-style (Assistant 1 = Sampling) | 50 | 6.06 | 6.12 | +0.06 | [-0.22, +0.36] | 18/10/22 |
| detailedness | swapped (Assistant 1 = Fox) | 50 | 6.24 | 6.52 | +0.28 | [-0.06, +0.60] | 26/5/19 |
| detailedness | order-averaged (our addition) | 50 | 6.15 | 6.32 | +0.17 | [-0.11, +0.45] | 20/11/19 |

Params actually sent: `{"detail": "auto", "max_tokens": 512, "temperature": 0.2}`. Raw replies are in the JSON beside this file.
