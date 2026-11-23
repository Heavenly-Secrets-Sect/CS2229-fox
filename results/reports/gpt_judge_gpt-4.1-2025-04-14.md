# GPT-judge substitute for Figure 6 / Table 5 (gpt-4.1-2025-04-14)

Judge `gpt-4.1-2025-04-14` (substitute for deprecated GPT-4V). 50 COCO val2014 images (first by sorted image_id of the CHAIR set), LLaVA-1.5 only, seed 42, sampling decoding. Calls 100, unparsed 0.

Scores are 1-10. Difference is Fox minus Sampling; interval is a paired bootstrap 90% CI over images.

| criterion | reading | n | Sampling | Fox | Fox - Sampling | 90% CI | Fox/tie/Sampling |
|---|---|---|---|---|---|---|---|
| accuracy | paper-style (Assistant 1 = Sampling) | 50 | 6.26 | 5.76 | -0.50 | [-1.02, +0.02] | 19/2/29 |
| accuracy | swapped (Assistant 1 = Fox) | 50 | 5.32 | 6.64 | +1.32 | [+0.78, +1.86] | 34/1/15 |
| accuracy | order-averaged (our addition) | 50 | 5.79 | 6.20 | +0.41 | [-0.10, +0.92] | 27/6/17 |
| detailedness | paper-style (Assistant 1 = Sampling) | 50 | 6.30 | 6.18 | -0.12 | [-0.44, +0.22] | 23/1/26 |
| detailedness | swapped (Assistant 1 = Fox) | 50 | 6.22 | 6.16 | -0.06 | [-0.36, +0.26] | 21/4/25 |
| detailedness | order-averaged (our addition) | 50 | 6.26 | 6.17 | -0.09 | [-0.37, +0.20] | 17/12/21 |

Params actually sent: `{"detail": "auto", "max_tokens": 512, "temperature": 0.2}`. Raw replies are in the JSON beside this file.
