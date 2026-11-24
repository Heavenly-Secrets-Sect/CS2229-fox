# GPT-judge substitute for Figure 6 / Table 5 (gpt-4.1-2025-04-14)

Judge `gpt-4.1-2025-04-14` (substitute for deprecated GPT-4V). 50 COCO val2014 images (first by sorted image_id of the CHAIR set), LLaVA-1.5 only, seed 42, sampling decoding. Calls 100, unparsed 0.

Scores are 1-10. Difference is Fox minus Sampling; interval is a paired bootstrap 90% CI over images.

| criterion | reading | n | Sampling | Fox | Fox - Sampling | 90% CI | Fox/tie/Sampling |
|---|---|---|---|---|---|---|---|
| accuracy | paper-style (Assistant 1 = Sampling) | 50 | 5.36 | 4.68 | -0.68 | [-1.36, +0.00] | 16/2/32 |
| accuracy | swapped (Assistant 1 = Fox) | 50 | 4.00 | 5.78 | +1.78 | [+1.20, +2.34] | 39/1/10 |
| accuracy | order-averaged (our addition) | 50 | 4.68 | 5.23 | +0.55 | [-0.05, +1.16] | 26/4/20 |
| detailedness | paper-style (Assistant 1 = Sampling) | 50 | 5.68 | 5.46 | -0.22 | [-0.66, +0.24] | 20/3/27 |
| detailedness | swapped (Assistant 1 = Fox) | 50 | 5.30 | 5.62 | +0.32 | [-0.08, +0.72] | 27/3/20 |
| detailedness | order-averaged (our addition) | 50 | 5.49 | 5.54 | +0.05 | [-0.34, +0.45] | 22/6/22 |

Params actually sent: `{"detail": "auto", "max_tokens": 512, "temperature": 0.2}`. Raw replies are in the JSON beside this file.
