# r13 diagnostics — derived report

`diagnostics_r13.txt` is the verbatim output of

```bash
python scripts/analyse_diagnostics.py --json results/reports/diagnostics_r13.json
```

over the r13 session-1 records: 3000 POPE-adversarial questions, one forward
pass each, 32 layers × 32 heads, no generation. `diagnostics_r13.json` is the
same content machine-readable.

These are committed because the 234 MB of raw per-head statistics behind them
cannot be (`../runs/README.md`). Re-running the script against a restored runs
root must reproduce them; if it does not, the runs root changed.

The reading is in `docs/VALIDATION.md` §"Results, round 13". In one paragraph:
Eq. 13's denominator is a ~4 % residual of 1024 near-cancelling terms, and its
sign — not any part of the mechanism — decides whether the published AUC is
0.13 or 0.87; `m_sys` moves *opposite* to the paper's stated direction in the
layers where its signal is strongest, while unweighted `m_sys` is at chance;
and the L2–15 window the released code ships scores below the all-layer
average on the paper's own window-selection criterion. The discriminative
signal is real and, sign-corrected, stronger than the paper reports (0.8735
held out against 0.8226) — but it is not the quantity the paper says it is.
