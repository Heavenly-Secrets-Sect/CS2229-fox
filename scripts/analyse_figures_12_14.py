#!/usr/bin/env python3
"""Paper Figures 12 and 14 from the round-13 diagnostic data. CPU only.

    python3 -I scripts/analyse_figures_12_14.py --data-dir <dir with the r13 jsonl> \
        --out-dir results/reports

``--data-dir`` holds ``m-diagnose__*.jsonl`` (per-head attention statistics,
one forward pass per sample, no generation) and the archived greedy
``m-sampling__b-pope__s-adversarial__seed-42__*greedy*.jsonl`` runs that supply
the labels. The raw data is 234 MB and lives only in the r13 zip; extract it
into a scratch directory, never into the repo.

Figure 12  distribution of m_V,tail (mean over all layers and heads) for
           faithful vs hallucinated samples, tail = 32 and tail = 1.
Figure 14  with S = m_sys * H_vis at tail = 1: (a) PR curve of the Top-32
           projected score, (b) the Top-32 heads by |dAUC|, (c) ROC AUC vs K.
Figure 15  needs a pass WITH the intervention; see ``docs/VALIDATION.md``.

Sample selection is the one thing the paper does not say ("1,000 Faithful and
1,000 Hallucinated"). POPE-adversarial has 3000 questions and the greedy
baseline gets ~459 of them wrong in the hallucinating direction (answers YES
about an absent object), so 1000 hallucinated samples do not exist in this
data. What is done instead is stated in the output, not hidden:

  hallucinated  = false positives of the greedy baseline (all of them);
  faithful      = one of three pools, reported side by side:
      all_rest   every other sample (misses included)  -- what Fig 3 used
      correct    answers that were right (TP + TN)
      tn         right "no" answers: same ground truth as every hallucinated
                 sample, so object presence cannot explain a difference, but
                 a different ANSWER (the probe may just read the answer)
      tp         right "yes" answers: same ANSWER as every hallucinated sample,
                 different truth -- the control that isolates hallucination
  plus a class-balanced subsample (as many faithful as hallucinated, seeded,
  averaged over many draws). AUC does not depend on class balance; the
  balanced draw is there so the plotted histograms are comparable in height.

The statistics (``auc_from_ranks``, ``ks_two_sample``, ``average_precision``,
``topk_projection`` ...) are pure functions over arrays so they can be tested
on synthetic data (tests/test_figures_12_14.py).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

STATS = ("m_vis_t1", "m_vis_t32", "m_sys_t1", "h_vis_t1")

# What the paper prints on Figure 12 and in Appendix A.4.
PAPER = {
    "fig12": {
        "tail32": {"faithful": (0.1653, 0.0086), "halluc": (0.1666, 0.0072),
                   "diff": 0.0013},
        "tail1": {"faithful": (0.1049, 0.0071), "halluc": (0.1067, 0.0063),
                  "diff": 0.0018},
    },
    "fig14": {"top32_auc": 0.8180, "all_heads_auc": 0.7852},
}


# ---------------------------------------------------------------- statistics

def average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n down axis 0 with ties given their average rank."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return average_ranks(x[:, None])[:, 0]
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        col = x[:, j]
        order = np.argsort(col, kind="mergesort")
        s = col[order]
        n = len(s)
        # start/end of each run of equal values
        new = np.empty(n, dtype=bool)
        new[0] = True
        new[1:] = s[1:] != s[:-1]
        starts = np.flatnonzero(new)
        ends = np.append(starts[1:], n)
        avg = (starts + ends + 1) / 2.0            # mean of ranks start+1..end
        run = np.cumsum(new) - 1
        r = np.empty(n)
        r[order] = avg[run]
        out[:, j] = r
    return out


def auc_from_ranks(ranks: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mann-Whitney AUC with ``labels`` True as the positive class.

    ``ranks`` is ``average_ranks`` of the scores; computing it once lets the
    permutation null reuse it, because permuting labels does not change ranks.
    """
    labels = np.asarray(labels, dtype=bool)
    n_p = int(labels.sum())
    n_n = len(labels) - n_p
    if n_p == 0 or n_n == 0:
        return np.full(ranks.shape[1:], np.nan)
    return (ranks[labels].sum(axis=0) - n_p * (n_p + 1) / 2.0) / (n_p * n_n)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(auc_from_ranks(average_ranks(scores), labels))


def ks_two_sample(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and its asymptotic p-value."""
    a = np.sort(np.asarray(a, float))
    b = np.sort(np.asarray(b, float))
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    en = math.sqrt(len(a) * len(b) / (len(a) + len(b)))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 1e-9:
        return d, 1.0
    p = 2.0 * sum((-1) ** (k - 1) * math.exp(-2.0 * (k * lam) ** 2)
                  for k in range(1, 101))
    return d, float(min(1.0, max(0.0, p)))


def overlap_coefficient(a: np.ndarray, b: np.ndarray, bins: int = 40) -> float:
    """Shared area of the two density histograms on a common grid (1 = same)."""
    lo = float(min(np.min(a), np.min(b)))
    hi = float(max(np.max(a), np.max(b)))
    if hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    pa = np.histogram(a, edges)[0] / len(a)
    pb = np.histogram(b, edges)[0] / len(b)
    return float(np.minimum(pa, pb).sum())


def cohens_d(pos: np.ndarray, neg: np.ndarray) -> float:
    vp, vn = np.var(pos, ddof=1), np.var(neg, ddof=1)
    pooled = math.sqrt(((len(pos) - 1) * vp + (len(neg) - 1) * vn)
                       / (len(pos) + len(neg) - 2))
    return float((np.mean(pos) - np.mean(neg)) / pooled) if pooled else 0.0


def mannwhitney_p(auc_value: float, n_p: int, n_n: int) -> float:
    """Two-sided normal-approximation p-value (no tie correction)."""
    mu = n_p * n_n / 2.0
    sd = math.sqrt(n_p * n_n * (n_p + n_n + 1) / 12.0)
    z = (auc_value * n_p * n_n - mu) / sd
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def bootstrap_auc_ci(scores: np.ndarray, labels: np.ndarray, reps: int = 300,
                     seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, bool)
    pos, neg = scores[labels], scores[~labels]
    vals = []
    for _ in range(reps):
        p = rng.choice(pos, len(pos))
        n = rng.choice(neg, len(neg))
        vals.append(auc(np.concatenate([p, n]),
                        np.r_[np.ones(len(p), bool), np.zeros(len(n), bool)]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def pr_curve(scores: np.ndarray, labels: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Precision, recall (one point per distinct threshold) and average
    precision, sklearn's step-wise definition ``sum (R_n - R_{n-1}) P_n``."""
    labels = np.asarray(labels, bool)
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    last = np.append(np.flatnonzero(s[1:] != s[:-1]), len(s) - 1)
    tp, fp = tp[last], fp[last]
    prec = tp / (tp + fp)
    rec = tp / max(1, int(labels.sum()))
    ap = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
    return prec, rec, ap


def delta_auc_matrix(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Equation 18 for every column of ``x`` [N, P]: AUC - 0.5, positive =
    hallucinated."""
    return auc_from_ranks(average_ranks(x), labels) - 0.5


def zscore(x: np.ndarray, ref: Optional[np.ndarray] = None
           ) -> np.ndarray:
    """Per-column z-score using the dataset statistics of ``ref`` (default x)."""
    ref = x if ref is None else ref
    mu = ref.mean(axis=0)
    sd = ref.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (x - mu) / sd


def topk_projection(z: np.ndarray, dauc: np.ndarray, k: int) -> np.ndarray:
    """Equation 19: s_n = sum_{top-K} z * dAUC / sum_{top-K} |dAUC|.

    The Top-K set is ranked by |dAUC|. The denominator is a sum of ABSOLUTE
    values, so unlike Eq. 13 it cannot change sign and reverse the ranking.
    """
    idx = np.argsort(-np.abs(dauc), kind="mergesort")[:k]
    den = np.abs(dauc[idx]).sum()
    return (z[:, idx] * dauc[idx]).sum(axis=1) / (den if den else 1.0)


def topk_ranked(dauc: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-np.abs(dauc), kind="mergesort")[:k]


# --------------------------------------------------------------------- data

def parse_yes_no(text: str) -> Optional[str]:
    t = str(text).strip().lower().lstrip("\"'").strip()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return None


def load_labels(path: Path) -> Dict[str, Tuple[str, str]]:
    """``sample_id -> (said, truth)``; unparsable rows are dropped."""
    out: Dict[str, Tuple[str, str]] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            said = parse_yes_no(r.get("output", ""))
            truth = str(r.get("label", "")).strip().lower()
            if said is None or truth not in ("yes", "no"):
                continue
            out[r["sample_id"]] = (said, truth)
    return out


def load_diagnostics(path: Path) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Sample ids and ``stat -> [N, L, H]`` arrays for the four used stats."""
    ids: List[str] = []
    cols: Dict[str, List[np.ndarray]] = {k: [] for k in STATS}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ex = r.get("method_extra") or {}
            dl = ex.get("diag_layers")
            if not dl:
                continue
            ids.append(r["sample_id"])
            layers = sorted(dl, key=int)
            for k in STATS:
                cols[k].append(np.array([dl[l][k] for l in layers], np.float64))
    return ids, {k: np.stack(v) for k, v in cols.items()}


def build_groups(said_truth: Sequence[Tuple[str, str]]) -> Dict[str, np.ndarray]:
    """Boolean masks over the samples: ``halluc`` and the faithful pools."""
    said = np.array([s for s, _ in said_truth])
    truth = np.array([t for _, t in said_truth])
    halluc = (said == "yes") & (truth == "no")
    correct = said == truth
    tn = (said == "no") & (truth == "no")
    tp = (said == "yes") & (truth == "yes")
    return {"halluc": halluc, "all_rest": ~halluc, "correct": correct, "tn": tn,
            "tp": tp}


# --------------------------------------------------------------- Figure 12

def fig12_for_tail(m: np.ndarray, groups: Dict[str, np.ndarray], tail: int,
                   n_draws: int = 200, seed: int = 0) -> dict:
    """``m`` is the sample-level m_V,tail, one number per sample."""
    h = groups["halluc"]
    out: dict = {"tail": tail, "n_halluc": int(h.sum()),
                 "halluc_mean": float(m[h].mean()),
                 "halluc_sd": float(m[h].std(ddof=1)), "pools": {}}
    rng = np.random.default_rng(seed)
    for name in ("all_rest", "correct", "tn", "tp"):
        f = groups[name]
        a = auc(m[h | f], h[h | f])          # AUC with halluc positive
        ks_d, ks_p = ks_two_sample(m[h], m[f])
        lo, hi = bootstrap_auc_ci(m[h | f], h[h | f])
        # balanced draws: as many faithful as hallucinated
        f_idx = np.flatnonzero(f)
        h_idx = np.flatnonzero(h)
        aucs, kss = [], []
        for _ in range(n_draws):
            pick = rng.choice(f_idx, len(h_idx), replace=False)
            idx = np.concatenate([h_idx, pick])
            y = np.r_[np.ones(len(h_idx), bool), np.zeros(len(pick), bool)]
            aucs.append(auc(m[idx], y))
            kss.append(ks_two_sample(m[h_idx], m[pick])[0])
        out["pools"][name] = {
            "n_faithful": int(f.sum()),
            "faithful_mean": float(m[f].mean()),
            "faithful_sd": float(m[f].std(ddof=1)),
            "diff_H_minus_F": float(m[h].mean() - m[f].mean()),
            "cohens_d": cohens_d(m[h], m[f]),
            "auc_halluc_positive": a,
            "auc_ci95": [lo, hi],
            "auc_mannwhitney_p": mannwhitney_p(a, int(h.sum()), int(f.sum())),
            "ks_statistic": ks_d, "ks_p": ks_p,
            "overlap_coefficient": overlap_coefficient(m[h], m[f]),
            "balanced_auc_mean": float(np.mean(aucs)),
            "balanced_auc_sd": float(np.std(aucs)),
            "balanced_ks_mean": float(np.mean(kss)),
        }
    return out


# --------------------------------------------------------------- Figure 14

KS_SWEEP = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def fig14_analysis(s: np.ndarray, labels: np.ndarray, *, seed: int = 0,
                   n_splits: int = 10, n_perm: int = 30) -> dict:
    """``s`` is [N, P] head-wise S = m_sys*H_vis (P = L*H), ``labels`` hallucinated."""
    n, p = s.shape
    z = zscore(s)
    ranks = average_ranks(s)
    dauc = auc_from_ranks(ranks, labels) - 0.5
    k32 = min(32, p)
    ks = sorted({k for k in KS_SWEEP if k <= p} | {k32})
    in_sample = {k: auc(topk_projection(z, dauc, k), labels) for k in ks}
    sc32 = topk_projection(z, dauc, k32)
    prec, rec, ap = pr_curve(sc32, labels)
    all_heads = auc(z.mean(axis=1), labels)             # replace H_K by all heads
    all_heads_weighted = auc(topk_projection(z, dauc, p), labels)

    # Held out: fit z-stats, dAUC and the Top-K on a random half, score the rest.
    rng = np.random.default_rng(seed)
    held = {k: [] for k in ks}
    held_ap32 = []
    pos_i, neg_i = np.flatnonzero(labels), np.flatnonzero(~labels)
    for _ in range(n_splits):
        tr = np.zeros(n, bool)
        tr[rng.permutation(pos_i)[: len(pos_i) // 2]] = True
        tr[rng.permutation(neg_i)[: len(neg_i) // 2]] = True
        te = ~tr
        d_tr = auc_from_ranks(average_ranks(s[tr]), labels[tr]) - 0.5
        z_te = zscore(s[te], ref=s[tr])
        for k in ks:
            held[k].append(auc(topk_projection(z_te, d_tr, k), labels[te]))
        held_ap32.append(pr_curve(topk_projection(z_te, d_tr, k32), labels[te])[2])

    # Permutation null: how much AUC does "pick the Top-32 of 1024 by |dAUC|
    # on the same labels" produce when there is nothing to find?
    null32, null_all = [], []
    for _ in range(n_perm):
        y = rng.permutation(labels)
        d = auc_from_ranks(ranks, y) - 0.5
        null32.append(auc(topk_projection(z, d, k32), y))
        null_all.append(auc(topk_projection(z, d, p), y))

    top = topk_ranked(dauc, k32)
    layer_hist = np.bincount(top // 32, minlength=p // 32)   # 32 heads per layer
    return {
        "n": int(n), "n_halluc": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "n_heads_total": int(p),
        "auc_in_sample_by_k": {str(k): in_sample[k] for k in ks},
        "auc_held_out_by_k": {str(k): float(np.mean(v)) for k, v in held.items()},
        "auc_held_out_sd_by_k": {str(k): float(np.std(v)) for k, v in held.items()},
        "top32_auc_in_sample": in_sample[k32],
        "top32_auc_held_out": float(np.mean(held[k32])),
        "all_heads_unweighted_mean_z_auc": all_heads,
        "all_heads_dauc_weighted_auc": all_heads_weighted,
        "top32_average_precision": ap,
        "top32_average_precision_held_out": float(np.mean(held_ap32)),
        "ap_chance_equals_prevalence": float(labels.mean()),
        "null_top32_auc_mean": float(np.mean(null32)),
        "null_top32_auc_max": float(np.max(null32)),
        "null_all_heads_weighted_auc_mean": float(np.mean(null_all)),
        "top32_heads": [{"layer": int(i // 32), "head": int(i % 32),
                         "dauc": float(dauc[i])} for i in top],
        "top32_by_layer": {str(l): int(c) for l, c in enumerate(layer_hist) if c},
        "top32_n_positive_dauc": int((dauc[top] > 0).sum()),
        "dauc_abs_quantiles": {str(q): float(np.quantile(np.abs(dauc), q))
                               for q in (0.5, 0.9, 0.99, 1.0)},
        "dauc_abs_share_of_top32": float(np.abs(dauc[top]).sum()
                                         / np.abs(dauc).sum()),
        "_pr": (prec, rec), "_dauc": dauc, "_top": top,
        "_ks": ks, "_held": {k: float(np.mean(v)) for k, v in held.items()},
        "_in": in_sample,
    }


# ------------------------------------------------------------------- output

def _clean(d):
    if isinstance(d, dict):
        return {k: _clean(v) for k, v in d.items() if not k.startswith("_")}
    if isinstance(d, (list, tuple)):
        return [_clean(v) for v in d]
    if isinstance(d, (np.floating,)):
        return float(d)
    if isinstance(d, (np.integer,)):
        return int(d)
    return d


def plot_fig12(path: Path, m: Dict[int, np.ndarray], groups) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), dpi=90)
    for ax, tail, title in ((axes[0], 32, "(a) tail=32"),
                            (axes[1], 1, "(b) tail=1")):
        v = m[tail]
        bins = np.linspace(v.min(), v.max(), 41)
        ax.hist(v[groups["correct"]], bins, density=True, alpha=0.55,
                color="#3b6fb6", label=f"faithful (correct, n={int(groups['correct'].sum())})")
        ax.hist(v[groups["halluc"]], bins, density=True, alpha=0.55,
                color="#e08a1e", label=f"hallucinated (FP, n={int(groups['halluc'].sum())})")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(f"mean m_vis over layers and heads (tail={tail})", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
    axes[0].set_ylabel("density", fontsize=8)
    fig.suptitle("Fig. 12 re-measured: POPE-adversarial, LLaVA-1.5-7B, greedy labels",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def plot_fig14(path: Path, r: dict) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2), dpi=90)
    prec, rec = r["_pr"]
    ax[0].plot(rec, prec, color="#3b6fb6")
    ax[0].axhline(r["prevalence"], ls="--", color="grey", lw=0.8)
    ax[0].set_xlabel("recall", fontsize=8)
    ax[0].set_ylabel("precision", fontsize=8)
    ax[0].set_title(f"(a) PR, Top-32 (AP={r['top32_average_precision']:.3f}; "
                    f"chance={r['prevalence']:.3f})", fontsize=8)
    top, dauc = r["_top"], r["_dauc"]
    vals = dauc[top]
    cols = ["#e08a1e" if v > 0 else "#3b6fb6" for v in vals]
    ax[1].bar(range(len(top)), vals, color=cols)
    ax[1].set_xticks(range(len(top)))
    ax[1].set_xticklabels([f"{i // 32}.{i % 32}" for i in top], rotation=90,
                          fontsize=5)
    ax[1].set_title("(b) Top-32 heads by |dAUC| (layer.head)", fontsize=8)
    ax[1].tick_params(axis="y", labelsize=7)
    ks = r["_ks"]
    ax[2].plot(ks, [r["_in"][k] for k in ks], "o-", ms=3, label="in-sample",
               color="#e08a1e")
    ax[2].plot(ks, [r["_held"][k] for k in ks], "s-", ms=3, label="held-out",
               color="#3b6fb6")
    ax[2].axhline(0.5, ls=":", color="grey", lw=0.8)
    ax[2].set_xscale("log", base=2)
    ax[2].set_xlabel("K", fontsize=8)
    ax[2].set_ylabel("ROC AUC", fontsize=8)
    ax[2].set_title("(c) ROC AUC vs K", fontsize=8)
    ax[2].legend(fontsize=7)
    for a in (ax[0], ax[2]):
        a.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def render_md(rep: dict) -> str:
    L: List[str] = []
    meta = rep["meta"]
    L.append("# Figures 12 and 14 re-measured from the r13 diagnostic data\n")
    L.append(f"Regenerate: `python3 -I scripts/analyse_figures_12_14.py "
             f"--data-dir <extracted r13 zip> --out-dir results/reports`.\n")
    L.append(f"Samples: {meta['n_samples']} POPE-adversarial questions, "
             f"{meta['n_halluc']} hallucinated (greedy false positives, "
             f"{meta['n_halluc'] / meta['n_samples']:.1%}). Labels from "
             f"`{meta['labels_file']}`. The paper's 1,000 + 1,000 cannot be "
             f"formed from this data; see the script docstring.\n")
    L.append("## Figure 12 - global visual attention mass m_V\n")
    L.append("Paper: tail=32 faithful 0.1653 +- 0.0086, hallucinated 0.1666 +- "
             "0.0072 (diff +0.0013); tail=1 0.1049 +- 0.0071 vs 0.1067 +- "
             "0.0063 (diff +0.0018).\n")
    L.append("| tail | faithful pool | n_F | mean F | mean H | H-F | sd F / sd H | "
             "Cohen d | AUC (H pos) [95% CI] | KS D (p) | overlap |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for tail in ("32", "1"):
        f12 = rep["figure12"][f"tail{tail}"]
        for name, p in f12["pools"].items():
            L.append(
                f"| {tail} | {name} | {p['n_faithful']} | {p['faithful_mean']:.4f} "
                f"| {f12['halluc_mean']:.4f} | {p['diff_H_minus_F']:+.4f} "
                f"| {p['faithful_sd']:.4f} / {f12['halluc_sd']:.4f} "
                f"| {p['cohens_d']:+.2f} | {p['auc_halluc_positive']:.3f} "
                f"[{p['auc_ci95'][0]:.3f}, {p['auc_ci95'][1]:.3f}] "
                f"| {p['ks_statistic']:.3f} ({p['ks_p']:.1e}) "
                f"| {p['overlap_coefficient']:.2f} |")
    L.append("")
    L.append("Class-balanced draws (as many faithful as hallucinated, 200 draws): "
             + "; ".join(
                 f"tail={t[4:]} {n}: AUC {p['balanced_auc_mean']:.3f} "
                 f"+- {p['balanced_auc_sd']:.3f}, KS {p['balanced_ks_mean']:.3f}"
                 for t in ("tail32", "tail1")
                 for n, p in rep["figure12"][t]["pools"].items()) + ".\n")
    f14 = rep["figure14"]
    L.append("## Figure 14 - Top-K projected joint risk S = m_sys * H_vis (tail=1)\n")
    L.append(f"Paper: Top-32 ROC AUC 0.8180; all heads 0.7852 (global "
             f"aggregation dilutes). Labels: {f14['n_halluc']} hallucinated of "
             f"{f14['n']} (prevalence {f14['prevalence']:.3f}).\n")
    L.append("| K | in-sample AUC | held-out AUC (10 half-splits) |")
    L.append("|---|---|---|")
    for k, v in f14["auc_in_sample_by_k"].items():
        L.append(f"| {k} | {v:.4f} | {f14['auc_held_out_by_k'][k]:.4f} "
                 f"+- {f14['auc_held_out_sd_by_k'][k]:.4f} |")
    L.append("")
    L.append(f"- Top-32 AP {f14['top32_average_precision']:.4f} in-sample, "
             f"{f14['top32_average_precision_held_out']:.4f} held out "
             f"(chance = prevalence {f14['prevalence']:.4f}).")
    L.append(f"- All 1024 heads, unweighted mean z: AUC "
             f"{f14['all_heads_unweighted_mean_z_auc']:.4f}; all heads with the "
             f"dAUC weights (Eq. 19 with H_K = everything): "
             f"{f14['all_heads_dauc_weighted_auc']:.4f}.")
    L.append(f"- Permutation null (labels shuffled, Top-32 re-selected on the "
             f"shuffled labels): mean AUC {f14['null_top32_auc_mean']:.4f}, max "
             f"{f14['null_top32_auc_max']:.4f}; all-heads weighted "
             f"{f14['null_all_heads_weighted_auc_mean']:.4f}. This is the "
             f"in-sample selection bias.")
    L.append(f"- Top-32 heads carry {f14['dauc_abs_share_of_top32']:.1%} of "
             f"sum|dAUC| over 1024 heads (uniform would be 3.1%); "
             f"{f14['top32_n_positive_dauc']}/32 have dAUC > 0; per layer: "
             f"{f14['top32_by_layer']}.")
    for key, what in (("figure14_tn_matched", "hallucinated vs right-'no' (TN)"),
                      ("figure14_tp_matched", "hallucinated vs right-'yes' (TP)")):
        m = rep[key]
        L.append(f"- Matched control, {what}: n={m['n']} (prevalence "
                 f"{m['prevalence']:.3f}); Top-32 AUC {m['top32_auc_in_sample']:.4f} "
                 f"in-sample, {m['top32_auc_held_out']:.4f} held out; AP "
                 f"{m['top32_average_precision_held_out']:.4f} held out; K=1 "
                 f"{m['auc_held_out_by_k']['1']:.4f}, K=4 "
                 f"{m['auc_held_out_by_k']['4']:.4f}, all-heads weighted "
                 f"{m['all_heads_dauc_weighted_auc']:.4f}.")
    L.append("- Top-32 heads (layer.head, dAUC): " + ", ".join(
        f"{h['layer']}.{h['head']} ({h['dauc']:+.3f})"
        for h in f14["top32_heads"]) + ".")
    return "\n".join(L) + "\n"


def run(data_dir: Path, out_dir: Path, labels_path: Optional[Path] = None,
        tag: str = "r13", plots: bool = True) -> dict:
    diag = sorted(data_dir.glob("m-diagnose__*.jsonl"))
    if not diag:
        raise SystemExit(f"no m-diagnose__*.jsonl in {data_dir}")
    if labels_path is None:
        cands = sorted(data_dir.glob(
            "m-sampling__b-pope__s-adversarial__seed-42__*greedy*.jsonl"))
        if not cands:
            raise SystemExit("no greedy sampling baseline to take labels from")
        labels_path = cands[0]   # same choice as scripts/analyse_diagnostics.py
    ids, arr = load_diagnostics(diag[0])
    lab = load_labels(labels_path)
    keep = [i for i, s in enumerate(ids) if s in lab]
    sel = np.array(keep)
    st = [lab[ids[i]] for i in keep]
    groups = build_groups(st)
    n_layers, n_heads = arr["m_vis_t1"].shape[1:]

    m = {t: arr[f"m_vis_t{t}"][sel].mean(axis=(1, 2)) for t in (32, 1)}
    rep: dict = {"meta": {
        "n_samples": len(keep), "n_halluc": int(groups["halluc"].sum()),
        "labels_file": labels_path.name, "diagnose_file": diag[0].name,
        "n_layers": int(n_layers), "n_heads": int(n_heads)},
        "paper": PAPER}
    rep["figure12"] = {f"tail{t}": fig12_for_tail(m[t], groups, t)
                       for t in (32, 1)}
    s = (arr["m_sys_t1"][sel] * arr["h_vis_t1"][sel]).reshape(len(keep), -1)
    labels = groups["halluc"]
    rep["figure14"] = fig14_analysis(s, labels)
    # Same analysis on a matched-ground-truth subset (hallucinated vs true "no").
    # Matched controls. Every hallucination is "said yes, truth no", so against
    # TN (same truth, other answer) the probe may just be reading the ANSWER the
    # forward pass is about to give; against TP (same answer, other truth) it
    # cannot. TP is the control that isolates hallucination from answering yes.
    for key, pool in (("figure14_tn_matched", "tn"), ("figure14_tp_matched", "tp")):
        sub = groups["halluc"] | groups[pool]
        rep[key] = _clean(fig14_analysis(s[sub], labels[sub]))
    out_dir.mkdir(parents=True, exist_ok=True)
    if plots:
        plot_fig12(out_dir / f"figure12_{tag}.png", m, groups)
        plot_fig14(out_dir / f"figure14_{tag}.png", rep["figure14"])
    (out_dir / f"figures_12_14_{tag}.md").write_text(render_md(_clean(rep)))
    (out_dir / f"figures_12_14_{tag}.json").write_text(
        json.dumps(_clean(rep), indent=1))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("results/reports"))
    ap.add_argument("--labels", type=Path, default=None,
                    help="greedy POPE run to take labels from (default: first "
                         "sorted match, as analyse_diagnostics.py)")
    ap.add_argument("--tag", default="r13")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    run(a.data_dir, a.out_dir, a.labels, a.tag, plots=not a.no_plots)
    print(f"[wrote] {a.out_dir}/figures_12_14_{a.tag}.{{json,md}}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
