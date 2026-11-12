#!/usr/bin/env python3
"""Turn the diagnostic sweep into the paper's Table 13 and Figures 3 and 7.

**CPU only. No GPU, no model.** Run it locally after a `diagnose` run is
ingested. The sweep records attention statistics per head per layer and
generates nothing; the hallucinated/faithful labels come from an archived
generation run joined on ``sample_id``, which is what makes it cheap.

    python scripts/analyse_diagnostics.py
    python scripts/analyse_diagnostics.py --json out.json

What it computes, and what the paper says:

Eq. 11  m_sys,tail per head -- attention mass on [0, img_start), averaged over
        the last `tail` queries. Already asserted equal to the released
        statistic at tail=1 (tests/test_diagnose.py).
Eq. 12  dAUC(l,h) = AUC(l,h) - 0.5, per head, with HALLUCINATION as the
        positive class.
Eq. 13  s_n = sum_lh m_sys(n)*dAUC / sum_lh dAUC, then the ROC-AUC of {s_n}.
        Paper: 0.8226 (tail=1), 0.7626 (tail=32), and 0.818 in the main text.
Fig. 7  |AUC - 0.5| for H_vis per layer, the input to the paper's automatic
        window rule: "retain the contiguous early-to-mid window above the
        selection threshold".
Tab. 13 mean m_vis / m_txt / m_sys. The paper reports 0.57 / 0.62 / 0.82.
        These three spans partition the keys, so the measured triple sums to
        one; the paper's sums to 2.01.

Two things the paper's own construction does that are worth reading off the
output rather than arguing about:

* Eq. 13 fits dAUC on the same labels the resulting AUC is scored against, so
  the headline number is in-sample. A held-out split is reported beside it.
* Eq. 13 divides by sum(dAUC), not sum(|dAUC|). With dAUC roughly symmetric
  about zero that denominator can approach zero and flip sign; the magnitude
  of the denominator is printed so the reader can see whether it did.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def auc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    """ROC-AUC by rank sum, ties shared. ``None`` when either class is empty.

    No sklearn: this project declares no dependencies, and the Mann-Whitney
    form is four lines and exact.
    """
    n_p, n_n = len(pos), len(neg)
    if n_p == 0 or n_n == 0:
        return None
    merged = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks: List[float] = [0.0] * len(merged)
    i = 0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = shared
        i = j + 1
    rank_pos = sum(r for r, (_, lab) in zip(ranks, merged) if lab == 1)
    return (rank_pos - n_p * (n_p + 1) / 2.0) / (n_p * n_n)


def parse_yes_no(text: str) -> Optional[str]:
    t = str(text).strip().lower().lstrip('"\'').strip()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return None


def load_labels(path: Path) -> Dict[str, bool]:
    """``sample_id -> hallucinated``.

    POPE asks whether an object is present. A hallucination is answering YES
    about an object that is not there -- a false positive, not any error. A
    false "no" is a miss, and calling it a hallucination would put two
    opposite failure modes in one class and blunt every AUC below.
    """
    out: Dict[str, bool] = {}
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        said = parse_yes_no(r.get("output", ""))
        truth = str(r.get("label", "")).strip().lower()
        if said is None or truth not in ("yes", "no"):
            continue
        out[r["sample_id"]] = (said == "yes" and truth == "no")
    return out


def load_diagnostics(path: Path) -> Tuple[Dict[str, dict], int, int]:
    """``sample_id -> {layer -> {stat -> [per-head]}}``, plus the shape."""
    rows: Dict[str, dict] = {}
    n_layers = n_heads = 0
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        ex = r.get("method_extra") or {}
        if "diag_layers" not in ex:
            continue
        rows[r["sample_id"]] = ex["diag_layers"]
        n_layers = int(ex.get("n_layers") or n_layers)
        n_heads = int(ex.get("n_heads") or n_heads)
    return rows, n_layers, n_heads


def head_series(rows: Dict[str, dict], ids: Sequence[str], stat: str,
                n_layers: int, n_heads: int) -> Dict[Tuple[int, int], List[float]]:
    out: Dict[Tuple[int, int], List[float]] = {}
    for l in range(n_layers):
        for h in range(n_heads):
            out[(l, h)] = [rows[s][str(l)][stat][h] for s in ids]
    return out


def delta_auc(series, ids, hall) -> Dict[Tuple[int, int], float]:
    pos_idx = [i for i, s in enumerate(ids) if hall[s]]
    neg_idx = [i for i, s in enumerate(ids) if not hall[s]]
    out = {}
    for key, vals in series.items():
        a = auc([vals[i] for i in pos_idx], [vals[i] for i in neg_idx])
        if a is not None:
            out[key] = a - 0.5
    return out


def sample_scores(series, ids, d, *, absolute: bool = False
                  ) -> Tuple[List[float], float]:
    """Equation 13, with its own denominator returned so it can be inspected.

    ``absolute`` divides by sum|dAUC| instead of sum(dAUC). The numerator is
    untouched, so this changes nothing but the scale -- unless sum(dAUC) is
    negative, in which case the published form reverses the ranking and the
    resulting AUC becomes 1 - AUC. Equation 13 as written has no term that
    pins that sign down, so both are reported.
    """
    denom = sum(abs(v) for v in d.values()) if absolute else sum(d.values())
    scores = []
    for i, _ in enumerate(ids):
        num = sum(series[k][i] * d[k] for k in d)
        scores.append(num / denom if denom else float("nan"))
    return scores, denom


def auc_of(scores: Sequence[float], ids: Sequence[str],
           hall: Dict[str, bool]) -> Optional[float]:
    return auc([s for s, i in zip(scores, ids) if hall[i]],
               [s for s, i in zip(scores, ids) if not hall[i]])


def say(label: str, value, note: str = "") -> None:
    v = "n/a" if value is None else (f"{value:.4f}" if isinstance(value, float) else value)
    print(f"  {label:<46s} {v}{('  ' + note) if note else ''}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    from foxrepro.storage import paths
    runs = args.runs_dir or paths().runs_root

    diag_files = sorted(runs.glob("m-diagnose__*.jsonl"))
    if not diag_files:
        print(f"no diagnose run in {runs}; run the `diagnostics` job first",
              file=sys.stderr)
        return 2
    rows, n_layers, n_heads = load_diagnostics(diag_files[0])

    base = sorted(runs.glob("m-sampling__b-pope__s-adversarial__seed-42__*greedy*.jsonl"))
    if not base:
        print("no greedy sampling baseline to take labels from", file=sys.stderr)
        return 2
    hall = load_labels(base[0])

    ids = sorted(set(rows) & set(hall))
    n_h = sum(1 for s in ids if hall[s])
    print(f"# Diagnostics from {diag_files[0].name}")
    print(f"# Labels from      {base[0].name}")
    print(f"# {len(ids)} samples, {n_layers} layers x {n_heads} heads, "
          f"{n_h} hallucinated ({n_h / max(1, len(ids)):.1%})\n")
    if not ids or n_h == 0:
        print("no overlapping labelled samples", file=sys.stderr)
        return 2

    report: Dict[str, object] = {"n_samples": len(ids), "n_hallucinated": n_h,
                                 "n_layers": n_layers, "n_heads": n_heads}

    print("Table 13 -- mean diagnostic score per token group (tail=1)")
    means = {}
    for stat, paper in (("m_vis", 0.57), ("m_txt", 0.62), ("m_sys", 0.82)):
        ser = head_series(rows, ids, f"{stat}_t1", n_layers, n_heads)
        m = sum(sum(v) for v in ser.values()) / (len(ser) * len(ids))
        means[stat] = m
        say(f"{stat}", m, f"(paper {paper})")
    total = sum(means.values())
    say("sum of the three", total,
        "-- they partition the keys; the paper's sums to 2.01")
    report["table13"] = {**means, "sum": total}
    print()

    print("Figure 3 / 13 -- ROC-AUC of the Eq. 13 sample score")
    for tail, paper in ((1, 0.8226), (32, 0.7626)):
        ser = head_series(rows, ids, f"m_sys_t{tail}", n_layers, n_heads)
        d = delta_auc(ser, ids, hall)
        scores, denom = sample_scores(ser, ids, d)
        a = auc_of(scores, ids, hall)
        say(f"tail={tail}  in-sample AUC", a, f"(paper {paper})")

        # The denominator is a sum of ~1024 roughly symmetric terms, so it is a
        # residual, and its SIGN decides whether the published score ranks
        # hallucination up or down. Print what it is a residual of.
        mag = sum(abs(v) for v in d.values())
        n_up = sum(1 for v in d.values() if v > 0)
        say(f"tail={tail}  sum(dAUC) denominator", denom,
            f"-- Eq. 13 divides by this; sum|dAUC| = {mag:.4f}, so the "
            f"denominator is a {abs(denom) / mag:.1%} residual")
        say(f"tail={tail}  heads with dAUC > 0", f"{n_up}/{len(d)}",
            "-- near-balanced, which is why the residual is small")
        sc_abs, _ = sample_scores(ser, ids, d, absolute=True)
        say(f"tail={tail}  AUC dividing by sum|dAUC|", auc_of(sc_abs, ids, hall),
            "-- same numerator, sign-safe scale")

        # Control: does the MAGNITUDE of system attention predict anything, or
        # is all the signal in the fitted per-head dAUC weights?
        flat = [sum(ser[k][i] for k in ser) / len(ser) for i, _ in enumerate(ids)]
        say(f"tail={tail}  unweighted mean m_sys AUC", auc_of(flat, ids, hall),
            "-- no Eq. 12 weights at all")
        # Held out: fit the weights on half, score the other half.
        half = len(ids) // 2
        fit_ids, test_ids = ids[:half], ids[half:]
        if any(hall[s] for s in fit_ids) and any(hall[s] for s in test_ids):
            ser_fit = head_series(rows, fit_ids, f"m_sys_t{tail}", n_layers, n_heads)
            d_fit = delta_auc(ser_fit, fit_ids, hall)
            ser_te = head_series(rows, test_ids, f"m_sys_t{tail}", n_layers, n_heads)
            sc, _ = sample_scores(ser_te, test_ids, d_fit)
            a_out = auc([s for s, i in zip(sc, test_ids) if hall[i]],
                        [s for s, i in zip(sc, test_ids) if not hall[i]])
            say(f"tail={tail}  held-out AUC", a_out,
                "-- weights fitted on the other half")
            report[f"auc_tail{tail}"] = {
                "in_sample": a, "held_out": a_out, "denominator": denom,
                "sum_abs_dauc": mag, "heads_dauc_positive": n_up,
                "auc_abs_denominator": auc_of(sc_abs, ids, hall),
                "auc_unweighted": auc_of(flat, ids, hall)}
    print()

    # Figure 7 is the input to Table 15's automatic window rule, so run the
    # rule on BOTH halves of the joint score S = m_sys * H_vis, not just H_vis:
    # the window the rule picks is only interesting next to the one the code
    # ships. Signed, because the sign is the mechanism claim.
    for stat, label in (("h_vis_t1", "H_vis"), ("m_sys_t1", "m_sys")):
        print(f"Figure 7 -- per-layer AUC for {label} (tail=1), "
              f"heads averaged within the layer")
        signed = []
        for l in range(n_layers):
            vals = [sum(rows[s][str(l)][stat]) / n_heads for s in ids]
            a = auc_of(vals, ids, hall)
            signed.append(None if a is None else a - 0.5)
        for l, v in enumerate(signed):
            v = 0.0 if v is None else v
            bar = "#" * int(round(abs(v) * 200))
            arrow = "up  " if v > 0 else ("down" if v < 0 else "    ")
            print(f"  L{l:<3d} {v:+.4f} {arrow} {bar}")

        mag = [0.0 if v is None else abs(v) for v in signed]
        peak = max(range(n_layers), key=lambda l: mag[l])
        up = [l for l, v in enumerate(signed) if v is not None and v > 0]
        print(f"\n  hallucination RAISES {label} in {len(up)}/{n_layers} "
              f"layers: {up}")
        print(f"  strongest layer L{peak} at {signed[peak]:+.4f}")

        def wmean(a: int, b: int) -> float:
            return sum(mag[a:b + 1]) / (b - a + 1)

        w = 14  # the released window's width, so the comparison is fair
        best_a = max(range(n_layers - w + 1), key=lambda a: wmean(a, a + w - 1))
        rows_out = [("released code, L2-15", wmean(2, 15)),
                    ("appendix manual, L2-13", wmean(2, 13)),
                    (f"best {w}-layer window, L{best_a}-{best_a + w - 1}",
                     wmean(best_a, best_a + w - 1)),
                    ("every layer, L0-31", wmean(0, n_layers - 1))]
        print(f"  mean |AUC-0.5| over a window -- what the rule maximises:")
        for name, v in rows_out:
            print(f"    {name:<34s} {v:.4f}")
        report[f"figure7_{stat}"] = {
            "signed_auc_minus_half": signed,
            "peak_layer": peak,
            "layers_raised_by_hallucination": up,
            "window_means": {name: v for name, v in rows_out}}
        print()

    print("  The rule is stated as keeping the contiguous early-to-mid run "
          "above a threshold the paper never gives.")
    print("  Released code ships L2-15; the appendix says the manual window "
          "is L2-13 and that the rule selects L3-15.")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\n[wrote] {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
