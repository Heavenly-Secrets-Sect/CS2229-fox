#!/usr/bin/env python3
"""Aggregate MME across seeds, on one seed set shared by every arm.

r21 scored MME at three seeds by hand. V17 widens it to ten, and widening
is exactly where an aggregate goes wrong quietly: r22's session 1 was cut
by the wall on its last job, so `fox off` has nine complete seeds where
`fox` and the baseline have ten. Averaging each arm over whatever it
happens to have produces a table whose columns have different denominators,
and the contrast that matters here -- `fox off` minus `fox` -- is precisely
the one that would absorb the difference.

So this restricts every arm to the INTERSECTION of complete seeds, says out
loud which seeds it dropped, and refuses to print anything if an arm is
empty. A run is complete only at the full 2374 records; a short run is
named and excluded rather than averaged in, because MME sums over subtasks
and a truncated run is a low score, not a missing one.

The contrast is paired by seed: both arms saw the same seed, so the
difference per seed is the unit, and the interval is a bootstrap over those
differences. Ten seeds is few, which is why the interval is reported rather
than a p-value alone.

Usage::

    python3 scripts/score_mme.py
    python3 scripts/score_mme.py --confidence 0.90 --n-boot 20000
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

#: (method, mode) -> the name used in the write-up.
ARMS = {
    ("sampling", "as_implemented"): "baseline",
    ("fox", "as_implemented"): "fox",
    ("fox", "off"): "fox off",
}

#: Contrasts worth an interval, as (a, b) meaning a - b.
CONTRASTS = [("fox", "baseline"), ("fox off", "baseline"), ("fox off", "fox")]

#: What a complete MME run holds. Checked, not assumed.
N_RECORDS = 2374

FIELDS = ("perception", "cognition", "overall")


def load(reports: Path) -> Tuple[Dict[str, Dict[int, dict]], List[str]]:
    """arm -> seed -> metrics, plus a note for every run left out."""
    out: Dict[str, Dict[int, dict]] = {name: {} for name in ARMS.values()}
    notes: List[str] = []
    for f in sorted(reports.glob("*b-mme*.metrics.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        prov, met = d["provenance"], d["metrics"]
        arm = ARMS.get((prov["method"], prov.get("mode") or "as_implemented"))
        if arm is None:
            notes.append(f"  unknown arm {prov['method']}/{prov.get('mode')}"
                         f" in {f.name}")
            continue
        n = int(prov["n_records"])
        if n != N_RECORDS:
            notes.append(f"  EXCLUDED {arm} seed {prov['seed']}: {n} of "
                         f"{N_RECORDS} records (run was cut short)")
            continue
        if met.get("subtasks_missing"):
            notes.append(f"  EXCLUDED {arm} seed {prov['seed']}: subtasks "
                         f"missing {met['subtasks_missing']} -- a sum over "
                         f"an empty subtask is a silent zero")
            continue
        out[arm][int(prov["seed"])] = met
    return out, notes


def paired_bootstrap(diffs: List[float], n_boot: int, confidence: float,
                     seed: int = 0) -> Tuple[float, float, float]:
    """Mean difference and a percentile interval, resampling SEEDS."""
    import random
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((1 - confidence) / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 + confidence) / 2 * n_boot))]
    return sum(diffs) / n, lo, hi


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports", type=Path,
                    default=Path.home() / ".cache/fox-repro/reports")
    ap.add_argument("--confidence", type=float, default=0.90)
    ap.add_argument("--n-boot", type=int, default=20000)
    args = ap.parse_args(argv)

    arms, notes = load(args.reports)
    if notes:
        print("Runs left out:")
        for n in notes:
            print(n)
        print()

    empty = [a for a, v in arms.items() if not v]
    if empty:
        print(f"no complete runs for {empty}; nothing to report")
        return 1

    shared = set.intersection(*(set(v) for v in arms.values()))
    if not shared:
        print("the arms share no complete seed; nothing comparable")
        return 1
    dropped = {a: sorted(set(v) - shared) for a, v in arms.items()}
    seeds = sorted(shared)

    print(f"MME over {len(seeds)} seeds shared by every arm: {seeds}")
    for a, d in dropped.items():
        if d:
            print(f"  {a} also has complete seeds {d}, NOT used: an arm "
                  f"averaged over more seeds than the one it is compared "
                  f"against is not a paired contrast")
    print()

    print("| arm | Perception | Cognition | overall | accuracy |")
    print("|---|---|---|---|---|")
    for arm in ("baseline", "fox", "fox off"):
        t = [arms[arm][s]["totals"] for s in seeds]
        acc = [arms[arm][s]["accuracy_ci"]["point"] for s in seeds]
        print(f"| `{arm}` | "
              + " | ".join(f"{statistics.fmean(x[f] for x in t):.1f}"
                           for f in FIELDS)
              + f" | {100 * statistics.fmean(acc):.2f} |")
    print()

    print(f"| contrast | field | mean diff | "
          f"{int(args.confidence * 100)}% CI | seeds |")
    print("|---|---|---|---|---|")
    for a, b in CONTRASTS:
        for f in FIELDS:
            diffs = [arms[a][s]["totals"][f] - arms[b][s]["totals"][f]
                     for s in seeds]
            m, lo, hi = paired_bootstrap(diffs, args.n_boot, args.confidence)
            excl = "" if lo > 0 or hi < 0 else "  (interval includes 0)"
            print(f"| `{a}` - `{b}` | {f} | {m:+.1f} | "
                  f"[{lo:+.1f}, {hi:+.1f}] | {len(seeds)} |{excl}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
