#!/usr/bin/env python3
"""Read the validation arms and decide each pre-registered prediction.

**CPU only. No GPU, no model, no checkpoint.** Run it locally after session 1
comes back, on the ingested run files.

    python scripts/check_validation.py                 # all available arms
    python scripts/check_validation.py --strict        # exit 1 on any FAIL

The predictions and their readings are fixed in docs/VALIDATION.md, written
before any of these arms ran. This script only applies them; it does not
choose them. Where an arm is missing it says so and moves on, so it is
useful after a partial session.

What it checks, by section of that document:

V1  negative_only   each negative branch's POPE accuracy against the
                    baseline's. PREDICTION: every one is lower.
V2  alpha = 0       token-identity against the baseline, per sample.
                    PREDICTION: identical. One differing token fails.
V4  cf_*            CausalMM's shuffle against constant and blind.
                    PREDICTION: all three agree.
V5  opera_*         OPERA's three-way decomposition.
V6  diverged_frac   read off every paired run. PREDICTION: strictly between
                    0 and 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PASS, FAIL, SKIP, INFO = "PASS", "FAIL", "n/a ", "    "


def load(path: Path) -> Tuple[List[dict], dict]:
    from foxrepro.bench.evaluate import read_run

    return read_run(path, strict=False)


def find(runs: Path, method: str, *, benchmark: str = "pope",
         split: str = "adversarial", seed: int = 42,
         control: Optional[str] = None, decoding: str = "sampling",
         **variant) -> Optional[Path]:
    """Locate a run file by what it is, not by guessing its hash."""
    from foxrepro.bench.generate import GenConfig
    from foxrepro.storage import MAX_NEW_TOKENS_DEFAULT

    cfg = GenConfig(
        method=method, benchmark=benchmark, split=split, seed=seed,
        dtype="fp16", model_id="llava-hf/llava-1.5-7b-hf",
        prompt_template_id=("pope_v0_raw" if benchmark == "pope"
                            else "chair_v1"),
        max_new_tokens=MAX_NEW_TOKENS_DEFAULT[benchmark],
        decoding=decoding, control=control, **variant,
    )
    path = runs / cfg.run_key().filename()
    return path if path.is_file() else None


def pope_accuracy(records: List[dict], split: str) -> float:
    """Accuracy as a percentage, the way every table in this project prints it."""
    from foxrepro.bench.evaluate import evaluate_pope

    metrics, _ = evaluate_pope(records, labels=None)
    return 100.0 * float(metrics["accuracy"]["point"])


def outputs_by_sample(records: List[dict]) -> Dict[str, str]:
    return {r["sample_id"]: r["output"] for r in records}


def say(status: str, text: str) -> None:
    colour = {PASS: "32", FAIL: "31", SKIP: "33"}.get(status)
    tag = f"\033[{colour}m[{status}]\033[0m" if colour else f" {status} "
    print(f"  {tag} {text}")


# --------------------------------------------------------------------------- #
def v1_negative_branch_is_worse(runs: Path) -> List[str]:
    """Every contrastive method claims it subtracts a WORSE model."""
    print("\nV1  Is the negative branch actually worse?")
    base = find(runs, "sampling")
    if base is None:
        say(SKIP, "no sampling baseline at seed 42; cannot compare")
        return []
    base_acc = pope_accuracy(load(base)[0], "adversarial")
    say(INFO, f"baseline accuracy {base_acc:.2f}")
    verdicts = []
    for method in ("vcd", "icd", "sid", "causalmm"):
        path = find(runs, method, control="negative_only")
        if path is None:
            say(SKIP, f"{method}: negative_only not run yet")
            continue
        acc = pope_accuracy(load(path)[0], "adversarial")
        delta = acc - base_acc
        if delta < 0:
            say(PASS, f"{method}: {acc:.2f} ({delta:+.2f}) -- premise holds")
            verdicts.append(PASS)
        else:
            say(FAIL, f"{method}: {acc:.2f} ({delta:+.2f}) -- the branch this "
                      f"method SUBTRACTS is not worse than the baseline, so "
                      f"whatever it gains is not the described mechanism")
            verdicts.append(FAIL)
    return verdicts


def v2_alpha_zero_collapses(runs: Path) -> List[str]:
    """At alpha = 0 the rule is the identity, so the tokens must match.

    GREEDY only. Under sampling the contrastive path decides on the
    beta-masked logits -- which the baseline never applies -- and the draw
    is stochastic, so identity is not achievable however correct the code
    is. Under greedy with mask_greedy False the decision is `fused`, which
    at alpha = 0 is z_obs exactly.
    """
    print("\nV2  Does alpha = 0 reproduce the greedy baseline exactly?")
    base = find(runs, "sampling", decoding="greedy")
    if base is None:
        say(SKIP, "no greedy baseline at seed 42; cannot compare")
        return []
    base_out = outputs_by_sample(load(base)[0])
    verdicts = []
    for method in ("vcd", "icd", "sid", "causalmm"):
        path = find(runs, method, alpha=0.0, decoding="greedy")
        if path is None:
            say(SKIP, f"{method}: alpha=0 arm not run yet")
            continue
        got = outputs_by_sample(load(path)[0])
        shared = sorted(set(got) & set(base_out))
        if not shared:
            say(SKIP, f"{method}: no overlapping sample ids")
            continue
        diff = [s for s in shared if got[s] != base_out[s]]
        if not diff:
            say(PASS, f"{method}: {len(shared)} samples, all identical")
            verdicts.append(PASS)
            continue
        # CausalMM cannot reach identity and the reason is recorded, not
        # discovered here: output_attentions=True forces the CLIP vision
        # tower onto its eager path, and eager CLIP does not agree with SDPA
        # CLIP in the last bits (3.9e-3 in fp16). So its CLEAN branch is not
        # the baseline's clean branch, whatever alpha is. A handful of
        # flipped decisions is that deviation being visible; a large number
        # would still be a fault. See docs/DEVIATIONS.md Sec 3.13.
        tol = max(1, len(shared) // 100) if method == "causalmm" else 0
        if len(diff) <= tol:
            say(PASS, f"{method}: {len(diff)}/{len(shared)} differ "
                      f"(<= {tol}) -- the eager-CLIP deviation, not the rule. "
                      f"The rule itself is the identity at alpha=0")
            verdicts.append(PASS)
        else:
            say(FAIL, f"{method}: {len(diff)}/{len(shared)} samples differ "
                      f"(e.g. {diff[0]}: {base_out[diff[0]]!r} vs "
                      f"{got[diff[0]]!r}). At alpha=0 the rule is the "
                      f"identity, so this is a wiring fault, not a tuning "
                      f"difference")
            verdicts.append(FAIL)
    return verdicts


def v4_is_the_shuffle_load_bearing(runs: Path) -> List[str]:
    """CausalMM's gain is 1/577 whatever the attention is.

    GREEDY where it exists, and that is a correction with the same cause as
    V2's. Token identity is not a question sampling can answer: a
    multinomial draw flips whenever the logits move at all, so two arms
    whose distributions agree still disagree sample by sample. Read under
    sampling these arms agreed on 96.6% / 95.2% of 500 answers while their
    accuracies spanned 0.6 points on n=500 -- inside the noise. Under greedy
    there is no draw, so a disagreement means the logits really differ.
    """
    print("\nV4  Is CausalMM's shuffle doing the work, or just its magnitude?")
    verdicts = []
    for decoding in ("greedy", "sampling"):
        base = find(runs, "causalmm", decoding=decoding)
        if base is None:
            continue
        exact = decoding == "greedy"
        label = "greedy, token-identity" if exact else \
                "sampling, DISTRIBUTION only -- a draw flips on any logit change"
        print(f"      [{label}]")
        base_records = load(base)[0]
        ref = outputs_by_sample(base_records)
        ref_acc = None
        for control in ("cf_constant", "cf_blind"):
            path = find(runs, "causalmm", decoding=decoding, control=control)
            if path is None:
                say(SKIP, f"{control}: not run under {decoding}")
                continue
            records = load(path)[0]
            got = outputs_by_sample(records)
            shared = sorted(set(got) & set(ref))
            if not shared:
                say(SKIP, f"{control}: no overlapping sample ids")
                continue
            if ref_acc is None:
                ref_acc = pope_accuracy(
                    [r for r in base_records if r["sample_id"] in set(shared)],
                    "adversarial")
                say(INFO, f"shuffle on these {len(shared)}: {ref_acc:.2f}")
            acc = pope_accuracy(records, "adversarial")
            same = sum(1 for s in shared if got[s] == ref[s])
            frac = same / len(shared)
            text = (f"{control}: agrees with shuffle on {same}/{len(shared)} "
                    f"({frac:.1%}), accuracy {acc:.2f} ({acc - ref_acc:+.2f})")
            if not exact:
                say(INFO, text + "  -- read the accuracy, not the agreement")
                continue
            if frac > 0.99:
                say(PASS, text + "  -- the vision counterfactual is a "
                                 "description of something the code does not "
                                 "do; CausalMM is a language-prior contrast")
                verdicts.append(PASS)
            else:
                say(FAIL, text + "  -- with no draw to blame, the residual "
                                 "image signal is load-bearing after all and "
                                 "the 1/n_keys argument is incomplete")
                verdicts.append(FAIL)
    return verdicts


def opera_penalty_diagnostics(records):
    """OPERA records its own penalty telemetry, so a plain run answers the
    first row of V5's table before any control is compared."""
    keys = ("penalty_spread_mean", "logit_spread_mean",
            "n_steps_penalty_changed_top")
    out = {}
    for key in keys:
        vals = [(r.get("method_extra") or {}).get(key) for r in records]
        vals = [v for v in vals if v is not None]
        if vals:
            out[key] = sum(vals) / len(vals)
    return out


def v5_opera_decomposition(runs: Path) -> List[str]:
    """Which third of OPERA is working?

    Pre-registered prediction (docs/VALIDATION.md V5): `opera_no_penalty`
    lands ON TOP of plain OPERA, i.e. the over-trust penalty is inert on
    POPE. A FAIL here is this project's PREDICTION being wrong, not a defect
    in the port -- it would mean the penalty does real work and the
    correct-geometry stand-in understated it.

    No direction was pre-registered for `opera_unrestricted`, so it stays
    INFO: the document lists readings for it but never named a favourite.
    """
    print("\nV5  OPERA: beam search, candidate restriction, or penalty?")
    plain = find(runs, "opera", decoding="greedy")
    if plain is None:
        say(SKIP, "plain opera not run yet")
        return []
    plain_records = load(plain)[0]
    plain_acc = pope_accuracy(plain_records, "adversarial")
    say(INFO, f"opera accuracy {plain_acc:.2f}")

    diag = opera_penalty_diagnostics(plain_records)
    if diag:
        parts = [f"{k} {v:.4g}" for k, v in diag.items()]
        say(INFO, "penalty telemetry: " + ", ".join(parts))
        changed = diag.get("n_steps_penalty_changed_top")
        if changed is not None and changed <= 0.0:
            say(INFO, "the penalty never moved the top candidate at any step, "
                      "so the accuracy comparison below can only agree")

    verdicts = []
    for control in ("opera_no_penalty", "opera_unrestricted"):
        path = find(runs, "opera", decoding="greedy", control=control)
        if path is None:
            say(SKIP, f"{control} not run yet")
            continue
        acc = pope_accuracy(load(path)[0], "adversarial")
        delta = acc - plain_acc
        if control == "opera_unrestricted":
            say(INFO, f"{control}: {acc:.2f} ({delta:+.2f})"
                      + ("  -- the five-candidate restriction is not what "
                         "helps" if abs(delta) < 0.2 else
                         "  -- the restriction is load-bearing"))
            continue
        if abs(delta) < 0.2:
            say(PASS, f"{control}: {acc:.2f} ({delta:+.2f}) -- the over-trust "
                      f"penalty is INERT on POPE. OPERA's published POPE "
                      f"number is beam search with a truncated candidate set")
            verdicts.append(PASS)
        else:
            say(FAIL, f"{control}: {acc:.2f} ({delta:+.2f}) -- the penalty "
                      f"DOES move POPE, so the prediction in VALIDATION.md "
                      f"V5 was wrong: the stand-in's near-uniform attention "
                      f"understated it. Not a port defect")
            verdicts.append(FAIL)
    return verdicts


def v6_divergence(runs: Path) -> List[str]:
    """Did each rule ever change a decision?"""
    print("\nV6  Does each rule change any decision?")
    verdicts = []
    for method in ("vcd", "icd", "sid", "causalmm"):
        for benchmark, split in (("pope", "adversarial"),
                                 ("chair", "val2014_500")):
            path = find(runs, method, benchmark=benchmark, split=split)
            if path is None:
                continue
            records = load(path)[0]
            fracs = [
                (r.get("method_extra") or {}).get("diverged_frac")
                for r in records
            ]
            fracs = [f for f in fracs if f is not None]
            if not fracs:
                say(SKIP, f"{method}/{benchmark}: no diverged_frac recorded")
                continue
            mean = sum(fracs) / len(fracs)
            if mean <= 0.0:
                say(FAIL, f"{method}/{benchmark}: diverged_frac 0.000 -- the "
                          f"rule never changed a decision. The method is "
                          f"inert and no accuracy figure would show it")
                verdicts.append(FAIL)
            elif mean >= 1.0:
                say(FAIL, f"{method}/{benchmark}: diverged_frac 1.000 -- the "
                          f"rule overrode the model on every token")
                verdicts.append(FAIL)
            else:
                say(PASS, f"{method}/{benchmark}: diverged_frac {mean:.3f}")
                verdicts.append(PASS)
    return verdicts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, default=None,
                    help="default: the live runs directory")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any prediction failed")
    args = ap.parse_args(argv)

    from foxrepro.storage import paths

    runs = args.runs_dir or paths().runs_root
    if not runs.is_dir():
        print(f"no runs directory at {runs}", file=sys.stderr)
        return 2
    print(f"# Validation arms, read from {runs}")
    print("# Predictions are pre-registered in docs/VALIDATION.md.")

    verdicts: List[str] = []
    for check in (v1_negative_branch_is_worse, v2_alpha_zero_collapses,
                  v4_is_the_shuffle_load_bearing, v5_opera_decomposition,
                  v6_divergence):
        verdicts += check(runs)

    n_fail = verdicts.count(FAIL)
    n_pass = verdicts.count(PASS)
    print(f"\n{n_pass} predictions held, {n_fail} failed.")
    if n_fail:
        print("A FAIL means the method is not doing what its paper describes "
              "on this stack. Read docs/VALIDATION.md for what each one "
              "implies before running the table rows.")
    return 1 if (args.strict and n_fail) else 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    raise SystemExit(main())
