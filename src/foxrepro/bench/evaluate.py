"""CPU-only metrics from generation JSONL. No GPU, no torch, no model.

Decision 1 of the project: generation and evaluation are separate programs.
This one reads the JSONL that ``bench/generate.py`` wrote and produces a
metrics JSON per run, plus a per-sample outcome file that
``bench/report.py`` uses for *paired* comparisons.

Metrics
-------
POPE (binary object existence, positive class = "yes"):
    accuracy, precision, recall, F1, and **yes-ratio** -- the fraction of
    "yes" answers. The yes-ratio is not decoration: a method can raise POPE
    accuracy simply by answering "no" more often, because the splits are
    balanced and the hard questions are the "yes" ones. Accuracy without the
    yes-ratio cannot distinguish better grounding from a shifted threshold.

CHAIR (open-ended captioning):
    CHAIR_S (captions with >= 1 hallucinated object / captions) and CHAIR_I
    (hallucinated object mentions / all object mentions), plus the
    completeness guards from the paper (recall, precision, F1) so that a drop
    in CHAIR achieved by saying less is visible.

Every metric is reported with its **numerator and denominator**. CHAIR_I's
denominator -- the total number of mentioned object instances -- is not stated
in the paper and cannot be recovered from a published CHAIR_I value, yet
nothing can be said about significance without it.

Interval notes
--------------
* POPE accuracy / yes-ratio and CHAIR_S are proportions over independent
  samples: Wilson intervals apply directly.
* CHAIR_I's trials (object mentions) are **clustered within captions**, so a
  Wilson interval over mentions is anti-conservative. Both are reported: the
  Wilson interval (comparable to what others publish) and a cluster bootstrap
  over captions (the honest one). The report prefers the bootstrap.
* F1 is a ratio of ratios, so it gets a bootstrap interval instead.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


from foxrepro.bench.chair import CHAIRScorer, CaptionScore  # noqa: E402
from foxrepro.bench.stats import (  # noqa: E402
    bootstrap_ci, wilson_interval,
)
from foxrepro.storage import (  # noqa: E402
    MME_COGNITION, MME_PERCEPTION,
    OPTIONAL_VARIANT_FIELDS, PROVENANCE_FIELDS, VARIANT_FIELDS, paths,
)


class LegacyProvenance(ValueError):
    """A run archived before one of today's provenance fields existed.

    Distinct from a missing field in a FRESH run, which is a real fault:
    ``generate.py`` asserts every field per record, so a new file cannot lack
    one. These are historical records in ``results/runs`` whose runs happened
    at commits that predate the field, and no re-run can repair them without
    spending the GPU hours again.
    """


#: Provenance fields added after records were already being archived, and
#: whose absence therefore dates a file rather than indicting it.
#:
#: Keep this list SHORT and justified. Each entry is a field the evaluator
#: will tolerate missing, so a field listed here by mistake turns a real
#: regression into a silent skip. Everything not listed still fails hard.
#:
#: ``wrapper_id`` entered PROVENANCE_FIELDS with the prompt-wrapper ablation;
#: six runs from commits 3300d76 and ff386a3 predate it. All six are
#: superseded -- the current plan's equivalent jobs carry different variant
#: hashes and have been re-run -- so nothing downstream reads them.
LEGACY_PROVENANCE_FIELDS: frozenset = frozenset({"wrapper_id"})

__all__ = [
    "parse_yes_no", "parse_yes_no_released", "POPE_PARSERS",
    "DEFAULT_POPE_PARSER", "read_run", "evaluate_pope", "evaluate_chair",
    "evaluate_run_file", "PopeOutcome", "evaluate_mmbench",
    "evaluate_mathvista", "evaluate_mmbench_circular",
    "LEGACY_METHOD_NAMES", "canonical_method",
]

_WORD_RE = re.compile(r"[a-z]+")

#: ``(recorded method, recorded variant) -> what that arm is now called``.
#:
#: Rounds 1-5 ran one compared-method arm under the name ``vcd``. It was never
#: VCD: it put VCD's noised image into **Fox's** additive gated fusion, and the
#: records say so in ``method_extra.variant``. Now that VCD's own contrastive
#: rule is implemented, ``vcd`` means VCD, and leaving the old files answering
#: to that name would put the hybrid into the VCD row of Table 1 -- a wrong
#: number under a real method's name, which is worse than a missing row.
#:
#: Keyed on the variant rather than on the file or the date, because the
#: variant is what the run itself recorded about what it computed.
LEGACY_METHOD_NAMES: Dict[Tuple[str, str], str] = {
    ("vcd", "vcd-noise-fox-fusion"): "vcd_fox_fusion",
}


def canonical_method(method: Any, variants: Sequence[str]) -> Any:
    """Today's name for an arm, given what the run recorded about itself.

    Returns ``method`` unchanged unless the run carries exactly one variant
    and that pair has been renamed. A run with two variants is already a
    mixed file and is refused upstream by the ``shared_keys`` check.
    """
    if len(variants) != 1 or not isinstance(method, str):
        return method
    return LEGACY_METHOD_NAMES.get((method, variants[0]), method)


# --------------------------------------------------------------------------- #
# answer parsing
# --------------------------------------------------------------------------- #
def parse_yes_no(text: str) -> Optional[str]:
    """Map a POPE answer to ``"yes"`` / ``"no"`` / ``None`` (unparsable).

    POPE is generated with ``max_new_tokens = 1``, so in practice the output is
    literally ``"Yes"`` or ``"No"``. The parser still handles longer answers,
    because the greedy control and any prompt change can produce them:

    1. the first alphabetic word decides, when it is "yes" or "no";
    2. otherwise the first occurrence of either word anywhere decides;
    3. otherwise the answer is unparsable and is **counted and reported**,
       never silently dropped.

    Args:
        text: raw model output.

    Returns:
        ``"yes"``, ``"no"`` or ``None``.
    """
    words = _WORD_RE.findall((text or "").lower())
    if not words:
        return None
    if words[0] in ("yes", "no"):
        return words[0]
    for w in words:
        if w in ("yes", "no"):
            return w
    return None


#: ``pope_eval.py:233``, verbatim. The released scorer has no unparsable
#: class: it looks for a negation token anywhere in the answer and calls
#: everything else "yes".
RELEASED_NEG_WORDS = ("No", "not", "no", "NO")


def parse_yes_no_released(text: str) -> str:
    """The released POPE rule (``pope_eval.py:232-242``), transcribed.

    Not a cleaned-up version of :func:`parse_yes_no`: a different function.
    The release strips ``.`` and ``,``, splits on a single space, and answers
    "no" if any token is in ``RELEASED_NEG_WORDS`` or ends in ``n't``. Every
    other answer -- including ``""``, ``"There"`` and ``"Maybe"`` -- is "yes".

    With ``max_new_tokens = 1`` and the "single word or phrase" suffix the two
    parsers agree on every sample (measured: 3000/3000 on both session-1 POPE
    runs). Without that suffix the first token is often neither "Yes" nor
    "No", and then they diverge: the release scores such an answer as a
    positive, this module's strict parser calls it unparsable. Since the
    paper's numbers came out of the released rule, that rule is the default.

    Args:
        text: raw model output.

    Returns:
        ``"yes"`` or ``"no"`` -- never ``None``.
    """
    line = (text or "").replace(".", "").replace(",", "")
    words = line.split(" ")
    if any(w in RELEASED_NEG_WORDS for w in words):
        return "no"
    if any(w.endswith("n't") for w in words):
        return "no"
    return "yes"


#: ``--pope-parser``. ``released`` reproduces the paper; ``strict`` is this
#: module's own rule, which reports unparsable answers instead of scoring them.
POPE_PARSERS = {
    "released": parse_yes_no_released,
    "strict": parse_yes_no,
}
DEFAULT_POPE_PARSER = "released"


# --------------------------------------------------------------------------- #
# reading runs
# --------------------------------------------------------------------------- #
def read_run(path: Path, strict: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Read a run JSONL, de-duplicating by ``sample_id``.

    Resume can in principle append a ``sample_id`` twice (for example if a
    file was concatenated by hand). The **first** occurrence wins, and the
    duplicate count is reported.

    Args:
        path: run file.
        strict: raise if the records disagree on the run-level provenance
            fields, or if a provenance field is missing.

    Returns:
        ``(records, meta)``. ``meta`` carries the shared provenance, the
        duplicate count and any trailing-partial-line byte count.
    """
    records: List[Dict[str, Any]] = []
    seen: Dict[str, int] = {}
    n_dupes = 0
    partial_bytes = 0
    with open(path, "rb") as fh:
        for raw in fh:
            if not raw.endswith(b"\n"):
                partial_bytes = len(raw)
                break
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec["sample_id"]
            if sid in seen:
                n_dupes += 1
                continue
            seen[sid] = len(records)
            records.append(rec)
    if not records:
        raise ValueError(f"{path} contains no records")

    if strict:
        missing = [f for f in PROVENANCE_FIELDS if f not in records[0]]
        if missing:
            if set(missing) <= LEGACY_PROVENANCE_FIELDS:
                raise LegacyProvenance(
                    f"{path}: records predate {sorted(missing)}, which entered "
                    f"PROVENANCE_FIELDS after this run was archived"
                )
            raise ValueError(f"{path}: records missing provenance fields {missing}")

    # Fields that MUST be constant: anything that changes what was computed.
    # A job routinely spans sessions, and sessions routinely run at different
    # commits -- that is the design, not a fault. `commit_hash` and
    # `transformers_path` are therefore recorded as sets rather than demanded
    # constant, which also makes this agree with the ingest guard in
    # plan.MERGE_COMPATIBLE_FIELDS, which already allows a newer commit to
    # merge. Before this, a resumed job was ingested successfully and then
    # refused at scoring time, which is the worst of both.
    shared_keys = (
        "method", "benchmark", "split", "seed", "dtype", "model_id",
        "prompt_template_id", "max_new_tokens", *VARIANT_FIELDS,
        # OPTIONAL_VARIANT_FIELDS too. `control` is left out of the run-file
        # NAME when unset, so that adding the field did not rename every
        # archived run -- but it still identifies the arm, and leaving it out
        # of the provenance made every control arm indistinguishable from its
        # own method downstream. report.py then found two results for
        # ('pope', 'adversarial', 'causalmm') and refused to build at all.
        *OPTIONAL_VARIANT_FIELDS,
    )
    #: Recorded, allowed to vary across a resumed run, surfaced for audit.
    multi_keys = ("commit_hash", "transformers_path")
    # The variant label travels with the records, so a relabelled method can
    # still be recognised in files written before the rename -- see
    # LEGACY_METHOD_NAMES below.
    variants = sorted({
        str((r.get("method_extra") or {}).get("variant"))
        for r in records
        if (r.get("method_extra") or {}).get("variant") is not None
    })
    meta: Dict[str, Any] = {"run_file": str(path)}
    for key in multi_keys:
        seen = sorted({str(r.get(key)) for r in records if r.get(key) is not None})
        if seen:
            meta[key] = seen[0] if len(seen) == 1 else seen
            if len(seen) > 1:
                meta.setdefault("resumed_across", {})[key] = seen
    for key in dict.fromkeys(shared_keys):
        values = {json.dumps(r.get(key), sort_keys=True) for r in records}
        if len(values) == 1:
            meta[key] = records[0].get(key)
        elif strict:
            raise ValueError(
                f"{path}: field {key!r} is not constant across records "
                f"({len(values)} distinct values); this file mixes runs."
            )
        else:
            meta[key] = sorted(json.loads(v) for v in values)  # type: ignore[arg-type]
    if variants:
        meta["method_variant"] = variants[0] if len(variants) == 1 else variants
    meta["method"] = canonical_method(meta.get("method"), variants)
    meta["n_records"] = len(records)
    meta["n_duplicate_sample_ids"] = n_dupes
    meta["trailing_partial_bytes"] = partial_bytes
    lat = [float(r["latency_ms"]) for r in records if r.get("latency_ms") is not None]
    gen = [int(r["n_gen_tokens"]) for r in records if r.get("n_gen_tokens") is not None]
    meta["latency_ms_mean"] = sum(lat) / len(lat) if lat else None
    meta["latency_ms_median"] = sorted(lat)[len(lat) // 2] if lat else None
    meta["n_gen_tokens_mean"] = sum(gen) / len(gen) if gen else None
    return records, meta


# --------------------------------------------------------------------------- #
# POPE
# --------------------------------------------------------------------------- #
@dataclass
class PopeOutcome:
    """Per-sample POPE outcome; the unit a paired comparison resamples."""

    sample_id: str
    image_id: int
    label: str
    pred: Optional[str]
    correct: int
    said_yes: int

    def as_dict(self) -> Dict[str, Any]:
        """JSON-ready outcome row."""
        return {
            "sample_id": self.sample_id, "image_id": self.image_id,
            "label": self.label, "pred": self.pred,
            "correct": self.correct, "said_yes": self.said_yes,
        }


def _f1_from_codes(codes: Sequence[int]) -> float:
    """F1 from per-sample confusion codes (0=TN, 1=TP, 2=FP, 3=FN)."""
    counts = Counter(codes)
    tp, fp, fn = counts[1], counts[2], counts[3]
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def evaluate_mme(
    records: Sequence[Dict[str, Any]],
    unparsed_policy: str = "negative",
    confidence: float = 0.95,
    n_boot: int = 10000,
    seed: int = 20260101,
    pope_parser: str = DEFAULT_POPE_PARSER,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """MME's score, which is NOT POPE accuracy under another name.

    MME asks exactly two yes/no questions per image -- one whose answer is
    yes, one whose answer is no -- and scores each subtask as

        subtask score = (acc + acc_plus) * 100

    where ``acc`` is per-question accuracy and ``acc_plus`` is the fraction
    of IMAGES where BOTH questions were right. The reported Perception and
    Cognition totals are sums over their ten and four subtasks, so each is
    out of 1000 and 400 rather than 100.

    That per-image term is the whole reason this cannot reuse
    :func:`evaluate_pope`. A model that answers "yes" to everything scores
    0.50 acc and **0.00** acc_plus; one that is genuinely right half the
    time at random scores 0.50 and 0.25. POPE accuracy cannot tell those
    apart and the MME score is built to.

    Parsing is shared with POPE deliberately -- the answers are the same
    yes/no shape -- so a parser change moves both benchmarks together rather
    than letting them drift.

    Returns ``(metrics, outcome_rows)``; the rows carry ``correct`` per
    question and the image grouping, so paired tests work as they do for
    POPE.
    """
    parse = POPE_PARSERS[pope_parser]
    by_image: Dict[str, List[Dict[str, Any]]] = {}
    rows: List[Dict[str, Any]] = []
    n_unparsed = 0
    for rec in records:
        label = str(rec.get("label", "")).strip().lower()
        if label not in ("yes", "no"):
            raise ValueError(
                f"MME record {rec.get('sample_id')!r} has non-binary label "
                f"{rec.get('label')!r}"
            )
        pred = parse(str(rec.get("output", "")))
        if pred is None:
            n_unparsed += 1
            # Same convention as POPE: an unparsable answer is a wrong
            # answer, not a dropped sample. Dropping would let a model
            # improve its score by refusing to answer.
            pred = "no" if unparsed_policy == "negative" else label
        correct = int(pred == label)
        # sample_id is "mme/<category>/<image>#<yes|no>", built by load_mme,
        # and is the one field every record has. The explicit fields are
        # preferred when present -- runs from r21.2 onward carry them -- but
        # deriving keeps the earlier runs scoreable without re-running them,
        # and keeps a missing field from silently bucketing into "?" where
        # it reads as Perception 0.0.
        extra = rec.get("method_extra") or {}
        sid = str(rec.get("sample_id") or "")
        derived_cat = derived_img = None
        if sid.startswith("mme/"):
            body = sid[len("mme/"):]
            derived_img = body.split("#", 1)[0] or None
            derived_cat = derived_img.split("/", 1)[0] if derived_img else None
        cat = str(rec.get("category") or extra.get("category")
                  or derived_cat or "?")
        img = str(rec.get("image_file") or extra.get("image_file")
                  or derived_img or rec.get("image_id"))
        if cat == "?":
            raise ValueError(
                f"MME record {sid!r} carries no category and none can be "
                f"derived from its sample_id. Scoring it would sum into "
                f"neither Perception nor Cognition and report a total of 0."
            )
        row = {"sample_id": rec.get("sample_id"), "category": cat,
               "image": img, "label": label, "pred": pred,
               "correct": correct, "parsed": pred is not None}
        rows.append(row)
        by_image.setdefault((cat, img), []).append(row)

    per_cat: Dict[str, Dict[str, float]] = {}
    for (cat, _img), qs in by_image.items():
        c = per_cat.setdefault(cat, {"n_q": 0, "n_right": 0,
                                     "n_img": 0, "n_img_both": 0})
        c["n_q"] += len(qs)
        c["n_right"] += sum(q["correct"] for q in qs)
        c["n_img"] += 1
        c["n_img_both"] += int(all(q["correct"] for q in qs))

    cats: Dict[str, Dict[str, float]] = {}
    for cat, c in sorted(per_cat.items()):
        acc = c["n_right"] / c["n_q"] if c["n_q"] else 0.0
        accp = c["n_img_both"] / c["n_img"] if c["n_img"] else 0.0
        cats[cat] = {
            "acc": round(100 * acc, 4), "acc_plus": round(100 * accp, 4),
            "score": round(100 * (acc + accp), 4),
            "n_questions": int(c["n_q"]), "n_images": int(c["n_img"]),
        }

    def total(group: Sequence[str]) -> float:
        return round(sum(cats[g]["score"] for g in group if g in cats), 4)

    n_q = sum(c["n_questions"] for c in cats.values())
    n_right = sum(r["correct"] for r in rows)
    metrics: Dict[str, Any] = {
        "benchmark": "mme",
        "n_samples": len(rows),
        "unparsed": n_unparsed,
        "per_category": cats,
        "totals": {
            "perception": total(MME_PERCEPTION),
            "cognition": total(MME_COGNITION),
            "overall": total(MME_PERCEPTION) + total(MME_COGNITION),
            "accuracy": round(100 * n_right / n_q, 4) if n_q else 0.0,
        },
        "subtasks_present": sorted(cats),
        "subtasks_missing": sorted(
            set(MME_PERCEPTION + MME_COGNITION) - set(cats)),
    }
    if n_q:
        metrics["accuracy_ci"] = wilson_interval(
            n_right, n_q, confidence=confidence, name="mme_accuracy"
        ).as_dict()
    return metrics, rows


# --------------------------------------------------------------------------- #
# MMBench and MathVista-MC (Table 11)
# --------------------------------------------------------------------------- #
def _mc_options(rec: Dict[str, Any]) -> Dict[str, str]:
    """The option set a multiple-choice record was asked with, or raise.

    Read from the record itself (``generate.py`` persists it). Never
    reconstructed: without it a letter cannot be told valid from invalid, and
    a guess would score the run on a question the model never saw.
    """
    opts = rec.get("options")
    if not isinstance(opts, dict) or len(opts) < 2:
        raise ValueError(
            f"record {rec.get('sample_id')!r} carries no `options`; a "
            f"multiple-choice run cannot be scored without them. Re-run it "
            f"with a generate.py that persists them.")
    return {str(k): str(v) for k, v in opts.items()}


def _mc_tally(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-rule counts. ``unparsed`` is the sum of the two failure rules."""
    rules = Counter(r["rule"] for r in rows)
    unparsed = rules.get("none", 0) + rules.get("invalid_letter", 0)
    return {"rule_counts": dict(sorted(rules.items())), "unparsed": unparsed}


def _acc(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Accuracy in percent with its numerator, or None for an empty group.

    None and not 0.0: a group with no questions has no accuracy, and a zero
    here would read as a measured failure. The reader of the metrics file
    must see that nothing was measured.
    """
    if not rows:
        return None
    k = sum(r["correct"] for r in rows)
    return {"acc": round(100 * k / len(rows), 4), "n_right": int(k),
            "n": len(rows)}


def evaluate_mmbench(
    records: Sequence[Dict[str, Any]],
    confidence: float = 0.95,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """MMBench dev accuracy: overall, perception and reasoning.

    Single pass over the questions as asked (NOT MMBench's CircularEval, which
    re-asks each question under option rotations; see docs/DEVIATIONS.md).
    Overall is accuracy over all questions; perception and reasoning are
    accuracy over the questions whose ``l2-category`` falls in each group,
    using ``storage.MMBENCH_PERCEPTION_L2`` / ``MMBENCH_REASONING_L2``.

    Raises rather than degrade: a record without its l2-category or option
    set, an l2-category in neither group, or a gold letter that is not one of
    the record's options. Each would otherwise move a question out of a group
    total silently.

    An unparseable answer is WRONG and counted (``unparsed``, with the
    per-rule breakdown), never dropped.
    """
    from foxrepro.bench.multichoice import extract_choice
    from foxrepro.storage import (
        MMBENCH_DEV_ROWS, MMBENCH_PERCEPTION_L2, MMBENCH_REASONING_L2)

    group_of = {**{c: "perception" for c in MMBENCH_PERCEPTION_L2},
                **{c: "reasoning" for c in MMBENCH_REASONING_L2}}
    rows: List[Dict[str, Any]] = []
    for rec in records:
        sid = rec.get("sample_id")
        for field in ("label", "category", "l2_category"):
            if rec.get(field) in (None, ""):
                raise ValueError(
                    f"MMBench record {sid!r} carries no {field!r}; scoring "
                    f"it would drop it from a group total without a trace.")
        l2 = str(rec["l2_category"])
        if l2 not in group_of:
            raise ValueError(
                f"MMBench record {sid!r} has l2-category {l2!r}, which is in "
                f"neither the perception nor the reasoning group "
                f"({sorted(group_of)}). Extend storage.MMBENCH_*_L2 after "
                f"checking MMBench's definition; do not bucket it.")
        options = _mc_options(rec)
        gold = str(rec["label"])
        if gold not in options:
            raise ValueError(
                f"MMBench record {sid!r}: gold {gold!r} is not among its "
                f"options {sorted(options)}")
        pred, rule = extract_choice(str(rec.get("output", "")), options)
        rows.append({
            "sample_id": sid, "category": str(rec["category"]),
            "l2_category": l2, "group": group_of[l2], "label": gold,
            "pred": pred, "rule": rule, "parsed": pred is not None,
            "correct": int(pred == gold),
        })

    def group(name: str) -> Optional[Dict[str, Any]]:
        return _acc([r for r in rows if r["group"] == name])

    per_l2: Dict[str, Any] = {}
    for l2 in sorted({r["l2_category"] for r in rows}):
        per_l2[l2] = _acc([r for r in rows if r["l2_category"] == l2])
    per_cat: Dict[str, Any] = {}
    for cat in sorted({r["category"] for r in rows}):
        per_cat[cat] = _acc([r for r in rows if r["category"] == cat])

    overall = _acc(rows)
    metrics: Dict[str, Any] = {
        "benchmark": "mmbench",
        "n_samples": len(rows),
        "n_expected": MMBENCH_DEV_ROWS,
        # A short run reads as a low score, not as an error, so say it.
        "complete": len(rows) == MMBENCH_DEV_ROWS,
        "totals": {"overall": overall, "perception": group("perception"),
                   "reasoning": group("reasoning")},
        "per_l2_category": per_l2,
        "per_category": per_cat,
        "l2_missing": sorted(set(group_of) - set(per_l2)),
        **_mc_tally(rows),
    }
    metrics["unparsed_fraction"] = (
        round(metrics["unparsed"] / len(rows), 6) if rows else None)
    if overall:
        metrics["accuracy_ci"] = wilson_interval(
            overall["n_right"], overall["n"], confidence=confidence,
            name="mmbench_accuracy").as_dict()
    return metrics, rows


def evaluate_mmbench_circular(
    records: Sequence[Dict[str, Any]],
    confidence: float = 0.95,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """MMBench CircularEval: a question counts only if EVERY rotation is right.

    Records are grouped by their persisted ``circ_index``. A question with N
    options must carry exactly the rotations ``0 .. N-1`` (``circ_n`` says N,
    and must equal the size of the record's option set); anything else RAISES
    and nothing is scored: a partial set is not a harder or easier score, it
    is a different question, and an incomplete group that were dropped (or
    counted as right on its present rotations) would move the total without a
    trace. A group whose rotations disagree on category, l2-category or on
    the multiset of option texts is refused too.

    Per-record parsing, the group map and the refusals for unknown
    categories, a missing field or a gold outside the options are
    :func:`evaluate_mmbench`'s, reused (not re-implemented) so the two cannot
    drift. An unparsed answer is a wrong rotation.

    Reports ``totals`` (circular overall / perception / reasoning), and, from
    the SAME records, ``single_pass`` (rotation 0 only: the question as
    asked, the single-pass benchmark's own item) and ``per_rotation`` (every
    record counted once) for a direct comparison.
    """
    from foxrepro.storage import MMBENCH_DEV_ROWS

    if not records:
        raise ValueError("MMBench circular: no records; refusing to score an "
                         "empty run (a zero would read as a measurement)")
    # Validates every record (categories, gold, options) and parses answers.
    _, rows = evaluate_mmbench(records, confidence=confidence)

    groups: Dict[int, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {}
    for rec, row in zip(records, rows):
        sid = rec.get("sample_id")
        for field in ("circ_index", "circ_rotation", "circ_n"):
            if rec.get(field) is None:
                raise ValueError(
                    f"circular record {sid!r} carries no {field!r}; it "
                    f"cannot be regrouped into its question. Re-run with a "
                    f"generate.py that persists it.")
        groups.setdefault(int(rec["circ_index"]), []).append((rec, row))

    q_rows: List[Dict[str, Any]] = []
    rot0: List[Dict[str, Any]] = []
    for index in sorted(groups):
        members = groups[index]
        n_decl = {int(r["circ_n"]) for r, _ in members}
        n_opts = {len(_mc_options(r)) for r, _ in members}
        if len(n_decl) != 1 or n_decl != n_opts:
            raise ValueError(
                f"question {index}: circ_n {sorted(n_decl)} disagrees with "
                f"its option counts {sorted(n_opts)}")
        n = n_decl.pop()
        have = sorted(int(r["circ_rotation"]) for r, _ in members)
        if have != list(range(n)):
            raise ValueError(
                f"question {index} has rotations {have} but needs "
                f"{list(range(n))}: a partial set is not scored (re-run the "
                f"job to completion; resume fills the missing rows)")
        for field in ("category", "l2_category"):
            if len({str(r[field]) for r, _ in members}) != 1:
                raise ValueError(
                    f"question {index}: its rotations disagree on {field}")
        if len({tuple(sorted(_mc_options(r).values()))
                for r, _ in members}) != 1:
            raise ValueError(
                f"question {index}: its rotations do not offer the same "
                f"option texts")
        by_rot = {int(r["circ_rotation"]): row for r, row in members}
        first = by_rot[0]
        q_rows.append({
            "sample_id": f"mmbench_circular/{index}", "index": index,
            "n_options": n, "category": first["category"],
            "l2_category": first["l2_category"], "group": first["group"],
            "n_correct_rotations": sum(by_rot[k]["correct"] for k in range(n)),
            "correct": int(all(by_rot[k]["correct"] for k in range(n))),
            "rule": first["rule"], "parsed": all(
                by_rot[k]["parsed"] for k in range(n)),
        })
        rot0.append(first)

    def group(rs: Sequence[Dict[str, Any]], name: str):
        return _acc([r for r in rs if r["group"] == name])

    def totals(rs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return {"overall": _acc(rs), "perception": group(rs, "perception"),
                "reasoning": group(rs, "reasoning")}

    per_l2 = {l2: _acc([r for r in q_rows if r["l2_category"] == l2])
              for l2 in sorted({r["l2_category"] for r in q_rows})}
    per_cat = {c: _acc([r for r in q_rows if r["category"] == c])
               for c in sorted({r["category"] for r in q_rows})}
    by_n = {str(n): _acc([r for r in q_rows if r["n_options"] == n])
            for n in sorted({r["n_options"] for r in q_rows})}
    # Expected score of a uniformly random guesser: right on a rotation with
    # probability 1/N, on all N independently with N**-N.
    chance = 100.0 * sum(r["n_options"] ** -r["n_options"]
                         for r in q_rows) / len(q_rows)
    overall = _acc(q_rows)
    unparsed_rot = sum(1 for r in rows if not r["parsed"])
    metrics: Dict[str, Any] = {
        "benchmark": "mmbench_circular",
        "n_samples": len(q_rows),                 # questions
        "n_rotation_records": len(rows),
        "n_expected": MMBENCH_DEV_ROWS,
        "complete": len(q_rows) == MMBENCH_DEV_ROWS,
        "totals": totals(q_rows),
        "single_pass": totals(rot0),
        "per_rotation": totals(rows),
        "per_l2_category": per_l2,
        "per_category": per_cat,
        "by_n_options": by_n,
        "circular_chance_random": round(chance, 4),
        **_mc_tally(rows),
    }
    metrics["unparsed_fraction"] = round(unparsed_rot / len(rows), 6)
    metrics["accuracy_ci"] = wilson_interval(
        overall["n_right"], overall["n"], confidence=confidence,
        name="mmbench_circular_accuracy").as_dict()
    return metrics, q_rows


def evaluate_mathvista(
    records: Sequence[Dict[str, Any]],
    confidence: float = 0.95,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """MathVista multiple-choice accuracy with a rule-based extractor.

    Accuracy is over ALL scored questions, with an unparseable answer counted
    wrong. ``coverage`` -- the parsed fraction -- is reported beside it
    because the official protocol's LLM extractor would recover some answers
    these rules cannot, so a low coverage means the number understates the
    model by an unknown amount (docs/DEVIATIONS.md). ``accuracy_parsed`` is
    accuracy over parsed answers only, so the two readings can be compared.

    A gold with more than one letter (a question whose correct choice text is
    duplicated) is correct if the prediction is any of them.
    """
    from foxrepro.bench.multichoice import extract_choice
    from foxrepro.storage import MATHVISTA_SUBSET_N

    rows: List[Dict[str, Any]] = []
    for rec in records:
        sid = rec.get("sample_id")
        if not rec.get("label"):
            raise ValueError(f"MathVista record {sid!r} carries no label")
        options = _mc_options(rec)
        gold = str(rec["label"]).split("|")
        if any(g not in options for g in gold):
            raise ValueError(
                f"MathVista record {sid!r}: gold {gold} is not among its "
                f"options {sorted(options)}")
        pred, rule = extract_choice(
            str(rec.get("output", "")), options, allow_tail=True)
        rows.append({
            "sample_id": sid, "label": "|".join(gold), "pred": pred,
            "rule": rule, "parsed": pred is not None,
            "correct": int(pred is not None and pred in gold),
        })
    overall = _acc(rows)
    parsed = _acc([r for r in rows if r["parsed"]])
    metrics: Dict[str, Any] = {
        "benchmark": "mathvista",
        "n_samples": len(rows),
        "n_expected": MATHVISTA_SUBSET_N,
        "complete": len(rows) == MATHVISTA_SUBSET_N,
        "totals": {"accuracy": overall, "accuracy_parsed": parsed},
        "coverage": (round(sum(r["parsed"] for r in rows) / len(rows), 6)
                     if rows else None),
        **_mc_tally(rows),
    }
    metrics["unparsed_fraction"] = (
        round(metrics["unparsed"] / len(rows), 6) if rows else None)
    if overall:
        metrics["accuracy_ci"] = wilson_interval(
            overall["n_right"], overall["n"], confidence=confidence,
            name="mathvista_accuracy").as_dict()
    return metrics, rows


def evaluate_pope(
    records: Sequence[Dict[str, Any]],
    labels: Optional[Dict[str, str]] = None,
    unparsed_policy: str = "negative",
    confidence: float = 0.95,
    n_boot: int = 10000,
    seed: int = 20260101,
    pope_parser: str = DEFAULT_POPE_PARSER,
) -> Tuple[Dict[str, Any], List[PopeOutcome]]:
    """Compute POPE metrics.

    Args:
        records: run records for one POPE split.
        labels: ``sample_id -> "yes"/"no"`` ground truth. Optional: the
            generator stores ``label`` in each record, and that copy is used
            when present. Supplying ``labels`` re-checks it and raises on
            disagreement, which catches a mismatched POPE file.
        unparsed_policy: ``"negative"`` treats an unparsable answer as "no"
            (what the common POPE scripts do, since they test ``"no" in
            text``); ``"exclude"`` drops it from every denominator. Either way
            the count is reported. Irrelevant under the ``released`` parser,
            which never returns ``None``.
        pope_parser: ``"released"`` (default, :func:`parse_yes_no_released`)
            or ``"strict"`` (:func:`parse_yes_no`). Recorded in the metrics,
            because the two rules can disagree and a number is meaningless
            without knowing which produced it.
        confidence: interval coverage.
        n_boot: bootstrap replicates for the F1 interval.
        seed: bootstrap seed.

    Returns:
        ``(metrics, outcomes)``.
    """
    if unparsed_policy not in ("negative", "exclude"):
        raise ValueError("unparsed_policy must be 'negative' or 'exclude'")
    try:
        parse = POPE_PARSERS[pope_parser]
    except KeyError:
        raise ValueError(
            f"unknown pope_parser {pope_parser!r}; known: {sorted(POPE_PARSERS)}"
        ) from None

    outcomes: List[PopeOutcome] = []
    n_unparsed = 0
    for rec in records:
        sid = rec["sample_id"]
        label = rec.get("label")
        if labels is not None:
            truth = labels.get(sid)
            if truth is None:
                raise KeyError(f"no ground-truth label for {sid}")
            if label is not None and truth != label:
                raise ValueError(
                    f"{sid}: record label {label!r} != annotation label {truth!r}; "
                    "the run and the POPE file disagree."
                )
            label = truth
        if label not in ("yes", "no"):
            raise ValueError(
                f"{sid}: no usable ground-truth label (got {label!r}). Pass "
                "--pope-dir so labels can be joined from the annotation file."
            )
        pred = parse(rec.get("output", ""))
        if pred is None:
            n_unparsed += 1
            if unparsed_policy == "exclude":
                continue
            pred_eff = "no"
        else:
            pred_eff = pred
        outcomes.append(
            PopeOutcome(
                sample_id=sid, image_id=int(rec.get("image_id", -1)), label=label,
                pred=pred, correct=int(pred_eff == label),
                said_yes=int(pred_eff == "yes"),
            )
        )

    n = len(outcomes)
    tp = sum(1 for o in outcomes if o.label == "yes" and o.said_yes)
    fp = sum(1 for o in outcomes if o.label == "no" and o.said_yes)
    fn = sum(1 for o in outcomes if o.label == "yes" and not o.said_yes)
    tn = sum(1 for o in outcomes if o.label == "no" and not o.said_yes)
    n_correct = tp + tn
    n_pred_yes = tp + fp
    n_gt_yes = tp + fn

    codes = [
        1 if (o.label == "yes" and o.said_yes) else
        2 if (o.label == "no" and o.said_yes) else
        3 if (o.label == "yes" and not o.said_yes) else 0
        for o in outcomes
    ]
    f1_point = _f1_from_codes(codes)
    f1_lo, f1_hi = bootstrap_ci(codes, _f1_from_codes, n_boot=n_boot,
                                confidence=confidence, seed=seed)

    acc = wilson_interval(n_correct, n, confidence, "pope_accuracy")
    prec = wilson_interval(tp, n_pred_yes, confidence, "pope_precision")
    rec_ = wilson_interval(tp, n_gt_yes, confidence, "pope_recall")
    yes = wilson_interval(n_pred_yes, n, confidence, "pope_yes_ratio")

    metrics: Dict[str, Any] = {
        "benchmark": "pope",
        "n_samples": n,
        "n_unparsed": n_unparsed,
        "unparsed_policy": unparsed_policy,
        "pope_parser": pope_parser,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "denominators": {
            "accuracy": n,
            "precision": n_pred_yes,
            "recall": n_gt_yes,
            "f1": n,
            "yes_ratio": n,
            "n_ground_truth_yes": n_gt_yes,
            "n_ground_truth_no": tn + fp,
        },
        "accuracy": acc.as_dict(),
        "precision": prec.as_dict(),
        "recall": rec_.as_dict(),
        "yes_ratio": yes.as_dict(),
        "f1": {
            "name": "pope_f1", "point": f1_point, "ci_lo": f1_lo, "ci_hi": f1_hi,
            "n": n, "confidence": confidence,
            "interval_method": f"percentile bootstrap (B={n_boot}, seed={seed})",
        },
        "interval_method": "Wilson score (except F1)",
    }
    return metrics, outcomes


def load_pope_labels(pope_dir: Path, split: str) -> Dict[str, str]:
    """``sample_id -> label`` for one POPE split (cross-check / repair)."""
    from foxrepro.data.loaders import load_pope  # local import keeps module import cheap

    # image paths are unused here (require_images=False); only labels are read
    samples = load_pope(pope_dir, Path("."), split, require_images=False)
    return {s.sample_id: str(s.label) for s in samples}


# --------------------------------------------------------------------------- #
# CHAIR
# --------------------------------------------------------------------------- #
def evaluate_chair(
    records: Sequence[Dict[str, Any]],
    scorer: CHAIRScorer,
    confidence: float = 0.95,
    n_boot: int = 10000,
    seed: int = 20260101,
) -> Tuple[Dict[str, Any], List[CaptionScore]]:
    """Compute CHAIR metrics for one run.

    Args:
        records: run records for the CHAIR split.
        scorer: configured :class:`~bench.chair.CHAIRScorer`.
        confidence: interval coverage.
        n_boot: replicates for the CHAIR_I cluster bootstrap.
        seed: bootstrap seed.

    Returns:
        ``(metrics, per_caption_scores)``.
    """
    scores = scorer.score_many(
        (r["sample_id"], int(r["image_id"]), str(r.get("output", ""))) for r in records
    )
    totals = scorer.totals(scores)

    chair_s = wilson_interval(
        totals.n_captions_with_hallucination, totals.n_captions, confidence, "chair_s"
    )
    chair_i_wilson = wilson_interval(
        totals.n_hallucinated_mentions, totals.n_mentions, confidence, "chair_i"
    )

    # cluster bootstrap over captions: the correct interval for a ratio of
    # sums whose trials are grouped inside captions
    num = [float(s.n_hallucinated) for s in scores]
    den = [float(s.n_mentions) for s in scores]
    idx_values = list(range(len(scores)))

    def _ratio(resampled: Sequence[int]) -> float:
        d = sum(den[i] for i in resampled)
        return sum(num[i] for i in resampled) / d if d else float("nan")

    ci_lo, ci_hi = bootstrap_ci(idx_values, _ratio, n_boot=n_boot,
                                confidence=confidence, seed=seed)

    metrics: Dict[str, Any] = {
        "benchmark": "chair",
        "n_samples": totals.n_captions,
        "denominators": {
            "chair_s": totals.n_captions,
            "chair_i": totals.n_mentions,
            "chair_i_note": (
                "total mentioned object instances, duplicates included; not "
                "stated in the paper, required for any interval"
            ),
            "recall": totals.n_gt_objects,
            "precision": totals.n_distinct_mentions,
        },
        "totals": totals.as_dict(),
        "chair_s_ci": chair_s.as_dict(),
        "chair_i_ci_wilson": chair_i_wilson.as_dict(),
        "chair_i_ci_cluster_bootstrap": {
            "name": "chair_i", "point": totals.chair_i, "ci_lo": ci_lo,
            "ci_hi": ci_hi, "n": totals.n_mentions, "confidence": confidence,
            "interval_method": (
                f"cluster bootstrap over captions (B={n_boot}, seed={seed}); "
                "preferred over Wilson because mentions are clustered in captions"
            ),
        },
        "scorer": scorer.config(),
    }
    return metrics, scores


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def evaluate_run_file(
    path: Path,
    scorer_factory: Optional[Callable[[], CHAIRScorer]] = None,
    pope_dir: Optional[Path] = None,
    unparsed_policy: str = "negative",
    confidence: float = 0.95,
    n_boot: int = 10000,
    seed: int = 20260101,
    strict: bool = True,
    pope_parser: str = DEFAULT_POPE_PARSER,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate one run file.

    Args:
        path: run JSONL.
        scorer_factory: zero-arg factory for a :class:`CHAIRScorer`; required
            for CHAIR runs, built lazily so POPE evaluation needs no COCO
            annotations at all.
        pope_dir: POPE annotation directory, for cross-checking labels.
        unparsed_policy: see :func:`evaluate_pope`.
        confidence: interval coverage.
        n_boot: bootstrap replicates.
        seed: bootstrap seed.
        strict: see :func:`read_run`.
        pope_parser: see :func:`evaluate_pope`.

    Returns:
        ``(result, outcome_rows)`` where ``result`` is the full metrics
        document and ``outcome_rows`` are per-sample rows for paired tests.
    """
    records, meta = read_run(Path(path), strict=strict)
    benchmark = meta["benchmark"]
    if benchmark == "pope":
        labels = (
            load_pope_labels(pope_dir, str(meta["split"])) if pope_dir else None
        )
        metrics, outcomes = evaluate_pope(
            records, labels=labels, unparsed_policy=unparsed_policy,
            confidence=confidence, n_boot=n_boot, seed=seed,
            pope_parser=pope_parser,
        )
        rows = [o.as_dict() for o in outcomes]
    elif benchmark == "chair":
        if scorer_factory is None:
            raise ValueError("CHAIR evaluation needs a scorer_factory")
        metrics, scores = evaluate_chair(
            records, scorer_factory(), confidence=confidence, n_boot=n_boot, seed=seed
        )
        rows = [s.as_dict() for s in scores]
    elif benchmark == "mme":
        metrics, rows = evaluate_mme(
            records, unparsed_policy=unparsed_policy, confidence=confidence,
            n_boot=n_boot, seed=seed, pope_parser=pope_parser,
        )
    elif benchmark == "mmbench":
        metrics, rows = evaluate_mmbench(records, confidence=confidence)
    elif benchmark == "mmbench_circular":
        metrics, rows = evaluate_mmbench_circular(records, confidence=confidence)
    elif benchmark == "mathvista":
        metrics, rows = evaluate_mathvista(records, confidence=confidence)
    else:
        raise ValueError(f"unknown benchmark {benchmark!r}")

    result = {
        "schema": "fox-metrics/1",
        "provenance": meta,
        "metrics": metrics,
        "eval_config": {
            "confidence": confidence, "n_boot": n_boot, "bootstrap_seed": seed,
            "unparsed_policy": unparsed_policy,
            "pope_parser": pope_parser,
        },
    }
    return result, rows


def _format_markdown(result: Dict[str, Any]) -> str:
    """Single-run metrics as a short Markdown block (for eyeballing a run)."""
    prov, met = result["provenance"], result["metrics"]
    head = (
        f"### {prov['method']} / {prov['benchmark']} / {prov['split']} "
        f"(seed {prov['seed']}, {prov['dtype']}, gamma={prov.get('gamma')})\n\n"
        f"- run file: `{Path(prov['run_file']).name}`\n"
        f"- model: `{prov.get('model_id')}`  commit: `{prov.get('commit_hash')}`\n"
        f"- transformers: `{prov.get('transformers_path')}`\n"
        f"- records: {prov['n_records']} (duplicates skipped: "
        f"{prov['n_duplicate_sample_ids']})\n\n"
    )
    lines = ["| metric | value | 95% CI | numerator | denominator |",
             "| --- | --- | --- | --- | --- |"]
    if met["benchmark"] == "pope":
        for name in ("accuracy", "precision", "recall", "yes_ratio"):
            m = met[name]
            lines.append(
                f"| {name} | {100 * m['point']:.2f} | "
                f"[{100 * m['ci_lo']:.2f}, {100 * m['ci_hi']:.2f}] | "
                f"{m['successes']:.0f} | {m['n']} |"
            )
        f1 = met["f1"]
        lines.append(
            f"| f1 | {100 * f1['point']:.2f} | "
            f"[{100 * f1['ci_lo']:.2f}, {100 * f1['ci_hi']:.2f}] | - | {f1['n']} |"
        )
        lines.append(
            f"| unparsed | {met['n_unparsed']} | - | - | {met['n_samples']} |"
        )
    elif met["benchmark"] == "mme":
        t = met["totals"]
        acc = met["accuracy_ci"]
        # Perception runs to 2000 and Cognition to 800: each subtask scores
        # (acc + acc_plus) x 100, summed over ten and four subtasks.
        lines.append(f"| perception | {t['perception']:.1f} | - | - | 2000 |")
        lines.append(f"| cognition | {t['cognition']:.1f} | - | - | 800 |")
        lines.append(f"| overall | {t['overall']:.1f} | - | - | 2800 |")
        lines.append(
            f"| accuracy | {100 * acc['point']:.2f} | "
            f"[{100 * acc['ci_lo']:.2f}, {100 * acc['ci_hi']:.2f}] | "
            f"{acc['successes']:.0f} | {acc['n']} |"
        )
        lines.append(
            f"| unparsed | {met['unparsed']} | - | - | {met['n_samples']} |")
        if met["subtasks_missing"]:
            # A missing subtask is a SILENT zero in a score that sums over
            # subtasks, so it is printed next to the total it deflates.
            lines.append(
                f"| subtasks missing | {', '.join(met['subtasks_missing'])} "
                f"| - | - | - |")
    elif met["benchmark"] in ("mmbench", "mmbench_circular", "mathvista"):
        def _cell(name: str, g: Any, ci: Any = None) -> str:
            # An empty group is "n/a", never a zero: nothing was measured.
            if g is None:
                return f"| {name} | n/a (no questions) | - | - | 0 |"
            band = ("-" if ci is None else
                    f"[{100 * ci['ci_lo']:.2f}, {100 * ci['ci_hi']:.2f}]")
            return (f"| {name} | {g['acc']:.2f} | {band} | "
                    f"{g['n_right']} | {g['n']} |")
        t = met["totals"]
        if met["benchmark"] == "mmbench_circular":
            lines.append(_cell("circular overall", t["overall"],
                               met.get("accuracy_ci")))
            lines.append(_cell("circular perception", t["perception"]))
            lines.append(_cell("circular reasoning", t["reasoning"]))
            for label, key in (("single-pass (rotation 0) ", "single_pass"),
                               ("per-rotation ", "per_rotation")):
                for g in ("overall", "perception", "reasoning"):
                    lines.append(_cell(label + g, met[key][g]))
            lines.append(f"| random-guess circular chance | "
                         f"{met['circular_chance_random']:.2f} | - | - | - |")
        elif met["benchmark"] == "mmbench":
            lines.append(_cell("overall", t["overall"], met.get("accuracy_ci")))
            lines.append(_cell("perception", t["perception"]))
            lines.append(_cell("reasoning", t["reasoning"]))
        else:
            lines.append(_cell("accuracy", t["accuracy"], met.get("accuracy_ci")))
            lines.append(_cell("accuracy (parsed only)", t["accuracy_parsed"]))
            cov = met["coverage"]
            lines.append(
                f"| coverage (parsed) | "
                f"{'n/a' if cov is None else format(100 * cov, '.2f')} "
                f"| - | - | {met['n_samples']} |")
        lines.append(
            f"| unparsed (counted wrong) | {met['unparsed']} | - | - | "
            f"{met['n_samples']} |")
        if not met["complete"]:
            # A short run is a low score in disguise.
            lines.append(
                f"| INCOMPLETE | {met['n_samples']} of {met['n_expected']} "
                f"questions | - | - | - |")
    elif met["benchmark"] == "chair":
        t = met["totals"]
        cs, ci = met["chair_s_ci"], met["chair_i_ci_cluster_bootstrap"]
        lines.append(
            f"| CHAIR_S | {100 * cs['point']:.2f} | "
            f"[{100 * cs['ci_lo']:.2f}, {100 * cs['ci_hi']:.2f}] | "
            f"{t['chair_s_numerator']} | {t['chair_s_denominator']} |"
        )
        lines.append(
            f"| CHAIR_I | {100 * ci['point']:.2f} | "
            f"[{100 * ci['ci_lo']:.2f}, {100 * ci['ci_hi']:.2f}] | "
            f"{t['chair_i_numerator']} | {t['chair_i_denominator']} |"
        )
        for name in ("recall", "precision", "f1"):
            lines.append(f"| {name} | {100 * t[name]:.2f} | - | - | "
                         f"{t.get(name + '_denominator', '-')} |")
        lines.append(f"| mean caption words | {t['mean_caption_words']:.1f} | - | - | - |")
    else:
        # This used to be the CHAIR branch, reached by `else`. MME metrics
        # therefore fell into it and died on met["chair_s_ci"] three frames
        # down, AFTER its own metrics file had been written -- so one run
        # scored, the exception left the loop, and the other twenty never
        # ran. A formatter is not where a new benchmark should be discovered.
        raise ValueError(
            f"no markdown formatter for benchmark "
            f"{met['benchmark']!r}; add one beside the pope/mme/chair "
            f"branches rather than letting it fall through"
        )
    return head + "\n".join(lines) + "\n"


def _main(argv: Optional[Sequence[str]] = None) -> int:
    p = paths()
    ap = argparse.ArgumentParser(
        description="Compute POPE / CHAIR metrics from generation JSONL (no GPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("runs", nargs="*", help="run JSONL files or globs")
    ap.add_argument("--runs-dir", type=Path, default=p.runs_root,
                    help="used when no explicit run files are given")
    ap.add_argument("--out-dir", type=Path, default=p.reports_dir)
    ap.add_argument("--format", choices=("json", "markdown", "both"), default="both")
    ap.add_argument(
        "--pope-parser", choices=tuple(POPE_PARSERS), default=DEFAULT_POPE_PARSER,
        help="POPE answer rule. 'released' transcribes pope_eval.py (any "
             "answer without a negation token counts as yes); 'strict' uses "
             "this module's parser, which reports unparsable answers.")
    ap.add_argument("--unparsed-policy", choices=("negative", "exclude"),
                    default="negative")
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--pope-dir", type=Path, default=None,
                    help="cross-check POPE labels against the annotation files")
    ap.add_argument("--synonyms", type=Path, default=p.chair_synonyms)
    # Prefers the 500-image subset that build_coco_subset.py packs, and falls
    # back to the full file that `fetch_images --with-annotations` leaves.
    ap.add_argument("--instances", type=Path,
                    default=p.coco_annotation("instances_val2014"))
    ap.add_argument("--captions", type=Path,
                    default=p.coco_annotation("captions_val2014"))
    ap.add_argument("--gt-source", choices=("both", "instances", "captions"),
                    default="both")
    ap.add_argument("--tokenizer", choices=("regex", "nltk"), default="regex")
    ap.add_argument(
        "--only-stale", action="store_true",
        help="skip run files whose .metrics.json is newer than the run. The "
             "fast path after ingesting a round. NOT the default and not "
             "safe after a scorer change, which leaves every run file "
             "untouched and every metric wrong.",
    )
    ap.add_argument("--no-strict", action="store_true",
                    help="tolerate runs whose provenance fields are not constant")
    args = ap.parse_args(argv)

    targets: List[Path] = []
    for pattern in args.runs:
        targets.extend(Path(x) for x in sorted(glob.glob(pattern)))
    if not targets:
        targets = sorted(Path(args.runs_dir).glob("*.jsonl"))
    if not targets:
        print(f"no run files found (looked in {args.runs_dir})", file=sys.stderr)
        return 1

    def scorer_factory() -> CHAIRScorer:
        return CHAIRScorer(
            synonyms_path=args.synonyms, instances_json=args.instances,
            captions_json=args.captions if args.gt_source != "instances" else None,
            tokenizer=args.tokenizer, gt_source=args.gt_source,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    exit_code = 0
    n_superseded = 0
    n_skipped = 0
    for target in targets:
        if args.only_stale:
            # Opt-in, never the default. Skipping by mtime is right when the
            # RUN changed and wrong when the SCORER did -- a new CHAIR
            # synonym list or a parser fix leaves every run file untouched
            # and every metric wrong. So the canonical pass stays a full
            # pass, and this is the fast path for "I just ingested a round".
            m = args.out_dir / f"{target.name[:-len('.jsonl')]}.metrics.json"
            try:
                if m.is_file() and m.stat().st_mtime >= target.stat().st_mtime:
                    n_skipped += 1
                    continue
            except OSError:
                pass
        try:
            result, rows = evaluate_run_file(
                target, scorer_factory=scorer_factory, pope_dir=args.pope_dir,
                unparsed_policy=args.unparsed_policy, confidence=args.confidence,
                pope_parser=args.pope_parser,
                n_boot=args.n_boot, seed=args.seed, strict=not args.no_strict,
            )
        except LegacyProvenance as exc:
            # Not a failure: a run archived before a provenance field existed.
            # No re-run can fix it, because the run is history -- so letting it
            # set the exit code would make a red exit PERMANENT, and a status
            # that is always red is a status nobody reads. Reported, counted,
            # and deliberately not fatal. Anything else missing still is.
            print(f"[evaluate] SUPERSEDED {target.name}: {exc}", file=sys.stderr)
            n_superseded += 1
            continue
        except Exception as exc:
            print(f"[evaluate] FAILED {target.name}: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        stem = target.name[: -len(".jsonl")]
        if args.format in ("json", "both"):
            out = args.out_dir / f"{stem}.metrics.json"
            out.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
            outcomes = args.out_dir / f"{stem}.outcomes.jsonl"
            with open(outcomes, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"[evaluate] wrote {out.name} and {outcomes.name}")
        if args.format in ("markdown", "both"):
            print()
            print(_format_markdown(result))
    if n_skipped:
        print(f"[evaluate] {n_skipped} run file(s) skipped as already scored "
              f"(--only-stale); a scorer change needs a full pass",
              file=sys.stderr)
    if n_superseded:
        print(f"[evaluate] {n_superseded} run file(s) skipped as superseded "
              f"(archived before a provenance field existed); not a failure",
              file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(_main())
