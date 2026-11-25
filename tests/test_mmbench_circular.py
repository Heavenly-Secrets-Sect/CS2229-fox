"""MMBench CircularEval (`mmbench_circular`), the test of V20's failed band.

What needs pinning is what a green run could hide:

* the rotations: the gold LETTER moves with the answer's position, every
  question gets exactly N of them, rotation 0 is the question as asked;
* sample ids are unique PER ROTATION and everything the scorer needs is
  persisted in the record;
* the scorer counts a question only if all rotations are right, RAISES on a
  partial set (never scores it), on an empty run and on an unknown category;
* a model that always answers "A" scores exactly the circular level it must
  (0.0), and a random guesser's expected level is the analytic N**-N;
* the single-pass benchmark is untouched.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from foxrepro.bench.evaluate import (
    _format_markdown, evaluate_mmbench, evaluate_mmbench_circular,
    evaluate_run_file,
)
from foxrepro.bench.generate import PERSISTED_SAMPLE_FIELDS
from foxrepro.bench.plan import COST_S, N_SAMPLES, PLAN, Job
from foxrepro.data import fetch_mmbench
from foxrepro.data.loaders import (
    DEFAULT_TEMPLATE_IDS, PROMPT_TEMPLATES, circular_rotations, load_benchmark,
    load_mmbench, load_mmbench_circular, template_benchmark,
)
from foxrepro.storage import (
    BENCHMARK_SPLITS, MMBENCH_CIRCULAR_ROWS_UPPER_BOUND, MMBENCH_DEV_ROWS,
)
import tests.test_table11 as t11
from tests.test_table11 import mm_rec, mmbench_rows


# --------------------------------------------------------------------------- #
# rotations
# --------------------------------------------------------------------------- #
class TestRotations(unittest.TestCase):
    def test_gold_letter_follows_the_answer_text(self) -> None:
        opts = {"A": "w", "B": "x", "C": "y", "D": "z"}
        for answer in "ABCD":
            rots = circular_rotations(opts, answer)
            self.assertEqual([k for k, _, _ in rots], [0, 1, 2, 3])
            for _, ro, gold in rots:
                self.assertEqual(ro[gold], opts[answer])
                self.assertEqual(sorted(ro.values()), sorted(opts.values()))
            # the gold takes every letter exactly once across the rotations
            self.assertEqual(sorted(g for _, _, g in rots), list("ABCD"))

    def test_rotation_zero_is_the_question_as_asked(self) -> None:
        opts = {"A": "w", "B": "x", "C": "y"}
        k, ro, gold = circular_rotations(opts, "B")[0]
        self.assertEqual((k, ro, gold), (0, opts, "B"))

    def test_n_options_give_n_rotations(self) -> None:
        for n in (2, 3, 4):
            opts = {c: f"t{c}" for c in "ABCD"[:n]}
            self.assertEqual(len(circular_rotations(opts, "A")), n)

    def test_a_repeated_option_text_still_tracks_the_position(self) -> None:
        opts = {"A": "same", "B": "same", "C": "other"}
        rots = circular_rotations(opts, "B")        # gold is the SECOND "same"
        self.assertEqual([g for _, _, g in rots], ["B", "A", "C"])

    def test_bad_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            circular_rotations({"A": "x"}, "A")
        with self.assertRaises(ValueError):
            circular_rotations({"A": "x", "C": "y"}, "A")
        with self.assertRaises(ValueError):
            circular_rotations({"A": "x", "B": "y"}, "C")


# --------------------------------------------------------------------------- #
# loader
# --------------------------------------------------------------------------- #
class TestLoader(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.dest = Path(self._td.name)
        fetch_mmbench.build(mmbench_rows(), self.dest)

    def test_expansion_size_ids_and_order(self) -> None:
        base = load_mmbench(self.dest)
        circ = load_mmbench_circular(self.dest)
        self.assertEqual(len(circ), sum(len(s.extra["options"]) for s in base))
        ids = [s.sample_id for s in circ]
        self.assertEqual(len(ids), len(set(ids)), "ids must be unique per rotation")
        self.assertTrue(all(i.startswith("mmbench_circular/") for i in ids))
        keys = [(s.extra["circ_index"], s.extra["circ_rotation"]) for s in circ]
        self.assertEqual(keys, sorted(keys))
        self.assertLessEqual(len(circ), MMBENCH_CIRCULAR_ROWS_UPPER_BOUND)

    def test_rotation_zero_matches_the_single_pass_item_exactly(self) -> None:
        base = {int(s.sample_id.split("/")[1]): s for s in load_mmbench(self.dest)}
        for s in load_mmbench_circular(self.dest):
            if s.extra["circ_rotation"] == 0:
                b = base[s.extra["circ_index"]]
                self.assertEqual(s.question, b.question)
                self.assertEqual(s.label, b.label)
                self.assertEqual(s.extra["options"], b.extra["options"])

    def test_every_rotation_is_consistent_with_its_own_question_text(self) -> None:
        base = {int(s.sample_id.split("/")[1]): s for s in load_mmbench(self.dest)}
        for s in load_mmbench_circular(self.dest):
            b = base[s.extra["circ_index"]]
            opts = s.extra["options"]
            self.assertEqual(opts[s.label],
                             b.extra["options"][b.label])      # gold text moved
            self.assertTrue(s.question.endswith(
                "\n".join(f"{c}. {opts[c]}" for c in sorted(opts))))
            self.assertTrue(s.question.startswith(
                b.question[: len(b.question)
                           - len("\n".join(f"{c}. {v}" for c, v in
                                          sorted(b.extra["options"].items())))]))
            self.assertEqual(s.extra["circ_n"], len(opts))
            self.assertEqual(s.extra["category"], b.extra["category"])
            self.assertEqual(s.extra["l2_category"], b.extra["l2_category"])

    def test_dispatch_through_load_benchmark(self) -> None:
        got = load_benchmark("mmbench_circular", "dev", Path("."), Path("."),
                             Path("."), "mmbench_circular_v0_letter",
                             mmbench_dir=self.dest)
        self.assertEqual(got[0].benchmark, "mmbench_circular")
        with self.assertRaises(ValueError):
            load_benchmark("mmbench_circular", "dev", Path("."), Path("."),
                           Path("."), "mmbench_circular_v0_letter")
        with self.assertRaises(ValueError):       # the single-pass template
            load_benchmark("mmbench_circular", "dev", Path("."), Path("."),
                           Path("."), "mmbench_v0_letter",
                           mmbench_dir=self.dest)

    def test_the_prompt_wording_is_the_single_pass_wording(self) -> None:
        self.assertEqual(PROMPT_TEMPLATES["mmbench_circular_v0_letter"],
                         PROMPT_TEMPLATES["mmbench_v0_letter"])
        self.assertEqual(template_benchmark("mmbench_circular_v0_letter"),
                         "mmbench_circular")
        self.assertEqual(DEFAULT_TEMPLATE_IDS["mmbench_circular"],
                         "mmbench_circular_v0_letter")

    def test_persisted_fields_cover_what_the_scorer_reads(self) -> None:
        for f in ("circ_index", "circ_rotation", "circ_n", "options",
                  "category", "l2_category"):
            self.assertIn(f, PERSISTED_SAMPLE_FIELDS)


# --------------------------------------------------------------------------- #
# scorer
# --------------------------------------------------------------------------- #
def circ_records(n_questions=12, answer_fn=None, l2s=None):
    """Records as generate.py persists them, from the fixture rows."""
    with tempfile.TemporaryDirectory() as td:
        # A limited build writes partial_<n>_mmbench_dev.jsonl; load_mmbench
        # insists on the full 4377 rows, so expand the manifest rows through
        # circular_rotations, the function the loader itself uses.
        fetch_mmbench.build(mmbench_rows(n_questions), Path(td), limit=n_questions)
        src = Path(td) / f"partial_{n_questions}_mmbench_dev.jsonl"
        lines = src.read_text().splitlines()
        recs = []
        for line in lines:
            row = json.loads(line)
            opts = row["options"]
            for k, ro, gold in circular_rotations(opts, row["answer"]):
                out = answer_fn(row, k, ro, gold) if answer_fn else gold
                recs.append({
                    "sample_id": f"mmbench_circular/{row['index']}/r{k}",
                    "label": gold, "output": out, "options": ro,
                    "category": row["category"],
                    "l2_category": row["l2_category"],
                    "circ_index": row["index"], "circ_rotation": k,
                    "circ_n": len(opts)})
        return recs


class TestScorer(unittest.TestCase):
    def test_all_correct_scores_100_and_matches_single_pass(self) -> None:
        recs = circ_records(12)
        m, rows = evaluate_mmbench_circular(recs)
        self.assertEqual(m["totals"]["overall"]["acc"], 100.0)
        self.assertEqual(m["single_pass"]["overall"]["acc"], 100.0)
        self.assertEqual(m["per_rotation"]["overall"]["acc"], 100.0)
        self.assertEqual(m["n_samples"], 12)
        self.assertEqual(m["n_rotation_records"], len(recs))
        self.assertFalse(m["complete"])
        self.assertEqual(len(rows), 12)

    def test_one_wrong_rotation_fails_the_whole_question(self) -> None:
        def wrong_once(row, k, ro, gold):
            if row["index"] == 1003 and k == 1:
                return next(c for c in ro if c != gold)
            return gold
        m, rows = evaluate_mmbench_circular(circ_records(12, wrong_once))
        self.assertEqual(m["totals"]["overall"]["n_right"], 11)
        bad = [r for r in rows if not r["correct"]]
        self.assertEqual([r["index"] for r in bad], [1003])
        self.assertEqual(bad[0]["n_correct_rotations"], bad[0]["n_options"] - 1)
        # rotation 0 was right, so single-pass still counts it
        self.assertEqual(m["single_pass"]["overall"]["n_right"], 12)

    def test_always_A_scores_exactly_zero_circular(self) -> None:
        recs = circ_records(12, lambda row, k, ro, gold: "A")
        m, _ = evaluate_mmbench_circular(recs)
        # "A" is the gold of exactly one rotation of N >= 2, never of all.
        self.assertEqual(m["totals"]["overall"]["acc"], 0.0)
        self.assertEqual(m["totals"]["perception"]["acc"], 0.0)
        self.assertEqual(m["totals"]["reasoning"]["acc"], 0.0)
        # per-rotation: exactly one right rotation per question, so the
        # accuracy is questions / records, to the digit.
        want = round(100 * 12 / len(recs), 4)
        self.assertEqual(m["per_rotation"]["overall"]["acc"], want)
        # single pass: right iff the original gold is A (fixture: even i).
        n_a = sum(1 for r in recs if r["circ_rotation"] == 0 and r["label"] == "A")
        self.assertEqual(m["single_pass"]["overall"]["n_right"], n_a)
        self.assertEqual(m["unparsed"], 0)

    def test_random_guess_chance_level_is_N_to_the_minus_N(self) -> None:
        m, _ = evaluate_mmbench_circular(circ_records(12))
        # fixture option counts cycle 2, 3, 4: 12 questions, four of each
        want = 100 * (4 * 2 ** -2 + 4 * 3 ** -3 + 4 * 4 ** -4) / 12
        self.assertAlmostEqual(m["circular_chance_random"], want, places=3)

    def test_a_partial_rotation_set_raises(self) -> None:
        recs = circ_records(12)
        dropped = [r for r in recs
                   if not (r["circ_index"] == 1004 and r["circ_rotation"] == 2)]
        self.assertEqual(len(dropped), len(recs) - 1)
        with self.assertRaisesRegex(ValueError, "partial set"):
            evaluate_mmbench_circular(dropped)

    def test_a_question_with_only_rotation_zero_raises(self) -> None:
        recs = circ_records(12)
        only0 = [r for r in recs
                 if not (r["circ_index"] == 1002 and r["circ_rotation"] > 0)]
        with self.assertRaises(ValueError):
            evaluate_mmbench_circular(only0)

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_mmbench_circular([])

    def test_unknown_category_raises(self) -> None:
        recs = circ_records(12)
        recs[0]["l2_category"] = "vibes"
        with self.assertRaises(ValueError):
            evaluate_mmbench_circular(recs)

    def test_a_missing_persisted_field_raises(self) -> None:
        for field in ("circ_index", "circ_rotation", "circ_n", "options",
                      "l2_category"):
            recs = circ_records(12)
            del recs[3][field]
            with self.assertRaises(ValueError, msg=field):
                evaluate_mmbench_circular(recs)

    def test_inconsistent_circ_n_raises(self) -> None:
        recs = circ_records(12)
        recs[0]["circ_n"] += 1
        with self.assertRaises(ValueError):
            evaluate_mmbench_circular(recs)

    def test_rotations_with_different_option_texts_raise(self) -> None:
        recs = circ_records(12)
        for r in recs:
            if r["circ_index"] == 1005 and r["circ_rotation"] == 1:
                r["options"] = {k: v + "!" for k, v in r["options"].items()}
        with self.assertRaises(ValueError):
            evaluate_mmbench_circular(recs)

    def test_unparsed_rotation_is_wrong_and_counted(self) -> None:
        def mumble(row, k, ro, gold):
            return "I do not know" if (row["index"] == 1000 and k == 0) else gold
        m, _ = evaluate_mmbench_circular(circ_records(12, mumble))
        self.assertEqual(m["unparsed"], 1)
        self.assertEqual(m["totals"]["overall"]["n_right"], 11)

    def test_groups_are_perception_and_reasoning_like_the_single_pass(self) -> None:
        recs = circ_records(12)
        m, _ = evaluate_mmbench_circular(recs)
        s, _ = evaluate_mmbench([r for r in recs if r["circ_rotation"] == 0])
        self.assertEqual(m["single_pass"]["perception"], s["totals"]["perception"])
        self.assertEqual(m["single_pass"]["reasoning"], s["totals"]["reasoning"])
        self.assertEqual(m["single_pass"]["overall"], s["totals"]["overall"])

    def test_markdown_formats(self) -> None:
        recs = circ_records(12)
        m, _ = evaluate_mmbench_circular(recs)
        text = _format_markdown({"provenance": {
            "method": "m", "benchmark": "mmbench_circular", "split": "dev",
            "seed": 42, "dtype": "fp16", "run_file": "x.jsonl",
            "n_records": len(recs), "n_duplicate_sample_ids": 0},
            "metrics": m})
        self.assertIn("circular overall", text)
        self.assertIn("single-pass (rotation 0) overall", text)
        self.assertIn("INCOMPLETE", text)


# --------------------------------------------------------------------------- #
# the single-pass benchmark is unchanged
# --------------------------------------------------------------------------- #
class TestSinglePassUnchanged(unittest.TestCase):
    def test_single_pass_scorer_is_untouched_by_circ_fields(self) -> None:
        """A record carrying circ_* fields scores in evaluate_mmbench exactly
        as one without; and a fixed fixture keeps its hand-computed numbers."""
        recs = [mm_rec(i, "A", "A" if i % 3 else "B",
                       l2="coarse_perception" if i % 2 else "logic_reasoning")
                for i in range(12)]
        m0, _ = evaluate_mmbench(recs)
        for r in recs:
            r.update({"circ_index": 1, "circ_rotation": 0, "circ_n": 3})
        m1, _ = evaluate_mmbench(recs)
        self.assertEqual(m0, m1)
        self.assertEqual(m0["benchmark"], "mmbench")
        # 12 questions; wrong where i % 3 == 0 -> 4 wrong, 8 right
        self.assertEqual(m0["totals"]["overall"],
                         {"acc": round(100 * 8 / 12, 4), "n_right": 8, "n": 12})

    def test_single_pass_loader_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fetch_mmbench.build(mmbench_rows(), Path(td))
            ss = load_mmbench(Path(td))
        self.assertEqual(len(ss), MMBENCH_DEV_ROWS)
        self.assertEqual(ss[0].sample_id, "mmbench/1000")
        self.assertTrue(all(s.benchmark == "mmbench" for s in ss))
        self.assertFalse(any("circ_index" in s.extra for s in ss))
        h = hashlib.sha1("".join(s.sample_id + s.question + str(s.label)
                                 for s in ss).encode()).hexdigest()
        # Pinned from the loader as it stood before circular was added
        # (the loader's code was not edited; this guards against drift).
        self.assertEqual(h, SINGLE_PASS_FIXTURE_SHA1)

    def test_single_pass_plan_and_registries_are_unchanged(self) -> None:
        self.assertEqual(BENCHMARK_SPLITS["mmbench"], ("dev",))
        self.assertEqual(N_SAMPLES[("mmbench", "dev")], 4377)
        self.assertEqual(
            len([j for j in PLAN if j.target == "table11_mmbench"]), 9)
        self.assertEqual(DEFAULT_TEMPLATE_IDS["mmbench"], "mmbench_v0_letter")


SINGLE_PASS_FIXTURE_SHA1 = "f0d58abb8eca9ba91880436eadbd35960ff79fdd"


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
class TestPlanJobs(unittest.TestCase):
    def jobs(self):
        return [j for j in PLAN if j.target == "table11_mmbench_circular"]

    def test_three_arms_seed_42_only(self) -> None:
        js = self.jobs()
        self.assertEqual({j.method for j in js}, {"sampling", "fox", "vcd"})
        self.assertEqual(len(js), 3)
        self.assertEqual({j.seed for j in js}, {42})
        self.assertEqual({(j.benchmark, j.split) for j in js},
                         {("mmbench_circular", "dev")})

    def test_ids_unique_and_no_collision_with_single_pass(self) -> None:
        ids = [j.job_id() for j in PLAN]
        self.assertEqual(len(ids), len(set(ids)))
        # A continuation chunk shares its first chunk's file on purpose.
        owners = [j for j in PLAN if not j.is_continuation]
        keys = {j.run_key().filename() for j in owners}
        self.assertEqual(len(keys), len(owners))

    def test_cost_is_a_ceiling_from_the_upper_bound(self) -> None:
        self.assertEqual(N_SAMPLES[("mmbench_circular", "dev")],
                         4 * MMBENCH_DEV_ROWS)
        s = Job("t", "sampling", "mmbench_circular", "dev").est_seconds()
        f = Job("t", "fox", "mmbench_circular", "dev").est_seconds()
        single = Job("t", "sampling", "mmbench", "dev").est_seconds()
        # Four rows per question, each at the per-row cost measured in r33
        # (which is why it is no longer MMBench's estimate).
        per_row = (COST_S[("mmbench_circular", "single")]
                   / COST_S[("mmbench", "single")])
        self.assertAlmostEqual(s / single, 4.0 * per_row)
        self.assertGreater(f, s)

    def test_command_and_run_key(self) -> None:
        j = self.jobs()[0]
        cmd = j.command()
        self.assertIn("--benchmark mmbench_circular --split dev", cmd)
        self.assertIn("--prompt-template-id mmbench_circular_v0_letter", cmd)
        self.assertEqual(j.run_key().variant_fields["max_new_tokens"], 8)

    def test_frozen_session_plan_is_not_touched(self) -> None:
        plan = json.loads((Path(__file__).resolve().parent.parent
                           / "scripts" / "run_plan.json").read_text())
        # r33 froze these (Sampling, Fox, VCD at seed 42). What must stay true
        # is that a frozen job's command names the circular benchmark, so a
        # session's data fetch sees it.
        for job in plan["jobs"].values():
            if "mmbench_circular" in job["id"]:
                self.assertIn("--benchmark mmbench_circular ", job["command"])


# --------------------------------------------------------------------------- #
# the CLI, as a process
# --------------------------------------------------------------------------- #
class TestCircularCli(t11.TestCliEndToEnd):
    # Inherit only the harness; do not re-run the base class's own tests.
    def test_mmbench_circular_dry_run_and_scores(self) -> None:
        d = self.root / "mmbench"
        fetch_mmbench.build(mmbench_rows(), d)
        # questions 1000..1002 have 2, 3, 4 options: 9 rows, 3 whole questions
        path = self._generate("mmbench_circular", "dev", "--mmbench-dir", d,
                              limit=9)
        recs = [json.loads(x) for x in path.read_text().splitlines()]
        self.assertEqual(len(recs), 9)
        self.assertEqual(len({r["sample_id"] for r in recs}), 9)
        for r in recs:
            for k in ("label", "category", "l2_category", "options",
                      "circ_index", "circ_rotation", "circ_n"):
                self.assertIn(k, r, f"record lacks {k!r}")
            self.assertIn(r["label"], r["options"])
        res, rows = evaluate_run_file(path)
        self.assertEqual(res["metrics"]["benchmark"], "mmbench_circular")
        self.assertEqual(res["metrics"]["n_samples"], 3)
        self.assertEqual(len(rows), 3)
        self._evaluate()
        self.assertTrue(list(self.reports.glob("*mmbench_circular*.metrics.json")))

    def test_a_killed_run_cut_mid_question_is_refused_not_scored(self) -> None:
        d = self.root / "mmbench"
        fetch_mmbench.build(mmbench_rows(), d)
        path = self._generate("mmbench_circular", "dev", "--mmbench-dir", d,
                              limit=7)           # cuts question 1002 at 2 of 4
        with self.assertRaisesRegex(ValueError, "partial set"):
            evaluate_run_file(path)


# Do not let unittest collect the base class's own tests a second time here.
for _name in [n for n in dir(t11.TestCliEndToEnd) if n.startswith("test_")]:
    if _name not in TestCircularCli.__dict__:
        setattr(TestCircularCli, _name, None)


if __name__ == "__main__":
    unittest.main()
