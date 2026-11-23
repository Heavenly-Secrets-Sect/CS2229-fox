"""Table 11: MMBench dev and MathVista multiple choice.

Both benchmarks answer with an option LETTER, so the harness is MME's with a
different scorer. What needs pinning is the part a green run could hide:

* sample ids are unique PER QUESTION (MME's `question_id` named an image and
  collided on every item);
* every field the scorer needs rides in the run record, and the scorer RAISES
  when one is missing or an l2-category is unknown, instead of bucketing it
  into a group total that reads as a smaller number;
* an unparseable answer is wrong and COUNTED, an empty group is None and not
  0.0, a short run says it is incomplete;
* the fetchers write the manifest last and assert their row counts;
* the CLI is run as a process, because a flag parsed and dropped is invisible
  to every unit test (r21 and r21.1 each lost a session to exactly that);
* an MMBench/MathVista-only session downloads no COCO.

Real-split facts these fixtures mirror (read from the Hub): MMBench dev has
4377 rows with unique `index`; absent options and hints are the string "nan";
`l2-category` has six values. MathVista testmini has 540 `multi_choice` rows,
whose `answer` is the choice TEXT.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from foxrepro.bench.evaluate import (
    _format_markdown, evaluate_mathvista, evaluate_mmbench, evaluate_run_file,
)
from foxrepro.bench.multichoice import extract_choice
from foxrepro.bench.plan import N_SAMPLES, PLAN, Job
from foxrepro.data import fetch_mathvista, fetch_mmbench
from foxrepro.data.fetch_images import _session_needs
from foxrepro.data.loaders import (
    DEFAULT_TEMPLATE_IDS, MATHVISTA_MANIFEST_NAME, MMBENCH_MANIFEST_NAME,
    build_mmbench_question, load_benchmark, load_mathvista, load_mmbench,
    mathvista_subset_pids, render_prompt, template_benchmark,
)
from foxrepro.storage import (
    BENCHMARK_SPLITS, MATHVISTA_MC_POOL, MATHVISTA_SUBSET_N,
    MATHVISTA_SUBSET_SEED, MAX_NEW_TOKENS_DEFAULT, MMBENCH_DEV_ROWS,
    MMBENCH_PERCEPTION_L2, MMBENCH_REASONING_L2, SELF_CONTAINED_BENCHMARKS,
)

ALL_L2 = MMBENCH_PERCEPTION_L2 + MMBENCH_REASONING_L2
JPEG = b"\xff\xd8fakejpeg"


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def mmbench_rows(n: int = MMBENCH_DEV_ROWS):
    """Dataset-shaped rows: 'nan' for absent options and hints, six l2s."""
    for i in range(n):
        n_opts = 2 + i % 3                       # 2, 3 or 4 options
        row = {
            "index": 1000 + i, "question": f"Question {i}?",
            "hint": "nan" if i % 2 else f"Hint {i}.",
            "A": "alpha", "B": "beta",
            "C": "gamma" if n_opts >= 3 else "nan",
            "D": "delta" if n_opts >= 4 else "nan",
            "answer": "AB"[i % 2], "category": f"cat{i % 20}",
            "l2-category": ALL_L2[i % len(ALL_L2)], "source": "fixture",
            "image": {"bytes": JPEG, "path": None},
        }
        yield row


def mathvista_rows(n_mc: int = MATHVISTA_MC_POOL, n_free: int = 20):
    for i in range(n_mc):
        choices = ["10", "20", "30", "40"][: 2 + i % 3]
        if i == 7:
            choices = ["5", "5", "9"]            # a repeated gold text
        gold = choices[0] if i == 7 else choices[i % len(choices)]
        query = ("Hint: Please answer the question and provide the correct "
                 "option letter, e.g., A, B, C, D, at the end.\n"
                 f"Question: Q{i}\nChoices:\n"
                 + "\n".join(f"({'ABCDEFGH'[k]}) {c}"
                             for k, c in enumerate(choices)))
        yield {"pid": str(i + 1), "question": f"Q{i}", "choices": choices,
               "answer": gold, "question_type": "multi_choice",
               "answer_type": "text", "query": query,
               "decoded_image": {"bytes": JPEG, "path": None}}
    for j in range(n_free):
        yield {"pid": str(10_000 + j), "question": "free", "choices": None,
               "answer": "12", "question_type": "free_form",
               "answer_type": "integer", "query": "free",
               "decoded_image": {"bytes": JPEG, "path": None}}


def mm_rec(sid, gold, out, l2="coarse_perception", cat="image_scene",
           options=None):
    return {"sample_id": f"mmbench/{sid}", "label": gold, "output": out,
            "category": cat, "l2_category": l2,
            "options": options or {"A": "alpha", "B": "beta", "C": "gamma"}}


def mv_rec(sid, gold, out, options=None):
    return {"sample_id": f"mathvista/{sid}", "label": gold, "output": out,
            "options": options or {"A": "10", "B": "20", "C": "30"}}


# --------------------------------------------------------------------------- #
# the extraction rule
# --------------------------------------------------------------------------- #
class TestExtractChoice(unittest.TestCase):
    OPTS = {"A": "a cat", "B": "a dog", "C": "97", "D": "102"}

    def check(self, text, want, rule=None, **kw):
        got, r = extract_choice(text, self.OPTS, **kw)
        self.assertEqual(got, want, (text, r))
        if rule:
            self.assertEqual(r, rule, text)

    def test_the_bare_letter_in_its_usual_spellings(self) -> None:
        for t in ("B", " B\n", "(B)", "b", "B.", '"B"', "**B**"):
            self.check(t, "B", "exact")

    def test_a_leading_letter_with_a_delimiter(self) -> None:
        for t in ("B. a dog", "C) 97", "(D) 102 is right", "A: a cat"):
            self.check(t, t[0] if t[0] != "(" else t[1], "lead")

    def test_the_article_A_is_not_option_A(self) -> None:
        """'A bird is shown.' opens with a space-delimited article."""
        self.check("A bird is shown in the image.", None, "none")

    def test_the_last_answer_phrase_wins(self) -> None:
        self.check("Option A is wrong. The answer is C.", "C", "phrase")
        self.check("the answer is B", "B", "phrase")

    def test_a_phrase_followed_by_a_sentence_is_not_a_letter(self) -> None:
        self.check("the answer is a cat", "A", "text")   # text rule, not 'a'

    def test_tail_rule_is_mathvista_only(self) -> None:
        text = "Since 5 plus 5 gives ten, I pick D"
        self.check(text, None)
        self.check(text, "D", "tail", allow_tail=True)

    def test_option_text_when_unique(self) -> None:
        self.check("It is 97 degrees", "C", "text")
        self.check("a dog", "B", "text")

    def test_ambiguous_text_is_not_guessed_through(self) -> None:
        self.check("a cat or a dog", None, "none")

    def test_a_number_inside_a_longer_number_is_not_a_mention(self) -> None:
        self.check("1970", None, "none")

    def test_a_letter_the_question_does_not_offer_is_invalid(self) -> None:
        got, rule = extract_choice("E", self.OPTS)
        self.assertEqual((got, rule), (None, "invalid_letter"))
        got, rule = extract_choice("C", {"A": "x", "B": "y"})
        self.assertEqual((got, rule), (None, "invalid_letter"))

    def test_empty_is_unparsed(self) -> None:
        self.check("", None, "none")
        self.check("   \n", None, "none")


# --------------------------------------------------------------------------- #
# the scorer
# --------------------------------------------------------------------------- #
class TestMMBenchScore(unittest.TestCase):
    def test_overall_perception_reasoning(self) -> None:
        recs = [
            mm_rec(1, "A", "A", l2="coarse_perception"),
            mm_rec(2, "B", "A", l2="finegrained_perception (instance-level)"),
            mm_rec(3, "A", "A", l2="logic_reasoning"),
            mm_rec(4, "C", "C", l2="relation_reasoning"),
            mm_rec(5, "B", "B", l2="attribute_reasoning"),
            mm_rec(6, "A", "B", l2="finegrained_perception (cross-instance)"),
        ]
        m, rows = evaluate_mmbench(recs)
        t = m["totals"]
        self.assertEqual(t["overall"]["acc"], round(100 * 4 / 6, 4))
        self.assertEqual(t["perception"], {"acc": round(100 * 1 / 3, 4),
                                           "n_right": 1, "n": 3})
        self.assertEqual(t["reasoning"], {"acc": 100.0, "n_right": 3, "n": 3})
        self.assertEqual(len(rows), 6)
        self.assertEqual(set(m["per_l2_category"]),
                         {r["l2_category"] for r in recs})

    def test_the_mapping_is_the_published_definition(self) -> None:
        self.assertEqual(len(MMBENCH_PERCEPTION_L2), 3)
        self.assertEqual(len(MMBENCH_REASONING_L2), 3)
        self.assertFalse(set(MMBENCH_PERCEPTION_L2) & set(MMBENCH_REASONING_L2))

    def test_an_unknown_l2_category_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            evaluate_mmbench([mm_rec(1, "A", "A", l2="vibes")])
        self.assertIn("vibes", str(ctx.exception))

    def test_a_record_without_its_l2_category_raises(self) -> None:
        r = mm_rec(1, "A", "A")
        del r["l2_category"]
        with self.assertRaises(ValueError):
            evaluate_mmbench([r])

    def test_a_record_without_options_raises(self) -> None:
        r = mm_rec(1, "A", "A")
        del r["options"]
        with self.assertRaises(ValueError) as ctx:
            evaluate_mmbench([r])
        self.assertIn("options", str(ctx.exception))

    def test_a_gold_letter_outside_the_options_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_mmbench([mm_rec(1, "D", "A")])

    def test_unparsed_is_wrong_counted_and_not_dropped(self) -> None:
        recs = [mm_rec(1, "A", "A"), mm_rec(2, "A", "I cannot tell"),
                mm_rec(3, "A", "D")]            # D: not offered here
        m, rows = evaluate_mmbench(recs)
        self.assertEqual(m["n_samples"], 3)
        self.assertEqual(m["unparsed"], 2)
        self.assertEqual(m["rule_counts"]["invalid_letter"], 1)
        self.assertEqual(m["totals"]["overall"]["n_right"], 1)
        self.assertAlmostEqual(m["unparsed_fraction"], 2 / 3, places=5)

    def test_an_empty_group_is_none_not_zero(self) -> None:
        m, _ = evaluate_mmbench([mm_rec(1, "A", "A")])
        self.assertIsNone(m["totals"]["reasoning"])
        self.assertIn("relation_reasoning", m["l2_missing"])
        text = _format_markdown(self._doc(m))
        self.assertIn("n/a", text)

    def test_a_short_run_says_it_is_incomplete(self) -> None:
        m, _ = evaluate_mmbench([mm_rec(1, "A", "A")])
        self.assertFalse(m["complete"])
        self.assertEqual(m["n_expected"], MMBENCH_DEV_ROWS)
        self.assertIn("INCOMPLETE", _format_markdown(self._doc(m)))

    def test_a_full_run_is_complete(self) -> None:
        recs = [mm_rec(i, "A", "A", l2=ALL_L2[i % 6])
                for i in range(MMBENCH_DEV_ROWS)]
        m, _ = evaluate_mmbench(recs)
        self.assertTrue(m["complete"])
        self.assertEqual(m["totals"]["overall"]["acc"], 100.0)
        self.assertEqual(m["l2_missing"], [])

    @staticmethod
    def _doc(metrics):
        return {"provenance": {"method": "m", "benchmark": "mmbench",
                               "split": "dev", "seed": 1, "dtype": "fp16",
                               "run_file": "x.jsonl", "n_records": 1,
                               "n_duplicate_sample_ids": 0},
                "metrics": metrics}


class TestMathVistaScore(unittest.TestCase):
    def test_accuracy_and_coverage(self) -> None:
        recs = [mv_rec(1, "A", "A"), mv_rec(2, "B", "A"),
                mv_rec(3, "C", "The answer is C"),
                mv_rec(4, "A", "no idea whatsoever")]
        m, rows = evaluate_mathvista(recs)
        self.assertEqual(m["totals"]["accuracy"]["n_right"], 2)
        self.assertEqual(m["totals"]["accuracy"]["acc"], 50.0)
        self.assertEqual(m["coverage"], 0.75)
        self.assertEqual(m["unparsed"], 1)
        # over parsed answers only: 2 of 3
        self.assertEqual(m["totals"]["accuracy_parsed"]["n"], 3)
        self.assertEqual(len(rows), 4)

    def test_a_repeated_gold_text_accepts_either_letter(self) -> None:
        opts = {"A": "5", "B": "5", "C": "9"}
        m, _ = evaluate_mathvista(
            [mv_rec(1, "A|B", "B", opts), mv_rec(2, "A|B", "A", opts),
             mv_rec(3, "A|B", "C", opts)])
        self.assertEqual(m["totals"]["accuracy"]["n_right"], 2)

    def test_the_trailing_letter_after_a_derivation(self) -> None:
        m, rows = evaluate_mathvista(
            [mv_rec(1, "B", "10 plus 10 is 20, so I pick B")])
        self.assertEqual(rows[0]["pred"], "B")
        self.assertEqual(rows[0]["rule"], "tail")

    def test_missing_options_raise(self) -> None:
        r = mv_rec(1, "A", "A")
        del r["options"]
        with self.assertRaises(ValueError):
            evaluate_mathvista([r])

    def test_empty_run_has_no_accuracy_not_a_zero(self) -> None:
        m, _ = evaluate_mathvista([])
        self.assertIsNone(m["totals"]["accuracy"])
        self.assertIsNone(m["coverage"])
        self.assertFalse(m["complete"])


# --------------------------------------------------------------------------- #
# fetchers and loaders
# --------------------------------------------------------------------------- #
class TestMMBenchFetchAndLoad(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.dest = Path(self._td.name)

    def test_full_layout_round_trips_through_the_loader(self) -> None:
        summary = fetch_mmbench.build(mmbench_rows(), self.dest)
        self.assertEqual(summary["n_rows"], MMBENCH_DEV_ROWS)
        self.assertEqual(set(summary["l2_categories"]), set(ALL_L2))
        samples = load_mmbench(self.dest)
        self.assertEqual(len(samples), MMBENCH_DEV_ROWS)
        ids = [s.sample_id for s in samples]
        self.assertEqual(len(set(ids)), len(ids), "sample ids collide")
        s = samples[0]
        self.assertEqual(s.benchmark, "mmbench")
        self.assertIn(s.label, s.extra["options"])
        for key in ("category", "l2_category", "options", "image_file"):
            self.assertIn(key, s.extra)

    def test_nan_options_and_hints_are_absent_not_the_string_nan(self) -> None:
        fetch_mmbench.build(mmbench_rows(), self.dest)
        samples = {s.sample_id: s for s in load_mmbench(self.dest)}
        two_opt = samples["mmbench/1000"]       # i=0: 2 options, hint present
        self.assertEqual(sorted(two_opt.extra["options"]), ["A", "B"])
        self.assertNotIn("nan", two_opt.question)
        self.assertTrue(two_opt.question.startswith("Hint 0."))
        no_hint = samples["mmbench/1001"]       # i=1: hint is 'nan'
        self.assertTrue(no_hint.question.startswith("Question 1?"))
        self.assertNotIn("nan", no_hint.question)

    def test_prompt_follows_the_llava_convention(self) -> None:
        q = build_mmbench_question("What?", {"A": "x", "B": "y"}, "Hint.")
        self.assertEqual(q, "Hint.\nWhat?\nA. x\nB. y")
        prompt = render_prompt("mmbench_v0_letter", q)
        self.assertEqual(
            prompt, "Hint.\nWhat?\nA. x\nB. y\nAnswer with the option's "
                    "letter from the given choices directly.")

    def test_a_short_download_is_refused_and_leaves_no_manifest(self) -> None:
        with self.assertRaises(ValueError):
            fetch_mmbench.build(mmbench_rows(100), self.dest)
        self.assertFalse((self.dest / MMBENCH_MANIFEST_NAME).exists())

    def test_a_failure_mid_extract_leaves_no_manifest(self) -> None:
        rows = list(mmbench_rows(50))
        rows[30]["answer"] = "D"                  # D is absent on that row
        rows[30]["C"] = rows[30]["D"] = "nan"
        with self.assertRaises(ValueError):
            fetch_mmbench.build(rows, self.dest, limit=None)
        self.assertEqual(list(self.dest.glob("*.jsonl")), [])

    def test_duplicate_index_is_refused(self) -> None:
        rows = list(mmbench_rows(5))
        rows[3]["index"] = rows[1]["index"]
        with self.assertRaises(ValueError):
            fetch_mmbench.build(rows, self.dest, limit=5)

    def test_unknown_l2_is_refused_by_the_fetcher(self) -> None:
        rows = list(mmbench_rows(3))
        rows[1]["l2-category"] = "vibes"
        with self.assertRaises(ValueError):
            fetch_mmbench.build(rows, self.dest, limit=3)

    def test_a_limited_fetch_is_never_loadable(self) -> None:
        fetch_mmbench.build(mmbench_rows(), self.dest, limit=10)
        self.assertTrue(
            (self.dest / f"partial_10_{MMBENCH_MANIFEST_NAME}").is_file())
        with self.assertRaises(FileNotFoundError):
            load_mmbench(self.dest)

    def test_loader_refuses_a_truncated_manifest(self) -> None:
        fetch_mmbench.build(mmbench_rows(), self.dest)
        path = self.dest / MMBENCH_MANIFEST_NAME
        lines = path.read_text().splitlines()[:-1]
        path.write_text("\n".join(lines) + "\n")
        with self.assertRaises(ValueError):
            load_mmbench(self.dest)

    def test_loader_refuses_a_missing_image(self) -> None:
        fetch_mmbench.build(mmbench_rows(), self.dest)
        next((self.dest / "images").iterdir()).unlink()
        with self.assertRaises(FileNotFoundError):
            load_mmbench(self.dest)

    def test_load_benchmark_dispatches_and_demands_the_dir(self) -> None:
        with self.assertRaises(ValueError):
            load_benchmark("mmbench", "dev", Path("."), Path("."), Path("."),
                           "mmbench_v0_letter")
        fetch_mmbench.build(mmbench_rows(), self.dest)
        got = load_benchmark("mmbench", "dev", Path("."), Path("."),
                             Path("."), "mmbench_v0_letter",
                             mmbench_dir=self.dest)
        self.assertEqual(len(got), MMBENCH_DEV_ROWS)


class TestMathVistaFetchAndLoad(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.dest = Path(self._td.name)

    def test_filters_to_multiple_choice_and_draws_500_of_540(self) -> None:
        summary = fetch_mathvista.build(mathvista_rows(), self.dest)
        self.assertEqual(summary["n_rows"], MATHVISTA_SUBSET_N)
        self.assertEqual(summary["pool_size"], MATHVISTA_MC_POOL)
        samples = load_mathvista(self.dest)
        self.assertEqual(len(samples), 500)
        self.assertTrue(all(int(s.extra["pid"]) < 10_000 for s in samples),
                        "a free-form question leaked into the MC subset")
        ids = [s.sample_id for s in samples]
        self.assertEqual(len(set(ids)), len(ids))

    def test_the_draw_depends_on_the_seed_alone_not_on_row_order(self) -> None:
        pids = [str(i) for i in range(1, 541)]
        a = mathvista_subset_pids(pids, 500, MATHVISTA_SUBSET_SEED)
        b = mathvista_subset_pids(list(reversed(pids)), 500,
                                  MATHVISTA_SUBSET_SEED)
        c = mathvista_subset_pids(pids, 500, MATHVISTA_SUBSET_SEED + 1)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(set(a)), 500)

    def test_the_seed_is_recorded_in_the_manifest(self) -> None:
        fetch_mathvista.build(mathvista_rows(), self.dest)
        rec = json.loads(
            (self.dest / MATHVISTA_MANIFEST_NAME).read_text().splitlines()[0])
        self.assertEqual(rec["subset_seed"], MATHVISTA_SUBSET_SEED)
        self.assertEqual(rec["pool_size"], MATHVISTA_MC_POOL)

    def test_a_manifest_from_another_seed_is_refused(self) -> None:
        fetch_mathvista.build(mathvista_rows(), self.dest)
        path = self.dest / MATHVISTA_MANIFEST_NAME
        recs = [json.loads(x) for x in path.read_text().splitlines()]
        for r in recs:
            r["subset_seed"] = 1
        path.write_text("".join(json.dumps(r) + "\n" for r in recs))
        with self.assertRaises(ValueError):
            load_mathvista(self.dest)

    def test_a_pool_of_the_wrong_size_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            fetch_mathvista.build(mathvista_rows(n_mc=539), self.dest)
        self.assertEqual(list(self.dest.glob("*.jsonl")), [])

    def test_gold_letters_come_from_the_choice_text(self) -> None:
        fetch_mathvista.build(mathvista_rows(), self.dest)
        by_pid = {s.extra["pid"]: s for s in load_mathvista(self.dest)}
        if "8" in by_pid:                       # the repeated-gold question
            self.assertEqual(by_pid["8"].extra["answer_letters"], ["A", "B"])
            self.assertEqual(by_pid["8"].label, "A|B")
        for s in by_pid.values():
            for letter in s.extra["answer_letters"]:
                self.assertIn(letter, s.extra["options"])

    def test_the_prompt_is_the_datasets_own_query(self) -> None:
        fetch_mathvista.build(mathvista_rows(), self.dest)
        s = load_mathvista(self.dest)[0]
        self.assertTrue(s.question.startswith("Hint: Please answer"))
        self.assertEqual(render_prompt("mathvista_v0_query", s.question),
                         s.question)

    def test_partial_fetch_is_not_loadable(self) -> None:
        fetch_mathvista.build(mathvista_rows(), self.dest, limit=5)
        with self.assertRaises(FileNotFoundError):
            load_mathvista(self.dest)


# --------------------------------------------------------------------------- #
# registries and plan
# --------------------------------------------------------------------------- #
class TestRegistries(unittest.TestCase):
    def test_splits_and_the_withheld_test_split(self) -> None:
        self.assertEqual(BENCHMARK_SPLITS["mmbench"], ("dev",))
        self.assertNotIn("test", BENCHMARK_SPLITS["mmbench"],
                         "the Hub test split has no answers")
        self.assertEqual(BENCHMARK_SPLITS["mathvista"], ("testmini_mc500",))

    def test_generation_lengths_are_small_and_from_the_registry(self) -> None:
        self.assertEqual(MAX_NEW_TOKENS_DEFAULT["mmbench"], 8)
        self.assertEqual(MAX_NEW_TOKENS_DEFAULT["mathvista"], 64)

    def test_templates(self) -> None:
        self.assertEqual(DEFAULT_TEMPLATE_IDS["mmbench"], "mmbench_v0_letter")
        self.assertEqual(DEFAULT_TEMPLATE_IDS["mathvista"], "mathvista_v0_query")
        self.assertEqual(template_benchmark("mmbench_v0_letter"), "mmbench")
        self.assertEqual(template_benchmark("mathvista_v0_query"), "mathvista")

    def test_self_contained_benchmarks(self) -> None:
        self.assertEqual(set(SELF_CONTAINED_BENCHMARKS),
                         {"mme", "mmbench", "mathvista"})

    def test_sample_counts(self) -> None:
        self.assertEqual(N_SAMPLES[("mmbench", "dev")], 4377)
        self.assertEqual(N_SAMPLES[("mathvista", "testmini_mc500")], 500)


class TestPlanJobs(unittest.TestCase):
    def jobs(self, target):
        return [j for j in PLAN if j.target == target]

    def test_mmbench_arms_and_seeds(self) -> None:
        js = self.jobs("table11_mmbench")
        self.assertEqual(len(js), 9)
        self.assertEqual({j.method for j in js}, {"sampling", "vcd", "fox"})
        self.assertEqual({j.seed for j in js}, {42, 43, 44})
        self.assertEqual({(j.benchmark, j.split) for j in js},
                         {("mmbench", "dev")})

    def test_mathvista_arms_and_seeds(self) -> None:
        js = self.jobs("table11_mathvista")
        self.assertEqual(len(js), 6)
        self.assertEqual({j.method for j in js}, {"sampling", "fox"})
        self.assertEqual({j.seed for j in js}, {42, 43, 44})

    def test_job_ids_are_unique(self) -> None:
        ids = [j.job_id() for j in PLAN]
        self.assertEqual(len(ids), len(set(ids)))

    def test_cost_is_priced_not_zero_and_is_estimate_shaped(self) -> None:
        for target in ("table11_mmbench", "table11_mathvista"):
            for j in self.jobs(target):
                self.assertGreater(j.est_seconds(), 0)
        sampling = Job("t", "sampling", "mmbench", "dev").est_seconds()
        fox = Job("t", "fox", "mmbench", "dev").est_seconds()
        self.assertGreater(fox, sampling, "dual branch must cost more")

    def test_commands_carry_the_benchmark_and_default_length(self) -> None:
        j = Job("table11_mmbench", "fox", "mmbench", "dev")
        cmd = j.command()
        self.assertIn("--benchmark mmbench --split dev", cmd)
        self.assertIn("--prompt-template-id mmbench_v0_letter", cmd)
        self.assertEqual(j.run_key().variant_fields["max_new_tokens"], 8)

    def test_a_frozen_round_may_carry_them_but_only_as_these_benchmarks(self) -> None:
        """Once a round freezes them (r25 does), they must be exactly the
        table11 jobs, so a session's data fetch (which reads the command)
        sees them."""
        import json
        plan = (Path(__file__).resolve().parent.parent / "scripts"
                / "run_plan.json")
        if not plan.is_file():
            self.skipTest("no frozen plan in this checkout")
        doc = json.loads(plan.read_text())
        for job in doc["jobs"].values():
            if job["id"].startswith(("table11_mmbench", "table11_mathvista")):
                self.assertTrue(
                    "--benchmark mmbench " in job["command"]
                    or "--benchmark mathvista " in job["command"],
                    job["id"])


class TestSessionsFetchNoCoco(unittest.TestCase):
    def _session(self, commands):
        plan = {"jobs": {}, "sessions": [{"session": 1, "jobs": []}]}
        for i, (bench, split) in enumerate(commands):
            jid = f"t:fox:{bench}:{split}"
            plan["jobs"][str(i)] = {"id": jid, "command": f"--benchmark {bench}"}
            plan["sessions"][0]["jobs"].append(str(i))
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "plan.json"
            p.write_text(json.dumps(plan))
            return _session_needs(1, p)

    def test_mmbench_and_mathvista_only_session_needs_no_coco(self) -> None:
        what, _ = self._session([("mmbench", "dev"),
                                 ("mathvista", "testmini_mc500")])
        self.assertEqual(what, "none")

    def test_a_mixed_session_still_fetches_what_pope_needs(self) -> None:
        what, _ = self._session([("mmbench", "dev"), ("pope", "random")])
        self.assertEqual(what, "pope")

    def test_the_runner_fetches_each_benchmark_only_for_its_sessions(self) -> None:
        sh = (Path(__file__).resolve().parent.parent
              / "run_kaggle_fox.sh").read_text()
        self.assertIn("foxrepro.data.fetch_mmbench", sh)
        self.assertIn("foxrepro.data.fetch_mathvista", sh)
        self.assertIn('needs_bench mmbench', sh)
        self.assertIn('needs_bench mathvista', sh)


# --------------------------------------------------------------------------- #
# the CLI, as a process
# --------------------------------------------------------------------------- #
class TestCliEndToEnd(unittest.TestCase):
    """`generate --dry-run` then `evaluate`, as subprocesses.

    Every benchmark needs a line here: a flag parsed and dropped (r21,
    r21.1) is invisible to the unit tests above.
    """

    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.root = Path(self._td.name)
        self.env = dict(os.environ)
        src = str(Path(__file__).resolve().parent.parent / "src")
        self.env["PYTHONPATH"] = src + os.pathsep + self.env.get("PYTHONPATH", "")
        self.env.update({"FOX_ENV": "local",
                         "FOX_PERSIST_ROOT": str(self.root / "persist"),
                         "FOX_LOCAL_ROOT": str(self.root / "local")})
        self.runs = self.root / "runs"
        self.reports = self.root / "reports"

    def _generate(self, benchmark, split, flag, directory, limit=8):
        cmd = [sys.executable, "-m", "foxrepro.bench.generate", "--dry-run",
               "--benchmark", benchmark, "--split", split, "--seed", "42",
               "--dtype", "fp16", flag, str(directory), "--limit", str(limit),
               "--runs-dir", str(self.runs), "--progress-every", "0",
               "--no-fsync"]
        proc = subprocess.run(cmd, env=self.env, capture_output=True,
                              text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
        runs = sorted(self.runs.glob(f"*b-{benchmark}*.jsonl"))
        self.assertEqual(len(runs), 1, f"no {benchmark} run file written")
        return runs[0]

    def _evaluate(self):
        proc = subprocess.run(
            [sys.executable, "-m", "foxrepro.bench.evaluate",
             "--runs-dir", str(self.runs), "--out-dir", str(self.reports),
             "--format", "both"],
            env=self.env, capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
        return proc.stdout

    def test_mmbench_runs_through_the_cli_and_scores(self) -> None:
        d = self.root / "mmbench"
        fetch_mmbench.build(mmbench_rows(), d)
        path = self._generate("mmbench", "dev", "--mmbench-dir", d)
        recs = [json.loads(x) for x in path.read_text().splitlines()]
        self.assertEqual(len(recs), 8)
        for r in recs:      # everything the scorer demands is in the record
            for k in ("label", "category", "l2_category", "options"):
                self.assertIn(k, r, f"record lacks {k!r}")
            self.assertIn(r["label"], r["options"])
        self.assertEqual({r["max_new_tokens"] for r in recs}, {8})
        self.assertEqual({r["prompt_template_id"] for r in recs},
                         {"mmbench_v0_letter"})
        self.assertTrue(all(r["prompt"].endswith("directly.") for r in recs))
        out = self._evaluate()
        doc = json.loads(next(self.reports.glob("*mmbench*.metrics.json"))
                         .read_text())
        m = doc["metrics"]
        self.assertEqual(m["benchmark"], "mmbench")
        self.assertEqual(m["n_samples"], 8)
        self.assertFalse(m["complete"])
        self.assertEqual(m["unparsed"], 0, "the echo letters must all parse")
        self.assertIn("INCOMPLETE", out)
        # The report CLI must survive an MMBench metrics file in its
        # directory: its resolution budget used to treat every non-POPE
        # benchmark as CHAIR and died on a key MMBench does not have.
        proc = subprocess.run(
            [sys.executable, "-m", "foxrepro.bench.report",
             "--reports-dir", str(self.reports),
             "--out", str(self.root / "report.md"), "--baseline", "echo"],
            env=self.env, capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])

    def test_mathvista_runs_through_the_cli_and_scores(self) -> None:
        d = self.root / "mathvista"
        fetch_mathvista.build(mathvista_rows(), d)
        path = self._generate("mathvista", "testmini_mc500",
                              "--mathvista-dir", d)
        recs = [json.loads(x) for x in path.read_text().splitlines()]
        self.assertEqual(len(recs), 8)
        for r in recs:
            for k in ("label", "options", "answer_letters"):
                self.assertIn(k, r, f"record lacks {k!r}")
        self.assertEqual({r["max_new_tokens"] for r in recs}, {64})
        self._evaluate()
        doc = json.loads(next(self.reports.glob("*mathvista*.metrics.json"))
                         .read_text())
        self.assertEqual(doc["metrics"]["coverage"], 1.0)
        self.assertFalse(doc["metrics"]["complete"])

    def test_a_record_stripped_of_options_is_refused_by_the_scorer(self) -> None:
        d = self.root / "mmbench"
        fetch_mmbench.build(mmbench_rows(), d)
        path = self._generate("mmbench", "dev", "--mmbench-dir", d, limit=3)
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        for r in lines:
            del r["options"]
        path.write_text("".join(json.dumps(r) + "\n" for r in lines))
        with self.assertRaises(ValueError):
            evaluate_run_file(path)


if __name__ == "__main__":
    unittest.main()
