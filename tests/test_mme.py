"""MME: yes/no like POPE, scored like nothing else here.

MME reuses POPE's shape almost entirely -- one image, one yes/no question,
one generated token, the same answer parser -- which is why its harness is
short. What it does NOT reuse is the metric, and that is the part worth
testing:

    subtask score = (acc + acc_plus) * 100

``acc`` is per QUESTION, ``acc_plus`` is per IMAGE and counts only images
where BOTH questions were right. A model that answers "yes" to everything
scores acc 0.50 and acc_plus 0.00; one that is right half the time at
random scores 0.50 and 0.25. POPE accuracy cannot separate those and the
MME score exists to.

Five things need pinning:

* the acc_plus term, because it is the only part a POPE-shaped harness
  would get wrong by doing nothing;
* the totals are SUMS over ten and four subtasks, so they run to 1000 and
  400, not 100 -- and a missing subtask therefore reads as a low score
  rather than as an error, which is why the fetcher checks the row count
  and the loader refuses a partial manifest;
* an unparsable answer is WRONG, not dropped. Dropping would let a model
  raise its score by refusing to answer;
* MME must not inherit CHAIR's 512-token default. The registry is the one
  place that decides, after a hard-coded `1 if pope else 512` silently gave
  MME 512;
* the subtask is a FIELD, never a split. The score sums over subtasks, so
  splitting them into jobs would make the total depend on which jobs ran.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from foxrepro.bench.evaluate import evaluate_mme  # noqa: E402
from foxrepro.bench.plan import N_SAMPLES, Job  # noqa: E402
from foxrepro.data.loaders import (  # noqa: E402
    MME_MANIFEST_NAME, DEFAULT_TEMPLATE_IDS, load_mme, render_prompt,
    template_benchmark,
)
from foxrepro.storage import (  # noqa: E402
    BENCHMARK_SPLITS, MAX_NEW_TOKENS_DEFAULT, MME_COGNITION, MME_PERCEPTION,
)


def rec(qid, img, label, cat="existence", out=None):
    return {"sample_id": f"mme/{cat}/{qid}", "label": label,
            "output": label.capitalize() if out is None else out,
            "category": cat, "image_file": img}


class TestTheScore(unittest.TestCase):
    def test_acc_plus_counts_images_not_questions(self) -> None:
        """The one term a POPE-shaped harness gets wrong by doing nothing."""
        # image a: both right. image b: one right, one wrong.
        recs = [rec("q1", "a", "yes"), rec("q2", "a", "no"),
                rec("q3", "b", "yes"), rec("q4", "b", "no", out="Yes")]
        m, _ = evaluate_mme(recs)
        cat = m["per_category"]["existence"]
        self.assertEqual(cat["acc"], 75.0)        # 3 of 4 questions
        self.assertEqual(cat["acc_plus"], 50.0)   # 1 of 2 images
        self.assertEqual(cat["score"], 125.0)

    def test_always_yes_scores_half_accuracy_and_no_acc_plus(self) -> None:
        """The degenerate model MME's second term exists to catch."""
        recs = [rec("q1", "a", "yes", out="Yes"), rec("q2", "a", "no", out="Yes"),
                rec("q3", "b", "yes", out="Yes"), rec("q4", "b", "no", out="Yes")]
        m, _ = evaluate_mme(recs)
        cat = m["per_category"]["existence"]
        self.assertEqual(cat["acc"], 50.0)
        self.assertEqual(cat["acc_plus"], 0.0, "always-yes earned acc_plus")
        self.assertEqual(cat["score"], 50.0)

    def test_a_perfect_run_scores_200_per_subtask(self) -> None:
        recs = [rec("q1", "a", "yes"), rec("q2", "a", "no")]
        m, _ = evaluate_mme(recs)
        self.assertEqual(m["per_category"]["existence"]["score"], 200.0)

    def test_totals_sum_over_subtasks_not_average(self) -> None:
        """Perception runs to 1000 and Cognition to 400, not to 100 each."""
        recs = []
        for i, cat in enumerate(MME_PERCEPTION + MME_COGNITION):
            recs += [rec(f"p{i}", f"img{i}", "yes", cat=cat),
                     rec(f"n{i}", f"img{i}", "no", cat=cat)]
        m, _ = evaluate_mme(recs)
        self.assertEqual(m["totals"]["perception"], 200.0 * len(MME_PERCEPTION))
        self.assertEqual(m["totals"]["cognition"], 200.0 * len(MME_COGNITION))
        self.assertEqual(m["totals"]["overall"], 2800.0)
        self.assertEqual(m["subtasks_missing"], [])

    def test_a_missing_subtask_is_reported_not_silently_dropped(self) -> None:
        """Because a short run reads as a low score, never as an error."""
        m, _ = evaluate_mme([rec("q1", "a", "yes"), rec("q2", "a", "no")])
        self.assertEqual(len(m["subtasks_missing"]), 13)
        self.assertIn("code_reasoning", m["subtasks_missing"])

    def test_an_unparsable_answer_is_wrong_not_dropped(self) -> None:
        """Otherwise refusing to answer raises the score.

        Only the `strict` parser can produce an unparsed answer: the
        `released` one defaults anything that is not clearly "no" to "yes",
        which is the released behaviour and the default here so MME and POPE
        parse identically. Tested under `strict` because that is the only
        parser where the branch is reachable at all.
        """
        recs = [rec("q1", "a", "yes", out="I cannot tell"),
                rec("q2", "a", "no")]
        m, _ = evaluate_mme(recs, pope_parser="strict")
        self.assertEqual(m["unparsed"], 1)
        self.assertEqual(m["n_samples"], 2, "an unparsed answer was dropped")
        self.assertEqual(m["per_category"]["existence"]["acc"], 50.0)

    def test_the_released_parser_never_reports_unparsed(self) -> None:
        """Stated so the zero is read as the parser, not as a clean run."""
        m, _ = evaluate_mme([rec("q1", "a", "yes", out="I cannot tell"),
                             rec("q2", "a", "no")])
        self.assertEqual(m["unparsed"], 0)
        self.assertEqual(m["per_category"]["existence"]["acc"], 100.0)

    def test_the_category_is_derived_from_sample_id_when_absent(self) -> None:
        """The first MME run wrote no category and scored 0.0 / 0.0.

        Every subtask bucketed into "?", so Perception and Cognition summed
        to zero -- a number that looks like a measurement. generate now
        persists the field, and the scorer derives it from sample_id, which
        keeps those runs scoreable without spending 5.4 GPU-hours again.
        """
        recs = [{"sample_id": "mme/OCR/0001.jpg#yes", "label": "yes",
                 "output": "Yes"},
                {"sample_id": "mme/OCR/0001.jpg#no", "label": "no",
                 "output": "No"}]
        m, _ = evaluate_mme(recs)
        self.assertIn("OCR", m["per_category"])
        self.assertEqual(m["per_category"]["OCR"]["acc_plus"], 100.0,
                         "the two rows were not recognised as one image")
        self.assertEqual(m["totals"]["perception"], 200.0)

    def test_an_explicit_category_wins_over_the_derived_one(self) -> None:
        m, _ = evaluate_mme([{"sample_id": "mme/OCR/0001.jpg#yes",
                              "category": "color", "label": "yes",
                              "output": "Yes"}])
        self.assertIn("color", m["per_category"])
        self.assertNotIn("OCR", m["per_category"])

    def test_a_record_with_no_recoverable_category_is_refused(self) -> None:
        """Rather than summing into neither total and reporting zero."""
        with self.assertRaises(ValueError) as ctx:
            evaluate_mme([{"sample_id": "something-else", "label": "yes",
                           "output": "Yes"}])
        self.assertIn("category", str(ctx.exception))

    def test_a_non_binary_label_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_mme([{"sample_id": "x", "label": "maybe", "output": "Yes"}])


class TestTheLoader(unittest.TestCase):
    def setUp(self) -> None:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.dir = Path(td.name)
        (self.dir / "images").mkdir()
        for n in ("a.jpg", "b.jpg"):
            (self.dir / "images" / n).write_bytes(b"\xff\xd8fake")

    def write(self, recs, name=MME_MANIFEST_NAME):
        (self.dir / name).write_text(
            "".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")

    def base(self):
        """One row per image, mirroring the real manifest's fields."""
        return [
            {"question_id": "existence/a.png", "item_id": "existence/a.png#yes",
             "image": "images/a.jpg", "answer": "Yes",
             "question": "Is there a dog? Please answer yes or no.",
             "category": "existence"},
            {"question_id": "code_reasoning/b.png",
             "item_id": "code_reasoning/b.png#no",
             "image": "images/b.jpg", "answer": "No",
             "question": "Is there a cat? Please answer yes or no.",
             "category": "code_reasoning"},
        ]

    def pair(self):
        """The real shape: ONE image, TWO questions, one Yes and one No.

        This is what killed r21's only session. MME's `question_id` is the
        IMAGE path -- "code_reasoning/0020.png" -- and every image carries
        exactly two rows. Verified over all four parquet shards: 1187
        images, 2374 rows, 2 per image, one of each answer. Code that treats
        question_id as unique dies on the second row of the split.
        """
        return [
            {"question_id": "code_reasoning/0020.png",
             "item_id": "code_reasoning/0020.png#yes",
             "image": "images/a.jpg", "answer": "Yes",
             "question": "Is a python code shown in the picture? "
                         "Please answer yes or no",
             "category": "code_reasoning"},
            {"question_id": "code_reasoning/0020.png",
             "item_id": "code_reasoning/0020.png#no",
             "image": "images/a.jpg", "answer": "No",
             "question": "Is a c++ code shown in the picture? "
                         "Please answer yes or no.",
             "category": "code_reasoning"},
        ]

    def test_two_questions_on_one_image_do_not_collide(self) -> None:
        """The exact failure that cost r21 a session, one layer down."""
        self.write(self.pair())
        got = load_mme(self.dir)
        self.assertEqual(len(got), 2, "an image's second question was lost")
        self.assertEqual(len({s.sample_id for s in got}), 2)
        self.assertEqual({s.label for s in got}, {"yes", "no"})
        # Same image, so the same image_id -- which is what acc_plus groups
        # on, and the reason the two must NOT be forced apart there.
        self.assertEqual(len({s.image_id for s in got}), 1)

    def test_a_manifest_without_item_id_is_refused(self) -> None:
        """An old manifest predates the fix; it must not load as half a split."""
        recs = [dict(r) for r in self.pair()]
        for r in recs:
            r.pop("item_id")
        self.write(recs)
        with self.assertRaises(ValueError) as ctx:
            load_mme(self.dir)
        self.assertIn("item_id", str(ctx.exception))

    def test_category_is_carried_on_the_sample_not_as_a_split(self) -> None:
        """The score sums over subtasks; splitting them would break totals."""
        self.write(self.base())
        got = load_mme(self.dir)
        self.assertEqual({s.split for s in got}, {"test"})
        self.assertEqual({s.extra["category"] for s in got},
                         {"existence", "code_reasoning"})
        self.assertEqual(BENCHMARK_SPLITS["mme"], ("test",))

    def test_labels_are_normalised_and_non_binary_refused(self) -> None:
        self.write(self.base())
        # Sorted by sample_id, so code_reasoning precedes existence.
        got = {s.extra["question_id"]: s.label for s in load_mme(self.dir)}
        self.assertEqual(got, {"existence/a.png": "yes",
                               "code_reasoning/b.png": "no"})
        bad = self.base()
        bad[0]["answer"] = "probably"
        self.write(bad)
        with self.assertRaises(ValueError):
            load_mme(self.dir)

    def test_a_missing_manifest_names_the_fetcher(self) -> None:
        with self.assertRaises(FileNotFoundError) as ctx:
            load_mme(self.dir)
        self.assertIn("fetch_mme", str(ctx.exception))

    def test_a_missing_image_is_refused_when_required(self) -> None:
        recs = self.base()
        recs[0]["image"] = "images/gone.jpg"
        self.write(recs)
        with self.assertRaises(FileNotFoundError):
            load_mme(self.dir)
        # require_images=False is for CPU-side inspection without the
        # images, so the sample is KEPT with a path that does not exist.
        relaxed = load_mme(self.dir, require_images=False)
        self.assertEqual(len(relaxed), 2)
        self.assertFalse(relaxed[0].image_path.is_file()
                         and relaxed[1].image_path.is_file())

    def test_duplicate_sample_ids_are_refused(self) -> None:
        recs = self.base()
        recs[1] = dict(recs[0])
        self.write(recs)
        with self.assertRaises(ValueError):
            load_mme(self.dir)

    def test_image_id_is_stable_across_loads(self) -> None:
        """MME images are not COCO; the id must still be a stable int."""
        self.write(self.base())
        a = {s.sample_id: s.image_id for s in load_mme(self.dir)}
        b = {s.sample_id: s.image_id for s in load_mme(self.dir)}
        self.assertEqual(a, b)
        self.assertTrue(all(isinstance(v, int) for v in a.values()))


class TestThePlanRegistration(unittest.TestCase):
    def test_mme_generates_one_token_not_512(self) -> None:
        """A hard-coded `1 if pope else 512` silently gave MME 512."""
        self.assertEqual(MAX_NEW_TOKENS_DEFAULT["mme"], 1)
        self.assertEqual(Job("t", "fox", "mme", "test").n_new_tokens(), 1)

    def test_the_whole_split_is_one_job_not_fourteen(self) -> None:
        self.assertEqual(N_SAMPLES[("mme", "test")], 2374)

    def test_the_default_template_is_raw(self) -> None:
        """MME ships its instruction inside the question text."""
        self.assertEqual(DEFAULT_TEMPLATE_IDS["mme"], "mme_v0_raw")
        self.assertEqual(template_benchmark("mme_v0_raw"), "mme")
        q = "Is there a dog? Please answer yes or no."
        self.assertEqual(render_prompt("mme_v0_raw", q), q)

    def test_a_job_builds_a_runnable_command(self) -> None:
        cmd = Job("t", "fox", "mme", "test").command()
        self.assertIn("--benchmark mme", cmd)
        self.assertIn("--prompt-template-id mme_v0_raw", cmd)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class TestTheMarkdownFormatterHandlesMME(unittest.TestCase):
    """r22 lost twenty of twenty-one scores to a missing formatter branch.

    `_format_markdown` tested `benchmark == "pope"` and treated everything
    else as CHAIR. MME metrics have no `chair_s_ci`, so the first MME run
    scored, wrote its metrics file, and then raised from the formatter --
    which is called AFTER the write and outside the loop's try/except. The
    exception left `_main` entirely, so the remaining twenty runs were never
    scored at all.

    The shape worth remembering is not the KeyError. It is that the failure
    landed three frames below the real mistake: `else` quietly meant
    "chair", so adding a benchmark could not produce a message naming it.
    An unknown benchmark now raises by name.
    """

    def _metrics(self, benchmark: str) -> dict:
        return {
            "provenance": {
                "method": "fox", "benchmark": benchmark, "split": "test",
                "seed": 42, "dtype": "fp16", "run_file": "x.jsonl",
                "n_records": 2374, "n_duplicate_sample_ids": 0,
            },
            "metrics": {
                "benchmark": benchmark,
                "n_samples": 2374, "unparsed": 0,
                "subtasks_missing": [],
                "totals": {"perception": 1376.9, "cognition": 297.9,
                           "overall": 1674.7, "accuracy": 0.7384},
                "accuracy_ci": {"point": 0.7384, "ci_lo": 0.7204,
                                "ci_hi": 0.7557, "successes": 1753,
                                "n": 2374},
            },
        }

    def test_an_mme_result_formats(self) -> None:
        from foxrepro.bench.evaluate import _format_markdown
        out = _format_markdown(self._metrics("mme"))
        self.assertIn("perception", out)
        self.assertIn("1376.9", out)
        self.assertIn("cognition", out)
        self.assertIn("2800", out)

    def test_a_missing_subtask_is_printed_next_to_the_total(self) -> None:
        """A subtask that produced no rows is a silent zero in a SUM."""
        from foxrepro.bench.evaluate import _format_markdown
        d = self._metrics("mme")
        d["metrics"]["subtasks_missing"] = ["code_reasoning"]
        out = _format_markdown(d)
        self.assertIn("subtasks missing", out)
        self.assertIn("code_reasoning", out)

    def test_an_unknown_benchmark_raises_by_name(self) -> None:
        from foxrepro.bench.evaluate import _format_markdown
        d = self._metrics("mmbench")
        with self.assertRaises(ValueError) as ctx:
            _format_markdown(d)
        self.assertIn("mmbench", str(ctx.exception))

    def test_chair_is_still_formatted(self) -> None:
        """The branch that used to be `else` keeps working."""
        from foxrepro.bench.evaluate import _format_markdown
        d = self._metrics("chair")
        d["metrics"] = {
            "benchmark": "chair",
            "chair_s_ci": {"point": 0.21, "ci_lo": 0.18, "ci_hi": 0.25},
            "chair_i_ci_cluster_bootstrap": {"point": 0.06, "ci_lo": 0.05,
                                             "ci_hi": 0.08},
            "totals": {"chair_s_numerator": 105, "chair_s_denominator": 500,
                       "chair_i_numerator": 190, "chair_i_denominator": 3100,
                       "recall": 0.72, "precision": 0.81, "f1": 0.76,
                       "mean_caption_words": 92.4},
        }
        out = _format_markdown(d)
        self.assertIn("CHAIR_S", out)
        self.assertIn("CHAIR_I", out)
