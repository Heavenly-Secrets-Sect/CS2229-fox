"""GPT-judge substitute: prompt fidelity, parsing, budget, summary. Fake client
only; no test makes a network call."""

from __future__ import annotations

import importlib.util
import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from foxrepro.bench import gpt_judge as gj

REPO = Path(__file__).resolve().parent.parent
PAPER = Path("/private/tmp/paper.txt")


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "gpt_judge_script", REPO / "scripts" / "gpt_judge.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


class FakeClient:
    """Judge that gives Assistant 1 a 5 and Assistant 2 a 8 unless told."""

    def __init__(self, replies=None, fatal_at=None):
        self.calls = []
        self.replies = replies
        self.fatal_at = fatal_at

    def complete(self, image_path, prompt):
        n = len(self.calls)
        self.calls.append((image_path, prompt))
        if self.fatal_at is not None and n == self.fatal_at:
            raise gj.FatalApiError("RateLimitError: quota")
        text = (self.replies[n] if self.replies else
                "Accuracy: 5 8\nReason: x\nDetailedness: 6 7\nReason: y")
        return gj.JudgeReply(text, "stop", {"prompt_tokens": 10,
                                            "completion_tokens": 5},
                             {"max_tokens": 512, "temperature": 0.2})


class TestPrompt(unittest.TestCase):
    def test_table5_wording(self):
        p = gj.build_prompt("AAA", "BBB")
        for frag in (
            "You are required to score the performance of two AI assistants "
            "in describing a given image.",
            "Note that hallucinated content does NOT count as valid details "
            "– only accurate information contributes to this score.",
            "ensuring that the order in which the responses were presented "
            "does not affect your judgment.",
            "[Assistant 1]\nAAA\n[End of Assistant 1]\n"
            "[Assistant 2]\nBBB\n[End of Assistant 2]\n"
            "Output format:\nAccuracy: <score 1> <score 2>\n"
            "Reason: <your explanation>\n"
            "Detailedness: <score 1> <score 2>\nReason: <your explanation>",
        ):
            self.assertIn(frag, p)
        self.assertTrue(p.endswith("Reason: <your explanation>"))
        self.assertNotIn("{R1}", p)

    @unittest.skipUnless(PAPER.is_file(), "paper text not on this machine")
    def test_matches_paper_text(self):
        txt = _norm(PAPER.read_text())
        start = txt.index("You are required to score")
        end = txt.index("Table 5. The prompt used")
        body = txt[start:end]
        # drop the PDF page furniture, keep the prompt body
        body = re.sub(r"\s*GPT-4V Prompt\s*", " ", body)
        want = _norm(gj.PROMPT_TEMPLATE.replace("{R1}", "{Response of Assistant 1}")
                     .replace("{R2}", "{Response of Assistant 2}"))
        self.assertEqual(_norm(body), want)


class TestParse(unittest.TestCase):
    def test_ok(self):
        r = gj.parse_reply("Accuracy: 7 9\nReason: a\nDetailedness: 6 8\nReason: b")
        self.assertEqual(r, {"accuracy": (7.0, 9.0), "detailedness": (6.0, 8.0)})

    def test_markdown_and_ten(self):
        r = gj.parse_reply("**Accuracy:** 10 4\nReason\n**Detailedness:** 3 10")
        self.assertEqual(r["accuracy"], (10.0, 4.0))

    def test_unparseable(self):
        self.assertIsNone(gj.parse_reply(None))
        self.assertIsNone(gj.parse_reply(""))
        self.assertIsNone(gj.parse_reply("Accuracy: 7\nDetailedness: 6 8"))
        self.assertIsNone(gj.parse_reply("Accuracy: 11 9\nDetailedness: 6 8"))
        self.assertIsNone(gj.parse_reply("Accuracy: 0 9\nDetailedness: 6 8"))
        self.assertIsNone(gj.parse_reply("Accuracy: 7 9"))


class TestSelection(unittest.TestCase):
    def test_first_n_sorted(self):
        self.assertEqual(gj.select_images([9, 3, 5, 3, 1], 3), [1, 3, 5])

    def test_too_few(self):
        with self.assertRaises(ValueError):
            gj.select_images([1, 2], 3)

    def test_archived_runs_exist_and_align(self):
        s, f = gj.load_captions(REPO / "results" / "runs")
        self.assertEqual(len(s), 500)
        self.assertEqual(set(s), set(f))


class TestRun(unittest.TestCase):
    def setUp(self):
        self.samp = {1: "s1", 2: "s2", 3: "s3"}
        self.fox = {1: "f1", 2: "f2", 3: "f3"}
        self.ids = [1, 2, 3]
        self.imgs = {i: Path(f"/x/{i}.jpg") for i in self.ids}
        self.calls = gj.plan_calls(self.ids, self.samp, self.fox)

    def test_both_orders_swap_positions(self):
        self.assertEqual(len(self.calls), 6)
        sf = next(c for c in self.calls if c.image_id == 1 and c.order == "SF")
        fs = next(c for c in self.calls if c.image_id == 1 and c.order == "FS")
        self.assertIn("[Assistant 1]\ns1\n", sf.prompt)
        self.assertIn("[Assistant 1]\nf1\n", fs.prompt)

    def test_budget_aborts_before_any_call(self):
        c = FakeClient()
        with self.assertRaises(gj.BudgetExceeded):
            gj.run_judge(c, self.calls, self.imgs, max_calls=5)
        self.assertEqual(c.calls, [])

    def test_scores_mapped_by_order(self):
        res = gj.run_judge(FakeClient(), self.calls, self.imgs)
        s = gj.summarise(res["records"], n_boot=200)
        # order SF: assistant 2 (Fox) gets 8 vs Sampling 5
        self.assertEqual(s["accuracy"]["paper_style_SF"]["fox_mean"], 8.0)
        self.assertEqual(s["accuracy"]["paper_style_SF"]["sampling_mean"], 5.0)
        # order FS: assistant 1 (Fox) gets 5 vs Sampling 8: position bias
        self.assertEqual(s["accuracy"]["swapped_FS"]["fox_mean"], 5.0)
        # the fake is pure position bias, so order averaging cancels it
        oa = s["accuracy"]["order_averaged"]
        self.assertEqual(oa["diff_mean"], 0.0)
        self.assertEqual(oa["n"], 3)
        self.assertEqual(s["n_unparsed"], 0)

    def test_unparsed_counted_not_dropped(self):
        ok = "Accuracy: 5 8\nReason\nDetailedness: 6 7\nReason"
        replies = ["garbage", ok, ok, ok, ok, ok]
        res = gj.run_judge(FakeClient(replies), self.calls, self.imgs)
        self.assertEqual(len(res["records"]), 6)
        s = gj.summarise(res["records"], n_boot=100)
        self.assertEqual(s["n_unparsed"], 1)
        self.assertEqual(s["unparsed"][0]["image_id"], 1)
        self.assertEqual(s["accuracy"]["paper_style_SF"]["n"], 2)
        self.assertEqual(s["accuracy"]["order_averaged"]["n"], 2)
        self.assertEqual(s["accuracy"]["swapped_FS"]["n"], 3)

    def test_refuses_average_over_zero(self):
        res = gj.run_judge(FakeClient(["bad"] * 6), self.calls, self.imgs)
        with self.assertRaises(ValueError):
            gj.summarise(res["records"], n_boot=10)
        with self.assertRaises(ValueError):
            gj.summarise([], n_boot=10)

    def test_fatal_stops_and_keeps_partial(self):
        res = gj.run_judge(FakeClient(fatal_at=2), self.calls, self.imgs)
        self.assertEqual(len(res["records"]), 2)
        self.assertIn("quota", res["fatal"])


class TestSecrets(unittest.TestCase):
    def test_redact(self):
        k = "sk-abcdefghijklmnop1234"
        self.assertNotIn(k, gj.redact(f"bad key {k} here", k))
        self.assertNotIn("sk-zzzzzzzzzz", gj.redact("Incorrect key sk-zzzzzzzzzz"))

    def test_env_file(self):
        import os
        old = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / ".env"
                p.write_text('OPENAI_API_KEY="sk-testtesttest"\n')
                self.assertEqual(gj.load_api_key([p]), "sk-testtesttest")
        finally:
            if old is not None:
                os.environ["OPENAI_API_KEY"] = old


class TestScript(unittest.TestCase):
    def _fixture(self, d: Path):
        runs = d / "runs"
        runs.mkdir()
        imgs = d / "imgs"
        imgs.mkdir()
        for name, method in ((gj.SAMPLING_RUN, "sampling"), (gj.FOX_RUN, "fox")):
            with open(runs / name, "w") as f:
                for i in range(1, 6):
                    f.write(json.dumps({"method": method, "benchmark": "chair",
                                        "seed": 42, "decoding": "sampling",
                                        "image_id": i,
                                        "output": f"{method} {i}"}) + "\n")
        from foxrepro.data.loaders import image_filename_for_id
        for i in range(1, 6):
            (imgs / image_filename_for_id(i)).write_bytes(b"\xff\xd8x\xff\xd9")
        return runs, imgs

    def test_dry_run_makes_no_calls(self):
        sc = _load_script()
        with tempfile.TemporaryDirectory() as d:
            runs, imgs = self._fixture(Path(d))
            out = io.StringIO()
            with redirect_stdout(out):
                rc = sc.main(["--model", "gpt-4.1", "--n-images", "3",
                              "--runs-dir", str(runs), "--dry-run"],
                             client=FakeClient())
            self.assertEqual(rc, 0)
            self.assertIn("calls planned: 6", out.getvalue())
            self.assertIn("would call", out.getvalue())

    def test_budget_abort(self):
        sc = _load_script()
        with tempfile.TemporaryDirectory() as d:
            runs, imgs = self._fixture(Path(d))
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                rc = sc.main(["--model", "gpt-4.1", "--n-images", "3",
                              "--max-calls", "4", "--runs-dir", str(runs)],
                             client=FakeClient())
            self.assertEqual(rc, 3)

    def test_full_pipeline_with_fake(self):
        sc = _load_script()
        with tempfile.TemporaryDirectory() as d:
            runs, imgs = self._fixture(Path(d))
            fc = FakeClient()
            with redirect_stdout(io.StringIO()):
                rc = sc.main(["--model", "gpt-4.1", "--n-images", "3",
                              "--runs-dir", str(runs), "--images-dir", str(imgs),
                              "--out-dir", str(Path(d) / "out"),
                              "--n-boot", "100"], client=fc)
            self.assertEqual(rc, 0)
            self.assertEqual(len(fc.calls), 6)
            js = json.loads((Path(d) / "out" / "gpt_judge_gpt-4.1_n3.json").read_text())
            self.assertEqual(js["judge_model"], "gpt-4.1")
            self.assertEqual(len(js["records"]), 6)
            self.assertTrue((Path(d) / "out" / "gpt_judge_gpt-4.1_n3.md").is_file())


if __name__ == "__main__":
    unittest.main()
