"""The audit must not call a thing unreachable once we have reached it.

This is here because the same error happened five times. `docs/PAPER_AUDIT.md`
has an "Out of reach" table listing what this reproduction cannot do, and four
separate times a row stayed in it after the work landed -- Table 7, Figure 9,
Table 15, the VCD/ICD rows -- and each time it nearly produced a wrong answer
to a direct question, because the audit is what gets read instead of the code.
The fifth was MME: scored in r21, still listed as impossible, two commits
after the number went into three documents.

A doc is not covered by the tests that cover the code, so it rots at the one
moment anyone cares about it: right after the result arrives. The check is
cheap and mechanical -- a benchmark with runs in `results/runs/` is a
benchmark we have run, and it cannot also be out of reach.

What this deliberately does NOT do is check prose. It checks the one claim
that is machine-decidable and was wrong every time: presence in that table.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "docs" / "PAPER_AUDIT.md"
RUNS = REPO / "results" / "runs"

#: Benchmark key as it appears in a run filename -> how the audit writes it.
#: Explicit, and asserted to be total below: a new benchmark fails this test
#: until someone states its name, which is the moment to ask whether the
#: audit still describes it correctly.
BENCH_NAMES = {
    "pope": "POPE",
    "chair": "CHAIR",
    "mme": "MME",
    "mmbench": "MMBench",
    "mmbench_circular": "MMBench CircularEval",
    "mathvista": "MathVista",
}


def benchmarks_with_runs() -> set:
    out = set()
    for p in RUNS.glob("*.jsonl"):
        m = re.search(r"__b-([a-z0-9]+)__", p.name)
        if m:
            out.add(m.group(1))
    return out


def section(name: str) -> str:
    """The body of one `## ` section, up to the next one."""
    text = AUDIT.read_text(encoding="utf-8")
    m = re.search(rf"^## {re.escape(name)}\s*$(.*?)(?=^## |\Z)",
                  text, re.M | re.S)
    if m is None:
        raise AssertionError(f"{AUDIT.name} has no '## {name}' section")
    return m.group(1)


class TestTheAuditKnowsWhatWeHaveRun(unittest.TestCase):
    def test_there_are_runs_to_check_against(self) -> None:
        """Guard the guard: an empty results dir would pass everything."""
        self.assertGreater(len(benchmarks_with_runs()), 0,
                           "no run files found; this test would be vacuous")

    def test_every_benchmark_with_runs_has_a_name(self) -> None:
        unnamed = benchmarks_with_runs() - set(BENCH_NAMES)
        self.assertEqual(unnamed, set(),
                         f"benchmark(s) {sorted(unnamed)} have runs but no "
                         f"entry in BENCH_NAMES -- add one, and while you are "
                         f"there check that PAPER_AUDIT.md still describes "
                         f"them correctly")

    def test_a_benchmark_we_have_run_is_not_out_of_reach(self) -> None:
        body = section("Out of reach")
        for key in sorted(benchmarks_with_runs()):
            name = BENCH_NAMES[key]
            # Word boundary: "MME" must not match inside another token, and
            # a row may legitimately mention a benchmark we HAVE run as
            # context ("the MathVista subset", not "MME").
            hits = [ln for ln in body.splitlines()
                    if re.search(rf"\b{re.escape(name)}\b", ln)]
            self.assertEqual(
                hits, [],
                f"{name} has runs in results/runs/ but PAPER_AUDIT.md still "
                f"lists it as out of reach:\n  " + "\n  ".join(hits))


class TestTheValidationLogAndTheAuditAgree(unittest.TestCase):
    """Both documents carry the MME verdict; neither may carry it alone."""

    def test_mme_is_recorded_in_both(self) -> None:
        if "mme" not in benchmarks_with_runs():
            self.skipTest("no MME runs")
        audit = AUDIT.read_text(encoding="utf-8")
        validation = (REPO / "docs" / "VALIDATION.md").read_text(encoding="utf-8")
        for doc, text in (("PAPER_AUDIT.md", audit), ("VALIDATION.md", validation)):
            with self.subTest(doc=doc):
                self.assertIn("MME", text,
                              f"{doc} does not mention MME although it has run")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
