"""scripts/verify_bitwise.py: the comparison must be exact and must refuse to
call a short or partial comparison 'identical'."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location(
    "verify_bitwise", ROOT / "scripts" / "verify_bitwise.py")
vb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vb)


def rec(i, out="Yes", **extra):
    return {"sample_id": f"s{i}", "output": out, "method_extra": dict(extra)}


class TestCompare(unittest.TestCase):
    def test_identical_records_are_identical(self) -> None:
        a = [rec(i, d_t_mean=0.1234567890123) for i in range(5)]
        b = [rec(i, d_t_mean=0.1234567890123) for i in range(5)]
        r = vb.compare(a, b)
        self.assertEqual(vb.verdict(r, 5), "identical")

    def test_one_changed_output_is_different(self) -> None:
        a = [rec(i) for i in range(5)]
        b = [rec(i) for i in range(5)]
        b[3]["output"] = "No"
        r = vb.compare(a, b)
        self.assertEqual(vb.verdict(r, 5), "DIFFERENT")
        self.assertEqual(r["first"]["sample_id"], "s3")

    def test_a_one_ulp_float_difference_is_not_tolerated(self) -> None:
        a = [rec(0, lam=1.0)]
        b = [rec(0, lam=1.0000000000000002)]
        self.assertEqual(vb.verdict(vb.compare(a, b), 1), "DIFFERENT")

    def test_missing_samples_are_incomplete_not_identical(self) -> None:
        a = [rec(i) for i in range(5)]
        b = [rec(i) for i in range(3)]
        self.assertEqual(vb.verdict(vb.compare(a, b), 5), "incomplete")

    def test_nothing_compared_is_incomplete(self) -> None:
        self.assertEqual(vb.verdict(vb.compare([rec(0)], []), 1), "incomplete")

    def test_command_strips_the_jobs_limit_and_redirects_output(self) -> None:
        from foxrepro.bench.plan import Job
        j = Job("t", "fox", "pope", "adversarial", decoding="greedy", limit=7)
        cmd = vb.command_for(j, Path("/tmp/x"), 300)
        self.assertEqual(cmd.count("--limit"), 1)
        self.assertEqual(cmd[cmd.index("--limit") + 1], "300")
        self.assertEqual(cmd[cmd.index("--runs-dir") + 1], "/tmp/x")
        self.assertEqual(cmd.count("-m"), 1)

    def test_every_listed_job_exists_in_the_plan(self) -> None:
        from foxrepro.bench.plan import PLAN
        ids = {j.job_id() for j in PLAN}
        for jid in vb.JOB_IDS:
            self.assertIn(jid, ids)


if __name__ == "__main__":
    unittest.main()
