"""The pure checks of scripts/round_cycle.py: each is a failure this project already paid for."""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import round_cycle as RC  # noqa: E402

NOW = 1_800_000_000_000.0
H = 3.6e6


def plan(budget=11.5, est=(8.0, 3.0)):
    return {"round": "r40", "budget_hours": budget,
            "sessions": [{"session": i + 1, "est_hours": e} for i, e in enumerate(est)]}


def manifest(**kw):
    m = {"sessions": "2", "est_hours": "11.0", "budget_hours": "12"}
    m.update({k: str(v) for k, v in kw.items()})
    return m


def doc(tag, states, age_h=1.0, created_days=1.0):
    return {"tag": tag, "createdAt": NOW - created_days * 8.64e7,
            "sessions": {str(i + 1): {"state": st, "accountName": "A", "pushedAt": NOW - age_h * H}
                         for i, st in enumerate(states)}}


class TestManifest(unittest.TestCase):
    def test_a_consistent_manifest_passes(self):
        self.assertEqual(RC.check_manifest(manifest(), plan()), [])

    def test_sessions_must_match_the_plan(self):
        self.assertTrue(any("runs empty" in b for b in RC.check_manifest(manifest(sessions=3), plan())))

    def test_est_hours_must_match_the_plan_total(self):
        self.assertTrue(any("est_hours" in b for b in RC.check_manifest(manifest(est_hours=9.9), plan())))

    def test_the_budget_cannot_exceed_kaggles_cap(self):
        self.assertTrue(any("cap" in b for b in RC.check_manifest(manifest(budget_hours=13), plan())))

    def test_a_plan_without_room_for_the_watchdog_is_flagged(self):
        # r32-r37: plan at 9 h inside a 10 h budget was fine; a plan AT the budget is always cut.
        self.assertTrue(any("watchdog" in b for b in RC.check_manifest(manifest(), plan(budget=12.0))))

    def test_one_session_over_budget_is_flagged(self):
        self.assertTrue(any("session 1" in b for b in RC.check_manifest(
            manifest(est_hours=14.0), plan(est=(13.0, 1.0)))))


class TestDue(unittest.TestCase):
    def test_all_terminal_and_unhandled_is_ready(self):
        r, w, z = RC.due_rounds([doc("r40", ["done", "failed"])], set(), NOW)
        self.assertEqual((r, w, z), (["r40"], [], []))

    def test_a_running_session_keeps_the_round_waiting(self):
        r, w, _ = RC.due_rounds([doc("r40", ["done", "running"])], set(), NOW)
        self.assertEqual((r, w), ([], ["r40"]))

    def test_a_handled_round_is_not_reported_again(self):
        r, w, _ = RC.due_rounds([doc("r40", ["done"])], {"r40"}, NOW)
        self.assertEqual((r, w), ([], []))

    def test_smoke_and_verify_rounds_are_not_cycle_rounds(self):
        r, w, _ = RC.due_rounds([doc("smoke-r40", ["done"]), doc("verify-r40", ["done"])], set(), NOW)
        self.assertEqual((r, w), ([], []))

    def test_a_long_running_session_is_a_zombie(self):
        _, _, z = RC.due_rounds([doc("r40", ["running"], age_h=34)], set(), NOW)
        self.assertEqual(len(z), 1)
        self.assertIn("34 h", z[0])

    def test_old_rounds_are_history_not_zombies(self):
        _, _, z = RC.due_rounds([doc("r16", ["running"], age_h=355, created_days=15)], set(), NOW)
        self.assertEqual(z, [])


class TestWatch(unittest.TestCase):
    def _run(self, docs_seq):
        import io
        from contextlib import redirect_stdout
        from unittest import mock

        calls = iter(docs_seq)

        def fake_docs():
            d = next(calls)
            if isinstance(d, Exception):
                raise d
            return d

        buf = io.StringIO()
        with mock.patch.object(RC, "_docs", fake_docs), \
                mock.patch.object(RC, "_handled", lambda: set()), \
                mock.patch.object(RC.time, "sleep", lambda s: None), \
                mock.patch.object(RC.time, "time", lambda: NOW / 1000), \
                redirect_stdout(buf):
            rc = RC.watch(0)
        return rc, buf.getvalue().splitlines()

    def test_it_waits_silently_then_reports_the_due_round(self):
        rc, out = self._run([[doc("r40", ["running"])], [doc("r40", ["running"])],
                             [doc("r40", ["done", "failed"])]])
        self.assertEqual((rc, out), (0, ["DUE r40"]))

    def test_a_zombie_is_reported_once_and_a_blip_does_not_kill_it(self):
        rc, out = self._run([[doc("r40", ["running"], age_h=20)], RuntimeError("db down"),
                             RuntimeError("db down"), [doc("r40", ["running"], age_h=20)],
                             [doc("r40", ["done"])]])
        self.assertEqual(rc, 0)
        self.assertEqual(sum(l.startswith("ZOMBIE") for l in out), 1)
        self.assertEqual(sum(l.startswith("watch error") for l in out), 1)
        self.assertEqual(out[-1], "DUE r40")


class TestScalars(unittest.TestCase):
    def test_reads_the_real_manifest(self):
        m = RC.scalars(REPO / "kaggle-run.yml")
        for k in ("sessions", "est_hours", "budget_hours"):
            self.assertIn(k, m)
        self.assertEqual(float(m["budget_hours"]) <= RC.KAGGLE_SESSION_HOURS, True)


if __name__ == "__main__":
    unittest.main()
