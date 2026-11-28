"""The PreToolUse guard: what it blocks, and that it does not block ordinary work."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "hooks"))
import guard_bash as G  # noqa: E402

HEAD = "a" * 40
OK = lambda tag: []                       # noqa: E731
BAD = lambda tag: ["no green test run"]   # noqa: E731


class TestRelease(unittest.TestCase):
    def test_a_round_release_needs_preflight(self):
        r = G.decide(f"gh release create r40 --target {HEAD} --title r40", HEAD, BAD)
        self.assertIn("blocked by preflight", r)
        self.assertIn("no green test run", r)

    def test_preflight_is_told_which_tag_is_being_released(self):
        seen = []
        G.decide(f"gh release create r41 --target {HEAD}", HEAD, lambda t: seen.append(t) or [])
        self.assertEqual(seen, ["r41"])

    def test_a_round_release_passes_when_preflight_is_clean(self):
        self.assertIsNone(G.decide(f"gh release create r40 --target {HEAD}", HEAD, OK))

    def test_a_short_target_is_refused(self):
        self.assertIn("full sha", G.decide("gh release create r40 --target 5db0868", HEAD, OK))

    def test_a_target_that_is_not_head_is_refused(self):
        self.assertIn("full sha", G.decide(f"gh release create r40 --target {'b' * 40}", HEAD, OK))

    def test_a_shell_variable_target_is_not_judged(self):
        # `SHA=$(git rev-parse HEAD); gh release create r40 --target $SHA` cannot be resolved here.
        self.assertIsNone(G.decide("SHA=$(git rev-parse HEAD); gh release create r40 --target $SHA", HEAD, OK))

    def test_smoke_and_verify_releases_are_not_rounds(self):
        for tag in ("smoke-r40", "verify-r40"):
            self.assertIsNone(G.decide(f"gh release create {tag} --target {HEAD}", HEAD, BAD))

    def test_it_finds_the_release_inside_a_chain(self):
        self.assertIsNotNone(G.decide(f"git push -q && gh release create r41 --target {HEAD}", HEAD, BAD))

    def test_other_gh_commands_are_untouched(self):
        for c in ("gh release view r39", "gh run list --workflow kaggle-run.yml", "gh release list"):
            self.assertIsNone(G.decide(c, HEAD, BAD))


class TestAdd(unittest.TestCase):
    def test_secrets_zips_and_pdfs_are_blocked(self):
        for c in ("git add .env", "git add runs_inbox/x", "git add fox_runs_fox_r39_s1.zip",
                  "git add docs/yu26c.pdf", "git add -f results/x.jsonl"):
            self.assertIsNotNone(G.decide(c, HEAD, OK), c)

    def test_ordinary_adds_pass(self):
        for c in ("git add scripts/round_cycle.py tests", "git add -A src tests docs",
                  "git add results/runs", "git add kaggle-run.yml run_kaggle_fox.sh"):
            self.assertIsNone(G.decide(c, HEAD, OK), c)


if __name__ == "__main__":
    unittest.main()
