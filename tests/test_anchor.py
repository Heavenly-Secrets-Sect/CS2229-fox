"""Tables 4 and 12: which query position the per-head score is read from.

The paper prints this table twice — "temporal anchor" in the main text,
"query position" in the appendix — and names five policies. The released
code implements one of them, `tail_n = 1`, which during generation is
`y_prev` and never `x_last`. So the configuration the paper DESCRIBES
(`x_last + y_{t-1}`) has never been run by anyone, including the authors'
own released harness.

Four things need pinning and none is visible by reading:

* the DEFAULT must not move. Every archived run and the bit-equality proof
  in test_kernel_equivalence.py depend on `y_prev` executing exactly the
  code it executed before anchors existed, so that path is a separate
  branch and is asserted identical here as well;
* `head_scores_rows` must agree with `head_scores` on the row they share.
  It is a second implementation of the same arithmetic over every row
  rather than the last, and a drift between them would silently change
  which heads are selected;
* `x_last` must survive decoding. It is captured once, at prefill, and
  every later step reuses it — a state object that reset per forward would
  degrade to `y_prev` and look like a working arm;
* the state must NOT survive a sample. Carrying one caption's prompt
  statistics into the next is the failure that produces plausible numbers
  and no error.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("torch is not installed") from None

from foxrepro.fox.kernel import (  # noqa: E402
    ANCHORS, AnchorState, head_scores, head_scores_rows, intervene,
)
from foxrepro.storage import ANCHORS as STORAGE_ANCHORS  # noqa: E402


def probs(q: int = 40, k: int = 100, seed: int = 0):
    torch.manual_seed(seed)
    return torch.softmax(torch.randn(1, 4, q, k), dim=-1)


class TestTheNameListsAgree(unittest.TestCase):
    def test_storage_mirrors_the_kernel(self) -> None:
        self.assertEqual(tuple(STORAGE_ANCHORS), tuple(ANCHORS))

    def test_the_released_policy_is_first_and_is_the_default(self) -> None:
        self.assertEqual(ANCHORS[0], "y_prev")


class TestRowsAgreeWithTheReleasedReduction(unittest.TestCase):
    def test_the_last_row_is_bit_equal_to_head_scores(self) -> None:
        """Two implementations of one quantity; they must not drift."""
        for q in (1, 7, 40):
            with self.subTest(q=q):
                p = probs(q=q)
                got = head_scores_rows(p, 6, 65)
                want = head_scores(p, 6, 65)
                self.assertIsNotNone(got)
                self.assertTrue(torch.equal(got[1][0], want[0]))
                self.assertTrue(torch.equal(got[1][1], want[1]))

    def test_rows_that_cannot_see_the_whole_image_are_dropped(self) -> None:
        """Before img_end, m_sys is computed over a truncated span.

        Averaging those in would mix a different quantity into the mean.
        The paper does not say what it does here; docs/DEVIATIONS.md records
        the choice.
        """
        p = probs(q=40, k=100)          # rows 0..39 sit at keys 60..99
        got = head_scores_rows(p, 6, 65)
        self.assertEqual(got[2], 34)    # img_end 65 -> first usable row is 6

    def test_a_single_decode_row_is_always_usable(self) -> None:
        got = head_scores_rows(probs(q=1, k=100), 6, 65)
        self.assertEqual(got[2], 1)

    def test_an_empty_visual_span_returns_none(self) -> None:
        self.assertIsNone(head_scores_rows(probs(), 6, 5))


class TestTheState(unittest.TestCase):
    def setUp(self) -> None:
        self.state = AnchorState()
        self.p_prefill = probs(q=40, k=100, seed=1)
        self.p_decode = probs(q=1, k=101, seed=2)

    def feed(self, p, layer=0):
        rows = head_scores_rows(p, 6, 65)
        self.state.observe(layer, (rows[0], rows[1]), rows[2],
                           is_prefill=p.shape[2] > 1)
        return rows[1]

    def test_x_last_is_captured_at_prefill_and_held(self) -> None:
        """The property the arm is: it must not decay to y_prev."""
        pre = self.feed(self.p_prefill)
        dec = self.feed(self.p_decode)
        held = self.state.resolve(0, "x_last", dec)
        self.assertTrue(torch.equal(held[0], pre[0]))
        self.assertFalse(torch.equal(held[0], dec[0]))

    def test_both_is_the_mean_of_the_two_anchors(self) -> None:
        pre = self.feed(self.p_prefill)
        dec = self.feed(self.p_decode)
        got = self.state.resolve(0, "both", dec)
        self.assertTrue(torch.allclose(got[0], 0.5 * (pre[0] + dec[0])))
        self.assertTrue(torch.allclose(got[1], 0.5 * (pre[1] + dec[1])))

    def test_mean_all_weighs_a_prefill_by_its_row_count(self) -> None:
        """600 prompt rows must not weigh the same as one decode step."""
        rows = head_scores_rows(self.p_prefill, 6, 65)
        self.feed(self.p_prefill)
        dec = self.feed(self.p_decode)
        n = rows[2]
        want = (rows[0][0] * n + dec[0]) / (n + 1)
        self.assertTrue(torch.allclose(
            self.state.resolve(0, "mean_all", dec)[0], want))

    def test_every_anchor_falls_back_before_the_first_prefill(self) -> None:
        """A decode-only call must not raise or return garbage."""
        dec = self.feed(self.p_decode)
        for a in ANCHORS:
            with self.subTest(anchor=a):
                got = self.state.resolve(0, a, dec)
                self.assertEqual(got[0].shape, dec[0].shape)

    def test_layers_do_not_share_state(self) -> None:
        self.feed(self.p_prefill, layer=0)
        other = self.feed(self.p_decode, layer=1)
        self.assertFalse(torch.equal(
            self.state.resolve(0, "x_last", other)[0],
            self.state.resolve(1, "x_last", other)[0]))

    def test_reset_clears_everything(self) -> None:
        """One caption's prompt must not leak into the next."""
        self.feed(self.p_prefill)
        self.state.reset()
        dec = self.feed(self.p_decode)
        self.assertTrue(torch.equal(
            self.state.resolve(0, "x_last", dec)[0], dec[0]))

    def test_an_unknown_anchor_is_refused(self) -> None:
        dec = self.feed(self.p_decode)
        with self.assertRaises(ValueError):
            self.state.resolve(0, "first_token", dec)


class TestTheDefaultPathIsUntouched(unittest.TestCase):
    """`y_prev` must execute the pre-anchor code, bit for bit."""

    def setUp(self) -> None:
        torch.manual_seed(3)
        self.logits = torch.randn(2, 4, 1, 100)

    def run_one(self, **kw):
        return intervene(self.logits.clone(), 6, 65, mode="as_implemented",
                         gamma=10000.0, ratio=0.5, **kw)[0]

    def test_no_anchor_argument_equals_y_prev(self) -> None:
        self.assertTrue(torch.equal(self.run_one(), self.run_one(anchor="y_prev")))

    def test_a_state_object_does_not_change_the_default(self) -> None:
        """Passing state while asking for y_prev must still take the old path."""
        self.assertTrue(torch.equal(
            self.run_one(),
            self.run_one(anchor="y_prev", anchor_state=AnchorState())))

    def test_a_non_default_anchor_without_state_falls_back(self) -> None:
        """No state means no accumulator, so the released path is correct."""
        self.assertTrue(torch.equal(
            self.run_one(), self.run_one(anchor="both", anchor_state=None)))


class TestThePlumbing(unittest.TestCase):
    @staticmethod
    def cfg(**kw):
        from foxrepro.bench.generate import GenConfig
        base = dict(method="fox", benchmark="pope", split="adversarial",
                    seed=42, dtype="fp16",
                    model_id="llava-hf/llava-1.5-7b-hf",
                    prompt_template_id="pope_v0_raw", max_new_tokens=1)
        return GenConfig(**{**base, **kw})

    def test_the_default_run_key_is_unchanged(self) -> None:
        self.assertNotIn("anchor", self.cfg()._variant_fields())

    def test_each_anchor_gets_its_own_file(self) -> None:
        names = {self.cfg(anchor=a).run_key().filename()
                 for a in ANCHORS if a != "y_prev"}
        names.add(self.cfg().run_key().filename())
        self.assertEqual(len(names), len(ANCHORS))

    def test_a_non_fox_method_may_not_set_it(self) -> None:
        with self.assertRaises(ValueError):
            self.cfg(method="sampling", anchor="both")

    def test_the_job_and_the_config_agree(self) -> None:
        from foxrepro.bench.plan import Job
        for a in ("x_last", "both", "mean_all", "random"):
            with self.subTest(anchor=a):
                job = Job("t4", "fox", "pope", "adversarial", anchor=a)
                self.assertEqual(job.run_key().filename(),
                                 self.cfg(anchor=a).run_key().filename())
                self.assertIn(f"--anchor {a}", job.command())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
