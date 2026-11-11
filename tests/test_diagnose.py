"""The forward-only diagnostic pass behind Table 13 and Figures 3 and 7.

Three things need pinning here and none is visible by reading:

* the three token-group masses PARTITION the keys, so they sum to one. The
  paper's Table 13 reports 0.57 + 0.62 + 0.82 = 2.01 for the same three
  names, so whatever that table measures, it is not Equation 11 applied to
  the three groups. The identity is what makes that a finding rather than a
  guess;
* measuring must not disturb a scored run. Diagnosis forces the eager path,
  and eager and SDPA reduce in different orders, so a diagnostic switched on
  during generation could move a token for a reason that has nothing to do
  with the method. The exact guarantee is narrower and is asserted below:
  measuring ALONGSIDE an intervention leaves the intervened output
  bit-identical, because `intervene` sees the same logits either way;
* every layer must be measured, not the intervention window. Figure 7 plots
  |AUC - 0.5| per layer and the paper's automatic rule picks a window from
  it; a sweep over the window the rule is meant to choose could not check it.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("torch is not installed") from None

try:
    import transformers  # noqa: F401
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("transformers is not installed") from None

from foxrepro.fox import attention as A  # noqa: E402
from foxrepro.fox.attention import fox_diagnose  # noqa: E402
from foxrepro.fox.kernel import diagnostic_stats, head_scores  # noqa: E402

from tests.test_tiny_llava import IMAGE_TOKEN_ID, N_IMG, tiny_llava  # noqa: E402

PRE, POST = 6, 4


def prompt():
    ids = torch.arange(1, PRE + N_IMG + POST + 1, dtype=torch.long) % 60
    ids = ids.unsqueeze(0)
    ids[0, PRE:PRE + N_IMG] = IMAGE_TOKEN_ID
    return ids


class TestTheMassesPartitionTheKeys(unittest.TestCase):
    """m_sys + m_vis + m_txt == 1, which Table 13's numbers cannot satisfy."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.probs = torch.softmax(torch.randn(2, 4, 40, 100), dim=-1)

    def test_the_three_masses_sum_to_one(self) -> None:
        for tail in (1, 8, 32, 1000):
            with self.subTest(tail=tail):
                d = diagnostic_stats(self.probs, 6, 65, tail=tail)
                total = d["m_sys"] + d["m_vis"] + d["m_txt"]
                self.assertTrue(
                    torch.allclose(total, torch.ones_like(total), atol=1e-6),
                    f"masses sum to {total.min():.8f}..{total.max():.8f}, so "
                    f"they no longer partition the keys"
                )

    def test_the_papers_table_13_triple_cannot_be_these_masses(self) -> None:
        """0.57 + 0.62 + 0.82 = 2.01. Stated as a test so it cannot drift."""
        self.assertAlmostEqual(0.57 + 0.62 + 0.82, 2.01, places=6)

    def test_m_sys_at_tail_1_is_the_released_statistic(self) -> None:
        """Equation 11 at tail=1 is exactly what head_scores already reads."""
        d = diagnostic_stats(self.probs, 6, 65, tail=1)
        m_sys, _ = head_scores(self.probs, 6, 65)
        self.assertTrue(torch.allclose(d["m_sys"], m_sys, atol=1e-7))

    def test_h_vis_differs_from_head_scores_by_the_double_normalisation(self) -> None:
        """head_scores renormalises twice to stay bit-equal to the release.

        This function does not, because it is not the oracle path. The gap is
        tiny and must stay tiny: a large one would mean the spans disagree.
        """
        d = diagnostic_stats(self.probs, 6, 65, tail=1)
        _, h_vis = head_scores(self.probs, 6, 65)
        gap = float((d["h_vis"] - h_vis).abs().max())
        self.assertLess(gap, 1e-5, f"h_vis gap {gap:g} is too large to be the "
                                   f"second renormalisation")

    def test_an_empty_visual_span_returns_none(self) -> None:
        self.assertIsNone(diagnostic_stats(self.probs, 6, 5))


class TestTheDiagnosticPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = tiny_llava()
        A.register()
        cls.model.set_attn_implementation(
            {"text_config": A.ATTN_NAME, "vision_config": "sdpa"}
        )

    def setUp(self) -> None:
        torch.manual_seed(1)
        ids = prompt()
        self.inputs = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "pixel_values": torch.randn(1, 3, 336, 336),
        }

    def run_diag(self, **kw):
        return fox_diagnose(self.model, self.inputs,
                            image_token_id=IMAGE_TOKEN_ID, **kw)

    def test_every_layer_is_measured(self) -> None:
        got = self.run_diag(tails=(1,))
        self.assertEqual(sorted(got["layers"]), list(range(got["n_layers"])),
                         "Figure 7 needs every layer, not the intervention "
                         "window")

    def test_each_requested_tail_is_recorded(self) -> None:
        got = self.run_diag(tails=(1, 4))
        for layer in got["layers"].values():
            self.assertEqual(
                sorted(layer),
                ["h_vis_t1", "h_vis_t4", "m_sys_t1", "m_sys_t4",
                 "m_txt_t1", "m_txt_t4", "m_vis_t1", "m_vis_t4"])

    def test_one_value_per_head(self) -> None:
        got = self.run_diag(tails=(1,))
        for layer in got["layers"].values():
            for name, values in layer.items():
                self.assertEqual(len(values), got["n_heads"], name)

    def test_the_masses_partition_on_the_real_model_too(self) -> None:
        got = self.run_diag(tails=(1, 4))
        for idx, layer in got["layers"].items():
            for tail in (1, 4):
                for h in range(got["n_heads"]):
                    total = sum(layer[f"{k}_t{tail}"][h]
                                for k in ("m_sys", "m_vis", "m_txt"))
                    self.assertAlmostEqual(
                        total, 1.0, places=5,
                        msg=f"layer {idx} head {h} tail {tail}: {total}")

    def test_measuring_does_not_perturb_an_intervention(self) -> None:
        """The property that protects every scored arm.

        Diagnosis forces the eager path, which is not bit-equal to SDPA. That
        is harmless while the diagnostic pass generates nothing, but it would
        not be if diagnosis were ever switched on during a scored run. What
        must hold -- and does, because `intervene` receives the same logits
        either way -- is that measuring beside an intervention leaves the
        intervened output untouched, bit for bit.
        """
        span = self.run_diag(tails=(1,))
        cfg = dict(
            enabled=True, mode="as_implemented", gamma=10000.0, ratio=0.35,
            logit_dtype="native", softmax_dtype="fp32",
            img_start=span["img_start"], img_end=span["img_end"],
            eager_layers=frozenset(range(span["n_layers"])),
        )
        with torch.inference_mode():
            with A.FOX.configured(diagnose=False, **cfg):
                plain = self.model(**self.inputs, use_cache=False).logits.clone()
            A.FOX.reset_diagnostics()
            with A.FOX.configured(diagnose=True, diagnose_tails=(1,), **cfg):
                measured = self.model(**self.inputs, use_cache=False).logits.clone()
        self.assertTrue(
            torch.equal(plain, measured),
            "switching the diagnostic on changed an intervened forward; the "
            "measurement is reading the tensor being mutated"
        )

    def test_diagnosis_alone_matches_the_untouched_path_to_kernel_tolerance(self) -> None:
        """Eager vs SDPA, stated rather than assumed.

        Not bit-equality: the two reduce in different orders. The bound is
        here so that a real regression -- a mask dropped, a span misread --
        cannot hide behind "floating point".
        """
        span = self.run_diag(tails=(1,))
        with torch.inference_mode():
            untouched = self.model(**self.inputs, use_cache=False).logits.clone()
            A.FOX.reset_diagnostics()
            with A.FOX.configured(
                enabled=False, mode="off", diagnose=True, diagnose_tails=(1,),
                img_start=span["img_start"], img_end=span["img_end"],
                eager_layers=frozenset(range(span["n_layers"])),
                softmax_dtype="fp32",
            ):
                measured = self.model(**self.inputs, use_cache=False).logits.clone()
        gap = float((untouched - measured).abs().max())
        self.assertLess(gap, 1e-4, f"diagnostic-only forward drifted {gap:g} "
                                   f"from the untouched path")

    def test_a_missing_visual_span_is_refused(self) -> None:
        ids = torch.arange(1, 12, dtype=torch.long).unsqueeze(0) % 60
        with self.assertRaises(Exception):
            fox_diagnose(
                self.model,
                {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                 "pixel_values": torch.randn(1, 3, 336, 336)},
                image_token_id=IMAGE_TOKEN_ID,
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class TestTheAnalysisArithmetic(unittest.TestCase):
    """The AUC is hand-rolled, so it is pinned against cases done by hand.

    No sklearn: this project declares no dependencies. The Mann-Whitney form
    is short, but ties are where rank-based AUCs go wrong silently -- a tied
    pair must count half, and a diagnostic where every head saturates would
    otherwise read as perfect separation.
    """

    @staticmethod
    def _auc(*a, **kw):
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "scripts"))
        from analyse_diagnostics import auc
        return auc(*a, **kw)

    def test_perfect_separation(self) -> None:
        self.assertEqual(self._auc([3, 4, 5], [0, 1, 2]), 1.0)

    def test_perfect_inversion(self) -> None:
        self.assertEqual(self._auc([0, 1, 2], [3, 4, 5]), 0.0)

    def test_all_ties_are_one_half(self) -> None:
        self.assertEqual(self._auc([1, 1, 1], [1, 1, 1]), 0.5)

    def test_an_empty_class_has_no_auc(self) -> None:
        self.assertIsNone(self._auc([], [1, 2]))
        self.assertIsNone(self._auc([1, 2], []))

    def test_an_interleaved_case_done_by_hand(self) -> None:
        # pos {1,3}, neg {2,4}: pairs (1,2) (1,4) (3,2) (3,4) -> 1 win of 4.
        self.assertEqual(self._auc([1, 3], [2, 4]), 0.25)

    def test_one_tie_counts_half(self) -> None:
        # pos {2}, neg {1,2}: one win, one tie -> (1 + 0.5) / 2.
        self.assertEqual(self._auc([2], [1, 2]), 0.75)

    def test_a_false_no_is_not_a_hallucination(self) -> None:
        """POPE's two failure modes are opposite and must not share a class.

        Answering "no" about an object that IS there is a miss. Pooling it
        with a false "yes" would put samples whose attention should look
        opposite into one positive class and flatten every AUC.
        """
        import json
        import sys
        import tempfile
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "scripts"))
        from analyse_diagnostics import load_labels

        rows = [
            {"sample_id": "a", "output": "Yes", "label": "no"},   # hallucination
            {"sample_id": "b", "output": "No", "label": "yes"},   # miss
            {"sample_id": "c", "output": "Yes", "label": "yes"},  # correct
            {"sample_id": "d", "output": "No", "label": "no"},    # correct
            {"sample_id": "e", "output": "maybe", "label": "no"},  # unparsable
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.jsonl"
            p.write_text("".join(json.dumps(r) + "\n" for r in rows))
            got = load_labels(p)
        self.assertEqual(got, {"a": True, "b": False, "c": False, "d": False},
                         "an unparsable answer must be dropped, and a false "
                         "'no' must not be labelled a hallucination")
