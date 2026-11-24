"""Statistics behind scripts/analyse_figures_12_14.py, on synthetic data.

The script reads 234 MB that is not in the repo, so what can be pinned here is
the arithmetic: AUC with ties, KS, average precision, the Top-K projection
(Eq. 19) and the leakage guard on the permutation null. Each is checked against
a brute-force definition rather than against a library.
"""

from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("numpy is not installed") from None

_SPEC = importlib.util.spec_from_file_location(
    "analyse_figures_12_14",
    Path(__file__).resolve().parent.parent / "scripts" / "analyse_figures_12_14.py")
F = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(F)  # type: ignore[union-attr]


def brute_auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


class TestAuc(unittest.TestCase):
    def test_matches_pairwise_definition_with_ties(self) -> None:
        rng = np.random.default_rng(1)
        x = np.round(rng.normal(size=60), 1)          # many exact ties
        y = rng.random(60) < 0.4
        self.assertAlmostEqual(F.auc(x, y), brute_auc(x, y), places=12)

    def test_perfect_and_reversed(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([False, False, True, True])
        self.assertEqual(F.auc(x, y), 1.0)
        self.assertEqual(F.auc(-x, y), 0.0)

    def test_column_wise_equals_single(self) -> None:
        rng = np.random.default_rng(2)
        x = rng.normal(size=(40, 5))
        y = rng.random(40) < 0.5
        got = F.auc_from_ranks(F.average_ranks(x), y)
        for j in range(5):
            self.assertAlmostEqual(got[j], brute_auc(x[:, j], y), places=12)

    def test_constant_scores_are_chance(self) -> None:
        self.assertEqual(F.auc(np.ones(10), np.arange(10) < 4), 0.5)

    def test_one_class_is_nan(self) -> None:
        self.assertTrue(np.isnan(F.auc(np.arange(5.0), np.zeros(5, bool))))


class TestKs(unittest.TestCase):
    def test_matches_brute_force(self) -> None:
        rng = np.random.default_rng(3)
        a, b = rng.normal(0, 1, 50), rng.normal(0.5, 1, 70)
        grid = np.concatenate([a, b])
        brute = max(abs((a <= g).mean() - (b <= g).mean()) for g in grid)
        d, p = F.ks_two_sample(a, b)
        self.assertAlmostEqual(d, brute, places=12)
        self.assertTrue(0.0 <= p <= 1.0)

    def test_identical_samples(self) -> None:
        a = np.linspace(0, 1, 30)
        d, p = F.ks_two_sample(a, a.copy())
        self.assertEqual(d, 0.0)
        self.assertEqual(p, 1.0)

    def test_disjoint_samples(self) -> None:
        d, p = F.ks_two_sample(np.arange(50.0), np.arange(50.0) + 100)
        self.assertEqual(d, 1.0)
        self.assertLess(p, 1e-10)

    def test_p_value_orders_effect_sizes(self) -> None:
        rng = np.random.default_rng(4)
        base = rng.normal(size=400)
        small = F.ks_two_sample(base, base + rng.normal(0.05, 1, 400))[1]
        big = F.ks_two_sample(base, rng.normal(1.0, 1, 400))[1]
        self.assertGreater(small, big)


class TestOverlapAndEffect(unittest.TestCase):
    def test_overlap_bounds(self) -> None:
        a = np.linspace(0, 1, 200)
        self.assertAlmostEqual(F.overlap_coefficient(a, a), 1.0)
        self.assertEqual(F.overlap_coefficient(a, a + 10), 0.0)

    def test_cohens_d(self) -> None:
        rng = np.random.default_rng(5)
        a = rng.normal(1.0, 1.0, 5000)
        b = rng.normal(0.0, 1.0, 5000)
        self.assertAlmostEqual(F.cohens_d(a, b), 1.0, delta=0.06)

    def test_mannwhitney_p_is_one_at_chance(self) -> None:
        self.assertAlmostEqual(F.mannwhitney_p(0.5, 100, 100), 1.0)
        self.assertLess(F.mannwhitney_p(0.7, 100, 100), 1e-4)


class TestPrecisionRecall(unittest.TestCase):
    def test_perfect_ranking_has_ap_one(self) -> None:
        s = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
        y = np.array([True, True, False, False, False])
        self.assertAlmostEqual(F.pr_curve(s, y)[2], 1.0)

    def test_known_small_case(self) -> None:
        # ranked: pos, neg, pos -> P@1=1 (R=.5), P@3=2/3 (R=1): AP=.5+.5*2/3
        s = np.array([3.0, 2.0, 1.0])
        y = np.array([True, False, True])
        self.assertAlmostEqual(F.pr_curve(s, y)[2], 0.5 + 0.5 * 2 / 3)

    def test_random_scores_ap_near_prevalence(self) -> None:
        rng = np.random.default_rng(6)
        y = rng.random(20000) < 0.15
        self.assertAlmostEqual(F.pr_curve(rng.random(20000), y)[2],
                               y.mean(), delta=0.02)

    def test_tied_scores_collapse_to_one_point(self) -> None:
        prec, rec, ap = F.pr_curve(np.ones(4), np.array([True, False, True, False]))
        self.assertEqual(len(prec), 1)
        self.assertAlmostEqual(ap, 0.5)


class TestTopKProjection(unittest.TestCase):
    def test_hand_computed(self) -> None:
        z = np.array([[1.0, 2.0, 3.0]])
        dauc = np.array([0.1, -0.4, 0.3])
        # Top-2 by |dAUC| are columns 1 and 2: (2*-0.4 + 3*0.3) / (0.4+0.3)
        got = F.topk_projection(z, dauc, 2)[0]
        self.assertAlmostEqual(got, (2 * -0.4 + 3 * 0.3) / 0.7)

    def test_negative_weight_head_is_used_with_its_sign(self) -> None:
        # one head whose value is LOWER for the positive class: dAUC < 0.
        rng = np.random.default_rng(7)
        y = rng.random(500) < 0.3
        x = rng.normal(size=(500, 1)) - 1.5 * y[:, None]
        d = F.delta_auc_matrix(x, y)
        self.assertLess(d[0], 0)
        self.assertGreater(F.auc(F.topk_projection(F.zscore(x), d, 1), y), 0.8)

    def test_denominator_is_absolute_so_ranking_cannot_flip(self) -> None:
        rng = np.random.default_rng(8)
        y = rng.random(400) < 0.3
        x = rng.normal(size=(400, 20)) + 0.8 * y[:, None] * rng.choice(
            [-1, 1], 20)[None, :]
        z = F.zscore(x)
        d = F.delta_auc_matrix(x, y)
        self.assertGreater(F.auc(F.topk_projection(z, d, 20), y), 0.5)

    def test_sparse_signal_saturates_in_k(self) -> None:
        rng = np.random.default_rng(9)
        n, p, k_true = 1200, 200, 5
        y = rng.random(n) < 0.3
        x = rng.normal(size=(n, p))
        x[:, :k_true] += 1.0 * y[:, None]
        z = F.zscore(x)
        d = F.delta_auc_matrix(x, y)
        a = {k: F.auc(F.topk_projection(z, d, k), y) for k in (1, 5, 20, 200)}
        self.assertGreater(a[5], a[1])
        self.assertGreater(a[20], a[5] - 0.03)    # saturated: no gain past K=5
        self.assertAlmostEqual(F.topk_ranked(d, 5).max(), k_true - 1, delta=0)

    def test_zscore_uses_reference_statistics(self) -> None:
        ref = np.array([[0.0], [2.0]])
        self.assertAlmostEqual(F.zscore(np.array([[4.0]]), ref)[0, 0], 3.0)


class TestFig14Analysis(unittest.TestCase):
    def test_signal_found_held_out_and_null_is_near_chance(self) -> None:
        rng = np.random.default_rng(10)
        n, p = 600, 64
        y = rng.random(n) < 0.3
        x = rng.normal(size=(n, p))
        x[:, :4] += 1.2 * y[:, None]
        r = F.fig14_analysis(x, y, n_splits=3, n_perm=5)
        self.assertGreater(r["top32_auc_held_out"], 0.8)
        self.assertLess(r["null_top32_auc_mean"], 0.7)   # selection bias only
        self.assertGreater(r["top32_average_precision"], r["prevalence"])
        self.assertEqual(len(r["top32_heads"]), min(32, p))

    def test_no_signal_stays_near_chance_held_out(self) -> None:
        rng = np.random.default_rng(11)
        y = rng.random(800) < 0.3
        x = rng.normal(size=(800, 64))
        r = F.fig14_analysis(x, y, n_splits=4, n_perm=3)
        self.assertAlmostEqual(r["top32_auc_held_out"], 0.5, delta=0.06)
        # ... while in-sample is inflated by the selection, which is the point.
        self.assertGreater(r["top32_auc_in_sample"], r["top32_auc_held_out"])


class TestGroupsAndLoading(unittest.TestCase):
    def test_groups(self) -> None:
        st = [("yes", "no"), ("no", "no"), ("yes", "yes"), ("no", "yes")]
        g = F.build_groups(st)
        self.assertEqual(g["halluc"].tolist(), [True, False, False, False])
        self.assertEqual(g["tn"].tolist(), [False, True, False, False])
        self.assertEqual(g["tp"].tolist(), [False, False, True, False])
        self.assertEqual(g["correct"].tolist(), [False, True, True, False])
        self.assertEqual(g["all_rest"].tolist(), [False, True, True, True])

    def test_parse(self) -> None:
        self.assertEqual(F.parse_yes_no(' "Yes."'), "yes")
        self.assertEqual(F.parse_yes_no("No"), "no")
        self.assertIsNone(F.parse_yes_no("maybe"))

    def test_end_to_end_on_synthetic_jsonl(self) -> None:
        rng = np.random.default_rng(12)
        n, L, H = 120, 2, 3
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with open(d / "m-diagnose__x.jsonl", "w") as fd, \
                    open(d / ("m-sampling__b-pope__s-adversarial__seed-42__"
                              "dt-fp16__v-greedy-aaaa.jsonl"), "w") as fl:
                for i in range(n):
                    hal = i % 4 == 0
                    layers = {str(l): {k: rng.random(H).round(5).tolist()
                                       for k in F.STATS} for l in range(L)}
                    fd.write(json.dumps({"sample_id": f"s{i}", "method_extra":
                                         {"diag_layers": layers}}) + "\n")
                    said, truth = ("yes", "no") if hal else (
                        ("no", "no") if i % 2 else ("yes", "yes"))
                    fl.write(json.dumps({"sample_id": f"s{i}", "output": said.title(),
                                         "label": truth}) + "\n")
            rep = F.run(d, d / "out", plots=False)
            self.assertEqual(rep["meta"]["n_samples"], n)
            self.assertEqual(rep["meta"]["n_halluc"], n // 4)
            self.assertTrue((d / "out" / "figures_12_14_r13.json").exists())
            md = (d / "out" / "figures_12_14_r13.md").read_text()
            self.assertIn("Figure 14", md)
            # pure noise: m_V AUC must sit near chance
            a = rep["figure12"]["tail1"]["pools"]["correct"]["auc_halluc_positive"]
            self.assertTrue(0.3 < a < 0.7 and not math.isnan(a))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
