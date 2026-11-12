"""Table 14 / Figure 19: the three rules for Fox's fusion weight.

The paper ablates its own fusion rule and the released code has no flag for
two of the three rows, so this axis is added here. Three things need pinning
and none is visible by reading:

* the DEFAULT must not move. 241 runs already exist under the hybrid rule,
  and this parameter must leave every one of their decisions -- and every one
  of their file names -- exactly where it was. ``strategy="hybrid"`` is
  therefore asserted bit-equal to ``reference_fuse``, the transcription of
  ``vcd_sample.py``, and the default run key is asserted unchanged;
* the two new arms must differ from the hybrid one on the RIGHT tokens.
  ``fixed`` can only differ where the gate fires (``js >= tau_js``) and
  ``dynamic`` only where it does not. An implementation that had the
  comparison backwards would still produce three different numbers, and the
  ablation would silently report them under swapped names;
* the name list in ``storage`` must match the one in ``kernel``. ``kernel``
  imports nothing from this package on purpose, so the two are separate
  tuples, and a value legal to the CLI but not to the kernel would fail 15
  GPU-hours into a run rather than at argument-parse time.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("torch is not installed") from None

from foxrepro.fox.kernel import FUSE_STRATEGIES, fuse  # noqa: E402
from foxrepro.fox.reference import reference_fuse  # noqa: E402
from foxrepro.storage import FUSE_STRATEGIES as STORAGE_STRATEGIES  # noqa: E402

ALPHA, TAU, BETA = 2.0, 0.15, 0.1


def logits(rows: int = 4, vocab: int = 32, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, vocab, generator=g),
            torch.randn(rows, vocab, generator=g))


class TestTheNameListsAgree(unittest.TestCase):
    def test_storage_mirrors_the_kernel(self) -> None:
        self.assertEqual(tuple(STORAGE_STRATEGIES), tuple(FUSE_STRATEGIES))

    def test_hybrid_is_first_so_it_reads_as_the_default(self) -> None:
        self.assertEqual(FUSE_STRATEGIES[0], "hybrid")


class TestHybridIsUnchanged(unittest.TestCase):
    """The released rule, which 241 archived runs were produced under."""

    def test_the_default_is_hybrid(self) -> None:
        z_obs, z_do = logits()
        js = torch.tensor([0.0, 0.1, 0.2, 0.5])
        a, _, lam_a = fuse(z_obs, z_do, js, alpha=ALPHA, tau_js=TAU, beta=BETA)
        b, _, lam_b = fuse(z_obs, z_do, js, alpha=ALPHA, tau_js=TAU,
                           beta=BETA, strategy="hybrid")
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(lam_a, lam_b))

    def test_hybrid_is_bit_equal_to_the_transcribed_release(self) -> None:
        """Both sides of the gate, against ``vcd_sample.py:208-215``."""
        for js_value in (0.0, 0.1499, 0.15, 0.3, 0.6931):
            with self.subTest(js=js_value):
                z_obs, z_do = logits(rows=1)
                got_raw, got_masked, _ = fuse(
                    z_obs, z_do, torch.tensor([js_value]),
                    alpha=ALPHA, tau_js=TAU, beta=BETA, strategy="hybrid")
                ref_raw, ref_masked = reference_fuse(
                    z_obs, z_do, js_value,
                    alpha=ALPHA, tau_js=TAU, beta=BETA)
                self.assertTrue(torch.equal(got_raw, ref_raw))
                self.assertTrue(torch.equal(got_masked, ref_masked))


class TestTheTwoNewArms(unittest.TestCase):
    def setUp(self) -> None:
        self.z_obs, self.z_do = logits()
        # Two rows below tau_js and two above, so every assertion below sees
        # both sides of the gate.
        self.js = torch.tensor([0.02, 0.14, 0.16, 0.60])
        self.below = self.js < TAU
        self.above = ~self.below

    def run_one(self, strategy):
        return fuse(self.z_obs, self.z_do, self.js, alpha=ALPHA,
                    tau_js=TAU, beta=BETA, strategy=strategy)

    def test_fixed_holds_lambda_at_alpha_everywhere(self) -> None:
        _, _, lam = self.run_one("fixed")
        self.assertTrue(torch.allclose(lam, torch.full_like(lam, ALPHA)))

    def test_dynamic_holds_lambda_at_the_jsd_everywhere(self) -> None:
        _, _, lam = self.run_one("dynamic")
        self.assertTrue(torch.allclose(lam, self.js.to(lam.dtype)))

    def test_fixed_differs_from_hybrid_only_where_the_gate_FIRES(self) -> None:
        """Backwards would still give three numbers, under swapped names."""
        _, _, lam_h = self.run_one("hybrid")
        _, _, lam_f = self.run_one("fixed")
        self.assertTrue(torch.equal(lam_h[self.below], lam_f[self.below]),
                        "fixed moved a consensus token, where hybrid already "
                        "uses alpha")
        self.assertFalse(torch.any(lam_h[self.above] == lam_f[self.above]),
                         "fixed left a conflict token alone, where it is the "
                         "only place it can act")

    def test_dynamic_differs_from_hybrid_only_where_the_gate_DOES_NOT(self) -> None:
        _, _, lam_h = self.run_one("hybrid")
        _, _, lam_d = self.run_one("dynamic")
        self.assertTrue(torch.equal(lam_h[self.above], lam_d[self.above]))
        self.assertFalse(torch.any(lam_h[self.below] == lam_d[self.below]))

    def test_every_strategy_keeps_the_same_plausibility_mask(self) -> None:
        """The beta cutoff reads z_obs, which no strategy touches.

        If a strategy changed which tokens survive the mask, the arms would
        differ in their candidate SET as well as their weighting, and Table
        14 would no longer be an ablation of one thing.
        """
        masks = []
        for strategy in FUSE_STRATEGIES:
            _, masked, _ = self.run_one(strategy)
            masks.append(torch.isinf(masked))
        for other in masks[1:]:
            self.assertTrue(torch.equal(masks[0], other))

    def test_an_unknown_strategy_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.run_one("always_on")


class TestItReachesTheKernelThroughGeneration(unittest.TestCase):
    """The parameter is only useful if it survives ``fox_generate``.

    Asserted on the recorded trace rather than on the text: the trace carries
    ``d_t`` and ``lambda_t`` per step, so the rule can be checked against the
    JSD the run actually saw. A parameter accepted at the CLI and dropped on
    the way down would otherwise produce a plausible run under a file name
    promising an arm that never happened -- the failure this project has
    already had once, with ``beta``.
    """

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import transformers  # noqa: F401
        except ImportError:  # pragma: no cover
            raise unittest.SkipTest("transformers is not installed") from None
        from tests.test_tiny_llava import IMAGE_TOKEN_ID, N_IMG, tiny_llava
        from foxrepro.fox import attention as A

        cls.A = A
        cls.model = tiny_llava()
        A.register()
        cls.model.set_attn_implementation(
            {"text_config": A.ATTN_NAME, "vision_config": "sdpa"}
        )
        pre, post = 6, 4
        ids = torch.arange(1, pre + N_IMG + post + 1, dtype=torch.long) % 60
        ids = ids.unsqueeze(0)
        ids[0, pre:pre + N_IMG] = IMAGE_TOKEN_ID
        cls.inputs = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "pixel_values": torch.randn(1, 3, 336, 336),
        }

    def trace_for(self, strategy):
        torch.manual_seed(3)
        out = self.A.fox_generate(
            self.model, dict(self.inputs), max_new_tokens=4,
            mode="as_implemented", alpha=ALPHA, tau_js=TAU, beta=BETA,
            fuse_strategy=strategy, collect_trace=True, seed=0,
            eager_layers=frozenset(range(self.model.config.text_config.num_hidden_layers)),
        )
        trace = out["trace"]
        self.assertTrue(trace, "no steps recorded")
        return trace

    def test_fixed_pins_every_step_to_alpha(self) -> None:
        for step in self.trace_for("fixed"):
            self.assertAlmostEqual(step["lambda_t"], ALPHA, places=5)

    def test_dynamic_follows_the_jsd_at_every_step(self) -> None:
        for step in self.trace_for("dynamic"):
            self.assertAlmostEqual(step["lambda_t"], step["d_t"], places=5)

    def test_hybrid_switches_on_the_threshold(self) -> None:
        for step in self.trace_for("hybrid"):
            want = ALPHA if step["d_t"] < TAU else step["d_t"]
            self.assertAlmostEqual(step["lambda_t"], want, places=5)


class TestThePlumbing(unittest.TestCase):
    """A legal value must survive the whole path, and change the file name."""

    @staticmethod
    def cfg(**kw):
        from foxrepro.bench.generate import GenConfig
        base = dict(method="fox", benchmark="pope", split="adversarial",
                    seed=42, dtype="fp16",
                    model_id="llava-hf/llava-1.5-7b-hf",
                    prompt_template_id="pope_v0_raw", max_new_tokens=1)
        return GenConfig(**{**base, **kw})

    def test_the_default_run_key_is_byte_identical(self) -> None:
        """Adding the field must not rename a single archived run."""
        self.assertNotIn("fuse_strategy",
                         self.cfg()._variant_fields())

    def test_each_strategy_gets_its_own_file(self) -> None:
        names = {self.cfg(fuse_strategy=s).run_key().filename()
                 for s in ("fixed", "dynamic")}
        names.add(self.cfg().run_key().filename())
        self.assertEqual(len(names), 3,
                         "two fusion rules would resume from each other")

    def test_a_non_fox_method_may_not_set_it(self) -> None:
        """Nothing else has a lambda; a file name promising one would lie."""
        with self.assertRaises(ValueError):
            self.cfg(method="sampling", fuse_strategy="fixed")

    def test_an_unknown_value_is_refused_before_the_model_loads(self) -> None:
        with self.assertRaises(ValueError):
            self.cfg(fuse_strategy="always_on")

    def test_the_job_and_the_config_build_the_same_key(self) -> None:
        """plan.Job.run_key must match generate.GenConfig.run_key or the
        plan reports a finished job as pending forever."""
        from foxrepro.bench.plan import Job
        for strategy in ("fixed", "dynamic"):
            with self.subTest(strategy=strategy):
                job = Job("table14", "fox", "pope", "adversarial",
                          fuse_strategy=strategy)
                self.assertEqual(job.run_key().filename(),
                                 self.cfg(fuse_strategy=strategy)
                                 .run_key().filename())

    def test_the_job_passes_the_flag_on_the_command_line(self) -> None:
        from foxrepro.bench.plan import Job
        job = Job("table14", "fox", "chair", "val2014_500",
                  fuse_strategy="dynamic")
        self.assertIn("--fuse-strategy dynamic", job.command())
        self.assertIn("fsdynamic", job.job_id())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
