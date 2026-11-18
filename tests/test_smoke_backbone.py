"""The pre-flight check for a new backbone, checked itself.

`scripts/smoke_backbone.py` exists to be run once on Kaggle before a round
commits twenty GPU-hours to a backbone nobody has executed here. Its value
is entirely in whether it FAILS when the image span is wrong, and an
assertion that has never failed is not evidence of anything -- this project
has now shipped two sessions killed by a script that parsed and did the
wrong thing.

So the two cases that matter are driven against the tiny LLaVA the rest of
the suite uses, with no download:

* a correct span passes every check;
* a span shifted off the image, or one token short, is REJECTED.

The shifted case is the one worth having, and it has already earned its
keep. The script first checked `m_sys + m_vis + m_txt == 1` and called that
proof the span sat on the image. The shifted-span test below failed to
fail, which showed the check was an identity: those ranges are
``[0, start)``, ``[start, end)`` and ``[end, n)``, so they tile the key
axis for any boundaries whatsoever. The real check compares the span to the
positions the processor wrote the image token at, which is exact.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("torch is not installed") from None

try:
    from transformers import (  # noqa: F401
        CLIPVisionConfig, LlamaConfig, LlavaConfig,
        LlavaForConditionalGeneration,
    )
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("transformers is not installed") from None

import foxrepro.fox.attention as A  # noqa: E402
from foxrepro.fox.attention import FOX  # noqa: E402

from tests.test_tiny_llava import IMAGE_TOKEN_ID, N_IMG, tiny_llava  # noqa: E402

_SRC = Path(__file__).resolve().parents[1] / "scripts" / "smoke_backbone.py"


def _load_script():
    """Import the script by path; `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location("smoke_backbone", _SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SMOKE = _load_script()


class _Diag(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = tiny_llava()
        # The diagnostics only exist if the decoder is running Fox's own
        # attention; a stock SDPA decoder reports nothing and every check
        # below would pass vacuously.
        A.register()
        cls.model.set_attn_implementation(
            {"text_config": A.ATTN_NAME, "vision_config": "sdpa"})
        torch.manual_seed(3)
        cls.px = torch.randn(1, 3, 336, 336)
        # 6 text tokens, the image, then 4 more. Ids avoid the image token.
        pre, post = 6, 4
        ids = ([1] * pre + [IMAGE_TOKEN_ID] * N_IMG + [2] * post)
        cls.ids = torch.tensor([ids], dtype=torch.long)
        cls.start, cls.end = pre, pre + N_IMG

    def diagnostics(self, start: int, end: int) -> dict:
        with FOX.configured(img_start=start, img_end=end, enabled=False,
                            diagnose=True, collect=True):
            FOX.reset_diagnostics()
            with torch.no_grad():
                self.model(input_ids=self.ids, pixel_values=self.px)
            return {k: {kk: vv.clone() for kk, vv in v.items()}
                    for k, v in FOX.diagnostics.items()}


class TestACorrectSpanPasses(_Diag):
    def test_every_layer_reports(self) -> None:
        diags = self.diagnostics(self.start, self.end)
        self.assertEqual(len(diags),
                         self.model.config.text_config.num_hidden_layers)

    def test_the_span_matches_the_image_tokens(self) -> None:
        width = SMOKE.check_span_matches_tokens(
            self.ids, IMAGE_TOKEN_ID, self.start, self.end)
        self.assertEqual(width, N_IMG)

    def test_the_span_carries_attention(self) -> None:
        diags = self.diagnostics(self.start, self.end)
        self.assertGreater(SMOKE.check_diagnostics_usable(diags), 0.0)


class TestAShiftedSpanIsCaught(_Diag):
    """Contiguous, in range, finite everywhere -- and wrong."""

    def test_a_span_shifted_off_the_image_raises(self) -> None:
        with self.assertRaises(AssertionError) as ctx:
            SMOKE.check_span_matches_tokens(
                self.ids, IMAGE_TOKEN_ID, self.start + 40, self.end + 40)
        self.assertIn("image tokens occupy", str(ctx.exception))

    def test_a_span_one_token_short_is_caught(self) -> None:
        """The realistic dynamic-backbone failure: off by part of a tile."""
        with self.assertRaises(AssertionError):
            SMOKE.check_span_matches_tokens(
                self.ids, IMAGE_TOKEN_ID, self.start, self.end - 1)

    def test_a_prompt_with_no_image_token_raises(self) -> None:
        with self.assertRaises(AssertionError) as ctx:
            SMOKE.check_span_matches_tokens(
                torch.tensor([[1, 2, 3]]), IMAGE_TOKEN_ID, 0, 0)
        self.assertIn("no token", str(ctx.exception))

    def test_the_mass_sum_would_NOT_have_caught_any_of_this(self) -> None:
        """Why the first version of the check was worthless, pinned.

        If this ever starts failing, the three ranges have stopped tiling
        the key axis and the sum has become informative -- at which point
        this test is the record of why it was not.
        """
        diags = self.diagnostics(self.start + 40, self.end + 40)
        for row in diags.values():
            for tail in sorted({k.rsplit("_t", 1)[1] for k in row}):
                total = sum(float(row[f"{n}_t{tail}"].mean())
                            for n in ("m_sys", "m_vis", "m_txt"))
                self.assertAlmostEqual(total, 1.0, places=2)

    def test_a_shifted_span_is_otherwise_indistinguishable(self) -> None:
        """Nothing else here would notice, which is why the check exists."""
        diags = self.diagnostics(self.start + 40, self.end + 40)
        self.assertEqual(len(diags),
                         self.model.config.text_config.num_hidden_layers)
        for row in diags.values():
            for v in row.values():
                self.assertTrue(torch.isfinite(v).all())


class TestTheCandidateIdsAreTheVerifiedOnes(unittest.TestCase):
    """`OpenGVLab/InternVL3-8B` is internvl_chat and load_backbone refuses
    it; the `-hf` repo is what transformers can build. Pinning the exact
    strings because getting this wrong costs a session, not a test run."""

    def test_candidates(self) -> None:
        self.assertEqual(SMOKE.CANDIDATES, [
            "Qwen/Qwen3-VL-8B-Instruct",
            "OpenGVLab/InternVL3-8B-hf",
        ])

    def test_internvl_without_the_hf_suffix_is_not_listed(self) -> None:
        self.assertNotIn("OpenGVLab/InternVL3-8B", SMOKE.CANDIDATES)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
