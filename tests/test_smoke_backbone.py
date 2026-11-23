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

    def test_instructblip_is_opt_in_and_is_the_private_fp16_copy(self) -> None:
        self.assertEqual(SMOKE.INSTRUCTBLIP_CANDIDATE,
                         "Salesforce/instructblip-vicuna-7b")
        self.assertNotIn(SMOKE.INSTRUCTBLIP_CANDIDATE, SMOKE.CANDIDATES)


class _IBProcessor:
    """Stands in for InstructBlipProcessor: the layout it really writes."""

    def __init__(self, shift: int = 0, short: bool = False) -> None:
        from tests.test_instructblip import EOS, tiny_inputs
        self._inputs = tiny_inputs
        self.seen: list = []
        self.shift, self.short = shift, short

        class _Tok:
            @staticmethod
            def decode(ids, skip_special_tokens=True):
                return " ".join(f"t{i}" for i in ids if i != EOS)
        self.tokenizer = _Tok()

    def __call__(self, images, text, return_tensors="pt"):
        self.seen.append(text)
        b = self._inputs()
        if self.shift:      # a text token in front: image no longer at key 0
            ids = b["input_ids"]
            b["input_ids"] = torch.cat([torch.tensor([[1] * self.shift]), ids], 1)
            b["attention_mask"] = torch.ones_like(b["input_ids"])
        if self.short:      # 31 image tokens
            b["input_ids"][0, 31] = 9
        return b


class TestTheInstructBlipEntry(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from tests.test_instructblip import tiny_instructblip
        cls.model = tiny_instructblip()

    def test_every_check_passes_on_a_correct_layout(self) -> None:
        proc = _IBProcessor()
        r = SMOKE.smoke_instructblip_loaded(self.model, proc, n_samples=3)
        self.assertTrue(r["ok"], r)
        self.assertEqual((r["img_start"], r["img_end"], r["derived_width"]),
                         (0, 31, 32))
        for name in ("span_exactly_32_contiguous_from_0",
                     "patched_forward_fires_on_4_to_10", "qformer_not_patched",
                     "query_tokens_nonzero",
                     "m_sys_zero_and_masses_partition",
                     "mode_off_bit_identical_to_baseline",
                     "gamma0_same_argmax", "intervention_generates"):
            self.assertTrue(r["checks"].get(name), name)
        # The release prompt, with the placeholder's double space.
        self.assertEqual(proc.seen,
                         ["USER:  Is there a dog in the image? ASSISTANT:"])
        # Measured, not asserted to be zero; recorded for the report.
        self.assertIn("gamma0_max_abs_diff", r)

    def test_a_short_span_is_rejected(self) -> None:
        with self.assertRaises(ValueError):          # locate_image_span
            SMOKE.smoke_instructblip_loaded(
                self.model, _IBProcessor(short=True))

    def test_an_image_not_at_key_zero_is_rejected(self) -> None:
        with self.assertRaisesRegex(AssertionError, "not 0"):
            SMOKE.smoke_instructblip_loaded(
                self.model, _IBProcessor(shift=2))

    def test_an_unpatched_decoder_is_rejected(self) -> None:
        """The failure the config string cannot show: attn says fox, runs sdpa."""
        m = self.model
        sub = m.language_model.config
        before = sub._attn_implementation_internal
        sub._attn_implementation_internal = "sdpa"
        try:
            with self.assertRaises(AssertionError):
                SMOKE.smoke_instructblip_loaded(m, _IBProcessor())
        finally:
            sub._attn_implementation_internal = before

    def test_zeroed_query_tokens_are_rejected(self) -> None:
        saved = self.model.query_tokens.data.clone()
        self.model.query_tokens.data.zero_()
        try:
            with self.assertRaisesRegex(AssertionError, "query_tokens"):
                SMOKE.smoke_instructblip_loaded(self.model, _IBProcessor())
        finally:
            self.model.query_tokens.data.copy_(saved)

    def test_a_patched_qformer_is_rejected(self) -> None:
        q = self.model.qformer.config
        before = q._attn_implementation_internal
        q._attn_implementation_internal = A.ATTN_NAME
        try:
            with self.assertRaisesRegex(AssertionError, "Q-Former"):
                SMOKE.smoke_instructblip_loaded(self.model, _IBProcessor())
        finally:
            q._attn_implementation_internal = before


class TestTheLlavaSmokeEndToEnd(unittest.TestCase):
    """``smoke_loaded`` had never been run as a whole; its helpers were tested
    with hand-built exclusive ends while it passed ``locate_image_span``'s
    INCLUSIVE end straight in, which would have failed on a real backbone."""

    def test_a_correct_llava_passes_the_whole_script(self) -> None:
        model = tiny_llava()
        A.register()
        model.set_attn_implementation(
            {"text_config": A.ATTN_NAME, "vision_config": "sdpa"})
        px = torch.randn(1, 3, 336, 336)
        ids = torch.tensor([[1] * 6 + [IMAGE_TOKEN_ID] * N_IMG + [2] * 4])

        class Proc:
            class tokenizer:                                    # noqa: N801
                pass

            def apply_chat_template(self, msgs, tokenize=False,
                                    add_generation_prompt=True):
                return "x"

            def __call__(self, images, text, return_tensors="pt"):
                return {"input_ids": ids,
                        "attention_mask": torch.ones_like(ids),
                        "pixel_values": px}

            def decode(self, t, skip_special_tokens=True):
                return "ok"

        r = SMOKE.smoke_loaded(model, Proc(), n_samples=2)
        self.assertTrue(r["ok"])
        self.assertEqual((r["img_start"], r["img_end"]), (6, 6 + N_IMG - 1))
        self.assertEqual(r["derived_width"], N_IMG)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
