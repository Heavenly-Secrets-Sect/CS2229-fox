"""Generalising the loader off LLaVA-1.5, and the guards that must survive it.

`scripts/probe_backbone.py` reported Qwen3-VL-8B and InternVL3-8B as
runnable with the Fox patch reaching the decoder, which is what made this
worth doing. What it could not report is the one thing that differs:

    LLaVA-1.5   336/14 = 24x24                 -> 576 image tokens, FIXED
    Qwen3-VL    patch 16, spatial_merge 2x2    -> dynamic with resolution
    InternVL3   448/14 tiles, tiled dynamically-> dynamic with tile count

So the 576 check cannot stay unconditional. It was never logic -- the span
has always been derived from `input_ids` -- but it is a real guard for
LLaVA, where a wrong width means the processor took the pre-4.44 legacy
path and every key index is silently wrong.

Four things need pinning:

* LLaVA keeps its check. Dropping it to make room for other backbones would
  trade a loud failure for a silent one on the backbone that produced every
  number in this repo;
* a dynamic backbone must be accepted WITHOUT a width, and still refuse a
  span that is not contiguous -- taking first and last positions across two
  images would measure m_sys over a span that is partly text;
* the image-token id must be read under either config name. LLaVA calls it
  `image_token_index`, Qwen3-VL and InternVL3 call it `image_token_id`, and
  reading only the first turns a port into an AttributeError three layers
  down;
* an unknown backbone must be REFUSED rather than guessed at. Fox patches
  the text tower only, so a wrong sub-config key would silently patch
  nothing, or patch the vision tower and change the features themselves.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("torch is not installed") from None

from foxrepro.fox.attention import (  # noqa: E402
    FIXED_SPAN_WIDTH, _DECODER_SUBCONFIG, expected_span_width,
    locate_image_span, resolve_image_token_id,
)


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Model:
    def __init__(self, **kw):
        self.config = _Cfg(**kw)


def ids(span_width: int, image_id: int = 32000, pre: int = 6, post: int = 4,
        gap: int = 0):
    """Prompt ids with one image span, optionally split into two blocks."""
    out = list(range(1, pre + 1))
    if gap:
        half = span_width // 2
        out += [image_id] * half + [999] * gap + [image_id] * (span_width - half)
    else:
        out += [image_id] * span_width
    out += list(range(1, post + 1))
    return torch.tensor([out], dtype=torch.long)


class TestTheFixedWidthGuardSurvives(unittest.TestCase):
    """LLaVA produced every number in this repo; its check stays loud."""

    def test_only_llava_instructblip_and_shikra_have_a_fixed_width(self) -> None:
        # InstructBLIP: the Q-Former emits num_query_tokens = 32 whatever the
        # image (tests/test_instructblip.py). Qwen3-VL and InternVL3 are
        # dynamic and must stay absent.
        self.assertEqual(FIXED_SPAN_WIDTH,
                         {"llava": 576, "instructblip": 32, "shikra": 256})

    def test_a_llava_model_reports_576(self) -> None:
        self.assertEqual(expected_span_width(_Model(model_type="llava")), 576)

    def test_a_dynamic_backbone_reports_none(self) -> None:
        for mt in ("qwen3_vl", "internvl"):
            with self.subTest(model_type=mt):
                self.assertIsNone(expected_span_width(_Model(model_type=mt)))

    def test_the_default_still_asserts_576(self) -> None:
        """Every pre-existing caller keeps the check it had."""
        with self.assertRaises(ValueError) as ctx:
            locate_image_span(ids(575), 32000)
        self.assertIn("575", str(ctx.exception))

    def test_a_correct_llava_span_passes(self) -> None:
        self.assertEqual(locate_image_span(ids(576), 32000), (6, 581))


class TestADynamicBackbone(unittest.TestCase):
    def test_any_width_is_accepted_without_an_expectation(self) -> None:
        for w in (64, 729, 1024, 3136):
            with self.subTest(width=w):
                start, end = locate_image_span(ids(w, image_id=151655), 151655,
                                               expected_width=None)
                self.assertEqual((start, end), (6, 5 + w))

    def test_a_non_contiguous_span_is_still_refused(self) -> None:
        """Two images would make m_sys a mix of image and text keys."""
        with self.assertRaises(ValueError) as ctx:
            locate_image_span(ids(64, image_id=151655, gap=5), 151655,
                              expected_width=None)
        self.assertIn("contiguous", str(ctx.exception))

    def test_contiguity_is_checked_for_llava_too(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            locate_image_span(ids(576, gap=3), 32000)
        self.assertIn("contiguous", str(ctx.exception))

    def test_no_image_token_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            locate_image_span(torch.tensor([[1, 2, 3]]), 32000,
                              expected_width=None)


class TestTheImageTokenIdIsReadUnderEitherName(unittest.TestCase):
    def test_llava_name(self) -> None:
        self.assertEqual(
            resolve_image_token_id(_Model(image_token_index=32000)), 32000)

    def test_qwen_and_internvl_name(self) -> None:
        self.assertEqual(
            resolve_image_token_id(_Model(image_token_id=151655)), 151655)

    def test_the_llava_name_wins_when_both_are_present(self) -> None:
        """Order is fixed so a config carrying both cannot flip behaviour."""
        self.assertEqual(
            resolve_image_token_id(
                _Model(image_token_index=32000, image_token_id=151655)), 32000)

    def test_a_config_with_neither_is_refused_by_name(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            resolve_image_token_id(_Model(model_type="mystery"))
        self.assertIn("mystery", str(ctx.exception))


class TestAnUnknownBackboneIsRefused(unittest.TestCase):
    """A guessed sub-config key patches nothing, or patches the wrong tower."""

    def test_the_known_layouts_are_the_probed_ones(self) -> None:
        self.assertEqual(sorted(_DECODER_SUBCONFIG),
                         ["instructblip", "internvl", "llava", "qwen3_vl"])

    def test_every_layout_patches_text_and_leaves_vision_on_sdpa(self) -> None:
        for mt, (text, vision) in _DECODER_SUBCONFIG.items():
            with self.subTest(model_type=mt):
                self.assertEqual(text, "text_config")
                self.assertEqual(vision, "vision_config")
                self.assertNotEqual(text, vision)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class TestDecodeEntryPointsRunWithoutAutograd(unittest.TestCase):
    """fox_generate and fox_diagnose must be wrapped by torch.inference_mode.

    A refactor once moved the decorator onto a helper defined between it and
    the function, so fox_generate ran with autograd enabled. Outputs were
    identical; memory grew ~1.26 GiB per CausalMM sample until the GPU
    filled, and every dual-pass job from r24 on died of it."""

    def test_the_entry_points_are_wrapped(self) -> None:
        from foxrepro.fox import attention as attn
        for name in ("fox_generate", "fox_diagnose"):
            with self.subTest(fn=name):
                self.assertTrue(
                    hasattr(getattr(attn, name), "__wrapped__"),
                    f"{name} is not under torch.inference_mode")

    def test_helpers_between_are_not_wrapped(self) -> None:
        from foxrepro.fox import attention as attn
        for name in ("_is_instructblip", "_instructblip_step"):
            with self.subTest(fn=name):
                self.assertFalse(hasattr(getattr(attn, name), "__wrapped__"))

    def test_generate_runs_in_inference_mode(self) -> None:
        """Behavioural: inside the wrapped function inference mode is on."""
        import torch
        from foxrepro.fox import attention as attn
        seen = []

        def probe(*a, **k):
            seen.append(torch.is_inference_mode_enabled())
            raise RuntimeError("stop")

        # Reach into the wrapper: call the decorated function with a model
        # whose first use records the mode, then aborts.
        class M:
            config = type("C", (), {"model_type": "llava"})()
            def __getattr__(self, n):
                seen.append(torch.is_inference_mode_enabled())
                raise RuntimeError("stop")

        try:
            attn.fox_generate(M(), {"input_ids": torch.zeros(1, 4, dtype=torch.long)})
        except Exception:
            pass
        self.assertTrue(seen and all(seen), seen)
