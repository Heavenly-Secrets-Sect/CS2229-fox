"""InstructBLIP-vicuna-7b, against a real -- but tiny -- InstructBlip.

No download and no GPU. The fixture is a genuine
``InstructBlipForConditionalGeneration`` built from config: real classes, the
real 32 query tokens and the real input layout (32 image ids PREPENDED before
BOS), toy widths. What this file can and cannot establish:

CAN: the loader pins the Q-Former and patches the inner LLaMA; the span is
exactly 32 and starts at key 0; the KV-cached decode loop agrees with a
no-cache recomputation; the Fox patch fires on the configured layers only and
never on the Q-Former; ``use_m_sys=True`` is refused where m_sys is
identically 0; the methods resolve the InstructBLIP copy's hyperparameters.

CANNOT: anything about the real 7B checkpoint -- that its keys load, that the
processor from the Hub emits what the fake emits, or any number a benchmark
would report.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

try:
    import torch
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("torch is not installed") from None

try:
    from transformers import (  # noqa: F401
        InstructBlipConfig, InstructBlipForConditionalGeneration,
        InstructBlipQFormerConfig, InstructBlipVisionConfig, LlamaConfig,
    )
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("transformers is not installed") from None

import foxrepro.fox.attention as A  # noqa: E402
from foxrepro.bench.generate import GenConfig, get_method  # noqa: E402
from foxrepro.data.loaders import Sample  # noqa: E402

IMAGE_TOKEN_ID = 65
N_Q = 32
N_LAYERS = 12
EOS = 2


def tiny_instructblip(seed: int = 0) -> "InstructBlipForConditionalGeneration":
    """Real class, real 32-query geometry, toy widths, 12 decoder layers."""
    v = InstructBlipVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, image_size=224, patch_size=14)
    q = InstructBlipQFormerConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, encoder_hidden_size=32, vocab_size=100)
    t = LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=N_LAYERS,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=70,
        max_position_embeddings=512, eos_token_id=EOS)
    cfg = InstructBlipConfig(
        vision_config=v.to_dict(), qformer_config=q.to_dict(),
        text_config=t.to_dict(), num_query_tokens=N_Q,
        image_token_index=IMAGE_TOKEN_ID)
    A.register()          # must precede construction, as in load_backbone
    torch.manual_seed(seed)
    m = InstructBlipForConditionalGeneration._from_config(
        cfg, attn_implementation={
            "text_config": A.ATTN_NAME, "vision_config": "sdpa",
            "qformer_config": "eager"},
    ).eval()
    # HF initialises query_tokens to zeros; a trained checkpoint's are not.
    m.query_tokens.data.normal_(0, 0.5)
    return m


def tiny_inputs(n_text: int = 6) -> Dict[str, torch.Tensor]:
    """What HF's InstructBlipProcessor writes: image ids, THEN BOS and text."""
    ids = [IMAGE_TOKEN_ID] * N_Q + [1] + [5, 6, 7, 8, 9][: n_text - 1]
    torch.manual_seed(7)
    return {
        "input_ids": torch.tensor([ids]),
        "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        "pixel_values": torch.randn(1, 3, 224, 224),
        "qformer_input_ids": torch.tensor([[3, 4, 5, 6]]),
        "qformer_attention_mask": torch.ones(1, 4, dtype=torch.long),
    }


class _Counting:
    """Wrap ``A.intervene`` to record which layers it ran on."""

    def __init__(self) -> None:
        self.layers: List[int] = []
        self.use_m_sys: List[bool] = []
        self._orig = A.intervene

    def __enter__(self):
        def wrapped(*a: Any, **kw: Any):
            self.layers.append(kw.get("layer"))
            self.use_m_sys.append(kw.get("use_m_sys"))
            return self._orig(*a, **kw)
        A.intervene = wrapped  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: Any) -> None:
        A.intervene = self._orig  # type: ignore[assignment]


class TestSpan(unittest.TestCase):
    def test_width_is_32_and_starts_at_zero(self) -> None:
        m = tiny_instructblip()
        self.assertEqual(A.expected_span_width(m), 32)
        self.assertEqual(A.resolve_image_token_id(m), IMAGE_TOKEN_ID)
        ids = tiny_inputs()["input_ids"]
        self.assertEqual(
            A.locate_image_span(ids, IMAGE_TOKEN_ID, A.expected_span_width(m)),
            (0, 31))

    def test_a_short_span_is_refused(self) -> None:
        m = tiny_instructblip()
        ids = tiny_inputs()["input_ids"].clone()
        ids[0, 31] = 9
        with self.assertRaises(ValueError):
            A.locate_image_span(ids, IMAGE_TOKEN_ID, A.expected_span_width(m))


class TestLoader(unittest.TestCase):
    def test_the_attention_dict_pins_all_three_towers(self) -> None:
        seen: Dict[str, Any] = {}

        def fake_from_pretrained(model_id: str, **kw: Any):
            seen.update(kw)
            return tiny_instructblip(), {"missing_keys": []}

        cfg = tiny_instructblip().config
        with mock.patch("transformers.AutoConfig.from_pretrained",
                        return_value=cfg), \
                mock.patch("transformers.AutoModelForImageTextToText"
                           ".from_pretrained", side_effect=fake_from_pretrained), \
                mock.patch("transformers.AutoProcessor.from_pretrained",
                           return_value=object()):
            A.load_backbone("someone/instructblip", dtype="fp32",
                            allow_cpu=True)
        self.assertEqual(seen["attn_implementation"], {
            "text_config": A.ATTN_NAME, "vision_config": "sdpa",
            "qformer_config": "eager"})

    def test_a_real_round_trip_patches_the_decoder_and_not_the_qformer(self) -> None:
        """save_pretrained -> load_backbone -> real from_pretrained.

        MEASURED on transformers 4.57.1: ``from_pretrained`` with the dict
        leaves InstructBLIP's inner LLaMA on ``sdpa`` (the wrapper builds it
        from ``config.text_config`` after the dict is gone), so
        ``load_backbone`` re-applies it with ``set_attn_implementation`` and
        verifies. Without that, every other assertion here and in a smoke run
        that only reads the dict would pass on an unpatched model.

        ``device_map`` is dropped for the call because it needs ``accelerate``,
        which this host may lack; sharding is the one thing not exercised.
        """
        from transformers import AutoModelForImageTextToText
        real = AutoModelForImageTextToText.from_pretrained

        def no_device_map(model_id: str, **kw: Any):
            kw.pop("device_map", None)
            return real(model_id, **kw)

        with tempfile.TemporaryDirectory() as td:
            tiny_instructblip().save_pretrained(td)
            with mock.patch("transformers.AutoProcessor.from_pretrained",
                            return_value=object()), \
                    mock.patch("transformers.AutoModelForImageTextToText"
                               ".from_pretrained", side_effect=no_device_map):
                model, _ = A.load_backbone(td, dtype="fp32", allow_cpu=True)
        # The patch must FIRE on the loaded model, not merely be configured.
        with _Counting() as c:
            A.fox_generate(model, tiny_inputs(), max_new_tokens=2, seed=1,
                           eager_layers=frozenset(range(4, 11)),
                           use_m_sys=False)
        self.assertEqual(set(c.layers), set(range(4, 11)))
        self.assertEqual(type(model).__name__,
                         "InstructBlipForConditionalGeneration")
        self.assertEqual(
            model.language_model.config._attn_implementation, A.ATTN_NAME)
        self.assertEqual(model.qformer.config._attn_implementation, "eager")
        self.assertEqual(
            model.vision_model.config._attn_implementation, "sdpa")
        # ...and the patch REACHES the inner LlamaForCausalLM's layers.
        layers = model.language_model.model.layers
        self.assertEqual(len(layers), N_LAYERS)
        self.assertTrue(all(
            l.self_attn.config._attn_implementation == A.ATTN_NAME
            for l in layers))

    def test_an_unpatchable_decoder_is_refused(self) -> None:
        class Stub:
            def __init__(self) -> None:
                sub = mock.Mock()
                sub._attn_implementation = "sdpa"
                self.config = mock.Mock(text_config=sub)

            def set_attn_implementation(self, impl: Any) -> None:
                pass  # silently does nothing, as an unsupported class would

        with self.assertRaisesRegex(RuntimeError, "not running"):
            A._ensure_decoder_is_patched(Stub(), {"text_config": A.ATTN_NAME},
                                         "text_config")

    def test_missing_qformer_keys_are_a_hard_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Q-Former"):
            A._assert_fully_loaded(
                "x/instructblip",
                {"missing_keys": ["qformer.encoder.layer.0.attention.q.weight",
                                  "query_tokens"]})
        with self.assertRaises(RuntimeError):
            A._assert_fully_loaded("x/instructblip",
                                   {"missing_keys": ["lm_head.weight"]})
        A._assert_fully_loaded("x/instructblip", {"missing_keys": []})


class TestGenerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = tiny_instructblip()
        A.register()

    def _gen(self, **kw: Any) -> Dict[str, Any]:
        return A.fox_generate(
            self.model, tiny_inputs(), max_new_tokens=4, seed=3,
            eager_layers=frozenset(range(4, 11)), **kw)

    def test_kv_cached_greedy_equals_full_recomputation(self) -> None:
        """The cached loop calls ``language_model`` after the prefill; the
        wrapper's own forward re-encodes the image every call. They must agree
        token for token or the cache path is wrong."""
        got = self._gen(single_branch=True, greedy=True)["token_ids"]
        inp = tiny_inputs()
        ids = inp["input_ids"]
        want: List[int] = []
        with torch.no_grad():
            for _ in range(len(got)):
                out = self.model(
                    pixel_values=inp["pixel_values"],
                    qformer_input_ids=inp["qformer_input_ids"],
                    qformer_attention_mask=inp["qformer_attention_mask"],
                    input_ids=ids, attention_mask=torch.ones_like(ids))
                tok = int(out.logits[0, -1].argmax())
                want.append(tok)
                ids = torch.cat([ids, torch.tensor([[tok]])], dim=1)
        self.assertEqual(got, want)

    def test_fox_fires_on_the_window_only_with_h_vis_scoring(self) -> None:
        with _Counting() as c:
            out = self._gen(use_m_sys=False)
        self.assertEqual(out["image_span"], (0, 31))
        self.assertGreater(out["n_gen_tokens"], 0)
        self.assertEqual(set(c.layers), set(range(4, 11)))
        self.assertEqual(set(c.use_m_sys), {False})
        # One call per layer per forward: prefill plus every decode step.
        self.assertEqual(len(c.layers), 7 * out["n_gen_tokens"])

    def test_sampling_never_calls_the_intervention(self) -> None:
        with _Counting() as c:
            self._gen(single_branch=True)
        self.assertEqual(c.layers, [])

    def test_m_sys_scoring_is_refused_where_m_sys_is_zero(self) -> None:
        with self.assertRaisesRegex(ValueError, "use_m_sys"):
            self._gen(use_m_sys=True)

    def test_the_other_baselines_are_refused_not_ignored(self) -> None:
        with self.assertRaises(ValueError):
            self._gen(sid=True, use_m_sys=False)

    def test_m_sys_is_zero_and_the_masses_partition(self) -> None:
        inp = tiny_inputs()
        with A.FOX.configured(img_start=0, img_end=31, enabled=False,
                              diagnose=True, collect=True,
                              eager_layers=frozenset()):
            A.FOX.reset_diagnostics()
            with torch.no_grad():
                self.model(**inp)
            diags = {k: {kk: vv.clone() for kk, vv in v.items()}
                     for k, v in A.FOX.diagnostics.items()}
        self.assertEqual(len(diags), N_LAYERS)
        for layer, row in diags.items():
            ms, mv, mt = (float(row[f"m_{n}_t1"].mean())
                          for n in ("sys", "vis", "txt"))
            self.assertEqual(ms, 0.0, f"layer {layer}: no keys precede the image")
            self.assertAlmostEqual(ms + mv + mt, 1.0, places=5)


class _FakeProcessor:
    def __init__(self) -> None:
        self.seen_text: List[str] = []

        class _Tok:
            @staticmethod
            def decode(ids, skip_special_tokens=True):
                return " ".join(f"t{i}" for i in ids if i != EOS)
        self.tokenizer = _Tok()

    def __call__(self, images: Any, text: str, return_tensors: str = "pt"):
        self.seen_text.append(text)
        return tiny_inputs()


class TestMethods(unittest.TestCase):
    def setUp(self) -> None:
        import foxrepro.bench.methods as M
        self.M = M
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        img = Path(td.name) / "COCO_val2014_000000000042.jpg"
        img.write_bytes(b"x")
        self.sample = Sample(
            sample_id="pope/random/q0000001", image_id=42, image_path=img,
            benchmark="pope", split="random",
            question="Is there a dog in the image?", label="no", extra={})
        self.model = tiny_instructblip()
        self.processor = _FakeProcessor()
        A.register()
        orig_load = A.load_llava
        A.load_llava = lambda *a, **k: (self.model, self.processor)  # type: ignore
        self.addCleanup(lambda: setattr(A, "load_llava", orig_load))
        orig_enc = M._LlavaMethod._encode
        M._LlavaMethod._encode = (  # type: ignore[assignment]
            lambda s, sample, text: s.processor(images=None, text=text))
        self.addCleanup(lambda: setattr(M._LlavaMethod, "_encode", orig_enc))

    def _cfg(self, method: str, **over: Any) -> GenConfig:
        base: Dict[str, Any] = dict(
            method=method, benchmark="pope", split="random", seed=42,
            dtype="fp32", model_id="fake/instructblip",
            prompt_template_id="pope_v0_raw", max_new_tokens=3,
            system_prompt_id="none", allow_cpu=True)
        base.update(over)
        return GenConfig(**base)

    def _run(self, method: str, **over: Any):
        m = get_method(method)(self._cfg(method, **over))
        m.load()
        return m, m.generate(self.sample, self.sample.question)

    def test_the_prompt_is_the_releases_with_two_spaces_and_no_system(self) -> None:
        self._run("sampling_instructblip")
        self.assertEqual(
            self.processor.seen_text,
            ["USER:  Is there a dog in the image? ASSISTANT:"])

    def test_the_llava_system_prompt_is_refused(self) -> None:
        for name in ("sampling_instructblip", "fox_instructblip"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                get_method(name)(self._cfg(name, system_prompt_id="llava_v1"))

    def test_fox_resolves_the_instructblip_copys_hyperparameters(self) -> None:
        m, out = self._run("fox_instructblip")
        d = m.describe()
        r = d["resolved_hyperparameters"]
        self.assertEqual(r["ratio"], 0.4)
        self.assertEqual(r["gamma"], 10000.0)
        self.assertAlmostEqual(r["tau_js"], 0.15)
        self.assertEqual(r["alpha"], 2.0)
        self.assertIs(r["use_m_sys"], False)
        self.assertIs(d["use_m_sys"], False)
        self.assertEqual(d["eager_layers"], list(range(4, 11)))
        self.assertEqual(d["backbone"], "instructblip")
        self.assertEqual(out.extra["image_span"], [0, 31])

    def test_explicit_values_still_override(self) -> None:
        m, _ = self._run("fox_instructblip", k=0.5, intervention_layers="2-13")
        self.assertEqual(m.describe()["resolved_hyperparameters"]["ratio"], 0.5)
        self.assertEqual(m.describe()["eager_layers"], list(range(2, 14)))

    def test_llava_fox_is_unchanged(self) -> None:
        from foxrepro.bench.methods import FoxMethod, FOX_DEFAULTS
        self.assertIs(FoxMethod.HPARAM_DEFAULTS, FOX_DEFAULTS)
        self.assertIsNone(FoxMethod.LAYER_DEFAULT)
        m = FoxMethod(self._cfg("fox", system_prompt_id="llava_v1"))
        self.assertNotIn("use_m_sys", m._decode_kwargs())

    def test_sampling_takes_one_row_and_fox_two(self) -> None:
        rows: List[int] = []
        orig = self.model.language_model.forward

        def spy(*a: Any, **kw: Any):
            e = kw.get("inputs_embeds")
            if e is None:
                e = kw["input_ids"]
            rows.append(int(e.shape[0]))
            return orig(*a, **kw)
        self.model.language_model.forward = spy  # type: ignore[method-assign]
        self._run("sampling_instructblip")
        self.assertEqual(set(rows), {1})
        rows.clear()
        self._run("fox_instructblip")
        self.assertEqual(set(rows), {2})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
