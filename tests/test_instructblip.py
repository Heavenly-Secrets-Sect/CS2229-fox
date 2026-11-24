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
                mock.patch("transformers.InstructBlipProcessor.from_pretrained",
                           return_value=mock.Mock(num_query_tokens=32)):
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
            with mock.patch("transformers.InstructBlipProcessor.from_pretrained",
                            return_value=mock.Mock(num_query_tokens=32)), \
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
        with self.assertRaises(ValueError):
            self._gen(cf_vision=True, use_m_sys=False)

    def _spy_embeds(self) -> List[torch.Tensor]:
        """Record every ``inputs_embeds`` the decoder is prefilled with."""
        seen: List[torch.Tensor] = []
        orig = self.model.language_model.forward

        def spy(*a: Any, **kw: Any):
            if kw.get("inputs_embeds") is not None:
                seen.append(kw["inputs_embeds"].detach().clone())
            return orig(*a, **kw)
        self.model.language_model.forward = spy  # type: ignore[method-assign]
        self.addCleanup(
            lambda: setattr(self.model.language_model, "forward", orig))
        return seen

    def _contrastive(self, **kw: Any) -> Dict[str, Any]:
        return A.fox_generate(
            self.model, tiny_inputs(), max_new_tokens=4, seed=3, mode="off",
            fusion="contrastive", greedy=True, use_m_sys=False,
            eager_layers=frozenset(), **kw)

    def test_vcd_noised_image_reaches_row_1_through_vit_and_qformer(self) -> None:
        seen = self._spy_embeds()
        # The toy ViT's patch embedding is near zero, so the image barely
        # moves its output; scale it up so the test can see an image change.
        w = self.model.vision_model.embeddings.patch_embedding.weight
        saved = w.data.clone()
        w.data.mul_(300.0)
        self.addCleanup(lambda: w.data.copy_(saved))
        px = tiny_inputs()["pixel_values"]
        noised = px + 0.5 * torch.randn_like(px)
        out = self._contrastive(pixel_values_do=noised)
        pre = seen[0]  # the batched prefill: [2, L, d]
        self.assertEqual(pre.shape[0], 2)
        # The 32 Q-Former tokens differ between rows; text embeddings do not.
        self.assertFalse(torch.allclose(pre[0, :32], pre[1, :32]))
        self.assertTrue(torch.equal(pre[0, 32:], pre[1, 32:]))
        self.assertEqual(out["image_span"], (0, 31))

    def test_vcd_with_an_identical_negative_image_is_the_plain_sample(self) -> None:
        """(1+a) z - a z = z: a wrongly wired row 1 would break this."""
        px = tiny_inputs()["pixel_values"]
        got = self._contrastive(pixel_values_do=px.clone())["token_ids"]
        want = self._gen(single_branch=True, greedy=True)["token_ids"]
        self.assertEqual(got, want)

    def _neg_text(self) -> Dict[str, torch.Tensor]:
        inp = tiny_inputs()
        ids = inp["input_ids"].clone()
        ids[0, 33:] = torch.tensor([11, 12, 13, 14, 15])[: ids.shape[1] - 33]
        return {
            "input_ids_do": ids,
            "attn_mask_do": torch.ones_like(ids),
            "qformer_input_ids_do": torch.tensor([[3, 9, 8, 7, 6]]),
            "qformer_attn_mask_do": torch.ones(1, 5, dtype=torch.long),
        }

    def test_icd_negative_prompt_reaches_the_qformer_and_the_decoder(self) -> None:
        seen = self._spy_embeds()
        out = self._contrastive(**self._neg_text())
        # Two separate prefills, one row each (split pass, two caches).
        pre = [e for e in seen if e.shape[1] > 1]
        self.assertEqual([e.shape[0] for e in pre], [1, 1])
        # Same image, but the Q-Former read different text, so even the 32
        # image tokens differ -- the negative prompt is not LLM-side only.
        self.assertFalse(torch.allclose(pre[0][0, :32], pre[1][0, :32]))
        self.assertGreater(out["n_gen_tokens"], 0)

    def test_icd_with_an_identical_negative_prompt_is_the_plain_sample(self) -> None:
        inp = tiny_inputs()
        same = {
            "input_ids_do": inp["input_ids"],
            "attn_mask_do": inp["attention_mask"],
            "qformer_input_ids_do": inp["qformer_input_ids"],
            "qformer_attn_mask_do": inp["qformer_attention_mask"],
        }
        got = self._contrastive(**same)["token_ids"]
        want = self._gen(single_branch=True, greedy=True)["token_ids"]
        self.assertEqual(got, want)

    def test_icd_without_qformer_text_is_refused(self) -> None:
        neg = self._neg_text()
        neg.pop("qformer_input_ids_do")
        with self.assertRaisesRegex(ValueError, "Q-Former"):
            self._contrastive(**neg)

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
        inp = tiny_inputs()
        if "confused" in text:
            inp["input_ids"][0, 33:] = torch.tensor([11, 12, 13, 14, 15])
            inp["qformer_input_ids"] = torch.tensor([[3, 9, 8, 7, 6]])
            inp["qformer_attention_mask"] = torch.ones(1, 5, dtype=torch.long)
        return inp


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

    def test_vcd_instructblip_is_its_own_method_with_llavas_vcd_values(self) -> None:
        m, out = self._run("vcd_instructblip")
        d = m.describe()
        self.assertEqual(m.name, "vcd_instructblip")
        self.assertEqual(d["backbone"], "instructblip")
        r = d["resolved_hyperparameters"]
        self.assertEqual((r["cd_alpha"], r["cd_beta"], r["noise_step"]),
                         (1.0, 0.1, 500))
        self.assertEqual(self.processor.seen_text,
                         ["USER:  Is there a dog in the image? ASSISTANT:"])
        self.assertEqual(out.extra["variant"], "vcd_instructblip")
        self.assertGreater(out.n_gen_tokens, 0)

    def test_vcd_instructblip_runs_two_rows(self) -> None:
        rows: List[int] = []
        orig = self.model.language_model.forward

        def spy(*a: Any, **kw: Any):
            e = kw.get("inputs_embeds")
            if e is None:
                e = kw["input_ids"]
            rows.append(int(e.shape[0]))
            return orig(*a, **kw)
        self.model.language_model.forward = spy  # type: ignore[method-assign]
        self._run("vcd_instructblip")
        self.assertEqual(set(rows), {2})

    def test_icd_negative_prompt_is_the_releases_without_user(self) -> None:
        m = get_method("icd_instructblip")(self._cfg("icd_instructblip"))
        self.assertEqual(
            m.disturbed_prompt("Is there a dog in the image?"),
            " You are a confused object detector. "
            "Is there a dog in the image? ASSISTANT:")

    def test_icd_instructblip_encodes_both_prompts_and_runs_one_row_each(self) -> None:
        rows: List[int] = []
        orig = self.model.language_model.forward

        def spy(*a: Any, **kw: Any):
            e = kw.get("inputs_embeds")
            if e is None:
                e = kw["input_ids"]
            rows.append(int(e.shape[0]))
            return orig(*a, **kw)
        self.model.language_model.forward = spy  # type: ignore[method-assign]
        m, out = self._run("icd_instructblip")
        self.assertEqual(set(rows), {1})
        self.assertEqual(len(self.processor.seen_text), 2)
        self.assertIn("confused object detector", self.processor.seen_text[1])
        self.assertNotIn("USER", self.processor.seen_text[1])
        d = m.describe()
        self.assertEqual(d["backbone"], "instructblip")
        self.assertIn("no 'USER:'", d["negative_prompt_form"])
        self.assertEqual(out.extra["variant"], "icd_instructblip")

    def test_icd_negative_only_control_swaps_the_qformer_text_too(self) -> None:
        got: List[torch.Tensor] = []
        orig = self.model.forward

        def spy(*a: Any, **kw: Any):
            got.append(kw["qformer_input_ids"].clone())
            return orig(*a, **kw)
        self.model.forward = spy  # type: ignore[method-assign]
        self._run("icd_instructblip", control="negative_only")
        self.assertTrue(torch.equal(got[0], torch.tensor([[3, 9, 8, 7, 6]])))

    def test_sid_and_causalmm_have_no_instructblip_method(self) -> None:
        from foxrepro.bench.generate import list_methods
        names = set(list_methods())
        self.assertNotIn("sid_instructblip", names)
        self.assertNotIn("causalmm_instructblip", names)
        for name in ("sid", "causalmm"):
            with self.subTest(name=name):
                m = get_method(name)(self._cfg(name, system_prompt_id="llava_v1"))
                m.load()  # stub load: returns the InstructBLIP model
                with self.assertRaises(ValueError):
                    m.generate(self.sample, self.sample.question)


class TestTheProcessorIsTheImageOne(unittest.TestCase):
    """load_backbone must not use AutoProcessor for InstructBLIP: the original
    checkpoint's processor_config.json names the VIDEO processor, which
    inserts no image tokens (smoke-r28)."""

    def test_load_backbone_names_the_explicit_class(self) -> None:
        import inspect
        from foxrepro.fox import attention as attn
        src = inspect.getsource(attn.load_backbone)
        self.assertIn("InstructBlipProcessor.from_pretrained", src)
        self.assertIn('mt == "instructblip"', src)
