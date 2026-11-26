"""Qwen3-VL-8B and InternVL3-8B (paper Table 10 / Table 3), on tiny real models.

No download and no GPU. The fixtures are genuine
``Qwen3VLForConditionalGeneration`` / ``InternVLForConditionalGeneration``
objects built from config: real classes, real multimodal RoPE, real image
scatter, toy widths. What this file can and cannot establish:

CAN: the decode loop agrees token for token with HF's own ``generate`` (this is
what caught the missing ``cache_position`` that makes Qwen3-VL's RoPE use the
wrong position after the first token); the image side-input
(``image_grid_thw``) reaches the model and is doubled with the pixel rows; the
span is derived per sample from the real ids and varies with the image; the
Fox patch fires on the configured layers; the methods build the chat-template
prompt, resolve D.1's hyperparameters and Table 15's windows, and keep their
own names; the plan wires them without touching an existing job.

CANNOT: anything about the real checkpoints -- their processors' output, the
real chat templates, real image sizes, memory on 2 x T4, or any benchmark
number.
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
        InternVLConfig, InternVLForConditionalGeneration, Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
    )
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("transformers lacks Qwen3-VL / InternVL") from None

import transformers  # noqa: E402
import foxrepro.fox.attention as A  # noqa: E402
from foxrepro.bench.generate import GenConfig, get_method  # noqa: E402
from foxrepro.data.loaders import Sample  # noqa: E402

IMG = 250
EOS = 2
N_LAYERS = 4


def _impl() -> Dict[str, str]:
    return {"text_config": A.ATTN_NAME, "vision_config": "sdpa"}


def tiny_qwen3vl(seed: int = 0):
    c = Qwen3VLConfig(
        text_config=dict(
            hidden_size=32, intermediate_size=64, num_hidden_layers=N_LAYERS,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            vocab_size=300, max_position_embeddings=512,
            rope_scaling=dict(rope_type="default", mrope_section=[2, 1, 1]),
            eos_token_id=EOS),
        vision_config=dict(
            depth=2, hidden_size=32, intermediate_size=64, num_heads=2,
            in_channels=3, patch_size=16, spatial_merge_size=2,
            temporal_patch_size=2, out_hidden_size=32,
            num_position_embeddings=64, deepstack_visual_indexes=[0, 1]),
        image_token_id=IMG, video_token_id=251, vision_start_token_id=252,
        vision_end_token_id=253)
    A.register()
    torch.manual_seed(seed)
    return Qwen3VLForConditionalGeneration._from_config(
        c, attn_implementation=_impl()).eval()


def qwen_inputs(h: int = 4, w: int = 4) -> Dict[str, torch.Tensor]:
    """What Qwen3VLProcessor writes: pixel_values is [n_patches, 1536] and
    image_grid_thw [1, 3]; the image id is repeated h*w/4 times (2x2 merge)."""
    n_tok = h * w // 4
    ids = [1, 5, 6, 252] + [IMG] * n_tok + [253, 7, 8, 9]
    torch.manual_seed(3)
    out = {
        "input_ids": torch.tensor([ids]),
        "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        "pixel_values": torch.randn(h * w, 1536),
        "image_grid_thw": torch.tensor([[1, h, w]]),
    }
    if int(transformers.__version__.split(".")[0]) >= 5:
        # transformers 5 refuses image_grid_thw without it; the real
        # processor returns it (1 on image tokens).
        out["mm_token_type_ids"] = (out["input_ids"] == IMG).long()
    return out


def tiny_internvl3(seed: int = 0):
    c = InternVLConfig(
        text_config=dict(
            model_type="qwen2", hidden_size=32, intermediate_size=64,
            num_hidden_layers=N_LAYERS, num_attention_heads=4,
            num_key_value_heads=2, vocab_size=300, eos_token_id=EOS),
        vision_config=dict(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=2, image_size=[28, 28], patch_size=[14, 14],
            num_channels=3),
        image_token_id=IMG, downsample_ratio=0.5)
    A.register()
    torch.manual_seed(seed)
    return InternVLForConditionalGeneration._from_config(
        c, attn_implementation=_impl()).eval()


def internvl_inputs(n_tiles: int = 2) -> Dict[str, torch.Tensor]:
    """One image token per 28x28 tile in the toy; the real one is 256."""
    ids = [1, 5, 6] + [IMG] * n_tiles + [7, 8, 9]
    torch.manual_seed(3)
    return {
        "input_ids": torch.tensor([ids]),
        "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        "pixel_values": torch.randn(n_tiles, 3, 28, 28),
    }


class _Counting:
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


class TestDecodeAgreesWithHF(unittest.TestCase):
    """Single-branch greedy is plain decoding, so it must equal generate()."""

    def _agree(self, model, inputs) -> None:
        n = int(inputs["input_ids"].shape[1])
        got = A.fox_generate(model, dict(inputs), max_new_tokens=5,
                             single_branch=True, greedy=True, seed=1)
        ref = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        self.assertEqual(got["token_ids"], ref[0, n:].tolist())

    def test_qwen3vl(self) -> None:
        self._agree(tiny_qwen3vl(), qwen_inputs())

    def test_qwen3vl_twice_in_a_row(self) -> None:
        """rope_deltas lives on the model; a stale one from the previous
        sample must not leak into the next prefill."""
        m = tiny_qwen3vl()
        self._agree(m, qwen_inputs(4, 4))
        self._agree(m, qwen_inputs(2, 4))

    def test_internvl3(self) -> None:
        self._agree(tiny_internvl3(), internvl_inputs())

    def test_fox_runs_two_rows_with_their_own_side_inputs(self) -> None:
        m = tiny_qwen3vl()
        seen: List[Dict[str, Any]] = []
        orig = m.model.forward

        def spy(*a: Any, **kw: Any):
            if kw.get("pixel_values") is not None:
                seen.append({"px": tuple(kw["pixel_values"].shape),
                             "grid": kw["image_grid_thw"].tolist(),
                             "rows": int(kw["input_ids"].shape[0])})
            return orig(*a, **kw)
        m.model.forward = spy  # type: ignore[method-assign]
        r = A.fox_generate(m, qwen_inputs(), max_new_tokens=3, seed=1,
                           eager_layers=frozenset({1, 2}))
        self.assertEqual(len(seen), 1, "image side inputs only on the prefill")
        self.assertEqual(seen[0]["rows"], 2)
        self.assertEqual(seen[0]["px"], (32, 1536))      # 16 patches x 2 rows
        self.assertEqual(seen[0]["grid"], [[1, 4, 4], [1, 4, 4]])
        self.assertEqual(len(r["token_ids"]), 3)

    def test_the_decode_entry_point_is_still_under_inference_mode(self) -> None:
        self.assertTrue(hasattr(A.fox_generate, "__wrapped__"))
        for name in ("_eos_ids",):
            self.assertFalse(hasattr(getattr(A, name), "__wrapped__"))
        seen: List[bool] = []
        m = tiny_internvl3()
        orig = m.forward

        def spy(*a: Any, **kw: Any):
            seen.append(torch.is_inference_mode_enabled())
            return orig(*a, **kw)
        m.forward = spy  # type: ignore[method-assign]
        A.fox_generate(m, internvl_inputs(), max_new_tokens=2, seed=1,
                       eager_layers=frozenset({1}))
        self.assertTrue(seen and all(seen))


class TestDecodeStepMaskForMRopeModels(unittest.TestCase):
    """r29: every Qwen3-VL CHAIR job died on its first sample with
    ``keys 419 vs mask 210`` (419 = 209 + 210). transformers 5.16-5.24 derive
    the 3D positions of a decode step from ``attention_mask.cumsum(-1) - 1``,
    which is as long as the whole sequence, so a single new token is expanded
    to P+1 queries. The installed transformers here may be older and not
    break, so this pins the cause rather than the symptom: a decode step must
    not hand the full-length mask to a multimodal-RoPE model."""

    @staticmethod
    def _decode_calls(model, inputs) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        orig = model.model.forward

        def spy(*a: Any, **kw: Any):
            # 5.x renames past_key_values to cache; a decode step is the
            # call that carries one new token.
            if kw.get("input_ids") is not None and kw["input_ids"].shape[1] == 1:
                m = kw.get("attention_mask")
                calls.append({
                    "q": int(kw["input_ids"].shape[1]),
                    "mask": None if m is None else int(m.shape[1]),
                })
            return orig(*a, **kw)
        model.model.forward = spy  # type: ignore[method-assign]
        A.fox_generate(model, inputs, max_new_tokens=4, seed=1,
                       eager_layers=frozenset({1, 2}))
        return calls

    def test_qwen3vl_decode_gets_no_full_length_mask(self) -> None:
        m = tiny_qwen3vl()
        self.assertTrue(A._uses_mrope(m))
        calls = self._decode_calls(m, qwen_inputs())
        self.assertTrue(calls)
        for c in calls:
            self.assertEqual(c["q"], 1)
            self.assertIsNone(c["mask"], "a P+1 mask next to one query token "
                              "is what transformers 5.x turns into P+1 queries")

    def test_models_without_mrope_still_get_their_mask(self) -> None:
        m = tiny_internvl3()
        self.assertFalse(A._uses_mrope(m))
        calls = self._decode_calls(m, internvl_inputs())
        self.assertTrue(calls)
        self.assertTrue(all(c["mask"] is not None for c in calls))

    def test_llava_is_not_an_mrope_model(self) -> None:
        try:
            from tests.test_tiny_llava import tiny_llava
        except Exception:  # pragma: no cover - llava stack missing
            self.skipTest("tiny llava unavailable")
        self.assertFalse(A._uses_mrope(tiny_llava()))

    def test_qwen3vl_sampling_survives_many_seeds(self) -> None:
        """The r29 failure was reported per seed, but it lives in the decode
        step, which every seed reaches; a spread of seeds must all complete."""
        m = tiny_qwen3vl()
        for s in range(8):
            r = A.fox_generate(m, qwen_inputs(), max_new_tokens=5, seed=s,
                               eager_layers=frozenset({1, 2}))
            self.assertGreaterEqual(len(r["token_ids"]), 1)


class TestMemoryBoundedPathsChangeNothing(unittest.TestCase):
    """r29 InternVL3 Fox OOM: ``intervene`` and the mask/softmax after it held
    several copies of a [2,H,Q,Q] tensor for 1.8k-3.3k-token prompts. Both now
    work on slices above a size threshold. Forcing the thresholds to 0 must
    leave every generated token and every trace value unchanged."""

    def _gen(self, model, inputs, seed):
        return A.fox_generate(model, dict(inputs), max_new_tokens=6, seed=seed,
                              eager_layers=frozenset({1, 2}),
                              collect_trace=True)

    def test_forced_small_thresholds_give_identical_generations(self) -> None:
        import foxrepro.fox.kernel as K
        saved = (K.LAST_ROW_ONLY_ELEMS, A.LARGE_TAIL_ELEMS, A._TAIL_CHUNK_ELEMS)
        for name, mk, inp in (("qwen3vl", tiny_qwen3vl, qwen_inputs),
                              ("internvl3", tiny_internvl3, internvl_inputs)):
            model = mk()
            for seed in (1, 2, 3):
                K.LAST_ROW_ONLY_ELEMS, A.LARGE_TAIL_ELEMS = 1 << 60, 1 << 60
                want = self._gen(model, inp(), seed)
                try:
                    K.LAST_ROW_ONLY_ELEMS, A.LARGE_TAIL_ELEMS = 0, 0
                    A._TAIL_CHUNK_ELEMS = 1       # one head per chunk
                    got = self._gen(model, inp(), seed)
                finally:
                    (K.LAST_ROW_ONLY_ELEMS, A.LARGE_TAIL_ELEMS,
                     A._TAIL_CHUNK_ELEMS) = saved
                with self.subTest(model=name, seed=seed):
                    self.assertEqual(want["token_ids"], got["token_ids"])
                    self.assertEqual(want["trace"], got["trace"])


class TestSpan(unittest.TestCase):
    def test_width_is_dynamic_and_derived_from_the_ids(self) -> None:
        for model, mk, widths in (
                (tiny_qwen3vl(), qwen_inputs, [(4, 4, 4), (2, 4, 2), (4, 8, 8)]),
                (tiny_internvl3(), None, [(1, 1), (2, 2), (5, 5)])):
            self.assertIsNone(A.expected_span_width(model))
            self.assertEqual(A.resolve_image_token_id(model), IMG)
            for spec in widths:
                with self.subTest(model=type(model).__name__, spec=spec):
                    if mk is qwen_inputs:
                        h, w, n = spec
                        ids = mk(h, w)["input_ids"]
                    else:
                        n_tiles, n = spec
                        ids = internvl_inputs(n_tiles)["input_ids"]
                    s, e = A.locate_image_span(ids, IMG, None)
                    self.assertEqual(e - s + 1, n)
                    self.assertGreater(s, 0, "system keys exist: m_sys is defined")

    def test_a_split_span_is_refused(self) -> None:
        ids = qwen_inputs()["input_ids"].clone()
        ids[0, 6] = 9
        with self.assertRaises(ValueError):
            A.locate_image_span(ids, IMG, None)


class TestPatchFires(unittest.TestCase):
    def test_fox_intervenes_on_exactly_the_window_with_m_sys(self) -> None:
        for model, inputs in ((tiny_qwen3vl(), qwen_inputs()),
                              (tiny_internvl3(), internvl_inputs())):
            with self.subTest(model=type(model).__name__), _Counting() as c:
                A.fox_generate(model, dict(inputs), max_new_tokens=2, seed=1,
                               eager_layers=frozenset({1, 2}))
            self.assertEqual(set(c.layers), {1, 2})
            self.assertTrue(all(c.use_m_sys))

    def test_eos_ids_union_the_text_and_generation_configs(self) -> None:
        m = mock.Mock()
        m.config.text_config.eos_token_id = 7
        m.generation_config.eos_token_id = [7, 9]
        self.assertEqual(A._eos_ids(m), frozenset({7, 9}))
        m.config.text_config.eos_token_id = None
        m.generation_config = None
        self.assertEqual(A._eos_ids(m), frozenset())


# --------------------------------------------------------------------------- #
# methods
# --------------------------------------------------------------------------- #
class _FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):  # noqa: ANN001
        return " ".join(str(i) for i in ids)


class _FakeProcessor:
    """apply_chat_template renders a Qwen-like turn; __call__ returns a
    prepared input dict, a different image size per call when asked."""

    chat_template = "{{ fake jinja }}"

    def __init__(self, make_inputs, sizes=None) -> None:
        self.tokenizer = _FakeTokenizer()
        self._make = make_inputs
        self._sizes = list(sizes or [None])
        self.seen_text: List[str] = []
        self.seen_messages: List[Any] = []

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False):
        assert tokenize is False and add_generation_prompt is True
        self.seen_messages.append(messages)
        parts = messages[0]["content"]
        assert [p["type"] for p in parts] == ["image", "text"]
        return ("<|im_start|>user\n<image>" + parts[1]["text"]
                + "<|im_end|>\n<|im_start|>assistant\n")

    def __call__(self, images=None, text=None, return_tensors=None):
        self.seen_text.append(text)
        size = self._sizes[(len(self.seen_text) - 1) % len(self._sizes)]
        return self._make(*size) if size else self._make()


class TestMethods(unittest.TestCase):
    def setUp(self) -> None:
        import foxrepro.bench.methods as M
        self.M = M
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        img = Path(td.name) / "COCO_val2014_000000000042.jpg"
        img.write_bytes(b"x")
        self.sample = Sample(
            sample_id="pope/adversarial/q0000001", image_id=42,
            image_path=img, benchmark="pope", split="adversarial",
            question="Is there a dog in the image?", label="no", extra={})
        A.register()
        orig_enc = M._LlavaMethod._encode
        M._LlavaMethod._encode = (  # type: ignore[assignment]
            lambda s, sample, text: s.processor(images=None, text=text))
        self.addCleanup(lambda: setattr(M._LlavaMethod, "_encode", orig_enc))

    def _load_with(self, model, processor) -> None:
        orig = A.load_llava
        A.load_llava = lambda *a, **k: (model, processor)  # type: ignore
        self.addCleanup(lambda: setattr(A, "load_llava", orig))

    def _cfg(self, method: str, model_id: str, **over: Any) -> GenConfig:
        base: Dict[str, Any] = dict(
            method=method, benchmark="pope", split="adversarial", seed=42,
            dtype="fp32", model_id=model_id, prompt_template_id="pope_v0_raw",
            max_new_tokens=3, system_prompt_id="none", allow_cpu=True)
        base.update(over)
        return GenConfig(**base)

    def _run(self, method, model, processor, model_id, **over):
        self._load_with(model, processor)
        m = get_method(method)(self._cfg(method, model_id, **over))
        m.load()
        return m, m.generate(self.sample, self.sample.question)

    def test_names_are_registered_and_distinct_from_fox(self) -> None:
        for n in ("sampling_qwen3vl", "fox_qwen3vl",
                  "sampling_internvl3", "fox_internvl3"):
            self.assertIn(n, __import__("foxrepro.bench.generate",
                                        fromlist=["x"])._GPU_METHODS)
            self.assertEqual(get_method(n).name, n)
            self.assertNotIn(n, ("sampling", "fox"))

    def test_qwen_prompt_is_the_chat_template_and_the_span_is_per_sample(self) -> None:
        proc = _FakeProcessor(qwen_inputs, sizes=[(4, 4), (2, 4)])
        m, out = self._run("sampling_qwen3vl", tiny_qwen3vl(), proc,
                           "fake/qwen3vl")
        self.assertEqual(proc.seen_text[0], (
            "<|im_start|>user\n<image>Is there a dog in the image?"
            "<|im_end|>\n<|im_start|>assistant\n"))
        self.assertEqual(out.extra["chat_prompt"], proc.seen_text[0])
        self.assertEqual(out.extra["image_span"], [4, 7])
        out2 = m.generate(self.sample, self.sample.question)
        self.assertEqual(out2.extra["image_span"], [4, 5])   # varied, derived
        d = m.describe()
        self.assertEqual(d["chat_template_id"], self.M.CHAT_TEMPLATE_ID)
        self.assertEqual(len(d["chat_template_sha256"]), 64)
        self.assertIn("<instruction>", d["chat_template_rendered_example"])
        self.assertEqual(d["backbone"], "qwen3_vl")

    def test_fox_resolves_d1_values_and_table15_windows(self) -> None:
        for name, model, mk, mid, layers in (
                ("fox_qwen3vl", tiny_qwen3vl(), qwen_inputs, "q", list(range(3, 11))),
                ("fox_internvl3", tiny_internvl3(), internvl_inputs, "i",
                 list(range(3, 12)))):
            with self.subTest(name=name):
                mt = "qwen3_vl" if "qwen" in name else "internvl"
                self.assertEqual(model.config.model_type, mt)
                m, out = self._run(name, model, _FakeProcessor(mk), mid)
                r = m.describe()["resolved_hyperparameters"]
                self.assertEqual(r["ratio"], 0.4)
                self.assertEqual(r["tau_js"], 0.2)
                self.assertEqual(r["alpha"], 2.0)
                self.assertEqual(r["beta"], 0.1)
                self.assertEqual(r["gamma"], 10000.0)
                self.assertNotIn("use_m_sys", r)         # Eq. 4, not H_vis-only
                self.assertEqual(m.describe()["eager_layers"], layers)
                self.assertEqual(out.n_gen_tokens, len(out.text.split()))

    def test_explicit_values_override(self) -> None:
        m, _ = self._run("fox_qwen3vl", tiny_qwen3vl(),
                         _FakeProcessor(qwen_inputs), "q", k=0.5,
                         intervention_layers="1-2")
        self.assertEqual(m.describe()["resolved_hyperparameters"]["ratio"], 0.5)
        self.assertEqual(m.describe()["eager_layers"], [1, 2])

    def test_sampling_takes_one_row_and_fox_two(self) -> None:
        for name, rows_expected in (("sampling_internvl3", {1}),
                                    ("fox_internvl3", {2})):
            model = tiny_internvl3()
            rows: List[int] = []
            orig = model.forward

            def spy(*a: Any, _o=orig, **kw: Any):
                rows.append(int(kw["input_ids"].shape[0]))
                return _o(*a, **kw)
            model.forward = spy  # type: ignore[method-assign]
            self._run(name, model, _FakeProcessor(internvl_inputs), "i")
            self.assertEqual(set(rows), rows_expected, name)

    def test_llava_wrapper_and_system_prompt_are_refused(self) -> None:
        for name in ("sampling_qwen3vl", "fox_qwen3vl",
                     "sampling_internvl3", "fox_internvl3"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                get_method(name)(self._cfg(name, "x", system_prompt_id="llava_v1"))

    def test_wrong_checkpoint_is_refused_at_load(self) -> None:
        self._load_with(tiny_internvl3(), _FakeProcessor(internvl_inputs))
        m = get_method("fox_qwen3vl")(self._cfg("fox_qwen3vl", "x"))
        with self.assertRaisesRegex(ValueError, "qwen3_vl"):
            m.load()

    def test_llava_fox_is_unchanged(self) -> None:
        from foxrepro.bench.methods import FoxMethod, FOX_DEFAULTS
        self.assertIs(FoxMethod.HPARAM_DEFAULTS, FOX_DEFAULTS)
        self.assertIsNone(FoxMethod.LAYER_DEFAULT)


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
class TestPlan(unittest.TestCase):
    def setUp(self) -> None:
        from foxrepro.bench import plan as P
        self.P = P
        self.jobs = [j for j in P.PLAN if j.target.startswith("table10_")]

    def test_the_jobs_are_exactly_table10(self) -> None:
        P = self.P
        self.assertEqual(len(self.jobs), 2 * 2 * 2 * 3)   # backbone x method x bench x seed
        by_target = {}
        for j in self.jobs:
            by_target.setdefault(j.target, []).append(j)
        self.assertEqual(set(by_target), {"table10_qwen3vl", "table10_internvl3"})
        for target, tag, mid in (
                ("table10_qwen3vl", "qwen3vl", P.QWEN3VL_MODEL_ID),
                ("table10_internvl3", "internvl3", P.INTERNVL3_MODEL_ID)):
            js = by_target[target]
            self.assertEqual({j.model_id for j in js}, {mid})
            self.assertEqual({j.method for j in js},
                             {f"sampling_{tag}", f"fox_{tag}"})
            self.assertEqual({(j.benchmark, j.split) for j in js},
                             {("pope", "adversarial"), ("chair", "val2014_500")})
            self.assertEqual({j.seed for j in js}, {42, 43, 44})
            self.assertEqual({j.system_prompt_id for j in js}, {"none"})
            # k / tau / window resolved by the method, so the Fox arm shares
            # its pair_key with the baseline.
            for j in js:
                self.assertIsNone(j.k)
                self.assertIsNone(j.tau_js)
                self.assertIsNone(j.intervention_layers)

    def test_ids_are_unique_and_no_existing_job_changed(self) -> None:
        ids = [j.job_id() for j in self.P.PLAN]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("table10_qwen3vl:fox_qwen3vl:pope:adversarial:q3vl:spnone",
                      ids)
        # A continuation chunk shares its first chunk's file on purpose.
        keys = [j.run_key().filename() for j in self.P.PLAN
                if not j.is_continuation]
        self.assertEqual(len(keys), len(set(keys)))

    def test_fox_is_dual_branch_and_cost_carries_the_measured_factor(self) -> None:
        P = self.P
        self.assertIn("fox_qwen3vl", P.DUAL_BRANCH)
        self.assertIn("fox_internvl3", P.FOX_METHODS)
        self.assertNotIn("sampling_qwen3vl", P.DUAL_BRANCH)
        j = P.Job("t", "fox_qwen3vl", "pope", "adversarial",
                  model_id=P.QWEN3VL_MODEL_ID)
        llava = P.Job("t", "fox", "pope", "adversarial")
        # Qwen3-VL Fox POPE was measured at 0.73x the LLaVA price (r32).
        self.assertAlmostEqual(
            j.est_seconds(),
            llava.est_seconds()
            * P.RECENT_BACKBONE_METHOD_COST_FACTOR[("fox_qwen3vl", "pope")])
        # The unmeasured InternVL3 rows are still labelled as guesses.
        self.assertIn("a guess", Path(P.__file__).read_text(encoding="utf-8"))

    def test_commands_round_trip_to_the_same_run_file(self) -> None:
        from tests.test_plan import cfg_from_command
        for j in self.jobs:
            with self.subTest(job=j.job_id()):
                self.assertEqual(
                    cfg_from_command(j.command()).run_key().filename(),
                    j.run_key().filename())
                self.assertIn(f"--model-id {j.model_id}", j.command())
                self.assertNotIn("llava-hf", j.command())


if __name__ == "__main__":
    unittest.main()
