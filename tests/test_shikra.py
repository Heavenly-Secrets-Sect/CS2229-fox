"""Shikra-7B against tiny stand-ins: the merge, the splice, the span, the plan.

No download and no GPU. What this file CAN establish: the streamed merge is
bit-identical to a literal transcription of the release's ``apply_delta.py``
and refuses every malformed delta; the prompt is tokenised the way the 2023
tokenizer did (pieces stripped, markers inserted by hand); the splice writes
exactly the 256 patch positions; the span is derived from the ids and is
(34, 289) behind the release's markers at 33 and 290; the Fox patch fires on
layers 3-10 only and never on the vision tower; the cached decode agrees with
a full recomputation; every decode path runs under inference mode; the jobs
are what the plan says.

What it CANNOT: that the real 13.5 GB delta loads and merges on a Kaggle host
(RAM, disk, the ``mmap`` of the 2023 pickles), that the real tokenizer files
tokenise as the stub does (it was measured once by hand against them: markers
at 33 and 290, see docs/DEVIATIONS.md 3.20), or any number a benchmark would
report.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("torch is not installed") from None

try:
    from safetensors.torch import save_file
    from transformers import CLIPVisionConfig, CLIPVisionModel, LlamaConfig, \
        LlamaForCausalLM
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("transformers is not installed") from None

import foxrepro.fox.attention as A  # noqa: E402
from foxrepro.bench.generate import GenConfig, get_method  # noqa: E402
from foxrepro.data.loaders import Sample  # noqa: E402
from foxrepro.fox import shikra as S  # noqa: E402

N_LAYERS = 12
EOS = 2
_SMOKE_SRC = Path(__file__).resolve().parents[1] / "scripts" / "smoke_backbone.py"
_SYSTEM_WORDS = 32          # tokens the stub gives "<system message>USER:"


def _load_smoke():
    spec = importlib.util.spec_from_file_location("smoke_backbone", _SMOKE_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SMOKE = _load_smoke()


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
class StubTokenizer:
    """Records the pieces it is asked to encode; one id per word."""

    bos_token_id = S.BOS_ID
    eos_token_id = S.EOS_ID

    def __init__(self) -> None:
        self.pieces: List[str] = []

    def encode(self, piece: str, add_special_tokens: bool = False) -> List[int]:
        assert add_special_tokens is False
        self.pieces.append(piece)
        if piece.startswith("A chat between"):
            return [100] * _SYSTEM_WORDS          # system message + "USER:"
        return [200 + (sum(map(ord, w)) % 50) for w in piece.split()]

    @staticmethod
    def decode(ids, skip_special_tokens: bool = True) -> str:
        return " ".join(f"t{i}" for i in ids if i != EOS)


class StubImageProcessor:
    def __call__(self, images: Any = None, return_tensors: str = "pt"):
        g = torch.Generator().manual_seed(7)
        return {"pixel_values": torch.randn(1, 3, 224, 224, generator=g)}


def tiny_shikra(seed: int = 0, vocab: int = S.VOCAB_SIZE) -> S.ShikraForFox:
    """A real LlamaForCausalLM carrying the Fox attention, a real CLIP vision
    tower at the real geometry (224 px, patch 14 -> 16 x 16), toy widths."""
    A.register()
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=N_LAYERS,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=vocab,
        max_position_embeddings=1024, bos_token_id=S.BOS_ID,
        eos_token_id=S.EOS_ID)
    lm = LlamaForCausalLM._from_config(
        cfg, attn_implementation=A.ATTN_NAME).eval()
    vcfg = CLIPVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=2, image_size=224, patch_size=14)
    vt = CLIPVisionModel._from_config(
        vcfg, attn_implementation="sdpa").eval()
    proj = nn.Linear(16, 32)
    return S.ShikraForFox(lm, vt, proj).eval()


def tiny_batch() -> Dict[str, torch.Tensor]:
    proc = S.ShikraProcessor(StubTokenizer(), StubImageProcessor())
    from foxrepro.bench.methods import SHIKRA_WRAPPER, _LLAVA_V1_SYSTEM
    return proc(images=None,
                text=_LLAVA_V1_SYSTEM + SHIKRA_WRAPPER.format(
                    instruction="Is there a dog in the image?"))


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


# --------------------------------------------------------------------------- #
# the prompt
# --------------------------------------------------------------------------- #
class TestPrompt(unittest.TestCase):
    TEXT = (S.SYSTEM_MESSAGE
            + "USER: <im_start><ImageHere><im_end> Is there a dog? ASSISTANT:")

    def test_pieces_are_stripped_as_the_2023_tokenizer_did(self) -> None:
        tok = StubTokenizer()
        S.encode_prompt(tok, self.TEXT)
        # No space before <im_start> and none after <im_end>: those are the
        # whitespace the modern LlamaTokenizer would turn into extra "▁"
        # tokens, shifting the span by one (DEVIATIONS 3.20).
        self.assertEqual(tok.pieces, [S.SYSTEM_MESSAGE + "USER:",
                                      "Is there a dog? ASSISTANT:"])

    def test_the_markers_land_where_the_release_has_them(self) -> None:
        ids = S.encode_prompt(StubTokenizer(), self.TEXT)
        self.assertEqual(ids[0], S.BOS_ID)
        s, e = ids.index(S.IM_START_ID), ids.index(S.IM_END_ID)
        self.assertEqual((s, e), (33, 290))      # Uncertain_head.py: 33, 290
        self.assertEqual(ids.count(S.IM_PATCH_ID), 256)
        self.assertEqual(set(ids[s + 1:e]), {S.IM_PATCH_ID})
        self.assertEqual(e - s + 1, 258)          # with the markers

    def test_exactly_one_placeholder_is_required(self) -> None:
        for bad in ("no image here", self.TEXT + S.SHIKRA_WRAPPER.format(
                instruction="x")):
            with self.subTest(bad=bad[:20]), self.assertRaises(ValueError):
                S.encode_prompt(StubTokenizer(), bad)

    def test_constants_agree_across_modules(self) -> None:
        from foxrepro.bench import methods as M
        from foxrepro.bench import plan as P
        self.assertEqual(M._LLAVA_V1_SYSTEM, S.SYSTEM_MESSAGE)
        self.assertEqual(M.SHIKRA_WRAPPER, S.SHIKRA_WRAPPER)
        self.assertEqual(P.SHIKRA_MODEL_ID, S.SHIKRA_DELTA_ID)
        self.assertEqual(SMOKE.SHIKRA_CANDIDATE, S.SHIKRA_DELTA_ID)

    def test_the_processor_returns_what_fox_generate_reads(self) -> None:
        b = tiny_batch()
        self.assertEqual(sorted(b), ["attention_mask", "input_ids",
                                     "pixel_values"])
        self.assertEqual(tuple(b["pixel_values"].shape), (1, 3, 224, 224))
        self.assertEqual(b["input_ids"].shape, b["attention_mask"].shape)


class TestTokenizerCheck(unittest.TestCase):
    def test_a_32000_vocab_tokenizer_is_refused(self) -> None:
        class T:
            bos_token_id, eos_token_id = 1, 2

            def __len__(self) -> int:
                return 32000

            @staticmethod
            def convert_tokens_to_ids(toks):
                return [0] * len(toks)
        with self.assertRaises(ValueError):
            S.check_tokenizer(T())


# --------------------------------------------------------------------------- #
# the merge
# --------------------------------------------------------------------------- #
H, FF, V0, V1, NL, HC = 4, 6, 10, 13, 2, 6


def _shape(name: str, vocab: int):
    if name.endswith("embed_tokens.weight") or name == "lm_head.weight":
        return (vocab, H)
    if "mlp.gate" in name or "mlp.up" in name:
        return (FF, H)
    if "mlp.down" in name:
        return (H, FF)
    if "layernorm" in name or name == "model.norm.weight":
        return (H,)
    return (H, H)


def release_apply_delta(base: Dict[str, torch.Tensor],
                        delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """``apply_delta.py`` of the release, with its loops kept as they are."""
    out = {k: v.clone() for k, v in delta.items()}
    for name, param in out.items():
        if name not in base:
            assert name in ["model.mm_projector.weight",
                            "model.mm_projector.bias"], name
            continue
        if param.data.shape == base[name].shape:
            param.data += base[name]
        else:
            assert name in ["model.embed_tokens.weight", "lm_head.weight"]
            bparam = base[name]
            param.data[:bparam.shape[0], :bparam.shape[1]] += bparam
    return out


def write_fixture(root: Path, *, drop: str = "", base_dtype=torch.float16,
                  odd_shape: str = "") -> Dict[str, Dict[str, torch.Tensor]]:
    """A 2-layer delta (pickled .bin shards + index) and base (safetensors)."""
    torch.manual_seed(5)
    names = S.expected_text_keys(NL)
    rope = [f"model.layers.{i}.self_attn.rotary_emb.inv_freq"
            for i in range(NL)]
    base = {n: torch.randn(*_shape(n, V0)).half() for n in names}
    base.update({n: torch.rand(2).half() for n in rope})
    delta = {n: torch.randn(*_shape(n, V1)).half() for n in names}
    delta.update({n: torch.zeros(2).half() for n in rope})
    delta["model.mm_projector.weight"] = torch.randn(H, HC).half()
    delta["model.mm_projector.bias"] = torch.randn(H).half()
    if odd_shape:
        delta[odd_shape] = torch.randn(H + 1, H).half()
    if base_dtype != torch.float16:
        base = {k: v.to(base_dtype) for k, v in base.items()}
    if drop:
        delta.pop(drop)

    (root / "delta").mkdir()
    (root / "base").mkdir()
    keys = sorted(delta)
    halves = [keys[:len(keys) // 2], keys[len(keys) // 2:]]
    wm = {}
    for i, part in enumerate(halves):
        fn = f"pytorch_model-0000{i + 1}-of-00002.bin"
        torch.save({k: delta[k] for k in part}, root / "delta" / fn)
        wm.update({k: fn for k in part})
    (root / "delta" / "pytorch_model.bin.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": wm}))

    bkeys = sorted(base)
    bparts = [bkeys[:len(bkeys) // 2], bkeys[len(bkeys) // 2:]]
    bwm = {}
    for i, part in enumerate(bparts):
        fn = f"model-0000{i + 1}-of-00002.safetensors"
        save_file({k: base[k].contiguous() for k in part},
                  str(root / "base" / fn))
        bwm.update({k: fn for k in part})
    (root / "base" / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": bwm}))
    return {"base": base, "delta": delta}


class TestMerge(unittest.TestCase):
    def test_bit_identical_to_the_releases_apply_delta(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fx = write_fixture(root)
            want = release_apply_delta(fx["base"], fx["delta"])
            llama, proj = S.merge_delta(root / "delta", root / "base",
                                        n_layers=NL)
        self.assertEqual(set(llama), set(S.expected_text_keys(NL)))
        for k, v in llama.items():
            self.assertTrue(torch.equal(v, want[k]), k)
            self.assertEqual(v.dtype, torch.float16)
        self.assertTrue(torch.equal(proj["weight"],
                                    want["model.mm_projector.weight"]))
        self.assertTrue(torch.equal(proj["bias"],
                                    want["model.mm_projector.bias"]))

    def test_the_three_new_rows_are_the_deltas_alone(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fx = write_fixture(root)
            llama, _ = S.merge_delta(root / "delta", root / "base",
                                     n_layers=NL)
        for k in ("model.embed_tokens.weight", "lm_head.weight"):
            self.assertEqual(tuple(llama[k].shape), (V1, H))
            self.assertTrue(torch.equal(llama[k][V0:], fx["delta"][k][V0:]))
            self.assertTrue(torch.equal(
                llama[k][:V0], fx["delta"][k][:V0] + fx["base"][k]))

    def test_the_rope_buffer_is_not_a_weight(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_fixture(root)
            names = [n for n, _ in S.iter_merged_tensors(
                root / "delta", root / "base", n_layers=NL)]
        self.assertFalse([n for n in names if "inv_freq" in n])
        self.assertEqual(len(names), len(set(names)))

    def test_a_tensor_missing_from_the_delta_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_fixture(root, drop="model.layers.1.mlp.up_proj.weight")
            idx = root / "delta" / "pytorch_model.bin.index.json"
            # the index must not mention it either: a delta that lost it.
            wm = json.loads(idx.read_text())["weight_map"]
            wm.pop("model.layers.1.mlp.up_proj.weight", None)
            idx.write_text(json.dumps({"weight_map": wm}))
            with self.assertRaisesRegex(RuntimeError, "merge incomplete"):
                S.merge_delta(root / "delta", root / "base", n_layers=NL)

    def test_an_empty_index_is_an_error_not_an_empty_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_fixture(root)
            (root / "delta" / "pytorch_model.bin.index.json").write_text(
                json.dumps({"weight_map": {}}))
            with self.assertRaises(ValueError):
                S.merge_delta(root / "delta", root / "base", n_layers=NL)

    def test_a_base_that_is_not_fp16_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_fixture(root, base_dtype=torch.bfloat16)
            with self.assertRaisesRegex(TypeError, "fp16"):
                S.merge_delta(root / "delta", root / "base", n_layers=NL)

    def test_a_shape_mismatch_outside_the_vocab_tensors_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_fixture(root, odd_shape="model.layers.0.self_attn.q_proj.weight")
            with self.assertRaisesRegex(ValueError, "3-new-rows"):
                S.merge_delta(root / "delta", root / "base", n_layers=NL)

    def test_a_wrong_base_is_named(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_fixture(root)
            idx = root / "base" / "model.safetensors.index.json"
            wm = json.loads(idx.read_text())["weight_map"]
            wm.pop("model.norm.weight")
            idx.write_text(json.dumps({"weight_map": wm}))
            with self.assertRaisesRegex(KeyError, "wrong base"):
                S.merge_delta(root / "delta", root / "base", n_layers=NL)

    def test_expected_keys_for_the_real_model_are_291(self) -> None:
        # 32 blocks x 9 + embed + norm + lm_head; the delta's other 34 tensors
        # are 32 rotary buffers and the 2 projector tensors (325 in all).
        self.assertEqual(len(S.expected_text_keys(32)), 291)
        self.assertEqual(291 + 32 + 2, 325)


# --------------------------------------------------------------------------- #
# the splice and the model
# --------------------------------------------------------------------------- #
class TestSplice(unittest.TestCase):
    def _ids(self, pre: int = 5, post: int = 3) -> torch.Tensor:
        return torch.tensor([[1] + [9] * pre + [S.IM_START_ID]
                             + [S.IM_PATCH_ID] * 4 + [S.IM_END_ID]
                             + [8] * post])

    def test_only_the_patch_positions_change(self) -> None:
        ids = self._ids()
        emb = torch.randn(1, ids.shape[1], 6)
        feats = torch.randn(1, 4, 6)
        out = S.splice_image_features(ids, emb, feats)
        s = 1 + 5 + 1
        self.assertTrue(torch.equal(out[0, s:s + 4], feats[0]))
        keep = torch.ones(ids.shape[1], dtype=torch.bool)
        keep[s:s + 4] = False
        self.assertTrue(torch.equal(out[0, keep], emb[0, keep]))   # markers too
        self.assertTrue(torch.equal(emb, emb.clone()))
        self.assertFalse(torch.equal(out, emb))

    def test_malformed_layouts_are_refused(self) -> None:
        emb = torch.randn(1, 14, 6)
        f4 = torch.randn(1, 4, 6)
        base = self._ids()
        cases = {
            "count": (base, torch.randn(1, 3, 6)),
            "batch": (base, torch.randn(2, 4, 6)),
        }
        no_start = base.clone()
        no_start[0, 6] = 5
        cases["no start"] = (no_start, f4)
        no_end = base.clone()
        no_end[0, 11] = 5
        cases["no end"] = (no_end, f4)
        gap = base.clone()
        gap[0, 8] = 5
        cases["gap"] = (gap, torch.randn(1, 3, 6))
        for name, (ids, feats) in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                S.splice_image_features(ids, emb, feats)


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = tiny_shikra()
        cls.batch = tiny_batch()

    def test_the_span_is_the_256_patches(self) -> None:
        ids = self.batch["input_ids"]
        self.assertEqual(A.resolve_image_token_id(self.model), S.IM_PATCH_ID)
        self.assertEqual(A.expected_span_width(self.model), 256)
        span = A.locate_image_span(ids, S.IM_PATCH_ID, 256)
        self.assertEqual(span, (34, 289))
        self.assertEqual(int(ids[0, span[0] - 1]), S.IM_START_ID)
        self.assertEqual(int(ids[0, span[1] + 1]), S.IM_END_ID)

    def test_the_image_reaches_the_decoder(self) -> None:
        with torch.no_grad():
            a = self.model(**self.batch).logits[0, -1]
            b = dict(self.batch)
            b["pixel_values"] = self.batch["pixel_values"].flip(-1)
            c = self.model(**b).logits[0, -1]
            text_only = self.model.language_model(
                input_ids=self.batch["input_ids"]).logits[0, -1]
        self.assertFalse(torch.equal(a, c))
        self.assertFalse(torch.equal(a, text_only))

    def test_patches_without_pixels_and_pixels_without_patches_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "no pixel_values"):
            self.model(input_ids=self.batch["input_ids"])
        with self.assertRaisesRegex(ValueError, "no <im_patch>"):
            self.model(input_ids=torch.tensor([[1, 5, 6]]),
                       pixel_values=self.batch["pixel_values"])
        with self.assertRaises(TypeError):
            self.model(input_ids=self.batch["input_ids"], images=1)

    def test_the_vision_tower_gives_256_patches_and_drops_cls(self) -> None:
        with torch.no_grad():
            f = self.model.encode_image(self.batch["pixel_values"])
        self.assertEqual(tuple(f.shape), (1, 256, 32))
        out = self.model.vision_tower(
            pixel_values=self.batch["pixel_values"], output_hidden_states=True)
        self.assertEqual(len(out.hidden_states), 5)        # embeddings + 4
        want = self.model.mm_projector(out.hidden_states[-2][:, 1:])
        self.assertTrue(torch.allclose(f, want.detach(), atol=1e-6))

    def test_eos_is_the_tokenizers_not_the_deltas(self) -> None:
        self.assertEqual(self.model.config.text_config.eos_token_id, 2)

    def test_the_weights_are_complete(self) -> None:
        self.assertEqual(set(self.model.language_model.state_dict()),
                         set(S.expected_text_keys(N_LAYERS)))


class TestGenerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = tiny_shikra()
        A.register()

    def _gen(self, **kw: Any) -> Dict[str, Any]:
        return A.fox_generate(
            self.model, tiny_batch(), max_new_tokens=4, seed=3,
            eager_layers=frozenset(range(3, 11)), ratio=0.4, **kw)

    def test_kv_cached_greedy_equals_full_recomputation(self) -> None:
        got = self._gen(single_branch=True, greedy=True)["token_ids"]
        inp = tiny_batch()
        ids = inp["input_ids"]
        want: List[int] = []
        with torch.no_grad():
            for _ in range(len(got)):
                out = self.model(input_ids=ids, pixel_values=inp["pixel_values"],
                                 attention_mask=torch.ones_like(ids))
                tok = int(out.logits[0, -1].argmax())
                want.append(tok)
                ids = torch.cat([ids, torch.tensor([[tok]])], dim=1)
        self.assertEqual(got, want)

    def test_fox_fires_on_3_to_10_with_m_sys_scoring(self) -> None:
        with _Counting() as c:
            out = self._gen()
        self.assertEqual(out["image_span"], (34, 289))
        self.assertGreater(out["n_gen_tokens"], 0)
        self.assertEqual(set(c.layers), set(range(3, 11)))
        self.assertEqual(len(c.layers), 8 * out["n_gen_tokens"])
        self.assertEqual(set(c.use_m_sys), {True})

    def test_sampling_never_calls_the_intervention(self) -> None:
        with _Counting() as c:
            self._gen(single_branch=True)
        self.assertEqual(c.layers, [])

    def test_the_vision_tower_is_never_on_the_fox_attention(self) -> None:
        for n, m in self.model.vision_tower.named_modules():
            self.assertNotEqual(
                getattr(getattr(m, "config", None), "_attn_implementation",
                        None), A.ATTN_NAME, n)
        self.assertEqual(
            self.model.language_model.config._attn_implementation, A.ATTN_NAME)

    def test_every_decode_path_runs_under_inference_mode(self) -> None:
        """The decorator must stay on fox_generate (a refactor once leaked
        1.26 GiB per sample), and the wrapper's own work runs inside it."""
        self.assertTrue(hasattr(A.fox_generate, "__wrapped__"))
        seen: List[bool] = []
        orig = S.ShikraForFox.encode_image

        def spy(self_: Any, px: torch.Tensor) -> torch.Tensor:
            seen.append(torch.is_inference_mode_enabled())
            return orig(self_, px)

        with mock.patch.object(S.ShikraForFox, "encode_image", spy):
            self._gen()
            self._gen(single_branch=True)
        self.assertEqual(seen, [True, True])

    def test_the_masses_partition_and_m_sys_is_positive(self) -> None:
        b = tiny_batch()
        with A.FOX.configured(img_start=34, img_end=289, enabled=False,
                              diagnose=True, collect=True,
                              eager_layers=frozenset()):
            A.FOX.reset_diagnostics()
            with torch.no_grad():
                self.model(**b)
            diags = {k: {kk: vv.clone() for kk, vv in v.items()}
                     for k, v in A.FOX.diagnostics.items()}
        self.assertEqual(len(diags), N_LAYERS)
        for layer, row in diags.items():
            ms, mv, mt = (float(row[f"m_{n}_t1"].mean())
                          for n in ("sys", "vis", "txt"))
            self.assertGreater(ms, 0.0, f"layer {layer}")
            self.assertAlmostEqual(ms + mv + mt, 1.0, places=5)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
class TestLoading(unittest.TestCase):
    def test_the_delta_id_is_routed_before_autoconfig(self) -> None:
        sentinel = (object(), object())
        with mock.patch("transformers.AutoConfig.from_pretrained",
                        side_effect=AssertionError("AutoConfig was called")), \
                mock.patch.object(S, "load_shikra", return_value=sentinel) as m:
            got = A.load_backbone(S.SHIKRA_DELTA_ID, dtype="fp16",
                                  allow_cpu=True)
        self.assertIs(got, sentinel)
        self.assertEqual(m.call_args.args[0], S.SHIKRA_DELTA_ID)

    def test_other_ids_are_not_routed(self) -> None:
        self.assertFalse(S.is_shikra_id("llava-hf/llava-1.5-7b-hf"))
        self.assertTrue(S.is_shikra_id(S.SHIKRA_DELTA_ID))

    def test_only_fp16_is_accepted(self) -> None:
        for dt in ("fp32", "bf16"):
            with self.subTest(dt), self.assertRaisesRegex(ValueError, "fp16"):
                S.load_shikra(S.SHIKRA_DELTA_ID, dtype=dt, allow_cpu=True)

    def test_a_session_merge_end_to_end_on_a_toy_checkpoint(self) -> None:
        """delta + base on disk -> merge -> LlamaForCausalLM from a state
        dict: the real from_pretrained(None, state_dict=...) path, on CPU."""
        A.register()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fx = write_fixture(root)
            cfg = LlamaConfig(hidden_size=H, intermediate_size=FF,
                              num_hidden_layers=NL, num_attention_heads=2,
                              num_key_value_heads=2, vocab_size=V1)
            sd, proj = S.merge_delta(root / "delta", root / "base",
                                     n_layers=NL)
            lm, info = LlamaForCausalLM.from_pretrained(
                None, config=cfg, state_dict=sd, dtype=torch.float16,
                attn_implementation=A.ATTN_NAME, output_loading_info=True)
        A._assert_fully_loaded("toy", info)
        self.assertEqual(info["unexpected_keys"], [])
        self.assertEqual(lm.config._attn_implementation, A.ATTN_NAME)
        want = release_apply_delta(fx["base"], fx["delta"])
        for k, v in lm.state_dict().items():
            self.assertTrue(torch.equal(v, want[k]), k)

    def test_load_shikra_end_to_end_on_a_toy_delta(self) -> None:
        """The whole of load_shikra on CPU: local delta + local base, the
        Hub-only pieces (CLIP, tokenizer files) stubbed. Not exercised: the
        GPU dispatch and the real checkpoint's size."""
        import os
        A.register()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fx = write_fixture(root)
            LlamaConfig(hidden_size=H, intermediate_size=FF,
                        num_hidden_layers=NL, num_attention_heads=2,
                        num_key_value_heads=2, vocab_size=V0,
                        bos_token_id=1, eos_token_id=2).save_pretrained(
                            root / "base")
            tiny_clip = CLIPVisionModel._from_config(CLIPVisionConfig(
                hidden_size=HC, intermediate_size=8, num_hidden_layers=2,
                num_attention_heads=2, image_size=224, patch_size=14)).half()
            stub = S.ShikraProcessor(StubTokenizer(), StubImageProcessor())
            with mock.patch.dict(os.environ,
                                 {"FOX_SHIKRA_BASE": str(root / "base")}), \
                    mock.patch.object(CLIPVisionModel, "from_pretrained",
                                      return_value=tiny_clip) as clip, \
                    mock.patch.object(S, "_build_processor", return_value=stub):
                model, proc = S.load_shikra(
                    str(root / "delta"), dtype="fp16", device_map="cpu",
                    allow_cpu=True, eager_layers=frozenset(range(3, 11)))
            want = release_apply_delta(fx["base"], fx["delta"])
        self.assertIs(proc, stub)
        self.assertEqual(clip.call_args.args[0], S.SHIKRA_CLIP_ID)
        self.assertEqual(model.config.model_type, "shikra")
        self.assertEqual(model.config.text_config.vocab_size, V1)
        self.assertEqual(model.config.text_config.eos_token_id, 2)
        self.assertEqual(model.language_model.config._attn_implementation,
                         A.ATTN_NAME)
        self.assertEqual(model.provenance["source"], "session-time merge")
        for k, v in model.language_model.state_dict().items():
            self.assertTrue(torch.equal(v, want[k]), k)
        self.assertTrue(torch.equal(model.mm_projector.weight,
                                    want["model.mm_projector.weight"]))
        self.assertTrue(torch.equal(model.mm_projector.bias,
                                    want["model.mm_projector.bias"]))
        self.assertEqual(model.mm_projector.weight.dtype, torch.float16)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        self.assertEqual(A.FOX.eager_layers, frozenset(range(3, 11)))

    def test_incomplete_loading_is_a_hard_error(self) -> None:
        with self.assertRaises(RuntimeError):
            A._assert_fully_loaded("x", {"missing_keys": ["lm_head.weight"]})


# --------------------------------------------------------------------------- #
# the build script
# --------------------------------------------------------------------------- #
_BUILD_SRC = (Path(__file__).resolve().parents[1] / "scripts"
              / "build_shikra_checkpoint.py")


def _load_build():
    spec = importlib.util.spec_from_file_location("build_shikra", _BUILD_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestBuildScript(unittest.TestCase):
    """build_shikra_checkpoint.py on a toy delta, offline, tokenizer skipped."""

    def _fixture(self, root: Path):
        fx = write_fixture(root)
        LlamaConfig(hidden_size=H, intermediate_size=FF,
                    num_hidden_layers=NL, num_attention_heads=2,
                    num_key_value_heads=2, vocab_size=V0,
                    bos_token_id=1, eos_token_id=2).save_pretrained(
                        root / "base")
        return fx

    def _args(self, root: Path, *extra: str) -> List[str]:
        return ["--delta", str(root / "delta"), "--base", str(root / "base"),
                "--out", str(root / "out"), "--skip-tokenizer",
                "--shard-gb", "0.0000002", *extra]

    def test_build_and_verify_pass_and_the_output_is_loadable(self) -> None:
        B = _load_build()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fx = self._fixture(root)
            self.assertEqual(B.main(self._args(root, "--verify")), 0)
            out = root / "out"
            names = sorted(p.name for p in out.iterdir())
            self.assertIn("shikra_projector.safetensors", names)
            self.assertIn("shikra_build.json", names)
            shards = [n for n in names if n.startswith("model-")]
            self.assertGreater(len(shards), 1)      # sharding actually ran
            self.assertTrue(all("-of-%05d" % len(shards) in n for n in shards))
            self.assertFalse([n for n in names if n.endswith(".tmp")])
            meta = json.loads((out / "shikra_build.json").read_text())
            self.assertEqual(meta["vocab_size"], V1)
            self.assertEqual(set(meta["sha256"]), set(shards)
                             | {"shikra_projector.safetensors"})
            # The merged copy loads through the same loader a session uses.
            A.register()
            tiny_clip = CLIPVisionModel._from_config(CLIPVisionConfig(
                hidden_size=HC, intermediate_size=8, num_hidden_layers=2,
                num_attention_heads=2, image_size=224, patch_size=14)).half()
            stub = S.ShikraProcessor(StubTokenizer(), StubImageProcessor())
            with mock.patch.object(CLIPVisionModel, "from_pretrained",
                                   return_value=tiny_clip), \
                    mock.patch.object(S, "_build_processor", return_value=stub):
                model, _ = S.load_shikra(str(out), dtype="fp16",
                                         device_map="cpu", allow_cpu=True)
            want = release_apply_delta(fx["base"], fx["delta"])
        self.assertEqual(model.provenance["source"], "merged copy")
        for k, v in model.language_model.state_dict().items():
            self.assertTrue(torch.equal(v, want[k]), k)
        self.assertTrue(torch.equal(model.mm_projector.weight,
                                    want["model.mm_projector.weight"]))

    def test_verify_catches_a_flipped_value(self) -> None:
        from safetensors.torch import load_file
        B = _load_build()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._fixture(root)
            B.main(self._args(root))
            out = root / "out"
            shard = sorted(out.glob("model-*.safetensors"))[0]
            tensors = load_file(str(shard))
            k = sorted(tensors)[0]
            tensors[k] = tensors[k].clone()
            flat = tensors[k].view(-1)
            flat[0] = flat[0] + torch.tensor(1.0, dtype=torch.float16)
            save_file(tensors, str(shard))
            self.assertEqual(B.main(self._args(root, "--verify-only")), 1)

    def test_a_missing_tokenizer_file_is_an_error_unless_skipped(self) -> None:
        B = _load_build()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._fixture(root)
            args = [a for a in self._args(root) if a != "--skip-tokenizer"]
            with self.assertRaises(FileNotFoundError):
                B.main(args)


# --------------------------------------------------------------------------- #
# the methods
# --------------------------------------------------------------------------- #
class _FakeProcessor:
    def __init__(self) -> None:
        self.seen_text: List[str] = []
        self.tokenizer = StubTokenizer()
        self._inner = S.ShikraProcessor(self.tokenizer, StubImageProcessor())

    def __call__(self, images: Any, text: str, return_tensors: str = "pt"):
        self.seen_text.append(text)
        return self._inner(images=images, text=text)


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
        self.model = tiny_shikra()
        self.model.provenance = {"source": "stub"}
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
            dtype="fp16", model_id=S.SHIKRA_DELTA_ID,
            prompt_template_id="pope_v0_raw", max_new_tokens=3,
            system_prompt_id="llava_v1", allow_cpu=True)
        base.update(over)
        return GenConfig(**base)

    def _run(self, method: str, **over: Any):
        m = get_method(method)(self._cfg(method, **over))
        m.load()
        return m, m.generate(self.sample, self.sample.question)

    def test_the_prompt_is_the_releases_behind_the_system_message(self) -> None:
        self._run("sampling_shikra")
        self.assertEqual(
            self.processor.seen_text,
            [S.SYSTEM_MESSAGE
             + "USER: <im_start><ImageHere><im_end> "
               "Is there a dog in the image? ASSISTANT:"])

    def test_other_prompts_are_refused(self) -> None:
        for name in ("sampling_shikra", "fox_shikra"):
            for over in ({"system_prompt_id": "none"},
                         {"wrapper_id": "newline"}):
                with self.subTest(name=name, over=over), \
                        self.assertRaises(ValueError):
                    get_method(name)(self._cfg(name, **over))

    def test_fox_resolves_the_papers_hyperparameters(self) -> None:
        m, out = self._run("fox_shikra")
        d = m.describe()
        r = d["resolved_hyperparameters"]
        self.assertEqual(r["ratio"], 0.4)
        self.assertAlmostEqual(r["tau_js"], 0.2)
        self.assertEqual((r["alpha"], r["beta"], r["gamma"]),
                         (2.0, 0.1, 10000.0))
        self.assertEqual(d["eager_layers"], list(range(3, 11)))
        self.assertIs(d["use_m_sys"], True)
        self.assertEqual(d["backbone"], "shikra")
        self.assertEqual(out.extra["image_span"], [34, 289])
        sk = d["shikra"]
        self.assertEqual(sk["select_layer"], -2)
        self.assertTrue(sk["drop_cls"])
        self.assertEqual(sk["n_patches"], 256)
        self.assertIn("1024 -> 4096", sk["projector"])
        self.assertEqual(sk["template"], S.SHIKRA_WRAPPER)
        self.assertEqual(sk["source"], "stub")      # the loader's provenance

    def test_explicit_values_still_override(self) -> None:
        m, _ = self._run("fox_shikra", k=0.5, intervention_layers="2-15")
        self.assertEqual(m.describe()["resolved_hyperparameters"]["ratio"], 0.5)
        self.assertEqual(m.describe()["eager_layers"], list(range(2, 16)))

    def test_the_methods_never_pool_with_fox(self) -> None:
        self.assertEqual({self.M.ShikraFoxMethod.name,
                          self.M.ShikraSamplingMethod.name},
                         {"fox_shikra", "sampling_shikra"})
        self.assertNotEqual(self.M.ShikraFoxMethod.HPARAM_DEFAULTS,
                            self.M.FOX_DEFAULTS)
        self.assertNotEqual(self.M.ShikraFoxMethod.name, "fox")

    def test_a_non_shikra_model_is_refused(self) -> None:
        self.model.config.model_type = "llava"
        m = get_method("sampling_shikra")(self._cfg("sampling_shikra"))
        with self.assertRaisesRegex(ValueError, "Shikra checkpoint"):
            m.load()

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
        self._run("sampling_shikra")
        self.assertEqual(set(rows), {1})
        rows.clear()
        self._run("fox_shikra")
        self.assertEqual(set(rows), {2})


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #
class TestPlan(unittest.TestCase):
    def setUp(self) -> None:
        from foxrepro.bench import plan as P
        self.P = P
        self.jobs = [j for j in P.PLAN if j.model_id == P.SHIKRA_MODEL_ID]

    def test_the_jobs_are_exactly_tables_1_and_2(self) -> None:
        jobs = self.jobs
        self.assertEqual({j.target for j in jobs},
                         {"table1_shikra", "table2_shikra"})
        self.assertEqual({j.method for j in jobs},
                         {"sampling_shikra", "fox_shikra"})
        pope = [j for j in jobs if j.benchmark == "pope"]
        chair = [j for j in jobs if j.benchmark == "chair"]
        self.assertEqual((len(pope), len(chair)), (18, 6))
        self.assertEqual({j.seed for j in jobs}, {42, 43, 44})
        self.assertEqual({j.split for j in pope},
                         {"random", "popular", "adversarial"})
        self.assertEqual({j.split for j in chair}, {"val2014_500"})
        self.assertEqual(len({j.job_id() for j in jobs}), len(jobs))
        self.assertEqual(len({j.run_key().filename() for j in self.P.PLAN}),
                         len(self.P.PLAN))

    def test_no_job_overrides_what_the_method_resolves(self) -> None:
        for j in self.jobs:
            self.assertEqual(j.system_prompt_id, "llava_v1")
            self.assertEqual(j.wrapper_id, "released")
            self.assertIsNone(j.k)
            self.assertIsNone(j.tau_js)
            self.assertIsNone(j.intervention_layers)

    def test_baseline_and_method_share_a_session_group(self) -> None:
        by_key: Dict[Any, set] = {}
        for j in self.jobs:
            by_key.setdefault(self.P.pair_key(j), set()).add(j.method)
        for methods in by_key.values():
            self.assertEqual(methods, {"sampling_shikra", "fox_shikra"})

    def test_fox_shikra_is_dual_branch_and_costed_as_an_estimate(self) -> None:
        self.assertIn("fox_shikra", self.P.DUAL_BRANCH)
        self.assertIn("fox_shikra", self.P.FOX_METHODS)
        sk = self.P.Job("t", "fox_shikra", "pope", "random",
                        model_id=self.P.SHIKRA_MODEL_ID)
        ll = self.P.Job("t", "fox", "pope", "random")
        self.assertEqual(sk.branches, "dual")
        self.assertLess(sk.est_seconds(), ll.est_seconds())
        self.assertGreater(sk.est_seconds(), 0)

    def test_the_id_carries_the_backbone_tag_and_no_llava_id_changes(self) -> None:
        self.assertTrue(all(j.job_id().split(":")[4] == "sk"
                            for j in self.jobs))
        for j in self.P.PLAN:
            if j.model_id == self.P.LLAVA_MODEL_ID:
                self.assertNotIn(":sk", j.job_id())

    def test_commands_name_the_delta_and_no_quantized_checkpoint(self) -> None:
        for j in self.jobs:
            c = j.command().lower()
            self.assertIn("--model-id shikras/shikra-7b-delta-v1-0708", c)
            for banned in ("gptq", "8bit", "4bit", "awq", "bnb"):
                self.assertNotIn(banned, c)

    def test_the_gpu_methods_are_registered(self) -> None:
        from foxrepro.bench import generate as G
        self.assertTrue({"sampling_shikra", "fox_shikra"} <= G._GPU_METHODS)
        self.assertIsNotNone(get_method("fox_shikra"))


# --------------------------------------------------------------------------- #
# the smoke entry
# --------------------------------------------------------------------------- #
class TestTheShikraSmokeEntry(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = tiny_shikra()

    def _proc(self) -> _FakeProcessor:
        return _FakeProcessor()

    def test_every_check_passes_on_a_correct_layout(self) -> None:
        r = SMOKE.smoke_shikra_loaded(self.model, self._proc(), n_samples=3)
        self.assertTrue(r["ok"], r)
        self.assertEqual((r["img_start"], r["img_end"], r["derived_width"]),
                         (34, 289, 256))
        self.assertEqual(r["n_tokens_with_markers"], 258)
        for name in ("span_256_flanked_by_markers_258_total",
                     "markers_at_the_releases_33_and_290",
                     "all_weights_present",
                     "patched_forward_fires_on_3_to_10",
                     "vision_tower_not_patched",
                     "m_sys_m_vis_m_txt_partition_to_1",
                     "splice_differs_from_text_only_and_depends_on_image",
                     "mode_off_bit_identical_to_baseline",
                     "gamma0_same_argmax", "intervention_generates"):
            self.assertTrue(r["checks"].get(name), name)
        self.assertGreater(r["min_m_sys"], 0.0)
        self.assertGreater(r["splice_vs_text_only_max_abs_diff"], 0.0)

    def test_a_one_token_tokenisation_drift_is_rejected(self) -> None:
        with self.assertRaisesRegex(AssertionError, "tokenisation"):
            SMOKE.smoke_shikra_loaded(self.model, self._proc(),
                                      release_positions=(34, 291))

    def test_a_zero_projector_is_rejected(self) -> None:
        saved = self.model.mm_projector.weight.data.clone()
        self.model.mm_projector.weight.data.zero_()
        try:
            with self.assertRaisesRegex(AssertionError, "mm_projector"):
                SMOKE.smoke_shikra_loaded(self.model, self._proc())
        finally:
            self.model.mm_projector.weight.data.copy_(saved)

    def test_zero_marker_rows_are_rejected(self) -> None:
        emb = self.model.language_model.get_input_embeddings().weight
        saved = emb.data[S.IM_PATCH_ID:S.IM_END_ID + 1].clone()
        emb.data[S.IM_PATCH_ID:S.IM_END_ID + 1] = 0
        try:
            with self.assertRaisesRegex(AssertionError, "markers"):
                SMOKE.smoke_shikra_loaded(self.model, self._proc())
        finally:
            emb.data[S.IM_PATCH_ID:S.IM_END_ID + 1] = saved

    def test_an_unpatched_decoder_is_rejected(self) -> None:
        cfg = self.model.language_model.config
        before = cfg._attn_implementation_internal
        cfg._attn_implementation_internal = "sdpa"
        try:
            with self.assertRaises(AssertionError):
                SMOKE.smoke_shikra_loaded(self.model, self._proc())
        finally:
            cfg._attn_implementation_internal = before

    def test_a_patched_vision_tower_is_rejected(self) -> None:
        cfg = self.model.vision_tower.config
        before = cfg._attn_implementation_internal
        cfg._attn_implementation_internal = A.ATTN_NAME
        try:
            with self.assertRaisesRegex(AssertionError, "vision tower"):
                SMOKE.smoke_shikra_loaded(self.model, self._proc())
        finally:
            cfg._attn_implementation_internal = before

    def test_shikra_is_opt_in(self) -> None:
        self.assertNotIn(SMOKE.SHIKRA_CANDIDATE, SMOKE.CANDIDATES)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
