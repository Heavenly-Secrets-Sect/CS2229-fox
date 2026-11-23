"""Shikra-7B on the modern stack: a port, because transformers has no such class.

Shikra is a LLaMA-7B decoder (Vicuna-style, vocab 32000 + 3 image markers),
a stock CLIP ViT-L/14 and ONE linear layer between them. The release runs it
through ``modeling_shikra.py`` on transformers 4.29; nothing here needs that
file. What has to be written is exactly what docs/DEVIATIONS.md 3.11 said it
was: the weight merge, the projector and the embedding splice.

Layout this module reproduces, each line checked against the release
(``minigpt4/models/modeling_shikra.py`` / ``shikra.py``) and against the files
on the Hub rather than remembered:

* the checkpoint is a DELTA. ``shikras/shikra-7b-delta-v1-0708`` plus
  ``huggyllama/llama-7b`` gives the model; the arithmetic is the release's own
  ``apply_delta.py`` (fp16 ``+=``, with the 3 extra vocabulary rows of
  ``embed_tokens`` / ``lm_head`` taken from the delta alone);
* image tokens: ``<im_patch>`` 32000, ``<im_start>`` 32001, ``<im_end>`` 32002;
* pixels: ``CLIPImageProcessor`` of ``openai/clip-vit-large-patch14`` (224 px,
  the checkpoint ships none), CLIP ViT-L/14 at fp16, ``hidden_states[-2]``
  with the CLS token dropped -> 256 patch features of width 1024;
* ``mm_projector``: ``nn.Linear(1024, 4096)``, the checkpoint's two extra
  tensors; its output REPLACES the embeddings of the 256 ``<im_patch>``
  positions, the markers keep their own token embeddings;
* the prompt (``INSTRUCTION_TEMPLATE["shikra"]`` in ``pope_eval.py`` and
  ``char_eval.py``): ``USER: <im_start><ImageHere><im_end> <question>
  ASSISTANT:`` with the system message concatenated in front, no
  separator, and BOS prepended.

Where the span comes from
-------------------------
The release finds the visual keys with ``img_s == 33 and img_e == 290``
(``Uncertain_head.py``). Those are the positions of ``<im_start>`` and
``<im_end>`` in THAT tokenizer's output, and its ``_span_visual_text`` then
takes the visual keys as ``[img_s + 1, img_e)`` = positions 34..289 = the 256
PATCHES, not the 258 tokens between and including the markers. Here the span
is derived from the ``<im_patch>`` ids, exactly as LLaVA's is from ``<image>``,
so it is (34, 289) -- the same set -- and nothing is hard-coded. The markers
stay outside it: ``<im_start>`` falls in ``[0, img_start)`` and so counts
towards ``m_sys`` (the release excludes it from all three regions); that
one-key difference is recorded in docs/DEVIATIONS.md 3.20.

The tokenisation detail that makes it 33 and not 34
---------------------------------------------------
The 2023 tokenizer strips the whitespace on both sides of an added token. The
current ``LlamaTokenizer`` does not: ``"USER: <im_start>"`` gains a lone ``"▁"``
token before the marker and another after ``<im_end>``, which moves the span by
one and changes the text the model sees. :func:`encode_prompt` therefore
splits the prompt at the markers, strips as the release did, encodes the two
text pieces separately and assembles the ids itself. Measured against the real
tokenizer files: ``<im_start>`` at 33, ``<im_end>`` at 290, as the release has.
"""

from __future__ import annotations

import json
import os
import types
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import nn

__all__ = [
    "SHIKRA_DELTA_ID", "SHIKRA_BASE_ID", "SHIKRA_CLIP_ID", "SYSTEM_MESSAGE",
    "SHIKRA_WRAPPER", "IM_PATCH_ID", "IM_START_ID", "IM_END_ID", "N_PATCHES",
    "SELECT_LAYER", "PROJECTOR_FILE", "ShikraForFox", "ShikraProcessor",
    "expected_text_keys", "iter_merged_tensors", "merge_delta",
    "splice_image_features", "encode_prompt", "load_shikra", "is_shikra_id",
    "describe_sources",
]

# --------------------------------------------------------------------------- #
# constants, each read off a file on the Hub (2026-11-24)
# --------------------------------------------------------------------------- #
#: The delta. Public, no token. 13,485,396,661 bytes of fp16 ``.bin`` shards.
SHIKRA_DELTA_ID = "shikras/shikra-7b-delta-v1-0708"
#: What the delta is relative to. 13,476,876,661 bytes of fp16 safetensors.
SHIKRA_BASE_ID = "huggyllama/llama-7b"
#: ``config.json`` of the delta: ``"mm_vision_tower"``.
SHIKRA_CLIP_ID = "openai/clip-vit-large-patch14"

#: ``added_tokens.json`` of the delta.
IM_PATCH_ID, IM_START_ID, IM_END_ID = 32000, 32001, 32002
VOCAB_SIZE = 32003
#: The tokenizer's ids. NOT the delta's ``config.json``, which says
#: ``bos_token_id: 0, eos_token_id: 1`` (an artefact of how the training
#: framework wrote it); the release generates with
#: ``eos_token_id=self.llama_tokenizer.eos_token_id`` = 2 and prepends the
#: tokenizer's BOS = 1.
BOS_ID, EOS_ID = 1, 2

#: ``vision_config.image_size // patch_size`` squared = (224 // 14) ** 2.
N_PATCHES = 256
#: ``config.mm_vision_select_layer``.
SELECT_LAYER = -2
MM_HIDDEN, LM_HIDDEN = 1024, 4096

#: ``eval_configs/shikra_eval.yaml: system_message``. Byte-identical to the
#: LLaVA one in ``methods._LLAVA_V1_SYSTEM``, and concatenated to the prompt
#: with a bare ``+`` (``shikra.py``: ``self.system_message + p``).
SYSTEM_MESSAGE = (
    "A chat between a curious user and an artificial intelligence "
    "assistant. The assistant gives helpful, detailed, and polite answers "
    "to the user's questions."
)
#: ``INSTRUCTION_TEMPLATE["shikra"]`` with ``<question>`` -> ``{instruction}``.
SHIKRA_WRAPPER = "USER: <im_start><ImageHere><im_end> {instruction} ASSISTANT:"
_PLACEHOLDER = "<im_start><ImageHere><im_end>"

#: The two tensors of ``ShikraLlamaModel.mm_projector``, saved on their own by
#: ``scripts/build_shikra_checkpoint.py``.
PROJECTOR_FILE = "shikra_projector.safetensors"
_PROJECTOR_KEYS = ("model.mm_projector.weight", "model.mm_projector.bias")
#: Tensors whose first axis grew from 32000 to 32003.
_VOCAB_TENSORS = ("model.embed_tokens.weight", "lm_head.weight")


def is_shikra_id(model_id: str) -> bool:
    """True for the delta repo, a local copy of it, or a merged build.

    Cheap and offline: it is what lets ``attention.load_backbone`` route
    before ``AutoConfig`` runs, which would raise on ``model_type: "shikra"``.
    """
    if model_id == SHIKRA_DELTA_ID:
        return True
    p = Path(model_id)
    return p.is_dir() and ((p / PROJECTOR_FILE).exists()
                           or (p / "pytorch_model.bin.index.json").exists())


# --------------------------------------------------------------------------- #
# the merge
# --------------------------------------------------------------------------- #
def expected_text_keys(n_layers: int) -> List[str]:
    """Every tensor name of a ``LlamaForCausalLM`` with ``n_layers`` blocks.

    The merge's completeness check compares against THIS, not against what
    the delta happened to contain, so a delta that lost a tensor is an error
    rather than a smaller model.
    """
    keys = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for i in range(n_layers):
        p = f"model.layers.{i}."
        keys += [p + f"self_attn.{n}_proj.weight" for n in "qkvo"]
        keys += [p + f"mlp.{n}_proj.weight" for n in ("gate", "up", "down")]
        keys += [p + "input_layernorm.weight",
                 p + "post_attention_layernorm.weight"]
    return keys


def _is_rope_buffer(name: str) -> bool:
    # 32 of the delta's 325 tensors. They are a recomputable buffer, not a
    # weight, and today's LlamaRotaryEmbedding does not store it. 325 = 293 +
    # 32: see docs/DEVIATIONS.md 3.20 for what is lost by recomputing it.
    return name.endswith("rotary_emb.inv_freq")


def _load_bin(path: Path) -> Dict[str, torch.Tensor]:
    """A pickled shard, safely. ``weights_only`` refuses arbitrary objects.

    ``mmap=True`` keeps the 10 GB shard on disk (page cache) instead of in
    anonymous memory; it needs the zip format torch has written since 1.6,
    which these 2023 files are. If the mmap is refused the shard is read into
    RAM instead and the caller is told, because on a 29 GB host that is the
    difference between 13 GB and 23 GB of peak.
    """
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True,
                          mmap=True)
    except (RuntimeError, ValueError) as exc:
        print(f"[shikra] WARNING: could not mmap {path.name} ({exc}); "
              f"reading it into RAM, peak memory is higher")
        return torch.load(str(path), map_location="cpu", weights_only=True)


def iter_merged_tensors(
    delta_dir: Path, base_dir: Path, *, n_layers: int = 32,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Yield ``(name, tensor)`` for every merged weight, one at a time.

    This is ``apply_delta.py`` rewritten to stream: the release's version
    loads both models (2 x 13.5 GB of RAM plus the output); this reads one
    delta shard at a time and one base tensor at a time, so the only thing
    that ever grows is whatever the caller keeps. The arithmetic is the same
    fp16 add, in the same order, on the same operands, so the results are
    bit-identical (``tests/test_shikra.py`` proves that against a literal
    transcription of the release function).

    Yields the ``model.mm_projector.*`` tensors unchanged and without a base
    (they have none), and everything else merged. ``rotary_emb.inv_freq`` is
    skipped. After the last yield it verifies that every name of
    :func:`expected_text_keys` and both projector tensors were produced, and
    raises if not -- a partial merge that loads is a model with random rows.
    """
    delta_dir, base_dir = Path(delta_dir), Path(base_dir)
    d_idx = json.loads((delta_dir / "pytorch_model.bin.index.json").read_text())
    b_idx = json.loads(
        (base_dir / "model.safetensors.index.json").read_text())["weight_map"]
    d_map = d_idx["weight_map"]
    if not d_map:
        raise ValueError(f"{delta_dir}: the delta index lists no tensors")

    from safetensors import safe_open

    handles: Dict[str, Any] = {}

    def base_tensor(name: str) -> torch.Tensor:
        try:
            shard = b_idx[name]
        except KeyError:
            raise KeyError(f"{name!r} is in the delta but not in the base "
                           f"({base_dir}); wrong base model?") from None
        if shard not in handles:
            handles[shard] = safe_open(str(base_dir / shard), framework="pt")
        return handles[shard].get_tensor(name)

    seen: List[str] = []
    try:
        for shard in sorted(set(d_map.values())):
            tensors = _load_bin(delta_dir / shard)
            for name in [n for n, s in d_map.items() if s == shard]:
                if _is_rope_buffer(name):
                    continue
                d = tensors[name]
                if name in _PROJECTOR_KEYS:
                    seen.append(name)
                    yield name, d.detach().clone()
                    continue
                b = base_tensor(name)
                if d.dtype != torch.float16 or b.dtype != torch.float16:
                    raise TypeError(
                        f"{name}: delta {d.dtype}, base {b.dtype}. The "
                        f"release merges in fp16 and so must this; casting "
                        f"first would change the rounding")
                if d.shape == b.shape:
                    out = d + b
                else:
                    if name not in _VOCAB_TENSORS or d.shape[1] != b.shape[1] \
                            or d.shape[0] < b.shape[0]:
                        raise ValueError(
                            f"{name}: delta {tuple(d.shape)} vs base "
                            f"{tuple(b.shape)} is not the 3-new-rows case "
                            f"apply_delta.py allows")
                    # Rows 32000..32002 exist only in the delta: they ARE the
                    # trained values of the three markers, with nothing added.
                    out = d.clone()
                    out[:b.shape[0], :b.shape[1]] += b
                seen.append(name)
                yield name, out
            del tensors
    finally:
        handles.clear()

    want = set(expected_text_keys(n_layers)) | set(_PROJECTOR_KEYS)
    got = set(seen)
    if got != want or len(seen) != len(got):
        raise RuntimeError(
            f"merge incomplete: missing {sorted(want - got)[:5]}, "
            f"unexpected {sorted(got - want)[:5]}, "
            f"duplicates {len(seen) - len(got)}")


def merge_delta(
    delta_dir: Path, base_dir: Path, *, n_layers: int = 32,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """``(llama_state_dict, projector_state_dict)``, held in RAM.

    12.56 GiB for the real model. The projector dict is keyed ``weight`` /
    ``bias`` so it loads straight into ``nn.Linear``.
    """
    llama: Dict[str, torch.Tensor] = {}
    proj: Dict[str, torch.Tensor] = {}
    for name, t in iter_merged_tensors(delta_dir, base_dir, n_layers=n_layers):
        if name in _PROJECTOR_KEYS:
            proj[name.rsplit(".", 1)[-1]] = t
        else:
            llama[name] = t
    return llama, proj


# --------------------------------------------------------------------------- #
# prompt -> ids
# --------------------------------------------------------------------------- #
def encode_prompt(tokenizer: Any, text: str, *, n_patches: int = N_PATCHES,
                  bos_id: int = BOS_ID) -> List[int]:
    """Ids of a Shikra prompt, assembled the way the 2023 tokenizer did.

    ``text`` holds ``<im_start><ImageHere><im_end>`` exactly once. The text
    left of it has its trailing whitespace removed, the text right of it its
    leading whitespace, each piece is encoded on its own (so each gets
    SentencePiece's leading ``▁``, as before), and the markers and
    ``n_patches`` ``<im_patch>`` ids are inserted by hand. See the module
    docstring for why the tokenizer is not simply called on ``text``.
    """
    if text.count(_PLACEHOLDER) != 1:
        raise ValueError(
            f"expected exactly one {_PLACEHOLDER!r} in the prompt, found "
            f"{text.count(_PLACEHOLDER)}: {text[:80]!r}")
    left, right = text.split(_PLACEHOLDER)
    left, right = left.rstrip(), right.lstrip()

    def enc(piece: str) -> List[int]:
        return list(tokenizer.encode(piece, add_special_tokens=False)) \
            if piece else []

    return ([bos_id] + enc(left) + [IM_START_ID] + [IM_PATCH_ID] * n_patches
            + [IM_END_ID] + enc(right))


class ShikraProcessor:
    """``processor(images=, text=, return_tensors="pt")``, as the methods call it.

    HF has no processor for this model. ``tokenizer`` is what ``generate``
    decodes with; ``image_processor`` is the CLIP one, loaded from the CLIP
    repo because the checkpoint ships none (docs/DEVIATIONS.md 3.11).
    """

    def __init__(self, tokenizer: Any, image_processor: Any) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __call__(self, images: Any, text: str,
                 return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        if return_tensors != "pt":
            raise ValueError("only return_tensors='pt' is implemented")
        ids = torch.tensor([encode_prompt(self.tokenizer, text)],
                           dtype=torch.long)
        px = self.image_processor(images=images, return_tensors="pt")
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                "pixel_values": px["pixel_values"]}


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
def splice_image_features(
    input_ids: torch.Tensor, embeds: torch.Tensor, feats: torch.Tensor,
) -> torch.Tensor:
    """Write ``feats`` over the ``<im_patch>`` positions of ``embeds``.

    The release's ``ShikraLlamaModel.forward`` with ``use_im_start_end``:
    for every row, the patch positions must be exactly ``feats.shape[1]``
    contiguous ids, preceded by ``<im_start>`` and followed by ``<im_end>``,
    else ``ValueError``; the markers keep their token embeddings.

    Returns a new tensor; ``embeds`` is not modified.
    """
    if feats.shape[0] != input_ids.shape[0]:
        raise ValueError(f"{feats.shape[0]} feature rows for "
                         f"{input_ids.shape[0]} prompts")
    n = int(feats.shape[1])
    out = embeds.clone()
    for b in range(input_ids.shape[0]):
        pos = (input_ids[b] == IM_PATCH_ID).nonzero(as_tuple=True)[0]
        if pos.numel() != n:
            raise ValueError(
                f"row {b}: {pos.numel()} <im_patch> ids for {n} image "
                f"features")
        s = int(pos[0])
        if int(pos[-1]) - s + 1 != n:
            raise ValueError(f"row {b}: <im_patch> ids are not contiguous")
        if s == 0 or int(input_ids[b, s - 1]) != IM_START_ID:
            raise ValueError(f"row {b}: <im_start> does not precede the patches")
        if s + n >= input_ids.shape[1] or int(input_ids[b, s + n]) != IM_END_ID:
            raise ValueError(f"row {b}: <im_end> does not follow the patches")
        out[b, s:s + n] = feats[b].to(device=out.device, dtype=out.dtype)
    return out


class ShikraForFox(nn.Module):
    """The decoder, the vision tower and the projector behind one ``forward``.

    It is shaped to be what ``fox_generate`` expects of any model: called as
    ``model(input_ids=, attention_mask=, pixel_values=, use_cache=True)`` for
    the prefill and ``model(input_ids=, attention_mask=, past_key_values=,
    use_cache=True)`` afterwards; ``.config.model_type`` /
    ``.config.image_token_index`` / ``.config.text_config.eos_token_id``.

    The decoder is a stock ``LlamaForCausalLM`` carrying the Fox attention
    (``language_model``, registered first so ``next(parameters())`` is its
    embedding, which is where the inputs belong). The vision tower is a
    separate stock ``CLIPVisionModel`` and is NEVER patched: the intervention
    is defined over image KEYS in the decoder, and patching the encoder would
    change the features themselves (the same rule as every other backbone).
    """

    def __init__(self, language_model: nn.Module, vision_tower: nn.Module,
                 mm_projector: nn.Linear, *,
                 select_layer: int = SELECT_LAYER) -> None:
        super().__init__()
        self.language_model = language_model
        self.vision_tower = vision_tower
        self.mm_projector = mm_projector
        self.select_layer = select_layer
        text_cfg = language_model.config
        self.config = types.SimpleNamespace(
            model_type="shikra", image_token_index=IM_PATCH_ID,
            text_config=text_cfg, mm_vision_select_layer=select_layer,
            mm_hidden_size=mm_projector.in_features,
            mm_use_im_start_end=True, num_image_patches=N_PATCHES,
        )

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``[B, 256, hidden]`` projected patch features."""
        vt = self.vision_tower
        p = next(vt.parameters())
        out = vt(pixel_values=pixel_values.to(device=p.device, dtype=p.dtype),
                 output_hidden_states=True)
        h = out.hidden_states[self.select_layer][:, 1:]      # drop CLS
        if h.shape[1] != N_PATCHES:
            raise ValueError(
                f"the vision tower produced {h.shape[1]} patch features, "
                f"expected {N_PATCHES}: wrong image size or tower")
        return self.mm_projector(h)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.Tensor] = None,
                past_key_values: Any = None, use_cache: bool = True,
                **unexpected: Any) -> Any:
        if unexpected:
            raise TypeError(f"unsupported arguments: {sorted(unexpected)}")
        lm = self.language_model
        has_patches = bool((input_ids == IM_PATCH_ID).any())
        if pixel_values is None:
            if has_patches:
                # Without this the <im_patch> embedding (a row the checkpoint
                # never trained for this) would be read as text.
                raise ValueError("input_ids carry <im_patch> but no "
                                 "pixel_values were given")
            return lm(input_ids=input_ids, attention_mask=attention_mask,
                      past_key_values=past_key_values, use_cache=use_cache)
        if not has_patches:
            raise ValueError("pixel_values given but input_ids carry no "
                             "<im_patch>: the image would be ignored")
        embeds = lm.get_input_embeddings()(input_ids)
        feats = self.encode_image(pixel_values)
        embeds = splice_image_features(input_ids, embeds, feats)
        return lm(inputs_embeds=embeds, attention_mask=attention_mask,
                  past_key_values=past_key_values, use_cache=use_cache)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _snapshot(repo: str, patterns: Sequence[str]) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, allow_patterns=list(patterns)))


_DELTA_PATTERNS = (
    "pytorch_model-*.bin", "pytorch_model.bin.index.json", "config.json",
    "added_tokens.json", "special_tokens_map.json", "tokenizer.model",
    "tokenizer_config.json", "generation_config.json")
#: safetensors only: the repo also holds a duplicate ``.bin`` set that would
#: double the transfer for nothing.
_BASE_PATTERNS = ("model-*.safetensors", "model.safetensors.index.json",
                  "config.json")


def _revision(path: Path) -> str:
    """A Hub snapshot's commit hash (its directory name), else ``"local"``."""
    return path.name if path.parent.name == "snapshots" else "local"


def describe_sources(model_id: str) -> Dict[str, Any]:
    """What a run would fetch, for provenance and for the disk estimate."""
    return {
        "delta": model_id if model_id == SHIKRA_DELTA_ID else str(model_id),
        "base": os.environ.get("FOX_SHIKRA_BASE", SHIKRA_BASE_ID),
        "clip": SHIKRA_CLIP_ID,
        "merge": "apply_delta.py semantics, fp16, streamed",
    }


def _text_config(base_dir: Path, vocab_size: int = VOCAB_SIZE) -> Any:
    from transformers import LlamaConfig

    cfg = LlamaConfig.from_pretrained(str(base_dir))
    cfg.vocab_size = vocab_size
    # The TOKENIZER's ids (see BOS_ID). The delta's config says 0 / 1.
    cfg.bos_token_id, cfg.eos_token_id = BOS_ID, EOS_ID
    cfg.pad_token_id = 0
    cfg.use_cache = True
    return cfg


def _build_processor(tok_dir: Path, clip_id: str) -> ShikraProcessor:
    from transformers import AutoTokenizer, CLIPImageProcessor

    # use_fast=False: the release's tokenizer is the slow SentencePiece one,
    # and the base repo's tokenizer.json is for 32000 ids.
    tok = AutoTokenizer.from_pretrained(str(tok_dir), use_fast=False)
    check_tokenizer(tok)
    return ShikraProcessor(tok, CLIPImageProcessor.from_pretrained(clip_id))


def check_tokenizer(tok: Any) -> None:
    """Refuse a tokenizer that does not carry this checkpoint's ids."""
    got = (len(tok), tok.bos_token_id, tok.eos_token_id,
           tok.convert_tokens_to_ids(["<im_patch>", "<im_start>", "<im_end>"]))
    want = (VOCAB_SIZE, BOS_ID, EOS_ID, [IM_PATCH_ID, IM_START_ID, IM_END_ID])
    if got != want:
        raise ValueError(f"tokenizer ids {got} != Shikra's {want}")


#: Headroom kept free on the first GPU for the CLIP tower (0.6 GiB at fp16),
#: the projector and the activations of the dual-branch prefill.
_FIRST_GPU_RESERVE = 1 << 30


def _dispatch(lm: nn.Module, device_map: Any) -> nn.Module:
    """Spread ``lm`` over the visible GPUs, balanced, as ``device_map="auto"``.

    ``LlamaDecoderLayer`` is never split. UNTESTED on a GPU host: written
    against accelerate's documented API on a machine that has neither, and
    the first thing the Shikra smoke session exercises.
    """
    if device_map in (None, "cpu") or not torch.cuda.is_available():
        return lm
    if device_map != "auto":
        raise ValueError(f"device_map must be 'auto', 'cpu' or None, "
                         f"got {device_map!r}")
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.utils import get_balanced_memory

    no_split = ["LlamaDecoderLayer"]
    mem = get_balanced_memory(lm, no_split_module_classes=no_split,
                              dtype=torch.float16)
    if 0 in mem:
        mem[0] = max(0, int(mem[0]) - _FIRST_GPU_RESERVE)
    dmap = infer_auto_device_map(lm, max_memory=mem, dtype=torch.float16,
                                 no_split_module_classes=no_split)
    if any(v in ("cpu", "disk") for v in dmap.values()):
        raise RuntimeError(
            f"the decoder does not fit on the visible GPUs ({mem}); "
            f"offloading is refused because it would change the timing "
            f"model of every job and quantization is not an option")
    return dispatch_model(lm, device_map=dmap)


def load_shikra(
    model_id: str = SHIKRA_DELTA_ID, *, dtype: str = "fp16",
    device_map: Any = "auto", eager_layers: Optional[frozenset] = None,
    allow_cpu: bool = False,
) -> Tuple[ShikraForFox, ShikraProcessor]:
    """Build the model from the delta (session-time merge) or a merged copy.

    ``model_id`` is the delta repo or a local copy of it (merged in RAM, no
    merged file written), or a directory produced by
    ``scripts/build_shikra_checkpoint.py`` (loaded as a plain LlamaForCausalLM
    plus ``shikra_projector.safetensors``). fp16 only: the merge is fp16
    arithmetic and a cast to anything else would not be the release's model.
    """
    from transformers import CLIPVisionModel, LlamaForCausalLM

    from foxrepro.fox import attention as A

    if dtype != "fp16":
        raise ValueError(f"Shikra is merged and run in fp16; got {dtype!r}")
    A._refuse_cpu_only_host(dtype, allow_cpu=allow_cpu)
    A.register()
    if eager_layers is not None:
        A.FOX.eager_layers = eager_layers

    p = Path(model_id)
    merged_copy = p.is_dir() and (p / PROJECTOR_FILE).exists()
    prov: Dict[str, Any] = {"source": "merged copy" if merged_copy
                            else "session-time merge"}
    kw: Dict[str, Any] = dict(dtype=torch.float16,
                              attn_implementation=A.ATTN_NAME,
                              output_loading_info=True)
    if merged_copy:
        from safetensors.torch import load_file

        lm, info = LlamaForCausalLM.from_pretrained(
            str(p), device_map=(None if device_map in (None, "cpu")
                                else device_map), **kw)
        proj = load_file(str(p / PROJECTOR_FILE))
        tok_dir = p
        prov["path"] = str(p)
    else:
        delta_dir = p if p.is_dir() else _snapshot(model_id, _DELTA_PATTERNS)
        base_id = os.environ.get("FOX_SHIKRA_BASE", SHIKRA_BASE_ID)
        base_dir = Path(base_id) if Path(base_id).is_dir() \
            else _snapshot(base_id, _BASE_PATTERNS)
        base_cfg = _text_config(base_dir)
        sd, proj = merge_delta(delta_dir, base_dir,
                               n_layers=base_cfg.num_hidden_layers)
        # The vocabulary is whatever the merged embedding says (32003 for the
        # real checkpoint); the smoke script asserts the markers' rows exist.
        cfg = _text_config(base_dir, sd["model.embed_tokens.weight"].shape[0])
        # No device_map here: the merged tensors are assigned on the CPU
        # (low_cpu_mem_usage, no second copy), freed, and the model is then
        # spread over the GPUs by _dispatch. That is two well-trodden steps
        # instead of the less-travelled from_pretrained(state_dict=,
        # device_map=) combination, which could not be exercised on the host
        # this was written on (no accelerate).
        lm, info = LlamaForCausalLM.from_pretrained(
            None, config=cfg, state_dict=sd, **kw)
        del sd
        tok_dir = delta_dir
        prov.update(delta=str(delta_dir), base=str(base_dir),
                    delta_revision=_revision(delta_dir),
                    base_revision=_revision(base_dir),
                    clip=SHIKRA_CLIP_ID)
    A._assert_fully_loaded(model_id, info)
    if info.get("unexpected_keys"):
        raise RuntimeError(f"unexpected keys: {info['unexpected_keys'][:5]}")
    lm = lm.eval()
    if not merged_copy:
        lm = _dispatch(lm, device_map)
    if lm.config._attn_implementation != A.ATTN_NAME:
        raise RuntimeError(
            f"the decoder reports attn_implementation="
            f"{lm.config._attn_implementation!r}, not {A.ATTN_NAME!r}: Fox "
            f"would not be running and every number would be unpatched")
    if set(proj) != {"weight", "bias"}:
        raise RuntimeError(f"projector tensors are {sorted(proj)}")

    dev = next(lm.parameters()).device
    vt = CLIPVisionModel.from_pretrained(
        SHIKRA_CLIP_ID, dtype=torch.float16, attn_implementation="sdpa")
    vt = vt.to(dev).eval()
    out_f, in_f = proj["weight"].shape
    if out_f != lm.config.hidden_size or in_f != vt.config.hidden_size:
        raise RuntimeError(
            f"projector {tuple(proj['weight'].shape)} does not map the vision "
            f"tower's {vt.config.hidden_size} to the decoder's "
            f"{lm.config.hidden_size}")
    linear = nn.Linear(in_f, out_f)
    linear.load_state_dict(proj)
    linear = linear.to(device=dev, dtype=torch.float16).eval()
    model = ShikraForFox(lm, vt, linear).eval()
    # Nothing here is trained and nothing may keep a graph (fox_generate is
    # under inference_mode anyway; this is the belt to that pair of braces).
    model.requires_grad_(False)
    model.provenance = prov
    return model, _build_processor(tok_dir, SHIKRA_CLIP_ID)
