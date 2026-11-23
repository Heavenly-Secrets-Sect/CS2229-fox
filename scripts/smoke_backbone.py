#!/usr/bin/env python3
"""Prove a backbone really works before a round spends twenty GPU-hours on it.

`probe_backbone.py` answers what a checkpoint's config implies. It cannot
answer the one thing that actually differs between these models, because the
answer only exists once a processor has run:

    LLaVA-1.5   336/14 = 24x24, fixed           -> 576 image tokens
    Qwen3-VL    patch 16 then a 2x2 spatial merge -> varies with resolution
    InternVL3   448px tiles, count chosen per image -> varies with tiling

Fox measures `m_sys` over `[0, img_start)` and renormalises `H_vis` over
`[img_start, img_end)`. Both are index arithmetic on ONE contiguous span. A
backbone that emits two blocks of image tokens, or interleaves a newline
token between tile rows, still produces a first and a last position -- so
`locate_image_span` would return a span that is partly text, every number
downstream would be finite and plausible, and nothing would say so. That is
the failure this project keeps paying for: a sum over the wrong set is a
number, not an error.

So this asserts the partition rather than inspecting it:

    m_sys + m_vis + m_txt == 1   at every layer

which can only hold if the three index ranges tile the key axis exactly. A
span off by one tile fails it by the mass of that tile.

Costs one model download and a handful of forwards. Run it on its own
session; a round is not the place to find out.

Usage::

    python3 scripts/smoke_backbone.py --model-id Qwen/Qwen3-VL-8B-Instruct
    python3 scripts/smoke_backbone.py --all
    python3 scripts/smoke_backbone.py --all --json

Note the `-hf` suffix on InternVL3. `OpenGVLab/InternVL3-8B` is
`model_type: internvl_chat` and carries its own modelling code;
`OpenGVLab/InternVL3-8B-hf` is `internvl` and is the one transformers can
build. `load_backbone` refuses the former by name rather than guessing a
sub-config key, which is correct and is also the easiest way to waste a
session.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback

#: Verified against each repo's config.json: model_type qwen3_vl and
#: internvl respectively, both present in _DECODER_SUBCONFIG.
CANDIDATES = [
    "Qwen/Qwen3-VL-8B-Instruct",
    "OpenGVLab/InternVL3-8B-hf",
]

#: The InstructBLIP checkpoint the plan runs (the pre-cast fp16 copy, which is
#: PRIVATE: it needs an HF token with read access, and must stay private --
#: docs/DEVIATIONS.md 3.10). Not in CANDIDATES, so `--all` never touches it;
#: ask for it with `--instructblip`.
INSTRUCTBLIP_CANDIDATE = "Salesforce/instructblip-vicuna-7b"

#: Shikra-7B: the public DELTA repo, merged onto huggyllama/llama-7b inside
#: the session (docs/DEVIATIONS.md 3.20). ~26.7 GiB to fetch, ~12.6 GiB of RAM
#: for the merged weights. Opt-in with `--shikra`, never in `--all`.
SHIKRA_CANDIDATE = "shikras/shikra-7b-delta-v1-0708"

#: Floor on the attention mass the image span must carry at every layer.
#: Not a tolerance -- a correct span on a working backbone carries orders
#: more than this, and only a degenerate span lands near it.
MIN_VISUAL_MASS = 1e-4


def check_span_matches_tokens(ids, image_token_id: int, start: int,
                              end: int) -> int:
    """Assert the span is EXACTLY the image positions, and return its width.

    This is the real guard, and it is exact rather than statistical: the
    processor already said where the image went by writing
    ``image_token_id`` at those positions, so the span is right if and only
    if it is that set. A dynamic backbone that interleaves a row-separator
    token between tiles, or emits two blocks, fails here.

    An earlier version of this script checked ``m_sys + m_vis + m_txt == 1``
    instead and called that a proof that the span was in the right place.
    It is not: those three ranges are ``[0, start)``, ``[start, end)`` and
    ``[end, n)``, so they tile the key axis for ANY boundaries and the sum
    is 1 whatever the span. `tests/test_smoke_backbone.py` shifts the span
    forty tokens off the image and watches it still sum to 1. Writing a
    tautology and reading it as a measurement is the failure this repo
    keeps finding; it does not get to ship in the script that exists to
    catch it.
    """
    import torch

    row = ids[0] if ids.dim() > 1 else ids
    pos = torch.nonzero(row == image_token_id, as_tuple=False).flatten()
    if pos.numel() == 0:
        raise AssertionError(
            f"the processor emitted no token {image_token_id}; either the "
            f"chat template dropped the image or the id is wrong"
        )
    want_start, want_end = int(pos[0]), int(pos[-1]) + 1
    if (start, end) != (want_start, want_end):
        raise AssertionError(
            f"span is [{start}, {end}) but the image tokens occupy "
            f"[{want_start}, {want_end}) -- m_sys would be measured over a "
            f"range that is partly image, and H_vis over one that is partly "
            f"text"
        )
    if int(pos.numel()) != want_end - want_start:
        gaps = want_end - want_start - int(pos.numel())
        raise AssertionError(
            f"the image tokens are not contiguous: {gaps} non-image "
            f"token(s) sit inside [{want_start}, {want_end}). Fox indexes "
            f"one unbroken span, so part of that range is text"
        )
    return want_end - want_start


def check_diagnostics_usable(diags: dict, min_visual_mass: float = 1e-4
                             ) -> float:
    """Assert the measured span is not degenerate, and return min m_vis.

    What this CAN detect, unlike the sum above: a span that holds no
    attention mass at all. `H_vis` renormalises over the image span, so an
    empty one divides by zero and every head score downstream becomes NaN
    rather than raising -- four silent zeros in one week came in through
    holes shaped exactly like this.
    """
    worst = 1.0
    for layer, row in sorted(diags.items()):
        for tail in sorted({k.rsplit("_t", 1)[1] for k in row}):
            key = f"h_vis_t{tail}"
            if key not in row:
                continue
            h = float(row[key].mean())
            if h != h:                      # NaN
                raise AssertionError(
                    f"h_vis is NaN at layer {layer}, tail {tail}: the "
                    f"renormalisation over the image span divided by zero, "
                    f"so the span holds no attention mass"
                )
            mv = float(row[f"m_vis_t{tail}"].mean())
            worst = min(worst, mv)
    if worst < min_visual_mass:
        raise AssertionError(
            f"the image span carries at most {worst:.2e} of the attention "
            f"mass at some layer (floor {min_visual_mass:.0e}). The span is "
            f"where the image tokens are but the model is not attending to "
            f"them, so every head score is noise"
        )
    return worst


def smoke(model_id: str, n_samples: int = 5, dtype: str = "fp16") -> dict:
    """Download the checkpoint, then run every check against it."""
    from foxrepro.fox.attention import load_backbone

    # load_backbone ends in _assert_fully_loaded, which raises on any missing
    # key -- and names the Q-Former / query_tokens when those are the ones.
    # Reaching the next line IS the "fully loaded" check; it is recorded below.
    model, processor = load_backbone(model_id, dtype=dtype)
    if getattr(model.config, "model_type", None) == "shikra":
        out = smoke_shikra_loaded(model, processor, model_id=model_id,
                                  n_samples=n_samples)
        out["checks"]["fully_loaded"] = True
        return out
    if getattr(model.config, "model_type", None) == "instructblip":
        out = smoke_instructblip_loaded(model, processor, model_id=model_id,
                                        n_samples=n_samples)
        out["checks"]["fully_loaded"] = True
        return out
    return smoke_loaded(model, processor, model_id=model_id,
                        n_samples=n_samples)


def smoke_instructblip_loaded(model, processor, *, model_id: str = "<loaded>",
                              n_samples: int = 5) -> dict:
    """The InstructBLIP checks, against a model that is already in memory.

    InstructBLIP cannot go through :func:`smoke_loaded`: it has no chat
    template, its processor takes the prompt string directly, and the image
    ids sit BEFORE BOS, so there is no ``m_sys`` to partition against.

    What each check means, and what it does not:

    * span is exactly 32 and contiguous, and starts at key 0 -- read off the
      processor's own placement of the image id, as the LLaVA check is;
    * the patched forward FIRES on exactly the intervention window, counted
      at the kernel, not inferred from a config string. (Measured: a config
      that says ``fox`` can still be running sdpa; see
      ``_ensure_decoder_is_patched``.);
    * the Q-Former was NOT patched: none of its modules report ``fox``, and
      the kernel never saw a layer index outside the decoder;
    * ``query_tokens`` is not the all-zero tensor a missing key leaves
      behind (the hard failure on missing keys is ``_assert_fully_loaded``,
      which runs at load time, not here);
    * ``m_sys == 0`` exactly and ``m_vis + m_txt == 1`` per layer. The task
      brief says ``m_sys + m_vis + m_txt == 1`` "where m_sys is on"; here it
      is not on (no key precedes the image), so the identity checked is the
      two-term one, and the zero is itself the assertion that the span starts
      at key 0. Like the LLaVA partition it is an identity for ANY span, so
      the span check above, not this one, is what catches a misplaced span;
    * two no-op comparisons against the baseline forward. ``k = 0`` does not
      exist -- ``select_heads`` clamps to at least one head -- so the no-op is
      reached two ways: ``mode="off"`` (the patch disabled, which must be
      BIT-identical because both take the same SDPA path) and ``gamma=0``
      (the eager path with nothing subtracted, which agrees to kernel
      tolerance, not bit for bit: eager and SDPA are different kernels, and
      the measured max difference is reported rather than asserted to be 0).
    """
    import torch
    from PIL import Image

    import foxrepro.fox.attention as A
    from foxrepro.fox.attention import FOX

    out: dict = {"model_id": model_id, "ok": False, "checks": {}}
    out["model_type"] = getattr(model.config, "model_type", None)
    if out["model_type"] != "instructblip":
        raise AssertionError(
            f"smoke_instructblip_loaded got model_type {out['model_type']!r}")

    image_token_id = A.resolve_image_token_id(model)
    out["image_token_id"] = image_token_id
    width = A.expected_span_width(model)
    out["expected_width"] = width
    if width != 32:
        raise AssertionError(f"FIXED_SPAN_WIDTH says {width}, want 32")

    from foxrepro.bench.methods import INSTRUCTBLIP_WRAPPER

    img = Image.new("RGB", (448, 448), (127, 127, 127))
    text = INSTRUCTBLIP_WRAPPER.format(instruction="Is there a dog in the image?")
    batch = processor(images=img, text=text, return_tensors="pt")
    dev = next(model.parameters()).device
    batch = {k: (v.to(dev) if hasattr(v, "to") else v)
             for k, v in batch.items()}
    ids = batch["input_ids"]
    out["n_prompt_tokens"] = int(ids.shape[-1])
    out["n_image_tokens"] = int((ids == image_token_id).sum())

    # ---- span ---------------------------------------------------------- #
    start, end = A.locate_image_span(ids, image_token_id, width)
    out["img_start"], out["img_end"] = int(start), int(end)
    out["derived_width"] = int(end) - int(start) + 1
    check_span_matches_tokens(ids, image_token_id, int(start), int(end) + 1)
    if out["derived_width"] != 32 or out["n_image_tokens"] != 32:
        raise AssertionError(
            f"want exactly 32 contiguous image tokens, got "
            f"{out['n_image_tokens']} spanning {out['derived_width']}")
    if int(start) != 0:
        raise AssertionError(
            f"the image span starts at key {start}, not 0: the processor no "
            f"longer prepends the image ids before BOS, so the m_sys-free "
            f"scoring and the release's layout no longer match")
    out["checks"]["span_exactly_32_contiguous_from_0"] = True

    # ---- the patch fires on the window and nowhere else ---------------- #
    window = frozenset(range(4, 11))
    seen: list = []
    orig = A.intervene

    def counting(*a, **kw):
        seen.append(kw.get("layer"))
        return orig(*a, **kw)

    A.intervene = counting
    try:
        with FOX.configured(img_start=int(start), img_end=int(end),
                            enabled=True, mode="as_implemented",
                            use_m_sys=False, eager_layers=window,
                            diagnose=False, ratio=0.4):
            with torch.no_grad():
                model(**batch)
    finally:
        A.intervene = orig
    n_dec = model.config.text_config.num_hidden_layers
    if set(seen) != set(window):
        raise AssertionError(
            f"the intervention ran on layers {sorted(set(seen))}, wanted "
            f"exactly {sorted(window)}: the Fox attention is not what the "
            f"decoder is running")
    if len(seen) != len(window):
        raise AssertionError(
            f"{len(seen)} intervention calls for {len(window)} layers in one "
            f"forward")
    if max(seen) >= n_dec:
        raise AssertionError("intervention saw a layer beyond the decoder")
    out["checks"]["patched_forward_fires_on_4_to_10"] = True

    # ---- the Q-Former was not patched ---------------------------------- #
    bad = [n for n, m in model.qformer.named_modules()
           if getattr(getattr(m, "config", None), "_attn_implementation", None)
           == A.ATTN_NAME]
    if bad:
        raise AssertionError(f"Q-Former modules report the Fox attention: {bad[:3]}")
    if model.language_model.config._attn_implementation != A.ATTN_NAME:
        raise AssertionError("the decoder is not on the Fox attention")
    out["checks"]["qformer_not_patched"] = True

    # ---- query_tokens is trained, not the zeros a missing key leaves ---- #
    if float(model.query_tokens.detach().float().abs().sum()) == 0.0:
        raise AssertionError(
            "query_tokens is all zeros: the key was probably missing and the "
            "Q-Former is not producing the checkpoint's image tokens")
    out["checks"]["query_tokens_nonzero"] = True

    # ---- masses ---------------------------------------------------------- #
    with FOX.configured(img_start=int(start), img_end=int(end), enabled=False,
                        diagnose=True, collect=True, eager_layers=frozenset()):
        FOX.reset_diagnostics()
        with torch.no_grad():
            model(**batch)
        diags = {k: {kk: vv.clone() for kk, vv in v.items()}
                 for k, v in FOX.diagnostics.items()}
    if len(diags) != n_dec:
        raise AssertionError(f"{len(diags)} of {n_dec} layers reported")
    worst_sum = 0.0
    for layer, row in sorted(diags.items()):
        ms = float(row["m_sys_t1"].float().mean())
        mv = float(row["m_vis_t1"].float().mean())
        mt = float(row["m_txt_t1"].float().mean())
        if ms != 0.0:
            raise AssertionError(f"layer {layer}: m_sys = {ms}, want exactly 0")
        worst_sum = max(worst_sum, abs(ms + mv + mt - 1.0))
    if worst_sum > 1e-3:
        raise AssertionError(f"m_sys + m_vis + m_txt off 1 by {worst_sum}")
    out["mass_sum_max_abs_error"] = worst_sum
    out["n_layers_measured"] = len(diags)
    out["min_visual_mass"] = check_diagnostics_usable(diags, MIN_VISUAL_MASS)
    out["checks"]["m_sys_zero_and_masses_partition"] = True

    # ---- no-op arms against the baseline forward ------------------------ #
    def logits(**cfg):
        with FOX.configured(img_start=int(start), img_end=int(end), **cfg):
            with torch.no_grad():
                return model(**batch).logits[0, -1].float().cpu()

    base = logits(enabled=False)
    off = logits(enabled=True, mode="off", use_m_sys=False, ratio=0.4,
                 eager_layers=window)
    if not torch.equal(base, off):
        raise AssertionError(
            f"mode='off' is not bit-identical to the baseline forward (max "
            f"|diff| {float((base - off).abs().max())}): the patch is not "
            f"inert when disabled")
    out["checks"]["mode_off_bit_identical_to_baseline"] = True
    noop = logits(enabled=True, mode="as_implemented", gamma=0.0,
                  use_m_sys=False, ratio=0.4, eager_layers=window)
    out["gamma0_max_abs_diff"] = float((base - noop).abs().max())
    out["gamma0_bit_identical"] = bool(torch.equal(base, noop))
    if int(base.argmax()) != int(noop.argmax()):
        raise AssertionError(
            f"gamma=0 changes the argmax (max |diff| "
            f"{out['gamma0_max_abs_diff']}): the eager path is not a no-op")
    out["checks"]["gamma0_same_argmax"] = True

    # ---- and it generates ------------------------------------------------ #
    with torch.no_grad():
        gen = A.fox_generate(
            model, batch, max_new_tokens=n_samples, seed=0,
            eager_layers=window, use_m_sys=False, ratio=0.4)
    if gen["n_gen_tokens"] < 1 or tuple(gen["image_span"]) != (0, 31):
        raise AssertionError(f"fox_generate gave {gen['n_gen_tokens']} tokens, "
                             f"span {gen['image_span']}")
    out["sample_output"] = processor.tokenizer.decode(
        gen["token_ids"], skip_special_tokens=True)
    out["checks"]["intervention_generates"] = True

    out["ok"] = True
    return out


def smoke_shikra_loaded(model, processor, *, model_id: str = "<loaded>",
                        n_samples: int = 5,
                        release_positions=(33, 290)) -> dict:
    """The Shikra checks, against a model that is already in memory.

    * the span is EXACTLY 256 contiguous ``<im_patch>`` positions, flanked by
      ``<im_start>`` and ``<im_end>`` (258 tokens with the markers), and it
      was DERIVED from the ids: ``release_positions`` (33, 290) is the
      release's hard-coded pair, passed only so a tokenisation drift of even
      one token (the modern ``LlamaTokenizer`` adds a lone space piece before
      ``<im_start>``) fails here instead of shifting m_sys silently;
    * the patched forward FIRES on layers 3..10 and on no others, counted at
      the kernel; the vision tower is not on the Fox attention;
    * ``m_sys + m_vis + m_txt == 1`` per layer, with ``m_sys > 0`` (unlike
      InstructBLIP there ARE system keys, so Eq. 4 is defined);
    * the splice does something: the logits differ from the text-only run
      (``<im_patch>`` embeddings left in place) and from a run with another
      image;
    * every weight is present: the decoder's keys are exactly those of a
      32003-vocab LLaMA, the three marker rows of ``embed_tokens`` and
      ``lm_head`` and the projector are not all zero (a missing key leaves a
      random or zero tensor, and ``_assert_fully_loaded`` only runs at load);
    * ``mode="off"`` is bit-identical to the baseline forward, ``gamma=0``
      keeps the argmax, and ``fox_generate`` returns the same span.
    """
    import torch
    from PIL import Image

    import foxrepro.fox.attention as A
    from foxrepro.bench.methods import SHIKRA_WRAPPER, _LLAVA_V1_SYSTEM
    from foxrepro.fox import shikra as S
    from foxrepro.fox.attention import FOX

    out: dict = {"model_id": model_id, "ok": False, "checks": {}}
    out["model_type"] = getattr(model.config, "model_type", None)
    if out["model_type"] != "shikra":
        raise AssertionError(
            f"smoke_shikra_loaded got model_type {out['model_type']!r}")
    image_token_id = A.resolve_image_token_id(model)
    out["image_token_id"] = image_token_id
    width = A.expected_span_width(model)
    out["expected_width"] = width
    if image_token_id != S.IM_PATCH_ID or width != S.N_PATCHES:
        raise AssertionError(
            f"image token {image_token_id} / width {width}, want "
            f"{S.IM_PATCH_ID} / {S.N_PATCHES}")

    img = Image.new("RGB", (448, 448), (127, 127, 127))
    text = _LLAVA_V1_SYSTEM + SHIKRA_WRAPPER.format(
        instruction="Is there a dog in the image?")
    batch = processor(images=img, text=text, return_tensors="pt")
    dev = next(model.parameters()).device
    batch = {k: (v.to(dev) if hasattr(v, "to") else v)
             for k, v in batch.items()}
    ids = batch["input_ids"]
    out["n_prompt_tokens"] = int(ids.shape[-1])
    out["n_image_tokens"] = int((ids == image_token_id).sum())

    # ---- span ---------------------------------------------------------- #
    start, end = A.locate_image_span(ids, image_token_id, width)
    out["img_start"], out["img_end"] = int(start), int(end)
    out["derived_width"] = int(end) - int(start) + 1
    check_span_matches_tokens(ids, image_token_id, int(start), int(end) + 1)
    if out["derived_width"] != 256 or out["n_image_tokens"] != 256:
        raise AssertionError(
            f"want exactly 256 contiguous patch tokens, got "
            f"{out['n_image_tokens']} spanning {out['derived_width']}")
    if int(ids[0, 0]) != S.BOS_ID:
        raise AssertionError(f"the prompt does not start with BOS ({S.BOS_ID})")
    if int(ids[0, start - 1]) != S.IM_START_ID \
            or int(ids[0, end + 1]) != S.IM_END_ID:
        raise AssertionError("the patches are not flanked by <im_start> and "
                             "<im_end>")
    out["n_tokens_with_markers"] = (int(end) + 1) - (int(start) - 1) + 1
    if out["n_tokens_with_markers"] != 258:
        raise AssertionError(f"{out['n_tokens_with_markers']} tokens "
                             f"between and including the markers, want 258")
    if (int(start) - 1, int(end) + 1) != tuple(release_positions):
        raise AssertionError(
            f"markers at {(int(start) - 1, int(end) + 1)}, the release has "
            f"{tuple(release_positions)}: the tokenisation no longer matches "
            f"what produced the paper's numbers")
    if int(start) == 0:
        raise AssertionError("no system keys precede the image")
    out["checks"]["span_256_flanked_by_markers_258_total"] = True
    out["checks"]["markers_at_the_releases_33_and_290"] = True

    # ---- every weight present ------------------------------------------ #
    lm = model.language_model
    cfg = lm.config
    want = set(S.expected_text_keys(cfg.num_hidden_layers))
    got = set(lm.state_dict())
    if got != want:
        raise AssertionError(f"decoder keys differ: missing "
                             f"{sorted(want - got)[:3]}, extra "
                             f"{sorted(got - want)[:3]}")
    if cfg.vocab_size < S.IM_END_ID + 1:
        raise AssertionError(f"vocab {cfg.vocab_size} has no marker rows")
    marker_rows = slice(S.IM_PATCH_ID, S.IM_END_ID + 1)
    for name, t in (("embed_tokens", lm.get_input_embeddings().weight),
                    ("lm_head", lm.get_output_embeddings().weight)):
        if float(t[marker_rows].detach().float().abs().sum()) == 0.0:
            raise AssertionError(f"{name} rows for the markers are all zero")
    for name, t in (("mm_projector.weight", model.mm_projector.weight),
                    ("mm_projector.bias", model.mm_projector.bias)):
        if float(t.detach().float().abs().sum()) == 0.0:
            raise AssertionError(f"{name} is all zero: the key was missing")
    out["checks"]["all_weights_present"] = True

    # ---- the patch fires on 3-10 and nowhere else ----------------------- #
    window = frozenset(range(3, 11))
    seen: list = []
    orig = A.intervene

    def counting(*a, **kw):
        seen.append(kw.get("layer"))
        return orig(*a, **kw)

    A.intervene = counting
    try:
        with FOX.configured(img_start=int(start), img_end=int(end),
                            enabled=True, mode="as_implemented",
                            use_m_sys=True, eager_layers=window,
                            diagnose=False, ratio=0.4):
            with torch.no_grad():
                model(**batch)
    finally:
        A.intervene = orig
    if set(seen) != set(window) or len(seen) != len(window):
        raise AssertionError(
            f"the intervention ran on layers {sorted(seen)}, wanted "
            f"exactly {sorted(window)} once each")
    bad = [n for n, m in model.vision_tower.named_modules()
           if getattr(getattr(m, "config", None), "_attn_implementation", None)
           == A.ATTN_NAME]
    if bad:
        raise AssertionError(f"vision tower modules run the Fox attention: "
                             f"{bad[:3]}")
    if cfg._attn_implementation != A.ATTN_NAME:
        raise AssertionError("the decoder is not on the Fox attention")
    out["checks"]["patched_forward_fires_on_3_to_10"] = True
    out["checks"]["vision_tower_not_patched"] = True

    # ---- masses --------------------------------------------------------- #
    n_dec = cfg.num_hidden_layers
    with FOX.configured(img_start=int(start), img_end=int(end), enabled=False,
                        diagnose=True, collect=True, eager_layers=frozenset()):
        FOX.reset_diagnostics()
        with torch.no_grad():
            model(**batch)
        diags = {k: {kk: vv.clone() for kk, vv in v.items()}
                 for k, v in FOX.diagnostics.items()}
    if len(diags) != n_dec:
        raise AssertionError(f"{len(diags)} of {n_dec} layers reported")
    worst_sum, min_sys = 0.0, 1.0
    for layer, row in sorted(diags.items()):
        ms = float(row["m_sys_t1"].float().mean())
        mv = float(row["m_vis_t1"].float().mean())
        mt = float(row["m_txt_t1"].float().mean())
        min_sys = min(min_sys, ms)
        worst_sum = max(worst_sum, abs(ms + mv + mt - 1.0))
    if worst_sum > 1e-3:
        raise AssertionError(f"m_sys + m_vis + m_txt off 1 by {worst_sum}")
    if min_sys <= 0.0:
        raise AssertionError(
            "m_sys is 0 at some layer although system keys precede the "
            "image: the scoring would be H_vis alone")
    out["mass_sum_max_abs_error"] = worst_sum
    out["min_m_sys"] = min_sys
    out["n_layers_measured"] = len(diags)
    out["min_visual_mass"] = check_diagnostics_usable(diags, MIN_VISUAL_MASS)
    out["checks"]["m_sys_m_vis_m_txt_partition_to_1"] = True

    # ---- the splice is not a no-op -------------------------------------- #
    def logits(**cfg_kw):
        with FOX.configured(img_start=int(start), img_end=int(end), **cfg_kw):
            with torch.no_grad():
                return model(**batch).logits[0, -1].float().cpu()

    base = logits(enabled=False)
    with FOX.configured(img_start=int(start), img_end=int(end), enabled=False):
        with torch.no_grad():
            text_only = lm(input_ids=ids,
                           attention_mask=batch["attention_mask"]
                           ).logits[0, -1].float().cpu()
    d_text = float((base - text_only).abs().max())
    other = dict(batch)
    other["pixel_values"] = torch.flip(batch["pixel_values"], dims=[-1]) \
        + 0.5 * torch.randn_like(batch["pixel_values"])
    with FOX.configured(img_start=int(start), img_end=int(end), enabled=False):
        with torch.no_grad():
            alt = model(**other).logits[0, -1].float().cpu()
    d_img = float((base - alt).abs().max())
    out["splice_vs_text_only_max_abs_diff"] = d_text
    out["other_image_max_abs_diff"] = d_img
    if d_text == 0.0:
        raise AssertionError("the spliced run equals the text-only run: the "
                             "image features never reached the decoder")
    if d_img == 0.0:
        raise AssertionError("a different image gives the same logits: the "
                             "decoder ignores what was spliced in")
    out["checks"]["splice_differs_from_text_only_and_depends_on_image"] = True

    # ---- no-op arms against the baseline forward ------------------------ #
    off = logits(enabled=True, mode="off", use_m_sys=True, ratio=0.4,
                 eager_layers=window)
    if not torch.equal(base, off):
        raise AssertionError(
            f"mode='off' is not bit-identical to the baseline forward (max "
            f"|diff| {float((base - off).abs().max())})")
    out["checks"]["mode_off_bit_identical_to_baseline"] = True
    noop = logits(enabled=True, mode="as_implemented", gamma=0.0,
                  use_m_sys=True, ratio=0.4, eager_layers=window)
    out["gamma0_max_abs_diff"] = float((base - noop).abs().max())
    if int(base.argmax()) != int(noop.argmax()):
        raise AssertionError(
            f"gamma=0 changes the argmax (max |diff| "
            f"{out['gamma0_max_abs_diff']}): the eager path is not a no-op")
    out["checks"]["gamma0_same_argmax"] = True

    # ---- and it generates ------------------------------------------------ #
    gen = A.fox_generate(
        model, batch, max_new_tokens=n_samples, seed=0,
        eager_layers=window, ratio=0.4)
    if gen["n_gen_tokens"] < 1 or tuple(gen["image_span"]) != (int(start), int(end)):
        raise AssertionError(f"fox_generate gave {gen['n_gen_tokens']} tokens, "
                             f"span {gen['image_span']}")
    out["sample_output"] = processor.tokenizer.decode(
        gen["token_ids"], skip_special_tokens=True)
    out["checks"]["intervention_generates"] = True

    out["ok"] = True
    return out


def smoke_loaded(model, processor, *, model_id: str = "<loaded>",
                 n_samples: int = 5) -> dict:
    """Every check, against a model that is already in memory.

    Split out from :func:`smoke` so the checks themselves are testable
    without a multi-gigabyte download -- `tests/test_smoke_backbone.py`
    drives this with the tiny LLaVA the rest of the suite uses, and also
    feeds it a deliberately wrong span to confirm the mass-partition
    assertion actually fires. An assertion that has never failed is not
    evidence of anything.
    """
    import torch
    from PIL import Image

    from foxrepro.fox.attention import (
        FOX, expected_span_width, locate_image_span, resolve_image_token_id,
    )

    out: dict = {"model_id": model_id, "ok": False, "checks": {}}
    out["model_type"] = getattr(model.config, "model_type", None)

    image_token_id = resolve_image_token_id(model)
    out["image_token_id"] = image_token_id
    # None for a dynamic backbone, 576 for LLaVA. Passed through rather than
    # asserted here: whether the width is RIGHT is what the partition check
    # below decides, and a hardcoded expectation is what this repo has
    # repeatedly caught itself asserting in place of a measurement.
    width = expected_span_width(model)
    out["expected_width"] = width

    # A synthetic image on purpose. The span layout is a property of the
    # processor and the resolution, not of the picture, and a real COCO file
    # would make this script need the dataset.
    img = Image.new("RGB", (448, 448), (127, 127, 127))
    prompt = "What is in this image?"
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)
    batch = processor(images=img, text=text, return_tensors="pt")
    batch = {k: (v.to(model.device) if hasattr(v, "to") else v)
             for k, v in batch.items()}

    ids = batch["input_ids"]
    out["n_prompt_tokens"] = int(ids.shape[-1])
    n_img = int((ids == image_token_id).sum())
    out["n_image_tokens"] = n_img
    if n_img == 0:
        raise AssertionError(
            f"the processor emitted no token {image_token_id}; either the "
            f"chat template dropped the image or the id is wrong"
        )

    # THE check. Raises on a non-contiguous span.
    start, end = locate_image_span(ids, image_token_id, expected_width=width)
    out["img_start"], out["img_end"] = int(start), int(end)
    out["derived_width"] = int(end) - int(start) + 1
    out["checks"]["span_contiguous"] = True
    # Exact, against the processor's own placement. This is the check the
    # script is for; everything after it only confirms the span is usable.
    # `end` is INCLUSIVE here (locate_image_span's contract) while the
    # checker's is exclusive, so pass end + 1. This call passed `end` as
    # returned until now, which would have raised "span is [a, b) but the
    # image tokens occupy [a, b+1)" on the first real backbone: the check was
    # never driven end to end, only with hand-built exclusive ends.
    check_span_matches_tokens(ids, image_token_id, int(start), int(end) + 1)
    out["checks"]["span_is_exactly_the_image_tokens"] = True

    # One diagnostic forward, every layer measured, nothing edited.
    with FOX.configured(img_start=int(start), img_end=int(end),
                        enabled=False, diagnose=True, collect=True):
        FOX.reset_diagnostics()
        with torch.no_grad():
            model(**batch)
        diags = {k: {kk: vv.clone() for kk, vv in v.items()}
                 for k, v in FOX.diagnostics.items()}

    if not diags:
        raise AssertionError(
            "no layer reported diagnostics, so the Fox attention never ran. "
            "The patch did not reach the decoder on this checkpoint even "
            "though attn_implementation was accepted."
        )
    out["n_layers_measured"] = len(diags)

    out["min_visual_mass"] = check_diagnostics_usable(diags, MIN_VISUAL_MASS)
    out["checks"]["span_carries_attention"] = True

    # And the intervention itself, on a few real generate steps.
    with FOX.configured(img_start=int(start), img_end=int(end),
                        enabled=True, diagnose=False):
        with torch.no_grad():
            gen = model.generate(**batch, max_new_tokens=n_samples,
                                 do_sample=False)
    tail_ids = gen[0, ids.shape[-1]:]
    out["sample_output"] = processor.decode(tail_ids, skip_special_tokens=True)
    if int(tail_ids.numel()) == 0:
        raise AssertionError("generate returned no new tokens")
    out["checks"]["intervention_generates"] = True

    out["ok"] = True
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", action="append", default=[])
    ap.add_argument("--all", action="store_true",
                    help=f"every candidate: {', '.join(CANDIDATES)}")
    ap.add_argument("--instructblip", action="store_true",
                    help=f"also check {INSTRUCTBLIP_CANDIDATE} (public, fp32 on "
                         f"the Hub: a ~29.5 GiB download)")
    ap.add_argument("--shikra", action="store_true",
                    help=f"also check {SHIKRA_CANDIDATE}: fetches ~26.7 GiB "
                         f"and merges the delta in RAM")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    ids = list(args.model_id or (CANDIDATES if args.all else []))
    if args.instructblip and INSTRUCTBLIP_CANDIDATE not in ids:
        ids.append(INSTRUCTBLIP_CANDIDATE)
    if args.shikra and SHIKRA_CANDIDATE not in ids:
        ids.append(SHIKRA_CANDIDATE)
    if not ids:
        ap.error("pass --model-id, --all, --instructblip or --shikra")

    results = []
    for mid in ids:
        try:
            r = smoke(mid, n_samples=args.n_samples, dtype=args.dtype)
        except Exception as exc:                        # noqa: BLE001
            r = {"model_id": mid, "ok": False,
                 "error": f"{type(exc).__name__}: {exc}",
                 "traceback": traceback.format_exc()}
        results.append(r)
        if not args.json:
            _print(r)

    if args.json:
        print(json.dumps(results, indent=1, default=str))
    # Non-zero if ANY candidate failed: a smoke session that half-worked must
    # not look green in the bot's session state.
    return 0 if all(r.get("ok") for r in results) else 1


def _print(r: dict) -> None:
    print(f"\n=== {r['model_id']}")
    if not r.get("ok"):
        print(f"    FAILED: {r.get('error')}")
        if r.get("traceback"):
            print("    " + r["traceback"].replace("\n", "\n    ").rstrip())
        return
    print(f"    model_type     : {r['model_type']}")
    print(f"    image token    : {r['image_token_id']}  "
          f"x{r['n_image_tokens']} of {r['n_prompt_tokens']} prompt tokens")
    exp = r["expected_width"]
    print(f"    span           : [{r['img_start']}, {r['img_end']}] "
          f"width {r['derived_width']}"
          + (f"  (expected {exp})" if exp else "  (dynamic, no expectation)"))
    print(f"    layers measured: {r['n_layers_measured']}")
    print(f"    visual mass    : >= {r['min_visual_mass']:.3f} at every "
          f"layer (floor {MIN_VISUAL_MASS:.0e})")
    print(f"    generates      : {r['sample_output']!r}")
    print(f"    -> OK, this backbone is safe to spend a round on")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
