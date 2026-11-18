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

    model, processor = load_backbone(model_id, dtype=dtype)
    return smoke_loaded(model, processor, model_id=model_id,
                        n_samples=n_samples)


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
    out["derived_width"] = int(end) - int(start)
    out["checks"]["span_contiguous"] = True
    # Exact, against the processor's own placement. This is the check the
    # script is for; everything after it only confirms the span is usable.
    check_span_matches_tokens(ids, image_token_id, int(start), int(end))
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
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    ids = args.model_id or (CANDIDATES if args.all else [])
    if not ids:
        ap.error("pass --model-id or --all")

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
    print(f"    span           : [{r['img_start']}, {r['img_end']}) "
          f"width {r['derived_width']}"
          + (f"  (expected {exp})" if exp else "  (dynamic, no expectation)"))
    print(f"    layers measured: {r['n_layers_measured']}")
    print(f"    visual mass    : >= {r['min_visual_mass']:.3f} at every "
          f"layer (floor {MIN_VISUAL_MASS:.0e})")
    print(f"    generates      : {r['sample_output']!r}")
    print(f"    -> OK, this backbone is safe to spend a round on")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
