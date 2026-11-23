#!/usr/bin/env python3
"""Merge the Shikra-7B delta onto LLaMA-7B ONCE, and keep the result.

Shikra's weights are published as a DELTA (``shikras/shikra-7b-delta-v1-0708``)
that must be added to ``huggyllama/llama-7b``; there is no merged repo and
publishing one would redistribute LLaMA-1 derived weights (the licensing
question of docs/DEVIATIONS.md 3.10). A Kaggle session therefore does the same
merge itself, in RAM, with no file written (``foxrepro.fox.shikra.load_shikra``
on the delta id), and THIS script is for the other cases: a person who wants
the merged model on a big disk (to inspect it, to run a smoke test off Kaggle,
or to host it somewhere private), and the one place the merge is checked
end to end against the real files.

What it writes to ``--out``::

    config.json                         a 32003-vocab LlamaConfig (fp16)
    model-0000i-of-0000N.safetensors    the 291 decoder tensors, fp16
    model.safetensors.index.json
    shikra_projector.safetensors        the two mm_projector tensors:
                                        "weight" [4096, 1024], "bias" [4096]
    added_tokens.json, special_tokens_map.json, tokenizer.model,
    tokenizer_config.json               the delta repo's, unchanged
    shikra_build.json                   sources, revisions, sha256, versions

``foxrepro.fox.shikra.load_shikra(<out>)`` loads that directory as a plain
``LlamaForCausalLM`` plus the projector. The arithmetic is the release's
``apply_delta.py``: fp16 ``+=``, the three extra vocabulary rows of
``embed_tokens`` / ``lm_head`` taken from the delta alone, rotary buffers
dropped (transformers recomputes them).

Usage (once, on a host with ~50 GB free disk and ~16 GB RAM)::

    python3 scripts/build_shikra_checkpoint.py --out ~/shikra-7b-merged --verify
    python3 scripts/build_shikra_checkpoint.py --dry-run          # sizes only
    python3 scripts/build_shikra_checkpoint.py --out ~/shikra-7b-merged \\
        --verify-only                                      # re-check later

Footprint, from the file listings of 2026-11-24 (``--dry-run`` re-measures):
download 13,485,396,661 B (delta ``.bin`` shards) + 13,476,876,661 B (base
safetensors) = 26.96 GB = 25.1 GiB; output ~13.49 GB = 12.6 GiB; so ~37.7 GiB
at peak if the sources are kept, which is the "~40 GB transient" the earlier
note quoted. RAM: the streaming merge never holds a model; one 10 GB delta
shard is memory-mapped (page cache), and one output shard (default 4 GB) is
buffered.

Private and not for upload to a public repo: LLaMA-1 is research-only.
Quantization is not an option anywhere in this project, and this script has
no such flag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_TOKENIZER_FILES = ("added_tokens.json", "special_tokens_map.json",
                    "tokenizer.model", "tokenizer_config.json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _revision(path: Path) -> str:
    """The commit hash of a Hub snapshot (its directory name), else 'local'."""
    return path.name if path.parent.name == "snapshots" else "local"


def _resolve(spec: str, patterns) -> Path:
    from foxrepro.fox import shikra as S

    p = Path(spec).expanduser()
    return p if p.is_dir() else S._snapshot(spec, patterns)


def _dry_run(delta: str, base: str) -> int:
    from huggingface_hub import model_info

    def size(repo: str, suffix: str) -> int:
        info = model_info(repo, files_metadata=True)
        return sum(f.size or 0 for f in info.siblings
                   if f.rfilename.endswith(suffix))

    d, b = size(delta, ".bin"), size(base, ".safetensors")
    out = d  # fp16 delta and fp16 merge hold the same number of bytes
    gib = 1024 ** 3
    print(f"[shikra] download   delta {d / gib:6.1f} GiB (.bin shards)")
    print(f"[shikra]            base  {b / gib:6.1f} GiB (safetensors only)")
    print(f"[shikra] output     {out / gib:6.1f} GiB")
    print(f"[shikra] peak disk  {(d + b + out) / gib:6.1f} GiB with sources "
          f"kept; a Kaggle session keeps no output: {(d + b) / gib:.1f} GiB")
    print("[shikra] RAM        one mmapped delta shard + one 4 GB output "
          "buffer; a session holds the whole 12.6 GiB merge instead")
    return 0


def build(delta: str, base: str, out: Path, *, shard_gb: float,
          skip_tokenizer: bool) -> dict:
    import torch
    from safetensors.torch import save_file

    from foxrepro.fox import shikra as S

    delta_dir = _resolve(delta, S._DELTA_PATTERNS)
    base_dir = _resolve(base, S._BASE_PATTERNS)
    cfg0 = S._text_config(base_dir)
    out.mkdir(parents=True, exist_ok=True)

    limit = int(shard_gb * 1e9)
    shards, cur, cur_bytes = [], {}, 0
    weight_map, total = {}, 0
    proj = {}
    vocab = None

    def flush() -> None:
        nonlocal cur, cur_bytes
        if not cur:
            return
        name = f"model-{len(shards) + 1:05d}.safetensors.tmp"
        save_file(cur, str(out / name))
        shards.append((name, list(cur)))
        cur, cur_bytes = {}, 0

    for name, t in S.iter_merged_tensors(
            delta_dir, base_dir, n_layers=cfg0.num_hidden_layers):
        if name in S._PROJECTOR_KEYS:
            proj[name.rsplit(".", 1)[-1]] = t.contiguous()
            continue
        if name == "model.embed_tokens.weight":
            vocab = int(t.shape[0])
        nbytes = t.numel() * t.element_size()
        if cur and cur_bytes + nbytes > limit:
            flush()
        cur[name] = t.contiguous()
        cur_bytes += nbytes
        total += nbytes
    flush()
    if vocab is None or not shards or set(proj) != {"weight", "bias"}:
        raise RuntimeError("the merge produced no embedding, no shard or an "
                           "incomplete projector")

    n = len(shards)
    for i, (tmp, names) in enumerate(shards, 1):
        final = f"model-{i:05d}-of-{n:05d}.safetensors"
        (out / tmp).rename(out / final)
        weight_map.update({k: final for k in names})
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map},
        indent=1, sort_keys=True))
    save_file(proj, str(out / S.PROJECTOR_FILE))

    cfg = S._text_config(base_dir, vocab)
    cfg.torch_dtype = "float16"
    cfg.save_pretrained(str(out))
    for fn in _TOKENIZER_FILES:
        src = delta_dir / fn
        if src.exists():
            shutil.copy2(src, out / fn)
        elif not skip_tokenizer:
            raise FileNotFoundError(f"{src}: the delta repo's tokenizer file "
                                    f"is missing (--skip-tokenizer for tests)")

    import transformers

    meta = {
        "delta": delta, "delta_revision": _revision(delta_dir),
        "base": base, "base_revision": _revision(base_dir),
        "merge": "apply_delta.py semantics, fp16; rotary inv_freq dropped",
        "vocab_size": vocab, "n_tensors": len(weight_map) + 2,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "sha256": {p.name: _sha256(p) for p in sorted(out.iterdir())
                   if p.suffix == ".safetensors"},
        "license_note": "LLaMA-1 derived: research use, do not redistribute",
    }
    (out / "shikra_build.json").write_text(json.dumps(meta, indent=1))
    print(f"[shikra] wrote {n} shards, {total / 1024 ** 3:.2f} GiB, vocab "
          f"{vocab}, to {out}")
    return meta


def verify(delta: str, base: str, out: Path, *, skip_tokenizer: bool) -> int:
    """Re-derive every tensor a second way and compare bit for bit.

    The build uses ``d + b``; the check uses the release's own in-place
    ``param += base`` on a clone, tensor by tensor, reading the sources again,
    so a bug in the streaming merge cannot agree with itself.
    """
    import torch
    from safetensors import safe_open

    from foxrepro.fox import shikra as S

    delta_dir = _resolve(delta, S._DELTA_PATTERNS)
    base_dir = _resolve(base, S._BASE_PATTERNS)
    d_map = json.loads((delta_dir / "pytorch_model.bin.index.json")
                       .read_text())["weight_map"]
    b_map = json.loads((base_dir / "model.safetensors.index.json")
                       .read_text())["weight_map"]
    o_map = json.loads((out / "model.safetensors.index.json")
                       .read_text())["weight_map"]
    cfg = S._text_config(base_dir)
    problems = []

    want = set(S.expected_text_keys(cfg.num_hidden_layers))
    if set(o_map) != want:
        problems.append(f"key set differs: missing {sorted(want - set(o_map))[:3]}"
                        f", extra {sorted(set(o_map) - want)[:3]}")

    handles = {}

    def opened(dirp: Path, shard: str):
        if (dirp, shard) not in handles:
            handles[(dirp, shard)] = safe_open(str(dirp / shard),
                                               framework="pt")
        return handles[(dirp, shard)]

    checked, rope_err = 0, 0.0
    theta = float(getattr(cfg, "rope_theta", 10000.0))
    hd = cfg.hidden_size // cfg.num_attention_heads
    ref_inv = 1.0 / (theta ** (torch.arange(0, hd, 2).float() / hd))
    for shard in sorted(set(d_map.values())):
        tensors = S._load_bin(delta_dir / shard)
        for name in [n for n, s in d_map.items() if s == shard]:
            if S._is_rope_buffer(name):
                b = opened(base_dir, b_map[name]).get_tensor(name)
                got = (tensors[name].float() + b.float())
                rope_err = max(rope_err, float(
                    ((got - ref_inv).abs() / ref_inv).max()))
                continue
            if name in S._PROJECTOR_KEYS:
                continue
            param = tensors[name].clone()
            bt = opened(base_dir, b_map[name]).get_tensor(name)
            if param.shape == bt.shape:
                param.data += bt
            else:
                param.data[:bt.shape[0], :bt.shape[1]] += bt
            ot = opened(out, o_map[name]).get_tensor(name)
            checked += 1
            if ot.dtype != torch.float16 or not torch.equal(ot, param):
                problems.append(f"{name} differs from apply_delta.py")
            if checked % 50 == 0:
                print(f"[shikra]   {checked}/{len(want)} tensors checked")
        del tensors
    if checked != len(want):
        problems.append(f"checked {checked} tensors, expected {len(want)}")

    vocab = opened(out, o_map["model.embed_tokens.weight"]).get_slice(
        "model.embed_tokens.weight").get_shape()[0]
    if vocab != S.VOCAB_SIZE and not skip_tokenizer:
        problems.append(f"vocab {vocab} != {S.VOCAB_SIZE}")

    pj = out / S.PROJECTOR_FILE
    with safe_open(str(pj), framework="pt") as fh:
        if sorted(fh.keys()) != ["bias", "weight"]:
            problems.append(f"projector keys {sorted(fh.keys())}")
        else:
            w, bias = fh.get_tensor("weight"), fh.get_tensor("bias")
            dw = S._load_bin(delta_dir / d_map["model.mm_projector.weight"])
            if not torch.equal(w, dw["model.mm_projector.weight"]):
                problems.append("projector weight differs from the delta's")
            if float(w.float().abs().sum()) == 0.0 or bias.dtype != torch.float16:
                problems.append("projector is zero or not fp16")

    # The model skeleton the loader builds has exactly these keys.
    from transformers import LlamaConfig, LlamaForCausalLM

    with torch.device("meta"):
        skel = set(LlamaForCausalLM(LlamaConfig.from_pretrained(str(out)))
                   .state_dict())
    if skel != set(o_map):
        problems.append("a LlamaForCausalLM skeleton and the output disagree "
                        "on keys")

    if not skip_tokenizer:
        from foxrepro.bench.methods import SHIKRA_WRAPPER, _LLAVA_V1_SYSTEM
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(out), use_fast=False)
        S.check_tokenizer(tok)
        ids = S.encode_prompt(tok, _LLAVA_V1_SYSTEM + SHIKRA_WRAPPER.format(
            instruction="Is there a dog in the image?"))
        pos = (ids.index(S.IM_START_ID), ids.index(S.IM_END_ID))
        if pos != (33, 290):
            problems.append(f"markers at {pos}, the release has (33, 290)")

    print(f"[shikra] rotary inv_freq: max relative difference between the "
          f"checkpoint's fp16 (delta + base) and fp32 recomputation: "
          f"{rope_err:.2e} (informational; see DEVIATIONS 3.20)")
    if problems:
        for p in problems:
            print(f"[shikra] FAIL: {p}", file=sys.stderr)
        return 1
    print(f"[shikra] OK: {checked} decoder tensors bit-identical to "
          f"apply_delta.py, projector and tokenizer checked")
    return 0


def main(argv=None) -> int:
    from foxrepro.fox import shikra as S

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delta", default=S.SHIKRA_DELTA_ID,
                    help="Hub id or local directory of the delta")
    ap.add_argument("--base", default=S.SHIKRA_BASE_ID,
                    help="Hub id or local directory of llama-7b")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--verify", action="store_true",
                    help="re-derive every tensor with the release's own "
                         "in-place add and compare bit for bit")
    ap.add_argument("--verify-only", action="store_true",
                    help="check an existing --out without rebuilding it")
    ap.add_argument("--shard-gb", type=float, default=4.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the transfer and disk plan, download nothing")
    ap.add_argument("--skip-tokenizer", action="store_true",
                    help="for tests on a toy checkpoint only")
    args = ap.parse_args(argv)

    if args.dry_run:
        return _dry_run(args.delta, args.base)
    if args.out is None:
        ap.error("--out is required")
    if not args.verify_only:
        build(args.delta, args.base, args.out, shard_gb=args.shard_gb,
              skip_tokenizer=args.skip_tokenizer)
    if args.verify or args.verify_only:
        return verify(args.delta, args.base, args.out,
                      skip_tokenizer=args.skip_tokenizer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
