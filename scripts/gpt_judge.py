#!/usr/bin/env python3
"""GPT-judge substitute for the paper's Figure 6 / Table 5. See
``foxrepro.bench.gpt_judge`` and docs/DEVIATIONS.md.

    python3 scripts/gpt_judge.py --list-models
    python3 scripts/gpt_judge.py --model gpt-4.1-2025-04-14 --dry-run
    python3 scripts/gpt_judge.py --model gpt-4.1-2025-04-14 --n-images 3
    python3 scripts/gpt_judge.py --model gpt-4.1-2025-04-14

The API key is read from OPENAI_API_KEY or a ``.env`` file and is never
printed or written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from foxrepro.bench import gpt_judge as gj  # noqa: E402
from foxrepro.data.loaders import image_filename_for_id  # noqa: E402


def main(argv=None, client=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", help="exact OpenAI model id (recorded in outputs)")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--n-images", type=int, default=gj.N_IMAGES)
    ap.add_argument("--max-calls", type=int, default=gj.DEFAULT_MAX_CALLS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--runs-dir", type=Path, default=REPO / "results" / "runs")
    ap.add_argument("--images-dir", type=Path, default=None,
                    help="COCO val2014 jpgs (default: foxrepro storage path)")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results" / "reports")
    ap.add_argument("--detail", default="auto", choices=["auto", "low", "high"])
    ap.add_argument("--price-in", type=float, help="USD per 1M input tokens")
    ap.add_argument("--price-out", type=float, help="USD per 1M output tokens")
    ap.add_argument("--sampling-run", help="run file name of the baseline arm")
    ap.add_argument("--fox-run", help="run file name of the Fox arm")
    ap.add_argument("--sampling-method", default=None)
    ap.add_argument("--fox-method", default=None)
    ap.add_argument("--tag", default="",
                    help="suffix for the output files, so another backbone "
                         "does not overwrite LLaVA's")
    ap.add_argument("--confidence", type=float, default=0.90)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args(argv)

    key = gj.load_api_key()
    if a.list_models:
        if not key:
            print("no OPENAI_API_KEY in environment or .env", file=sys.stderr)
            return 2
        try:
            ids = gj.OpenAIJudgeClient(key, "none").list_models()
        except Exception as e:  # noqa: BLE001
            print("model listing failed:", gj.redact(e, key)[:300], file=sys.stderr)
            return 2
        print("\n".join(i for i in ids if i.startswith(("gpt-", "o"))))
        return 0
    if not a.model:
        ap.error("--model is required")

    if a.sampling_run:
        gj.SAMPLING_RUN = a.sampling_run
    if a.fox_run:
        gj.FOX_RUN = a.fox_run
    if a.sampling_method:
        gj.SAMPLING_METHOD = a.sampling_method
    if a.fox_method:
        gj.FOX_METHOD = a.fox_method
    samp, fox = gj.load_captions(a.runs_dir)
    ids = gj.select_images(sorted(set(samp) & set(fox)), a.n_images)
    calls = gj.plan_calls(ids, samp, fox)
    price = ((a.price_in, a.price_out)
             if a.price_in is not None and a.price_out is not None else None)
    est = gj.estimate_cost(calls, a.model, price)
    print(f"judge model: {a.model}")
    print(f"images: {len(ids)} (ids {ids[0]}..{ids[-1]}), calls planned: "
          f"{len(calls)} (both orders), budget: {a.max_calls}")
    print(f"estimate: ~{est['input_tokens']} input + ~{est['output_tokens']} "
          f"output tokens, cost "
          + ("unknown (no price known; pass --price-in/--price-out)"
             if est["usd"] is None else f"~${est['usd']:.2f} (assumed list price)"))
    if len(calls) > a.max_calls:
        print(f"ABORT: {len(calls)} calls exceed --max-calls {a.max_calls}",
              file=sys.stderr)
        return 3
    if a.dry_run:
        for c in calls:
            print(f"  would call: image {c.image_id} order {c.order}")
        return 0

    if a.images_dir is None:
        from foxrepro.storage import paths
        a.images_dir = paths().coco_images_dir
    img = {i: a.images_dir / image_filename_for_id(i) for i in ids}
    missing = [p.name for p in img.values() if not p.is_file()]
    if missing:
        print(f"missing images in {a.images_dir}: {missing[:5]}", file=sys.stderr)
        return 2
    if client is None:
        if not key:
            print("no OPENAI_API_KEY in environment or .env", file=sys.stderr)
            return 2
        client = gj.OpenAIJudgeClient(key, a.model, a.detail)

    res = gj.run_judge(client, calls, img, max_calls=a.max_calls)
    recs = res["records"]
    tok_in = sum(r["usage"].get("prompt_tokens", 0) for r in recs)
    tok_out = sum(r["usage"].get("completion_tokens", 0) for r in recs)
    tag = "" if a.n_images == gj.N_IMAGES else f"_n{a.n_images}"
    stem = a.out_dir / f"gpt_judge_{a.model}{a.tag}{tag}"
    a.out_dir.mkdir(parents=True, exist_ok=True)
    summary, err = None, None
    try:
        summary = gj.summarise(recs, n_boot=a.n_boot, confidence=a.confidence)
    except ValueError as e:
        err = str(e)
    params = recs[0]["params"] if recs else None
    payload = {
        "judge_model": a.model,
        "protocol": "Table 5 prompt, 1-10 Accuracy/Detailedness",
        "max_tokens": gj.MAX_TOKENS, "temperature": gj.TEMPERATURE,
        "params_sent": params, "images": ids,
        "image_rule": "first N by ascending image_id of the CHAIR "
                      "val2014_500 set",
        "sampling_run": gj.SAMPLING_RUN, "fox_run": gj.FOX_RUN,
        "n_calls": res["n_calls"], "fatal": res["fatal"],
        "tokens": {"input": tok_in, "output": tok_out},
        "summary": summary, "summary_error": err, "records": recs,
    }
    Path(str(stem) + ".json").write_text(json.dumps(payload, indent=1))
    if summary is not None:
        Path(str(stem) + ".md").write_text(
            gj.render_markdown(a.model, {"params": params}, summary))
    print(f"calls made: {res['n_calls']}, tokens in/out: {tok_in}/{tok_out}")
    p = price or est["price_per_M"]
    if p:
        print(f"actual cost at assumed price: "
              f"~${(tok_in * p[0] + tok_out * p[1]) / 1e6:.3f}")
    if res["fatal"]:
        print("STOPPED by API error:", res["fatal"], file=sys.stderr)
        return 4
    if summary is None:
        print("no summary:", err, file=sys.stderr)
        return 5
    print(f"unparsed: {summary['n_unparsed']} / {summary['n_calls']}")
    for crit in ("accuracy", "detailedness"):
        for k in ("paper_style_SF", "order_averaged"):
            v = summary[crit][k]
            if v:
                print(f"{crit:13s} {k:15s} n={v['n']} "
                      f"sampling={v['sampling_mean']:.2f} fox={v['fox_mean']:.2f} "
                      f"diff={v['diff_mean']:+.2f} "
                      f"CI90=[{v['diff_ci'][0]:+.2f},{v['diff_ci'][1]:+.2f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
