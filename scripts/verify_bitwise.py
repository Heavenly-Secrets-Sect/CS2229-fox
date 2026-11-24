#!/usr/bin/env python3
"""Rerun greedy jobs on a prefix and require the outputs to equal the archive.

Greedy decoding is deterministic, so a code change that should only move memory
(the autograd fix) or restore a missing argument (cache_position) must leave
every generated token and every recorded number exactly where it was. This is
the check, run on the code under test against runs the archive made BEFORE the
change. Anything but identical is a finding, not a tolerance to widen.

It reruns the first ``--n`` samples of a few greedy POPE-adversarial jobs into
a scratch directory (never the archive), then compares sample by sample:
``output`` exactly, and every float in ``method_extra`` exactly.

    python3 scripts/verify_bitwise.py --n 300          # on a GPU host
    python3 scripts/verify_bitwise.py --dry-run        # print the commands
    python3 scripts/verify_bitwise.py --ref 1e9901e    # also run an OLD commit

Comparing to the archive alone mixes two causes: the code changed, and the host
(torch, transformers, the Kaggle image) changed since the archive was made.
``--ref`` runs the same jobs with an older commit's ``src`` in the SAME session
and prints three comparisons: current vs archive, ref vs archive, current vs
ref. Only the last one is the effect of the code, because host and hardware
are held fixed.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

#: Greedy, POPE-adversarial, one per decode path that goes through fox_generate.
JOB_IDS = (
    "table9:sampling:pope:adversarial:greedy",
    "table9:fox:pope:adversarial:greedy",
    "table9_methods:vcd:pope:adversarial:greedy",
    "table9_methods:sid:pope:adversarial:greedy",
)


def read_records(path: Path, n: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def _floats(obj: Any, prefix: str = "") -> Iterator[Tuple[str, float]]:
    if isinstance(obj, bool):
        return
    if isinstance(obj, float):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _floats(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _floats(v, f"{prefix}[{i}]")


def compare(archived: List[Dict[str, Any]], rerun: List[Dict[str, Any]]
            ) -> Dict[str, Any]:
    """Exact comparison by sample_id. Returns counts and the first difference."""
    by_id = {r["sample_id"]: r for r in rerun}
    res: Dict[str, Any] = {"compared": 0, "missing": 0, "output_diff": 0,
                           "float_diff": 0, "first": None}
    for a in archived:
        b = by_id.get(a["sample_id"])
        if b is None:
            res["missing"] += 1
            continue
        res["compared"] += 1
        if a.get("output") != b.get("output"):
            res["output_diff"] += 1
            res["first"] = res["first"] or {
                "sample_id": a["sample_id"], "what": "output",
                "archived": a.get("output"), "rerun": b.get("output")}
        fa = dict(_floats(a.get("method_extra") or {}))
        fb = dict(_floats(b.get("method_extra") or {}))
        for k in fa:
            if k in fb and fa[k] != fb[k]:
                res["float_diff"] += 1
                res["first"] = res["first"] or {
                    "sample_id": a["sample_id"], "what": k,
                    "archived": fa[k], "rerun": fb[k]}
                break
    return res


def verdict(res: Dict[str, Any], n: int) -> str:
    """'identical' only if every requested sample was compared and none differ."""
    if res["compared"] == 0 or res["compared"] < min(n, res["compared"] + res["missing"]):
        return "incomplete"
    if res["missing"]:
        return "incomplete"
    if res["output_diff"] or res["float_diff"]:
        return "DIFFERENT"
    return "identical"


def command_for(job: Any, scratch: Path, n: int) -> List[str]:
    parts = shlex.split(job.command())
    # Drop any limit the job carries; this script sets its own.
    out: List[str] = []
    skip = False
    for p in parts:
        if skip:
            skip = False
            continue
        if p == "--limit":
            skip = True
            continue
        out.append(p)
    if out and out[0] == "python":
        out[0] = sys.executable
    return out + ["--runs-dir", str(scratch), "--limit", str(n)]


def _worktree(ref: str, where: Path) -> Path:
    """Check ``ref`` out beside the repo, fetching it first if the clone lacks it."""
    if where.exists():
        return where
    have = subprocess.run(["git", "cat-file", "-e", f"{ref}^{{commit}}"],
                          cwd=str(REPO), capture_output=True).returncode == 0
    if not have:
        print(f"[verify] {ref} is not in this clone; fetching it", flush=True)
        for cmd in (["git", "fetch", "-q", "origin", ref],
                    ["git", "fetch", "-q", "--unshallow", "origin"]):
            try:
                subprocess.run(cmd, cwd=str(REPO), check=False, timeout=600)
            except subprocess.TimeoutExpired:
                print(f"[verify] {' '.join(cmd)} timed out after 600 s", flush=True)
    print(f"[verify] checking out {ref} into {where}", flush=True)
    subprocess.run(["git", "worktree", "add", "--detach", str(where), ref],
                   cwd=str(REPO), check=True, timeout=600)
    return where


def run_job(cmd: List[str], cwd: Path, src: Optional[Path]) -> int:
    env = dict(os.environ)
    if src is not None:
        env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
    # A hung job must fail the check, not hold a Kaggle session for twelve
    # hours (verify-r30 did): 300 greedy samples take about 5 minutes.
    try:
        return subprocess.run(cmd, cwd=str(cwd), env=env, timeout=2400).returncode
    except subprocess.TimeoutExpired:
        print("[verify] job timed out after 2400 s", flush=True)
        return 124


def _line(label: str, res: Dict[str, Any], n: int) -> str:
    v = verdict(res, n)
    return (f"   {label:<22s} {v}: compared {res['compared']}, output differs "
            f"{res['output_diff']}, float fields differ {res['float_diff']}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--scratch", type=Path, default=Path("/tmp/fox_verify"))
    ap.add_argument("--archive", type=Path, default=REPO / "results" / "runs")
    ap.add_argument("--ref", action="append", default=[],
                    help="an older commit to run in the same session (repeatable)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from foxrepro.bench.plan import PLAN
    jobs = {j.job_id(): j for j in PLAN}
    missing = [i for i in JOB_IDS if i not in jobs]
    if missing:
        print(f"unknown job ids (plan changed?): {missing}", file=sys.stderr)
        return 2

    args.scratch.mkdir(parents=True, exist_ok=True)
    trees: Dict[str, Optional[Path]] = {"HEAD": None}
    if not args.dry_run:
        for ref in args.ref:
            trees[ref] = _worktree(ref, args.scratch.parent / f"fox_ref_{ref}")

    failures = 0
    host_drift = 0
    for jid in JOB_IDS:
        job = jobs[jid]
        fname = job.run_key().filename()
        print(f"\n== {jid}")
        recs: Dict[str, List[Dict[str, Any]]] = {}
        arch = args.archive / fname
        if not args.dry_run and not arch.is_file():
            print(f"   archive file {fname} is missing: nothing to compare against")
            failures += 1
            continue
        for name, tree in trees.items():
            out_dir = args.scratch / name
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = command_for(job, out_dir, args.n)
            print(f"   [{name}] {' '.join(cmd)}")
            if args.dry_run:
                continue
            rc = run_job(cmd, tree or REPO, (tree / "src") if tree else None)
            new = out_dir / fname
            if rc != 0 or not new.is_file():
                print(f"   [{name}] generate exited {rc}, no rerun file")
                failures += 1
                continue
            recs[name] = read_records(new, args.n)
        if args.dry_run:
            continue
        archived = read_records(arch, args.n)
        for name in recs:
            res = compare(archived, recs[name])
            print(_line(f"{name} vs archive", res, args.n))
            if res["first"] and name == "HEAD":
                print(f"   first difference: {json.dumps(res['first'])[:300]}")
        for name in recs:
            if name == "HEAD":
                continue
            res = compare(recs[name], recs["HEAD"]) if "HEAD" in recs else None
            if res is not None:
                print(_line(f"HEAD vs {name} (code only)", res, args.n))
                if res["first"]:
                    print(f"   first difference: {json.dumps(res['first'])[:300]}")
        if "HEAD" in recs and verdict(compare(archived, recs["HEAD"]), args.n) != "identical":
            refs = [n for n in recs if n != "HEAD"]
            code_same = bool(refs) and all(
                verdict(compare(recs[n], recs["HEAD"]), args.n) == "identical" for n in refs
            )
            if code_same:
                host_drift += 1
            else:
                failures += 1
    if failures:
        print(f"\n{failures} job(s) NOT reproducible: the code differs from an older commit, or no older commit was given to tell code from host")
    elif host_drift:
        print(f"\nCODE UNCHANGED: HEAD matches every --ref commit on this host; {host_drift} job(s) differ from the archive, which is host drift, not code")
    else:
        print("\nALL IDENTICAL TO THE ARCHIVE")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
