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
"""
from __future__ import annotations

import argparse
import json
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


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--scratch", type=Path, default=Path("/tmp/fox_verify"))
    ap.add_argument("--archive", type=Path, default=REPO / "results" / "runs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from foxrepro.bench.plan import PLAN
    jobs = {j.job_id(): j for j in PLAN}
    missing = [i for i in JOB_IDS if i not in jobs]
    if missing:
        print(f"unknown job ids (plan changed?): {missing}", file=sys.stderr)
        return 2

    args.scratch.mkdir(parents=True, exist_ok=True)
    failures = 0
    for jid in JOB_IDS:
        job = jobs[jid]
        cmd = command_for(job, args.scratch, args.n)
        fname = job.run_key().filename()
        print(f"\n== {jid}\n   {' '.join(cmd)}")
        if args.dry_run:
            continue
        arch = args.archive / fname
        if not arch.is_file():
            print(f"   archive file {fname} is missing: nothing to compare against")
            failures += 1
            continue
        rc = subprocess.run(cmd, cwd=str(REPO)).returncode
        new = args.scratch / fname
        if rc != 0 or not new.is_file():
            print(f"   generate exited {rc}, no rerun file")
            failures += 1
            continue
        res = compare(read_records(arch, args.n), read_records(new, args.n))
        v = verdict(res, args.n)
        print(f"   {v}: compared {res['compared']}, missing {res['missing']}, "
              f"output differs {res['output_diff']}, float fields differ "
              f"{res['float_diff']}")
        if res["first"]:
            print(f"   first difference: {json.dumps(res['first'])[:400]}")
        if v != "identical":
            failures += 1
    print(f"\n{'ALL IDENTICAL' if failures == 0 else f'{failures} job(s) NOT identical'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
