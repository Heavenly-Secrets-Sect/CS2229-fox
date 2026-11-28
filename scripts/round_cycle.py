#!/usr/bin/env python3
"""Checks that guard the round cycle: when a round is due, and whether a freeze is safe.

The cycle is: a round finishes -> fetch zips -> `plan ingest` -> `plan archive`
-> commit and push `results/runs/` -> score -> `plan plan` -> test -> release.
Each check below is a failure this project has already paid for, written as a
test a machine can run instead of a sentence an agent has to remember.

    due             exit 0 only if a round has every session finished and has not
                    been handled; says why not otherwise. Cheap, for a scheduler.
    archive-check   exit 0 only if the committed archive holds everything in the
                    live run directory (r37 re-ran 6 GPU-hours of MMBench and its
                    OPERA chunk found no predecessor because this was skipped).
    preflight       archive-check plus the manifest, the git state, the tag and a
                    recorded green test run for HEAD. Exit 0 means "safe to release".
    watch           block until a round is due, then print `DUE rNN` and exit 0; run it
                    under a Monitor so a session is woken only when there is work.
    test            run the suite and, only if green, stamp HEAD as tested.
    verify-landed   after a release: the log line proving the tag guard ran,
                    `already` empty, and the bot's session count equal to the plan.
    mark-handled    record a round as processed, so `due` stops reporting it.

Nothing here writes to MongoDB, GitHub or the archive.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO = Path(__file__).resolve().parents[1]
HANDLED = Path.home() / ".cache" / "fox-repro" / "handled_rounds.txt"

#: Kaggle's own cap on one session (docs/CATCHUP.md).
KAGGLE_SESSION_HOURS = 12.0
#: `run_session` stops jobs this long before `budget_hours` (plan.py watchdog).
WATCHDOG_MARGIN_HOURS = 0.25
TERMINAL = ("done", "failed", "cancelled")
ROUND_TAG = re.compile(r"^r\d+$")


# --------------------------------------------------------------------------- #
# pure checks (unit-tested)
# --------------------------------------------------------------------------- #

def scalars(path: Path) -> Dict[str, str]:
    """Top-level ``key: value`` lines of a manifest; no YAML dependency."""
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z_][\w]*):\s*(.*?)\s*(#.*)?$", line)
        if m and m.group(2) != "":
            out[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return out


def check_manifest(manifest: Dict[str, str], plan: Dict[str, object]) -> List[str]:
    """Problems between ``kaggle-run.yml`` and the frozen plan; empty means fine."""
    bad: List[str] = []
    sessions = list(plan.get("sessions") or [])  # type: ignore[arg-type]
    if int(manifest.get("sessions", -1)) != len(sessions):
        bad.append(f"manifest sessions={manifest.get('sessions')} but the plan has "
                   f"{len(sessions)}: a session past the plan runs empty and burns quota")
    total = sum(float(s["est_hours"]) for s in sessions)
    if abs(float(manifest.get("est_hours", -1)) - total) > 0.1:
        bad.append(f"manifest est_hours={manifest.get('est_hours')} but the plan totals {total:.2f}")
    budget = float(manifest.get("budget_hours", 0))
    if budget > KAGGLE_SESSION_HOURS:
        bad.append(f"manifest budget_hours={budget} is over Kaggle's {KAGGLE_SESSION_HOURS} h cap")
    plan_budget = float(plan.get("budget_hours", 0))
    if plan_budget > budget - WATCHDOG_MARGIN_HOURS:
        bad.append(f"plan budget {plan_budget} h leaves no room for the {WATCHDOG_MARGIN_HOURS} h "
                   f"watchdog inside the manifest's {budget} h: the last job is always cut")
    for s in sessions:
        if float(s["est_hours"]) > budget:
            bad.append(f"session {s['session']} is estimated at {s['est_hours']} h, over {budget} h")
    return bad


def due_rounds(docs: Iterable[dict], handled: Set[str], now_ms: float,
               zombie_hours: float = 14.0,
               zombie_max_age_days: float = 14.0) -> Tuple[List[str], List[str], List[str]]:
    """``(ready, waiting, zombies)`` from the bot's round documents.

    ``ready``: numbered rounds (``rNN``) whose sessions have all finished and
    that are not in ``handled``. ``waiting``: numbered rounds still running.
    ``zombies``: sessions ``running`` longer than ``zombie_hours``; a session
    like that never finishes by itself (r33 s5/s6 ran 34 h) and keeps its
    round out of ``ready`` for good, so it is surfaced, never auto-repaired.
    Rounds older than ``zombie_max_age_days`` are not reported: r16's sessions
    have said ``running`` for 350 h and are history, not a live problem.
    """
    ready: List[str] = []
    waiting: List[str] = []
    zombies: List[str] = []
    for d in docs:
        tag = str(d.get("tag") or "")
        sessions = d.get("sessions") or {}
        if not ROUND_TAG.match(tag) or not sessions:
            continue
        recent = (now_ms - float(d.get("createdAt") or now_ms)) / 8.64e7 <= zombie_max_age_days
        for n, s in sessions.items():
            if not recent:
                break
            age_h = (now_ms - float(s.get("pushedAt") or now_ms)) / 3.6e6
            if s.get("state") == "running" and age_h > zombie_hours:
                zombies.append(f"{tag} session {n} ({s.get('accountName')}) running {age_h:.0f} h")
        if tag in handled:
            continue
        if all(s.get("state") in TERMINAL for s in sessions.values()):
            ready.append(tag)
        else:
            waiting.append(tag)
    return ready, waiting, zombies


# --------------------------------------------------------------------------- #
# checks that touch the machine
# --------------------------------------------------------------------------- #

STAMPS = Path.home() / ".cache" / "fox-repro" / "tests_green"


def head_sha() -> str:
    return _run(["git", "rev-parse", "HEAD"])[1].strip()


def tests_green_for_head() -> bool:
    """A stamp written only by `round_cycle.py test`, named by the commit it ran on."""
    return (STAMPS / head_sha()).is_file()


def run_tests_and_stamp() -> int:
    rc, out = _run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."],)
    tail = out.splitlines()[-1] if out else "no output"
    if rc == 0:
        STAMPS.mkdir(parents=True, exist_ok=True)
        (STAMPS / head_sha()).write_text(tail + "\n")
    print(("GREEN " if rc == 0 else "RED ") + head_sha()[:7] + ": " + tail)
    return rc


def landed_problems(tag: str, sha: str, expected_sessions: int) -> List[str]:
    """After a release: the log line proving the tag guard ran, and the sessions the bot registered."""
    rc, out = _run(["gh", "run", "list", "--workflow", "kaggle-run.yml", "-L", "10",
                    "--json", "databaseId,headSha,status,conclusion"])
    runs = [r for r in json.loads(out) if r["headSha"] == sha] if rc == 0 else []
    if not runs:
        return [f"no kaggle-run.yml workflow run for {sha[:7]} yet (wait, then retry)"]
    rc, log = _run(["gh", "run", "view", str(runs[0]["databaseId"]), "--log"])
    bad: List[str] = []
    if not re.search(rf"tag '{re.escape(tag)}' khớp round '{re.escape(tag)}'", log):
        bad.append(f"log lacks the line \"tag '{tag}' khớp round '{tag}'\": the tag guard did not run")
    if not re.search(r'"already":\s*\[\s*\]', log):
        bad.append('log shows a non-empty "already": the bot registered nothing new (same sha as a round)')
    docs = [d for d in _docs() if d.get("tag") == tag and d.get("sha") == sha]
    n = len(docs[0].get("sessions") or {}) if docs else 0
    if n != expected_sessions:
        bad.append(f"the bot registered {n} session(s) for {tag}@{sha[:7]}, the plan has {expected_sessions}")
    return bad


def _run(cmd: Sequence[str]) -> Tuple[int, str]:
    p = subprocess.run(list(cmd), cwd=REPO, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def archive_problems() -> List[str]:
    """Files whose live records are not all in the committed archive."""
    sys.path.insert(0, str(REPO / "src"))
    from foxrepro.bench.plan import archive_runs

    rep = archive_runs(dry_run=True)
    behind = [f for f in rep.get("files", [])  # type: ignore[union-attr]
              if str(f).startswith(("copied", "merged"))]
    bad = [f"archive is behind the live runs: {len(behind)} file(s), e.g. {behind[0]}"] if behind else []
    # `archive_runs` only dry-runs the copy; the files must also be COMMITTED.
    rc, out = _run(["git", "status", "--porcelain", "results/runs"])
    if out:
        bad.append(f"results/runs has {len(out.splitlines())} uncommitted change(s): commit them")
    rc, out = _run(["git", "rev-list", "--count", "@{u}..HEAD"])
    if rc == 0 and out.strip() not in ("", "0"):
        bad.append(f"{out.strip()} commit(s) not pushed: Kaggle clones the remote")
    return bad


def preflight_problems(plan_path: Path, run_tests: bool, tag: Optional[str] = None) -> List[str]:
    bad = archive_problems()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    bad += check_manifest(scalars(REPO / "kaggle-run.yml"), plan)
    plan_round = str(plan.get("round") or "")
    if not ROUND_TAG.match(plan_round):
        bad.append(f"plan round {plan_round!r} is not rNN, so the tag guard cannot match it")
    if tag is not None and tag != plan_round:
        bad.append(f"releasing {tag} but the frozen plan is {plan_round}: the workflow's tag guard "
                   f"would fail, and the zips would be named for the wrong round")
    tag = tag or plan_round
    if ROUND_TAG.match(tag):
        rc, _ = _run(["gh", "release", "view", tag])
        if rc == 0:
            bad.append(f"release {tag} already exists")
    rc, out = _run(["git", "tag", "--points-at", "HEAD"])
    if out:
        bad.append(f"HEAD already carries tag(s) {out.split()}: the bot keys a round by "
                   f"repo@sha, so a second release on this commit registers nothing")
    rc, out = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    if out:
        bad.append(f"{len(out.splitlines())} tracked file(s) modified: commit before releasing")
    if run_tests:
        run_tests_and_stamp()
    if not tests_green_for_head():
        bad.append(f"no green test run is recorded for HEAD {head_sha()[:7]}: run "
                   f"`python3 scripts/round_cycle.py test` after the last commit")
    return bad


def _handled() -> Set[str]:
    return set(HANDLED.read_text().split()) if HANDLED.is_file() else set()


def _docs() -> List[dict]:
    sys.path.insert(0, str(REPO / "scripts"))
    import fetch_session_zip as F

    return F.rounds(F.load_env())


def watch(interval: float) -> int:
    """Block until a round is due, printing one line per new event; for a Monitor.

    Prints `ZOMBIE ...` once per session and `DUE rNN ...` then exits 0, which is
    the Monitor's wake-up. A database or network error is printed once per distinct
    message and retried: a watcher that dies on a blip is a watcher nobody notices
    is gone. Silent while nothing happens, so waiting costs no tokens.
    """
    seen: Set[str] = set()
    while True:
        try:
            ready, _, zombies = due_rounds(_docs(), _handled(), time.time() * 1000)
            events = [f"ZOMBIE {z}" for z in zombies] + ([f"DUE {' '.join(ready)}"] if ready else [])
        except Exception as e:                                  # noqa: BLE001
            events = [f"watch error (retrying): {type(e).__name__}: {str(e)[:120]}"]
        for ev in events:
            if ev not in seen:
                print(ev, flush=True)
                seen.add(ev)
        if any(ev.startswith("DUE") for ev in events):
            return 0
        time.sleep(interval)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("due")
    sub.add_parser("archive-check")
    pf = sub.add_parser("preflight")
    pf.add_argument("--plan", type=Path, default=REPO / "scripts" / "run_plan.json")
    pf.add_argument("--run-tests", action="store_true")
    wp = sub.add_parser("watch")
    wp.add_argument("--interval", type=float, default=300.0, help="seconds between checks")
    sub.add_parser("test")
    lp = sub.add_parser("verify-landed")
    lp.add_argument("tag")
    lp.add_argument("--plan", type=Path, default=REPO / "scripts" / "run_plan.json")
    mh = sub.add_parser("mark-handled")
    mh.add_argument("round")
    args = ap.parse_args(argv)

    if args.cmd == "due":
        ready, waiting, zombies = due_rounds(_docs(), _handled(), time.time() * 1000)
        for z in zombies:
            print(f"ZOMBIE {z}: cannot be cancelled with our tokens; mark it stale (ask the user)")
        if ready:
            print("DUE " + " ".join(ready))
            return 0
        print("not due: " + (f"still running: {' '.join(waiting)}" if waiting else "nothing unhandled"))
        return 1
    if args.cmd == "watch":
        return watch(args.interval)
    if args.cmd == "archive-check":
        bad = archive_problems()
    elif args.cmd == "preflight":
        bad = preflight_problems(args.plan, args.run_tests)
    elif args.cmd == "test":
        return run_tests_and_stamp()
    elif args.cmd == "verify-landed":
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        bad = landed_problems(args.tag, head_sha(), len(plan["sessions"]))
    else:
        HANDLED.parent.mkdir(parents=True, exist_ok=True)
        with HANDLED.open("a") as fh:
            fh.write(args.round + "\n")
        print(f"marked {args.round} handled")
        return 0
    for b in bad:
        print("FAIL " + b)
    print("OK" if not bad else f"{len(bad)} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
