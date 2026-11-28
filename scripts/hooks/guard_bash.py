#!/usr/bin/env python3
"""PreToolUse hook for Bash: block the commands this project has been burned by.

Reads the hook JSON on stdin. Exit 2 blocks the call and sends stderr back to the
agent; exit 0 allows it. It never edits anything, and an internal error allows the
call (exit 1 is non-blocking) rather than stopping the work on a bug in the guard.

Blocked:
  * `gh release create rNN` (a GPU round) unless `round_cycle.py preflight` passes
    for HEAD, and unless a literal --target is the full HEAD sha. smoke*/verify*
    releases are not rounds and are left alone. Run commit and push in an EARLIER
    command: this hook sees the repository before the command runs.
  * `git add` of secrets, Kaggle result zips or PDFs, or any `git add -f`.
"""
from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path
from typing import Callable, List, Optional

REPO = Path(__file__).resolve().parents[2]
ROUND_TAG = re.compile(r"^r\d+$")
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
FORBIDDEN_ADD = re.compile(r"(^|/)\.env($|\.)|\.zip$|\.pdf$|(^|/)runs_inbox(/|$)")


def _segments(command: str) -> List[List[str]]:
    """Each shell statement as tokens (split on ; && || | and newlines)."""
    out: List[List[str]] = []
    for part in re.split(r"&&|\|\||;|\||\n", command):
        try:
            toks = shlex.split(part)
        except ValueError:
            toks = part.split()
        if toks:
            out.append(toks)
    return out


def decide(command: str, head: str,
           preflight: Callable[[str], List[str]]) -> Optional[str]:
    """A reason to block, or None."""
    for toks in _segments(command):
        if toks[:3] == ["gh", "release", "create"] and len(toks) > 3:
            tag = next((t for t in toks[3:] if not t.startswith("-")), "")
            if not ROUND_TAG.match(tag):
                continue
            target = next((toks[i + 1] for i, t in enumerate(toks[:-1]) if t == "--target"), None)
            if target and not target.startswith("$") and (
                    not FULL_SHA.match(target) or target != head):
                return (f"release {tag}: --target {target!r} is not the full sha of HEAD "
                        f"({head[:7]}); a short sha is refused by GitHub and another commit "
                        f"is not the one that was tested")
            problems = preflight(tag)
            if problems:
                return (f"release {tag} blocked by preflight:\n  - " + "\n  - ".join(problems)
                        + "\nFix these, commit and push in an earlier command, run "
                          "`python3 scripts/round_cycle.py test`, then release.")
        if toks[:2] == ["git", "add"]:
            args = toks[2:]
            if "-f" in args or "--force" in args:
                return "git add -f is blocked: ignored files (results zips, .env) are ignored on purpose"
            bad = [a for a in args if not a.startswith("-") and FORBIDDEN_ADD.search(a)]
            if bad:
                return f"git add {bad}: secrets, Kaggle result zips and PDFs are never committed"
    return None


def main() -> int:
    try:
        event = json.load(sys.stdin)
        command = (event.get("tool_input") or {}).get("command") or ""
        if "gh release create" not in command and "git add" not in command:
            return 0
        sys.path.insert(0, str(REPO / "scripts"))
        import round_cycle as RC

        reason = decide(command, RC.head_sha(),
                        lambda tag: RC.preflight_problems(REPO / "scripts" / "run_plan.json", False, tag))
    except Exception as e:                                     # noqa: BLE001
        print(f"guard_bash: internal error, allowing the call: {e}", file=sys.stderr)
        return 1
    if reason:
        print(reason, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
