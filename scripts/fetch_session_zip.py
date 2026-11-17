#!/usr/bin/env python3
"""Ask MongoDB what a round's sessions did, and pull the zips that exist.

The loop this closes: a session finishes, the bot posts a download button to
Discord, and the zip only reaches `runs_inbox/` if somebody clicks it. When
nobody does, the work is done and invisible -- r16's session 5 and r20's
session 2 were both "missing" for a while before anyone checked whether they
had actually failed.

This reads the bot's own state instead of guessing:

    round:<repo>@<sha>:<notebook>   { sessions: { "1": { state, account,
                                      owner, version, versionId, ... } } }

and for every session in a terminal state it asks Kaggle, with that
session's OWN account token, for the version's output files -- which is the
same call the bot's /dl endpoint makes.

Credentials come from `message-talisman/.env` (MONGODB_URI and one
KAGGLE_TOKEN_<ACCOUNT> per member). Nothing is printed from them.

Usage::

    python3 scripts/fetch_session_zip.py --status            # what exists
    python3 scripts/fetch_session_zip.py --status --round r20
    python3 scripts/fetch_session_zip.py --download --round r20
    python3 scripts/fetch_session_zip.py --download --round r20 --session 2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

ENV_PATH = Path.home() / "Desktop/Workspace/message-talisman/.env"
DB_NAME = "message_talisman"
#: Rounds used to live as `round:<repo>@<sha>:<notebook>` keys inside the
#: generic kv collection; the bot has since given them their own. The old
#: keys survive in `kaggle_state_legacy` and stop at r16, so reading the kv
#: collection silently returns nothing for every round since -- which is how
#: the first version of this script reported "no round documents matched"
#: while twenty-six of them sat one collection away.
COLLECTION = "rounds"
LEGACY_COLLECTION = "kaggle_state_legacy"
KAGGLE_MCP = "https://www.kaggle.com/mcp"
UA = "fox-repro-fetch/1.0"


def load_env(path: Path = ENV_PATH) -> Dict[str, str]:
    """Parse KEY=VALUE lines. Values are used, never printed."""
    if not path.is_file():
        raise SystemExit(
            f"{path} not found. It holds MONGODB_URI and the KAGGLE_TOKEN_*\n"
            f"secrets; without it this script cannot reach either service."
        )
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def rounds(env: Dict[str, str], want_round: Optional[str] = None) -> List[dict]:
    """Every round document, newest first, optionally filtered by tag."""
    try:
        from pymongo import MongoClient
    except ImportError:
        raise SystemExit("pip install pymongo")
    uri = env.get("MONGODB_URI")
    if not uri:
        raise SystemExit("MONGODB_URI is not in the .env")
    cli = MongoClient(uri, serverSelectionTimeoutMS=20000)
    try:
        db = cli[DB_NAME]
        docs = list(db[COLLECTION].find({}))
        # Rounds older than the bot's schema change still only exist under
        # the legacy keys, so a question about r13..r16 is answerable too.
        docs += list(db[LEGACY_COLLECTION].find({"_id": {"$regex": "^round:"}}))
    finally:
        cli.close()
    docs.sort(key=lambda d: d.get("createdAt", 0), reverse=True)
    if want_round:
        docs = [d for d in docs if str(d.get("tag") or "") == want_round
                or str(d.get("tag") or "").startswith(want_round + ".")]
    return docs


def _mcp(token: str, owner: str, slug: str, version: int) -> dict:
    """`list_notebook_session_output`, the call the bot's /dl page makes.

    Three things this needs that a plain POST does not, each of which cost
    an attempt to discover:

    * an MCP handshake first -- `initialize`, then the `Mcp-Session-Id` the
      response carries, then `notifications/initialized`;
    * arguments wrapped as ``{"request": {...}}``. Passing them bare returns
      "An error occurred invoking ...", which is not obviously a shape
      problem;
    * `isError` checked. Without it a failed call parses as zero files and
      reports "this version has no zip" -- a wrong answer that looks like a
      finding.
    """
    sess = {"id": None}

    def raw(body: dict) -> Optional[dict]:
        headers = {"Authorization": f"Bearer {token}",
                   "Content-Type": "application/json",
                   "Accept": "application/json, text/event-stream",
                   "User-Agent": UA}
        if sess["id"]:
            headers["Mcp-Session-Id"] = sess["id"]
        req = urllib.request.Request(
            KAGGLE_MCP, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=120) as r:
            sess["id"] = r.headers.get("mcp-session-id") or sess["id"]
            text = r.read().decode("utf-8", "replace")
        lines = [l[5:].strip() for l in text.splitlines()
                 if l.startswith("data:")]
        if not lines and not text.strip():
            return None
        return json.loads(lines[-1] if lines else text)

    raw({"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                    "clientInfo": {"name": "fox-repro-fetch",
                                   "version": "1.0"}}})
    try:
        raw({"jsonrpc": "2.0", "method": "notifications/initialized"})
    except Exception:                                   # noqa: BLE001
        pass

    out = raw({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
               "params": {"name": "list_notebook_session_output",
                          "arguments": {"request": {
                              "userName": owner, "kernelSlug": slug,
                              "versionLabel": f"v{version}",
                              "hasVersionLabel": True}}}})
    result = (out or {}).get("result") or {}
    if result.get("isError"):
        detail = "".join(c.get("text", "")
                         for c in result.get("content", []))[:200]
        raise RuntimeError(f"Kaggle refused the call: {detail}")
    text = "".join(c.get("text", "") for c in result.get("content", []))
    return json.loads(text) if text else {}


def session_rows(doc: dict) -> List[dict]:
    out = []
    for n, s in sorted((doc.get("sessions") or {}).items(),
                       key=lambda kv: int(kv[0])):
        out.append({"session": int(n), **s})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", action="store_true", help="list, download nothing")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--round", help="tag, e.g. r20 (also matches r20.1)")
    ap.add_argument("--session", type=int, help="one session; default all done ones")
    ap.add_argument("--dest", type=Path,
                    default=Path(__file__).resolve().parents[1] / "runs_inbox")
    args = ap.parse_args(argv)
    if not (args.status or args.download):
        args.status = True

    env = load_env()
    docs = rounds(env, args.round)
    if not docs:
        print("no round documents matched", file=sys.stderr)
        return 1

    for doc in docs:
        tag = doc.get("tag") or "?"
        print(f"\n=== {tag}  {doc.get('repo')}@{str(doc.get('sha'))[:7]} "
              f"notebook={doc.get('notebook')}")
        for s in session_rows(doc):
            print(f"  session {s['session']}: {s.get('state'):<9} "
                  f"{str(s.get('accountName') or s.get('account') or '?'):<9} "
                  f"v{s.get('version')}  owner={s.get('owner')}")
            if not args.download or s.get("state") != "done":
                continue
            if args.session is not None and s["session"] != args.session:
                continue
            acct = str(s.get("accountName") or s.get("account") or "").upper()
            token = env.get(f"KAGGLE_TOKEN_{acct}")
            if not token:
                print(f"      no KAGGLE_TOKEN_{acct} in the .env; skipping")
                continue
            try:
                out = _mcp(token, s["owner"], doc["notebook"], int(s["version"]))
            except Exception as exc:                       # noqa: BLE001
                print(f"      Kaggle call failed: {type(exc).__name__}: {exc}")
                continue
            files = [f for f in (out.get("files") or [])
                     if str(f.get("url", "")).startswith("https://")]
            zips = [f for f in files
                    if str(f.get("file_name") or f.get("fileName") or "")
                    .endswith(".zip")]
            if not zips:
                print(f"      version has {len(files)} file(s), none a zip")
                continue
            args.dest.mkdir(parents=True, exist_ok=True)
            for f in zips:
                name = str(f.get("file_name") or f.get("fileName"))
                dest = args.dest / name
                if dest.exists():
                    print(f"      {name} already in runs_inbox; left alone")
                    continue
                req = urllib.request.Request(f["url"],
                                             headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=600) as r, \
                        dest.open("wb") as fh:
                    while chunk := r.read(1 << 20):
                        fh.write(chunk)
                print(f"      downloaded {name} "
                      f"({dest.stat().st_size / 1e6:.1f} MB) -> {dest}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
