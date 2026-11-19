#!/usr/bin/env python3
"""Is this zip's content already in results/runs? Answer before deleting it.

A zip is redundant only if EVERY .jsonl inside it is present in the archive
with at least as many records. Anything less and the zip holds work the
archive does not, which is the one thing a cleanup must not throw away.

Three things found on the first run of this, none of them guessable:

* `fox_runs_fox_r13_s1.zip` is NOT redundant and must never be deleted. It
  is the only copy outside `~/.cache` of r13's 234 MB diagnostic run, which
  `archive_runs` refuses because GitHub rejects files over 100 MB.
  `results/reports/diagnostics_r13.*` is derived from it and cannot be
  re-scored back into it. A zip loose in the workspace root looks like
  clutter and is not;
* `r21_s1` existed in two versions nine files apart, same filename,
  different directories. Deleting "the duplicate" by name would have been a
  coin flip over which one survived;
* `r16_s5` had been re-run, downloaded, and never moved into `runs_inbox`,
  so it had never been ingested at all.

Run this AFTER `plan archive` and AFTER committing, so the archive it
checks against is itself backed up. Prints one line per zip and exits
non-zero if any zip is NOT redundant.

Usage::

    python3 scripts/verify_zip_redundant.py runs_inbox/*.zip
    python3 scripts/verify_zip_redundant.py ~/Desktop/Workspace/ML_Algo/*.zip
"""
import subprocess
import sys
import zipfile
from pathlib import Path

ARCHIVE = Path.home() / "Desktop/Workspace/ML_Algo/fox-repro/results/runs"


def lines(path: Path) -> int:
    out = subprocess.run(["wc", "-l", str(path)], capture_output=True,
                         text=True)
    return int(out.stdout.split()[0]) if out.returncode == 0 else -1


def check(zpath: Path) -> tuple:
    missing, short = [], []
    with zipfile.ZipFile(zpath) as z:
        members = [n for n in z.namelist() if n.endswith(".jsonl")]
        for name in members:
            base = Path(name).name
            dest = ARCHIVE / base
            if not dest.is_file():
                missing.append(base)
                continue
            # Count the member's records without extracting it to disk.
            with z.open(name) as fh:
                n_zip = sum(1 for _ in fh)
            n_arch = lines(dest)
            if n_arch < n_zip:
                short.append((base, n_zip, n_arch))
    return members, missing, short


def main(argv) -> int:
    bad = 0
    for arg in argv:
        zpath = Path(arg)
        members, missing, short = check(zpath)
        if not missing and not short:
            print(f"REDUNDANT  {zpath.name}  ({len(members)} files all in "
                  f"the archive at full length)")
        else:
            bad += 1
            print(f"KEEP       {zpath.name}  ({len(members)} files; "
                  f"{len(missing)} absent, {len(short)} shorter in archive)")
            for b in missing[:5]:
                print(f"             absent : {b}")
            for b, nz, na in short[:5]:
                print(f"             shorter: {b}  zip={nz} archive={na}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
