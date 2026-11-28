---
name: fox-round-cycle
description: Process a finished Kaggle round of the CS2229-fox (foxrepro) reproduction end to end: fetch zips, ingest, archive, commit, score, plan the next round, release it. Use when a round has finished, when asked to "check status of the rounds", "ingest", "archive", "plan the next round" or "release rNN", or when `scripts/round_cycle.py due` exits 0. Also use before any `gh release create rNN`.
---

# Round cycle (CS2229-fox)

One pass of this skill handles ONE finished round and ends with the next round
released or a stated reason it was not. Every gate below is a command that exits
non-zero; do not replace a gate with your own judgement, and do not report a
step done without the command's output.

Work from the repo root with `export PYTHONPATH=src`. Read `CLAUDE.md` and the
memory index first: they hold the hard rules (never delete results, never commit
`.env`/zips/PDFs, never edit `results/runs/` by hand).

## 0. Is a round due?

```bash
python3 scripts/round_cycle.py due
```

Exit 0 prints `DUE rNN`. Exit 1: stop, there is nothing to do (it says what is
still running). A `ZOMBIE` line is a session `running` past 14 h that cannot be
cancelled with our tokens; **ask the user** before marking it stale in MongoDB
(see memory `kaggle-zombie-sessions`), never mark it yourself.

## 1. Fetch and read the round

```bash
python3 scripts/fetch_session_zip.py --status --round rNN
python3 scripts/fetch_session_zip.py --download --round rNN
```

For every zip read `_job_times.tsv` (column 2 = minutes, column 3 = exit code).
`failed` on a session is the bot's label, not a finding: exit codes and logs are.
Classify each non-zero job as (a) config error that died in seconds, (b) cut by
the watchdog (`-15`, resumable), or (c) a real crash. Cite the log line.

## 2. Ingest, archive, commit, push — in this order, no skipping

```bash
python3 -m foxrepro.bench.plan ingest                 # zips -> ~/.cache/fox-repro/runs
python3 -m foxrepro.bench.plan archive                # cache  -> results/runs
git add results/runs && git commit -m "results: rNN archived ..." ; git push
python3 scripts/round_cycle.py archive-check          # GATE: must print OK
```

`archive-check` exists because skipping this made r37 re-run 6 GPU-hours and left
OPERA chunk 2 with no predecessor on Kaggle. Move processed zips to
`runs_inbox/ingested/` so the next ingest is quick. Ingest zips of failed
sessions too: they hold every job that finished.

## 3. Score what completed

Use `python3 -m foxrepro.bench.evaluate` on the new run files and compare with
the pre-registered predictions in `docs/VALIDATION.md`. Score against the paper
(`ML_Algo/yu26c.pdf`, text via `pdftotext`). Record outcomes in `docs/VALIDATION.md`,
`docs/PAPER_AUDIT.md`, `docs/DEVIATIONS.md` (a change to the plan or criteria goes
in DEVIATIONS, never silently), then `python3 -m unittest tests.test_docs_are_not_stale`.
Decide by the project rule: a gap caused by our code or an unrun protocol gets an
experiment; a gap caused by information the paper does not contain is explained and
left. Look at raw outputs before explaining a gap (Qwen3-VL's 30% "Based" answers).

## 4. Plan the next round

Remaining work comes from the planner, never from memory:

```bash
python3 -m foxrepro.bench.plan plan --order balanced --sessions N --budget-hours 11.5 \
    --round rMM --in-flight <file of job ids still running elsewhere> --out /tmp/plan.json
```

* Kaggle's session cap is **12 h**; the watchdog stops jobs at `budget_hours - 0.25`
  (manifest `budget_hours: 12`). Plan at 11.5 so the last job is not always cut.
* Parallelise independent work across sessions (`balanced`, N = accounts with quota);
  do not pack everything into one session because the budget grew. Sequential
  dependencies (OPERA chunks share one run file) are the only exception.
* Costs come from measured `_job_times.tsv`; if a job ran 15% over its estimate,
  correct `plan.py` before freezing, not after the next cut.
* Exclude only jobs in sessions that are genuinely still running (check their age).
* Copy to `scripts/run_plan.json`, then sync `kaggle-run.yml` (`sessions`,
  `est_hours` = plan total) and the `(1..N)` comment in `run_kaggle_fox.sh`.

## 5. Gate, then release

```bash
python3 scripts/round_cycle.py test            # runs the suite; stamps HEAD only if green
git add kaggle-run.yml run_kaggle_fox.sh scripts/run_plan.json
git commit -m "rMM: ..." ; git push            # in THIS command; the release is the NEXT one
python3 scripts/round_cycle.py test            # again: the stamp is per commit
python3 scripts/round_cycle.py preflight       # GATE: must print OK
gh release create rMM --target "$(git rev-parse HEAD)" --title rMM --notes "..."
```

A PreToolUse hook (`scripts/hooks/guard_bash.py`) refuses `gh release create rNN`
unless preflight passes for HEAD and the tag equals the frozen plan's round.
Never use `grep ... &&` to gate on a result; test the command's own exit status.
Release only a round whose sha is new: the bot keys a round by `repo@sha`.

## 6. Prove it landed

```bash
python3 scripts/round_cycle.py verify-landed rMM
python3 scripts/fetch_session_zip.py --status --round rMM
```

`verify-landed` quotes nothing for you: it fails unless the workflow log has
`tag 'rMM' khớp round 'rMM'`, `"already": []`, and the bot registered as many
sessions as the plan has. Paste those log lines in your report.

## 7. Close the round

```bash
python3 scripts/round_cycle.py mark-handled rNN
```

Then tell the user, in their language (Vietnamese): what finished, what failed and
why (with log evidence), what was released and its GPU-hours, and what you could not
verify. Update the knowledge base note `20_Nghien-cuu/cs2229-fox/cs2229-fox-index.md`
when results changed.

## Stop and ask the user when

* a session must be marked stale or cancelled (their accounts, their DB);
* the plan would change a pre-registered criterion, or add a labelled variant;
* a gate fails and the fix is not mechanical;
* anything would delete or hand-edit committed results.

Releases of normal rounds are authorised (memory `run-experiments-and-smokes-autonomously`);
say what was released. There is no GPU-hour ceiling: the goal is to keep quota busy,
and the gates above guard correctness, not spending.
