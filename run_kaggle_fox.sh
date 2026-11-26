#!/usr/bin/env bash
# Kaggle (GPU T4 x2, Internet on): one session's share of the Fox reproduction.
# Sessions are disjoint and cost-balanced, so several can run concurrently --
# on separate accounts, or one after another on the same one.
#
#   local:   python3 -m foxrepro.bench.plan plan --budget-hours 10 --round r6
#            python3 -m foxrepro.bench.plan plan --budget-hours 9 \
#                    --calibrate <logs>/_job_times.tsv --round r2   (after session 1)
#   Kaggle:  bash run_kaggle_fox.sh <N>        N = session number (1..12)
#            SMOKE=<model-id> bash run_kaggle_fox.sh 0   backbone pre-flight,
#                                      no data fetch, no plan jobs
#            DRY=1 bash run_kaggle_fox.sh <N>  print the commands only
#            BUDGET=10 bash run_kaggle_fox.sh <N>     hours, default from the plan
#
# Run files are re-packed to /kaggle/working/fox_runs_<tag>.zip after EVERY
# finished job, so a cut session loses at most the job in flight. Download each
# zip into runs_inbox/ locally, then merge them with one command:
#   python3 -m foxrepro.bench.plan ingest
#   python3 -m foxrepro.bench.evaluate && python3 -m foxrepro.bench.report
#
# Resumable: relaunch the identical command. Completed samples are skipped, and
# a job the previous session left half-done is continued, not restarted,
# because resume is keyed on the run file rather than on the session.
set -uo pipefail
cd "$(dirname "$0")"

N="${1:?usage: bash run_kaggle_fox.sh <session number>}"
export FOX_ENV="${FOX_ENV:-kaggle}"

# A backbone smoke session: no benchmark data, no plan jobs, one checkpoint.
# Table 10 needs Qwen3-VL-8B and InternVL3-8B, and the only risk unit tests
# cannot reach is whether the image span is contiguous on a real checkpoint.
# Isolated into its own session on purpose -- a round is not the place to
# find out, and this costs one download instead of twenty GPU-hours.
#
#   SMOKE=Qwen/Qwen3-VL-8B-Instruct bash run_kaggle_fox.sh 0
#   SMOKE=all bash run_kaggle_fox.sh 0
#
# One model per session by default. Each 8B checkpoint is ~16 GiB at fp16 and
# the HF cache keeps both, so `all` wants the disk checked first.
if [ -n "${SMOKE:-}" ]; then
  echo "[smoke] backbone pre-flight; no benchmark data will be fetched"
  # Checked before the DRY branch below, because a smoke session has no
  # session number in the frozen plan and `plan run --session 0` would die
  # on a lookup instead of saying so.
  if [ "${DRY:-0}" = "1" ]; then
    [ "$SMOKE" = "all" ] && echo "python3 scripts/smoke_backbone.py --all" \
      || echo "python3 scripts/smoke_backbone.py --model-id $SMOKE"
    exit 0
  fi
  pip install -q -e . || { echo "editable install failed"; exit 1; }
  if [ "${ALLOW_CPU:-0}" != "1" ]; then
    python3 scripts/require_gpu.py || exit 3
  fi
  if [ "$SMOKE" = "all" ]; then
    python3 scripts/smoke_backbone.py --all
  else
    python3 scripts/smoke_backbone.py --model-id "$SMOKE"
  fi
  rc=$?
  echo "[smoke] exit $rc"
  exit $rc
fi

if [ "${DRY:-0}" = "1" ]; then
  python3 -m foxrepro.bench.plan run --session "$N" --dry-run
  exit 0
fi

pip install -q -e . || { echo "editable install failed"; exit 1; }

# Fail before 87 GPU-hours of work discovers it: the checkpoint must be the
# HF-converted repo. The original liuhaotian/llava-v1.5-7b loads with a
# randomly initialised vision tower and produces fluent captions unrelated to
# the image, which no metric downstream would catch.
python3 scripts/probe_host.py || echo "[warn] probe reported problems; read them"

# A missing accelerator is not a warning. r4 sessions 1 and 2 ran to the
# ten-hour wall on a notebook whose GPU was never attached -- 724 s per caption
# against 7.8 s on two T4s -- and finished about a tenth of one job each. The
# probe said so, and the session carried on, because the probe only warned.
if [ "${ALLOW_CPU:-0}" != "1" ]; then
  python3 scripts/require_gpu.py || exit 3
fi

python3 - <<'PY'
import sys
from foxrepro.storage import paths, readiness_report
p = paths()
c = readiness_report(p)["checks"]
if not c["pope_files"]:
    sys.exit("POPE annotations missing from the package -- reinstall")
if not p.chair_image_list.is_file():
    sys.exit("the CHAIR image list is missing -- reinstall; it ships in assets/")
PY

# --what auto reads the frozen plan and fetches exactly what THIS session's
# jobs need. Hardcoding --what pope here is what made every CHAIR job in
# session 1 die within seconds: the data step and the run step disagreed.
python3 -m foxrepro.data.fetch_images --what auto --session "$N" \
    --workers 8 --progress-every 100

# MME ships its images inside parquet, so it needs its own fetch and the
# `datasets` package -- neither of which the POPE/CHAIR path touches. Only
# when THIS session actually has MME jobs: the download is ~864 MB and every
# other session would pay it for nothing. Same lesson as --what auto above,
# where a hardcoded benchmark killed every CHAIR job in session 1.
NEEDS_MME=$(python3 -c 'import json,sys; p=json.load(open("scripts/run_plan.json")); n=int(sys.argv[1]); s=next((x for x in p["sessions"] if int(x["session"])==n), {}); print(int(any("--benchmark mme" in p["jobs"][j]["command"] for j in s.get("jobs", []))))' "$N")
if [ "$NEEDS_MME" = "1" ]; then
  echo "[mme] session $N has MME jobs; fetching the split"
  pip install -q datasets || { echo "datasets install failed"; exit 1; }
  python3 -m foxrepro.data.fetch_mme || { echo "MME fetch failed"; exit 1; }
else
  echo "[mme] no MME jobs in session $N; skipping the 864 MB download"
fi

# MMBench (~98 MB) and MathVista (~142 MB) are the same shape: images inside
# Hub parquet, fetched only by a session that has jobs for them, and neither
# touches COCO (fetch_images.py subtracts them in --what auto).
needs_bench() {
  python3 -c 'import json,sys; p=json.load(open("scripts/run_plan.json")); n=int(sys.argv[1]); s=next((x for x in p["sessions"] if int(x["session"])==n), {}); print(int(any("--benchmark "+sys.argv[2]+" " in p["jobs"][j]["command"] for j in s.get("jobs", []))))' "$N" "$1"
}
# mmbench_circular reads the SAME manifest as mmbench, so either one needs the
# fetch. `needs_bench` matches "--benchmark <name> " with a trailing space and
# so does not see the longer name from the shorter one.
if [ "$(needs_bench mmbench)" = "1" ] || [ "$(needs_bench mmbench_circular)" = "1" ]; then
  echo "[mmbench] session $N has MMBench jobs; fetching dev"
  pip install -q datasets || { echo "datasets install failed"; exit 1; }
  python3 -m foxrepro.data.fetch_mmbench || { echo "MMBench fetch failed"; exit 1; }
fi
if [ "$(needs_bench mathvista)" = "1" ]; then
  echo "[mathvista] session $N has MathVista jobs; fetching MC500"
  pip install -q datasets || { echo "datasets install failed"; exit 1; }
  python3 -m foxrepro.data.fetch_mathvista || { echo "MathVista fetch failed"; exit 1; }
fi

# Kaggle gives a session no memory of the last one: /kaggle/working starts
# empty, so resume would see no run files and redo work already paid for.
# results/runs/ is the committed archive, so a plain `git clone` already
# carries every finished job and this picks them up -- no Dataset to attach,
# nothing to remember. /kaggle/input is still read for results that have not
# been committed yet.
python3 -m foxrepro.bench.plan ingest results/runs || true
if [ -d /kaggle/input ]; then
  python3 -m foxrepro.bench.plan ingest /kaggle/input || true
fi

python3 -m foxrepro.bench.plan run --session "$N" --budget-hours "${BUDGET:-10}"
RC=$?

bash scripts/pack_runs.sh "$(python3 -m foxrepro.bench.plan run --session "$N" --tag)"
exit $RC
