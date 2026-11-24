#!/usr/bin/env bash
# One backbone pre-flight per CI session. kaggle-smoke.yml declares one session
# per case below, and tests/test_plan.py keeps the two counts equal.
#
# A smoke session loads one real checkpoint and asserts what unit tests cannot
# reach: that the image span is contiguous on a real model, that the Fox patch
# reaches the decoder, that no weights are missing. It fetches no benchmark
# data and runs no plan jobs.
#
# Output is exactly ONE zip in /kaggle/working (what the bot's download button
# points at): the JSON report and the log. It is named fox_smoke_<sha>_s<N>.zip: the
# prefix keeps it from overwriting a round's fox_runs_*.zip, and the commit sha
# keeps one smoke release from overwriting the last (smoke-r25 and smoke-r27
# both wrote fox_smoke_s1.zip, and a download skipped the new one as "already
# there"). The exit code is the smoke
# result, so a failed assertion shows as a failed session, and the zip is
# written either way because the log is what explains the failure.
set -uo pipefail
cd "$(dirname "$0")/.."

N="${1:?usage: bash scripts/ci_smoke.sh <case number>}"
case "$N" in
  1) ARGS=(--model-id Qwen/Qwen3-VL-8B-Instruct) ;;
  2) ARGS=(--model-id OpenGVLab/InternVL3-8B-hf) ;;
  3) ARGS=(--instructblip) ;;
  4) ARGS=(--shikra) ;;     # delta + llama-7b merged in RAM: DEVIATIONS 3.20
  5) ARGS=(verify) ;;   # not a backbone smoke: scripts/verify_bitwise.py
  *) echo "[smoke] no case $N"; exit 2 ;;
esac

export FOX_ENV="${FOX_ENV:-kaggle}"
pip install -q -e . || { echo "editable install failed"; exit 1; }
if [ "${ALLOW_CPU:-0}" != "1" ]; then
  python3 scripts/require_gpu.py || exit 3
fi

WORK="/kaggle/temp/smoke"; [ -d /kaggle/temp ] || WORK="${TMPDIR:-/tmp}/fox_smoke"
OUTDIR="/kaggle/working"; [ -d "$OUTDIR" ] || OUTDIR="."
mkdir -p "$WORK"
SHA="$(git rev-parse --short HEAD 2>/dev/null || echo nosha)"
if [ "$N" = "5" ]; then
  # Bit-for-bit check of greedy reruns against the archive (see the script).
  python3 -m foxrepro.data.fetch_images --what pope --workers 8 --progress-every 100 \
      > "$WORK/smoke_s${N}.log" 2>&1 || { echo "image fetch failed"; tail -n 20 "$WORK/smoke_s${N}.log"; exit 1; }
  # Same session, three code versions: HEAD, r17's last commit, and the commit
  # just before the InstructBLIP port. HEAD vs a ref is the effect of the code
  # alone; either vs the archive also contains the host drift since it was made.
  python3 scripts/verify_bitwise.py --n 200 --scratch "$WORK/verify" \
      --ref 1e9901e --ref a8ac2e5 \
      > "$WORK/smoke_s${N}.json" 2>> "$WORK/smoke_s${N}.log"
  rc=$?
else
  python3 scripts/smoke_backbone.py "${ARGS[@]}" --json \
      > "$WORK/smoke_s${N}.json" 2> "$WORK/smoke_s${N}.log"
  rc=$?
fi
echo "[smoke] case $N (${ARGS[*]}) exit $rc"
tail -n 40 "$WORK/smoke_s${N}.log"

rm -f "$OUTDIR/fox_smoke_${SHA}_s${N}.zip"
command -v zip >/dev/null 2>&1 || { echo "[smoke] zip not installed"; exit 1; }
zip -q -j "$OUTDIR/fox_smoke_${SHA}_s${N}.zip" "$WORK/smoke_s${N}.json" "$WORK/smoke_s${N}.log"
ls -lh "$OUTDIR/fox_smoke_${SHA}_s${N}.zip"
exit $rc
