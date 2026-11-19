# runs_inbox

Drop the zips downloaded from each Kaggle session here, then run:

```bash
python -m foxrepro.bench.plan ingest
```

With no arguments `ingest` reads this folder, expands every `*.zip`, and merges
the run JSONL into the runs root so `evaluate` and `report` see every session's
work at once. It is idempotent: re-ingesting the same zip adds nothing.

Partial files for the same job -- which happens when a plan edit moves a job
between sessions -- are merged by `sample_id` union rather than one session's
file overwriting the other's. Both halves are real GPU time.

The zips themselves are gitignored; the merged JSONL in the runs root is the
artefact worth keeping, and it is irreplaceable, so back it up deliberately.

## Deleting a zip once it is ingested

These add up -- 39 of them were 1.5 GB -- and they are safe to delete, but
only once the work inside them is somewhere else. Check, do not assume:

```bash
python -m foxrepro.bench.plan ingest      # runs_inbox -> runs root
python -m foxrepro.bench.plan archive     # runs root  -> results/runs
git add results/runs && git commit && git push
python3 scripts/verify_zip_redundant.py runs_inbox/*.zip
```

`verify_zip_redundant.py` prints `REDUNDANT` only when every run file in the
zip is in `results/runs` at full record count. Delete those; keep anything
it marks `KEEP`. It will mark `fox_runs_fox_r13_s1.zip` KEEP forever, for a
reason written at the top of that script.

Look in `~/Desktop/Workspace/ML_Algo/` too. Browser downloads land there,
not here, so a zip can be on the machine and invisible to anything that
only reads this folder.
