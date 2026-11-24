"""A job longer than one session, and OPERA's CHAIR run, which is one.

``run_session`` DEFERs any job whose estimate is above the budget still left
in the session. For a job that cannot fit a session at all -- OPERA on CHAIR
is 40.5 GPU-hours at the 512-token upper bound against 10 h sessions -- that
means "never starts, and the session runs empty". The scheme that replaces it:
the job is written as chunks on ONE run file with a cumulative ``--limit``
(resume is keyed on the file, so chunk k generates only what is missing), and
a chunk is only scheduled once its predecessors' records are in the archive.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from foxrepro.bench.plan import (
    BEAM_COST_S, OPERA_CHAIR_CHUNK, PLAN, Job, freeze_plan, next_jobs,
    partition_by_priority, partition_jobs, pending_jobs, run_session,
    scan_progress,
)
from tests.test_plan import cfg_from_command

BUDGET_H = 10.0


def chunks():
    return [j for j in PLAN if j.target == "table2_opera"]


def write_records(job: Job, n: int, where: Path) -> Path:
    """``n`` valid-looking records in ``job``'s run file."""
    where.mkdir(parents=True, exist_ok=True)
    prov = cfg_from_command(job.command()).provenance()
    f = where / job.run_key().filename()
    with f.open("w") as fh:
        for i in range(n):
            rec = dict(prov)
            rec.update(sample_id=f"chair/val2014_500/q{i:07d}", image_id=i,
                       prompt="p", output="a caption", latency_ms=1.0,
                       n_prompt_tokens=600, n_gen_tokens=90,
                       timestamp="2026-01-01T00:00:00Z", commit_hash="x",
                       transformers_path="y", label="")
            fh.write(json.dumps(rec) + "\n")
    return f


class TestTheOperaChairJobs(unittest.TestCase):
    def test_five_chunks_cover_the_whole_split_exactly_once(self) -> None:
        cs = chunks()
        self.assertEqual(len(cs), 5)
        self.assertEqual([j.resume_from for j in cs],
                         [0, 100, 200, 300, 400])
        self.assertEqual([j.n_samples for j in cs],
                         [100, 200, 300, 400, 500])
        self.assertEqual(sum(j.chunk_samples for j in cs), 500)

    def test_they_all_write_one_run_file(self) -> None:
        files = {j.run_key().filename() for j in chunks()}
        self.assertEqual(len(files), 1)
        (name,) = files
        self.assertIn("m-opera__b-chair__s-val2014_500", name)

    def test_each_chunks_command_round_trips_to_that_file(self) -> None:
        for j in chunks():
            with self.subTest(job=j.job_id()):
                self.assertEqual(
                    cfg_from_command(j.command()).run_key().filename(),
                    j.run_key().filename())

    def test_the_total_is_the_audits_40_5_hours_from_beam_cost(self) -> None:
        want = 500 * (BEAM_COST_S["prefill"]
                      + 512 * 6 * BEAM_COST_S["forward"]) / 3600
        got = sum(j.est_seconds() for j in chunks()) / 3600
        self.assertAlmostEqual(got, want, places=6)
        self.assertAlmostEqual(got, 40.5, delta=0.1)

    def test_no_job_in_the_plan_is_longer_than_a_session(self) -> None:
        """Else run_session defers it forever. The guard for the next one."""
        for j in PLAN:
            self.assertLessEqual(
                j.est_seconds() / 3600, BUDGET_H,
                f"{j.job_id()} is {j.est_seconds() / 3600:.1f} h; write it "
                "as chunks (see OPERA_CHAIR_CHUNK)")

    def test_it_runs_at_chairs_512_tokens_with_the_rollback_regime(self) -> None:
        j = chunks()[0]
        self.assertEqual(j.n_new_tokens(), 512)
        self.assertGreater(j.n_new_tokens(), 10)
        self.assertEqual(j.decoding, "greedy")

    def test_job_ids_stay_unique(self) -> None:
        ids = [j.job_id() for j in chunks()]
        self.assertEqual(len(set(ids)), 5)

    def test_a_chunk_must_start_inside_its_own_limit(self) -> None:
        bad = Job("t", "opera", "chair", "val2014_500", decoding="greedy",
                  limit=100, resume_from=100)
        with self.assertRaises(ValueError):
            bad.chunk_samples

    def test_every_shared_file_is_one_contiguous_chain(self) -> None:
        """No gap, no overlap, and the chain ends at the whole split."""
        from collections import defaultdict
        from foxrepro.bench.plan import N_SAMPLES

        by_file = defaultdict(list)
        for j in PLAN:
            by_file[j.run_key().filename()].append(j)
        shared = {f: sorted(js, key=lambda j: j.resume_from)
                  for f, js in by_file.items() if len(js) > 1}
        self.assertTrue(shared)
        for f, js in shared.items():
            with self.subTest(file=f):
                self.assertEqual(js[0].resume_from, 0)
                for a, b in zip(js, js[1:]):
                    self.assertEqual(b.resume_from, a.n_samples)
                self.assertEqual(
                    js[-1].n_samples,
                    N_SAMPLES[(js[-1].benchmark, js[-1].split)])

    def test_the_chunk_size_fits_a_session_with_room_to_spare(self) -> None:
        h = OPERA_CHAIR_CHUNK * (
            BEAM_COST_S["prefill"] + 512 * 6 * BEAM_COST_S["forward"]) / 3600
        self.assertLess(h, BUDGET_H - 1.0)


class _Archive(unittest.TestCase):
    def setUp(self) -> None:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.tmp = Path(td.name)
        self.runs = self.tmp / "runs"
        self.runs.mkdir()
        self.cs = chunks()

    def ids(self, jobs):
        return [j.job_id() for j in jobs]


class TestReadiness(_Archive):
    def test_with_nothing_archived_only_the_first_chunk_is_scheduled(self) -> None:
        keep, waiting = pending_jobs(self.cs, self.runs)
        self.assertEqual(self.ids(keep), self.ids(self.cs[:1]))
        self.assertEqual(self.ids(waiting), self.ids(self.cs[1:]))

    def test_a_finished_chunk_releases_the_next_one_only(self) -> None:
        write_records(self.cs[0], 100, self.runs)
        keep, waiting = pending_jobs(self.cs, self.runs)
        self.assertEqual(self.ids(keep), self.ids(self.cs[1:2]))
        self.assertEqual(self.ids(waiting), self.ids(self.cs[2:]))

    def test_a_half_done_chunk_is_still_scheduled_and_resumes(self) -> None:
        write_records(self.cs[0], 100, self.runs)
        write_records(self.cs[1], 130, self.runs)   # same file, 130 records
        keep, _ = pending_jobs(self.cs, self.runs)
        self.assertEqual(self.ids(keep), self.ids(self.cs[1:2]))
        row = scan_progress(self.cs, runs_root=self.runs)["rows"][1]
        self.assertEqual(row["done"], 130)
        # 70 of the chunk's 100 samples remain, priced per sample
        self.assertAlmostEqual(
            row["remaining_s"], self.cs[1].est_seconds() * 0.7, places=3)

    def test_finishing_the_file_completes_every_chunk(self) -> None:
        write_records(self.cs[0], 500, self.runs)
        keep, waiting = pending_jobs(self.cs, self.runs)
        self.assertEqual(keep, [])
        self.assertEqual(waiting, [])

    def test_completeness_is_per_job_not_per_file(self) -> None:
        """Chunk 1 done must not drop chunks 2..5 for sharing its file."""
        write_records(self.cs[0], 100, self.runs)
        for groups in (partition_by_priority(BUDGET_H, self.cs, self.runs),
                       partition_jobs(3, self.cs, runs_root=self.runs,
                                      skip_done=True)):
            scheduled = [j for g in groups for j in g]
            self.assertEqual(self.ids(scheduled), self.ids(self.cs[1:2]))

    def test_one_chunk_per_round_and_it_fits_a_session(self) -> None:
        for n_done, expect in ((0, 0), (100, 1), (250, 2), (400, 4)):
            with self.subTest(done=n_done):
                f = write_records(self.cs[0], n_done, self.runs) if n_done \
                    else None
                groups = partition_by_priority(BUDGET_H, self.cs, self.runs)
                flat = [j for g in groups for j in g]
                self.assertEqual(len(flat), 1)
                self.assertEqual(flat[0].job_id(), self.cs[expect].job_id())
                self.assertEqual(len(groups), 1)
                self.assertLessEqual(
                    sum(j.est_seconds() for j in flat) / 3600, BUDGET_H)
                if f:
                    f.unlink()

    def test_next_jobs_skips_chunks_that_are_not_ready(self) -> None:
        out = next_jobs(BUDGET_H, self.cs, runs_root=self.runs)
        self.assertEqual([o["target"] for o in out], ["table2_opera"])
        self.assertEqual(len(out), 1)

    def test_a_freeze_records_the_chunks_it_held_back(self) -> None:
        payload = freeze_plan(None, BUDGET_H, self.tmp / "p.json",
                              runs_root=self.runs)
        self.assertEqual(
            payload["chunks_waiting"], sorted(self.ids(self.cs[1:])))
        scheduled = [i for i in payload["jobs"] if i.startswith("table2_opera")]
        self.assertEqual(scheduled, [self.cs[0].job_id()])
        self.assertEqual(payload["jobs"][scheduled[0]]["resume_from"], 0)


class TestRunSessionWithChunks(_Archive):
    """run_session itself, with ``echo`` standing in for the GPU command."""

    def setUp(self) -> None:
        super().setUp()
        self._env = {k: os.environ.get(k)
                     for k in ("FOX_PERSIST_ROOT", "FOX_RUNS_ROOT")}
        os.environ["FOX_PERSIST_ROOT"] = str(self.tmp / "persist")
        os.environ["FOX_RUNS_ROOT"] = str(self.runs)
        from foxrepro.storage import paths
        paths(refresh=True)

        def restore():
            for k, v in self._env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            paths(refresh=True)
        self.addCleanup(restore)

    def _plan(self, job: Job, est_s: float, command: str) -> Path:
        plan = {
            "round": "t", "budget_hours": BUDGET_H,
            "jobs": {job.job_id(): {
                "id": job.job_id(), "command": command, "est_s": est_s,
                "run_file": job.run_key().filename(),
                "n_samples": job.n_samples, "resume_from": job.resume_from,
            }},
            "sessions": [{"session": 1, "jobs": [job.job_id()]}],
        }
        p = self.tmp / "plan.json"
        p.write_text(json.dumps(plan))
        return p

    def _run(self, plan: Path) -> str:
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_session(1, plan, pack=False, runs_root=self.runs)
        return buf.getvalue()

    def test_an_unready_chunk_is_skipped_not_run(self) -> None:
        marker = self.tmp / "ran"
        plan = self._plan(self.cs[2], 100.0, f"touch {marker}")
        out = self._run(plan)
        self.assertIn("chunk not ready", out)
        self.assertFalse(marker.exists(), "a not-ready chunk was started")

    def test_a_ready_chunk_runs(self) -> None:
        write_records(self.cs[0], 100, self.runs)
        marker = self.tmp / "ran"
        plan = self._plan(self.cs[1], 100.0, f"touch {marker}")
        out = self._run(plan)
        self.assertIn("START", out)
        self.assertTrue(marker.exists())

    def test_the_whole_40_hour_job_would_have_been_deferred(self) -> None:
        """What the chunks exist to avoid: the pre-chunking behaviour."""
        whole = Job("t", "opera", "chair", "val2014_500", decoding="greedy")
        marker = self.tmp / "ran"
        out = self._run(self._plan(whole, whole.est_seconds(),
                                   f"touch {marker}"))
        self.assertIn("DEFER", out)
        self.assertFalse(marker.exists())
        # ... and one chunk is not.
        marker2 = self.tmp / "ran2"
        out = self._run(self._plan(self.cs[0], self.cs[0].est_seconds(),
                                   f"touch {marker2}"))
        self.assertIn("START", out)
        self.assertTrue(marker2.exists())


class TestResumeIsKeyedOnTheRunFile(unittest.TestCase):
    """The property the whole scheme rests on, asserted rather than assumed."""

    def test_limit_is_cumulative_in_generate(self) -> None:
        import inspect

        from foxrepro.bench import generate
        src = inspect.getsource(generate.run)
        self.assertIn("cfg.limit - len(scan.done_ids)", src)

    def test_limit_is_not_part_of_the_run_key(self) -> None:
        a = Job("t", "opera", "chair", "val2014_500", decoding="greedy",
                limit=100)
        b = Job("t", "opera", "chair", "val2014_500", decoding="greedy")
        self.assertEqual(a.run_key().filename(), b.run_key().filename())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
