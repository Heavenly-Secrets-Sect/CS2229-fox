"""Fetch MME and lay it out the way the rest of the pipeline expects.

MME ships as four parquet shards on the Hub with the images EMBEDDED in the
table (``lmms-lab/MME``, split ``test``, 2374 rows, ~864 MB download). Every
other benchmark here reads images from disk by path, so this extracts them
once and writes a manifest beside them:

    <mme_dir>/images/<question_id>.<ext>
    <mme_dir>/mme_test.jsonl       {question_id, image, question, answer, category}

Written as a module rather than a notebook cell because a Kaggle session
runs it unattended and a half-extracted directory is worse than none: the
manifest is written **last**, so `load_mme` either finds a complete layout
or refuses.

Usage::

    python3 -m foxrepro.data.fetch_mme
    python3 -m foxrepro.data.fetch_mme --dest /kaggle/temp/mme --limit 40
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from foxrepro.data.loaders import MME_MANIFEST_NAME
from foxrepro.storage import MME_COGNITION, MME_PERCEPTION, paths

#: The Hub dataset. Pinned by name only; the loader verifies what arrives.
MME_DATASET = "lmms-lab/MME"

#: What the released split holds. Checked rather than trusted: a silently
#: truncated download would otherwise produce an MME score that looks like a
#: result and is a fraction of one, since the score SUMS over subtasks.
MME_N_ROWS = 2374


def _image_ext(data: bytes) -> str:
    if data[:2] == b"\xff\xd8":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    return "bin"


def fetch(dest: Path, limit: Optional[int] = None,
          dataset: str = MME_DATASET) -> Dict[str, object]:
    """Download MME, extract the images, write the manifest last.

    Args:
        dest: directory to populate. Created if absent.
        limit: stop after this many rows (for smoke tests). A limited run
            writes the manifest under a different name so it can never be
            mistaken for the full split.
        dataset: Hub id.

    Returns:
        A summary dict: counts, per-subtask totals, and the manifest path.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "MME needs `datasets` to read the parquet shards: pip install "
            "datasets. It is not a dependency of the rest of this package."
        ) from exc

    dest = Path(dest)
    images = dest / "images"
    images.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset, split="test")
    n_total = len(ds)
    if limit is None and n_total != MME_N_ROWS:
        raise ValueError(
            f"{dataset} split 'test' has {n_total} rows, expected "
            f"{MME_N_ROWS}. The MME score sums over subtasks, so a short "
            f"download would read as a low score rather than as an error."
        )

    rows: List[Dict[str, str]] = []
    seen_qid: set = set()
    for i, rec in enumerate(ds):
        if limit is not None and i >= limit:
            break
        qid = str(rec["question_id"])
        if qid in seen_qid:
            raise ValueError(f"duplicate question_id {qid!r} at row {i}")
        seen_qid.add(qid)

        img = rec["image"]
        buf = getattr(img, "_bytes", None)
        if buf is None:                      # a decoded PIL image
            import io
            bio = io.BytesIO()
            img.convert("RGB").save(bio, format="JPEG", quality=95)
            buf = bio.getvalue()
        ext = _image_ext(buf)
        # Name by question_id, not by a counter: a resumed or re-run fetch
        # then overwrites the same file instead of growing a second copy.
        name = f"{hashlib.sha1(qid.encode()).hexdigest()[:16]}.{ext}"
        (images / name).write_bytes(buf)

        rows.append({
            "question_id": qid,
            "image": f"images/{name}",
            "question": str(rec["question"]),
            "answer": str(rec["answer"]),
            "category": str(rec["category"]),
        })

    cats: Dict[str, int] = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    unknown = sorted(set(cats) - set(MME_PERCEPTION + MME_COGNITION))
    if unknown:
        raise ValueError(
            f"{dataset} has subtasks this repo does not know: {unknown}. "
            f"Add them to MME_PERCEPTION or MME_COGNITION in storage.py -- "
            f"an unknown subtask is silently dropped from both totals."
        )

    # Manifest LAST. A session killed mid-extract leaves no manifest, and
    # `load_mme` refuses rather than scoring a fraction of the split.
    name = MME_MANIFEST_NAME if limit is None else f"partial_{limit}_{MME_MANIFEST_NAME}"
    out = dest / name
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    return {
        "dest": str(dest), "manifest": str(out), "n_rows": len(rows),
        "n_images": len({r["image"] for r in rows}),
        "subtasks": dict(sorted(cats.items())),
        "partial": limit is not None,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=None,
                    help="default: paths().mme_dir")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N rows; writes a partial_ manifest")
    ap.add_argument("--dataset", default=MME_DATASET)
    args = ap.parse_args(argv)
    dest = args.dest or paths().mme_dir
    summary = fetch(dest, limit=args.limit, dataset=args.dataset)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
