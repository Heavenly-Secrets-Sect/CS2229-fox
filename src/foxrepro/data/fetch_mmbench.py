"""Fetch MMBench (English) dev and lay it out the way the pipeline expects.

``lmms-lab/MMBench_EN``, split ``dev`` only: the ``test`` split withholds its
answers (they are scored on a server), so it cannot be scored here. 4377 rows,
one parquet shard of ~98 MB with the images embedded. Written to::

    <mmbench_dir>/images/<index>.<ext>
    <mmbench_dir>/mmbench_dev.jsonl
        {index, image, question, hint, options{A..D}, answer, category,
         l2_category, source}

What the real split looks like (read from the Hub, not assumed): ``index`` is
unique per question (4377 of 4377); options absent from a question are the
STRING ``"nan"`` (C and D on 2-3 option questions), as is a missing ``hint``;
``answer`` is a letter among the present options; ``l2-category`` takes six
values, which ``storage.MMBENCH_PERCEPTION_L2`` / ``MMBENCH_REASONING_L2``
partition into MMBench's perception and reasoning groups.

The manifest is written **last**, so a killed session leaves no manifest and
``load_mmbench`` refuses rather than scoring a fragment. A ``--limit`` run
writes ``partial_<N>_mmbench_dev.jsonl``, which nothing loads.

Usage::

    python3 -m foxrepro.data.fetch_mmbench
    python3 -m foxrepro.data.fetch_mmbench --dest /kaggle/temp/mmbench --limit 40
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from foxrepro.data.fetch_mme import _image_ext
from foxrepro.data.loaders import MMBENCH_MANIFEST_NAME
from foxrepro.storage import (
    MMBENCH_DEV_ROWS, MMBENCH_PERCEPTION_L2, MMBENCH_REASONING_L2, paths,
)

MMBENCH_DATASET = "lmms-lab/MMBench_EN"
_NAN = "nan"


def _clean(value: Any) -> Optional[str]:
    """``None`` for the dataset's missing-value spellings, else the string."""
    if value is None:
        return None
    text = str(value)
    return None if text.strip().lower() in (_NAN, "none", "") else text


def _image_bytes(img: Any) -> bytes:
    if isinstance(img, dict) and img.get("bytes"):
        return img["bytes"]
    buf = getattr(img, "_bytes", None)
    if buf:
        return buf
    if hasattr(img, "convert"):                      # a decoded PIL image
        import io
        bio = io.BytesIO()
        img.convert("RGB").save(bio, format="JPEG", quality=95)
        return bio.getvalue()
    raise ValueError(f"cannot get bytes out of an image of type {type(img)}")


def build(rows: Iterable[Dict[str, Any]], dest: Path,
          limit: Optional[int] = None) -> Dict[str, object]:
    """Extract images and write the manifest from an iterable of dataset rows.

    Separate from :func:`fetch` so the layout and every assertion can be
    tested on fixtures without ``datasets`` or the network.
    """
    dest = Path(dest)
    images = dest / "images"
    images.mkdir(parents=True, exist_ok=True)
    known_l2 = set(MMBENCH_PERCEPTION_L2) | set(MMBENCH_REASONING_L2)

    out_rows: List[Dict[str, object]] = []
    seen: set = set()
    for i, rec in enumerate(rows):
        if limit is not None and i >= limit:
            break
        index = int(rec["index"])
        if index in seen:
            raise ValueError(
                f"duplicate MMBench index {index} at row {i}; sample ids are "
                f"built from it and must be unique per question")
        seen.add(index)

        options: Dict[str, str] = {}
        for letter in "ABCD":
            text = _clean(rec.get(letter))
            if text is not None:
                options[letter] = text
        if sorted(options) not in (["A", "B"], ["A", "B", "C"],
                                   ["A", "B", "C", "D"]):
            raise ValueError(
                f"index {index}: options {sorted(options)} are not a prefix "
                f"of A-D")
        answer = str(rec["answer"]).strip()
        if answer not in options:
            raise ValueError(
                f"index {index}: answer {answer!r} is not among the options "
                f"present {sorted(options)}")
        l2 = str(rec["l2-category"])
        if l2 not in known_l2:
            raise ValueError(
                f"index {index}: l2-category {l2!r} is in neither MMBench "
                f"group. Known: {sorted(known_l2)}. Extend storage.py after "
                f"checking MMBench's definition; the scorer refuses it too.")

        buf = _image_bytes(rec["image"])
        name = f"{index}.{_image_ext(buf)}"
        (images / name).write_bytes(buf)
        out_rows.append({
            "index": index,
            "image": f"images/{name}",
            "question": str(rec["question"]),
            "hint": _clean(rec.get("hint")),
            "options": options,
            "answer": answer,
            "category": str(rec["category"]),
            "l2_category": l2,
            "source": str(rec.get("source", "")),
            # A content hash, so a re-fetch that changed an image is visible.
            "image_sha1": hashlib.sha1(buf).hexdigest()[:12],
        })

    if limit is None and len(out_rows) != MMBENCH_DEV_ROWS:
        raise ValueError(
            f"MMBench dev has {len(out_rows)} rows, expected "
            f"{MMBENCH_DEV_ROWS}. A short download scores as a low "
            f"accuracy rather than as an error, so it is refused here.")

    # Manifest LAST.
    name = (MMBENCH_MANIFEST_NAME if limit is None
            else f"partial_{limit}_{MMBENCH_MANIFEST_NAME}")
    out = dest / name
    with out.open("w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    l2_counts: Dict[str, int] = {}
    for r in out_rows:
        l2_counts[str(r["l2_category"])] = l2_counts.get(
            str(r["l2_category"]), 0) + 1
    return {"dest": str(dest), "manifest": str(out), "n_rows": len(out_rows),
            "l2_categories": dict(sorted(l2_counts.items())),
            "partial": limit is not None}


def fetch(dest: Path, limit: Optional[int] = None,
          dataset: str = MMBENCH_DATASET) -> Dict[str, object]:
    """Download MMBench dev and write the layout. See the module docstring."""
    try:
        from datasets import Image, load_dataset
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "MMBench needs `datasets` to read the parquet shard: pip "
            "install datasets. It is not a dependency of the rest of this "
            "package.") from exc
    ds = load_dataset(dataset, split="dev")
    ds = ds.cast_column("image", Image(decode=False))   # raw bytes, no PIL
    return build(ds, dest, limit=limit)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=None,
                    help="default: paths().mmbench_dir")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N rows; writes a partial_ manifest")
    ap.add_argument("--dataset", default=MMBENCH_DATASET)
    args = ap.parse_args(argv)
    dest = args.dest or paths().mmbench_dir
    print(json.dumps(fetch(dest, limit=args.limit, dataset=args.dataset),
                     indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
