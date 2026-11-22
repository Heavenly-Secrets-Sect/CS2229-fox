"""Fetch the MathVista multiple-choice subset used for Table 11.

``AI4Math/MathVista``, split ``testmini`` (1000 rows). Of those, 540 have
``question_type == "multi_choice"`` (the dataset's spelling; all 540 also have
``answer_type == "text"``). The paper evaluates a random 500-question subset
and does not publish which; this fetcher draws 500 of the 540 with
``random.Random(storage.MATHVISTA_SUBSET_SEED)`` over the numerically sorted
pids (``loaders.mathvista_subset_pids``), so every arm and seed scores the
same 500 and the draw is reproducible from the seed alone. Written to::

    <mathvista_dir>/images/<pid>.<ext>
    <mathvista_dir>/mathvista_testmini_mc500.jsonl
        {pid, image, query, question, options{A..}, choices[], answer,
         answer_letters[], subset_seed, subset_n, pool_size}

Facts read from the real split: ``answer`` is the TEXT of the right choice,
not a letter; 3 of the 540 questions repeat a choice text, and in one of
those the gold text is repeated, so ``answer_letters`` lists every letter
whose text equals it and the scorer accepts any. ``query`` is the benchmark's
own prompt ("Hint: ... Question: ... Choices:\\n(A) ...") and is sent
verbatim. Choice counts run from 2 to 8.

The manifest is written **last**; a ``--limit`` run writes a ``partial_``
manifest that nothing loads.

Usage::

    python3 -m foxrepro.data.fetch_mathvista
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from foxrepro.data.fetch_mme import _image_ext
from foxrepro.data.fetch_mmbench import _image_bytes
from foxrepro.data.loaders import MATHVISTA_MANIFEST_NAME, mathvista_subset_pids
from foxrepro.storage import (
    MATHVISTA_MC_POOL, MATHVISTA_SUBSET_N, MATHVISTA_SUBSET_SEED, paths,
)

MATHVISTA_DATASET = "AI4Math/MathVista"
_LETTERS = "ABCDEFGH"


def build(rows: Iterable[Dict[str, Any]], dest: Path,
          limit: Optional[int] = None, pool_size: int = MATHVISTA_MC_POOL,
          subset_n: int = MATHVISTA_SUBSET_N,
          seed: int = MATHVISTA_SUBSET_SEED) -> Dict[str, object]:
    """Filter to multiple choice, draw the subset, extract, write manifest.

    ``pool_size`` and ``subset_n`` default to the real split's numbers and
    are overridable only so a fixture can exercise the assertions at small
    scale; a full run (``limit is None``) with the defaults checks the real
    ones.
    """
    dest = Path(dest)
    images = dest / "images"
    images.mkdir(parents=True, exist_ok=True)

    pool: Dict[str, Dict[str, Any]] = {}
    for rec in rows:
        if rec["question_type"] != "multi_choice":
            continue
        pid = str(rec["pid"])
        if pid in pool:
            raise ValueError(f"duplicate MathVista pid {pid!r}")
        pool[pid] = rec
    if len(pool) != pool_size:
        raise ValueError(
            f"found {len(pool)} multiple-choice rows in testmini, expected "
            f"{pool_size}. If the dataset was revised, the subset draw "
            f"changes with it and every earlier run stops being comparable.")

    chosen = mathvista_subset_pids(list(pool), subset_n, seed)
    if limit is not None:
        chosen = chosen[:limit]

    out_rows: List[Dict[str, object]] = []
    for pid in chosen:
        rec = pool[pid]
        choices = [str(c) for c in rec["choices"]]
        if not 2 <= len(choices) <= len(_LETTERS):
            raise ValueError(f"pid {pid}: {len(choices)} choices")
        options = {_LETTERS[i]: c for i, c in enumerate(choices)}
        gold = str(rec["answer"])
        letters = [k for k, v in options.items() if v == gold]
        if not letters:
            raise ValueError(
                f"pid {pid}: answer {gold!r} is not among its choices "
                f"{choices}")
        query = str(rec["query"])
        if "\n(A) " not in query:
            raise ValueError(
                f"pid {pid}: the query does not list its choices as '(A) ' "
                f"lines, so the prompt this scorer assumes is not the one "
                f"the dataset ships")
        buf = _image_bytes(rec["decoded_image"])
        name = f"{pid}.{_image_ext(buf)}"
        (images / name).write_bytes(buf)
        out_rows.append({
            "pid": pid, "image": f"images/{name}", "query": query,
            "question": str(rec["question"]), "options": options,
            "choices": choices, "answer": gold, "answer_letters": letters,
            "subset_seed": seed, "subset_n": subset_n, "pool_size": pool_size,
        })

    if limit is None and len(out_rows) != subset_n:
        raise ValueError(f"wrote {len(out_rows)} rows, expected {subset_n}")

    # Manifest LAST.
    name = (MATHVISTA_MANIFEST_NAME if limit is None
            else f"partial_{limit}_{MATHVISTA_MANIFEST_NAME}")
    out = dest / name
    with out.open("w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    return {"dest": str(dest), "manifest": str(out), "n_rows": len(out_rows),
            "pool_size": len(pool), "subset_seed": seed,
            "n_multi_gold": sum(len(r["answer_letters"]) > 1  # type: ignore[arg-type]
                                for r in out_rows),
            "partial": limit is not None}


def fetch(dest: Path, limit: Optional[int] = None,
          dataset: str = MATHVISTA_DATASET) -> Dict[str, object]:
    """Download MathVista testmini and write the MC500 layout."""
    try:
        from datasets import Image, load_dataset
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "MathVista needs `datasets`: pip install datasets.") from exc
    ds = load_dataset(dataset, split="testmini")
    ds = ds.cast_column("decoded_image", Image(decode=False))
    return build(ds, dest, limit=limit)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=None,
                    help="default: paths().mathvista_dir")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N of the drawn subset; writes a partial_ "
                         "manifest")
    ap.add_argument("--dataset", default=MATHVISTA_DATASET)
    args = ap.parse_args(argv)
    dest = args.dest or paths().mathvista_dir
    print(json.dumps(fetch(dest, limit=args.limit, dataset=args.dataset),
                     indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
