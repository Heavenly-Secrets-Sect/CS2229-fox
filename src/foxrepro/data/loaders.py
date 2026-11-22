"""Deterministic iterators over the POPE and CHAIR evaluation sets.

Two hard requirements shape this module:

* **Stable ``sample_id``.** Resume (Decision 4) keys on
  ``(method, benchmark, split, seed, dtype, sample_id)``, so a ``sample_id``
  must be a pure function of the benchmark item, never of its position in a
  list or of a wall-clock value. POPE ids come from the official
  ``question_id``; CHAIR ids come from the COCO ``image_id``.
* **Deterministic ordering.** Items are always yielded in ascending
  ``sample_id`` order, so a partially finished run and a fresh run agree on
  what "the next sample" is.

Generation length is a per-benchmark protocol constant, not a free knob:
POPE is generated with ``max_new_tokens = 1`` (the answer is one yes/no token,
so POPE is prefill-dominated and cheap per sample), CHAIR with
``max_new_tokens = 512``. See ``storage.MAX_NEW_TOKENS_DEFAULT``.

Prompt templates live here rather than in the model code, and every record
stores the ``prompt_template_id`` that produced it: the exact wording of the
captioning prompt is not stated in the Fox paper and is known to move CHAIR by
several points, so it must be part of the provenance.
"""

from __future__ import annotations

import json
import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

__all__ = [
    "PROMPT_TEMPLATES",
    "DEFAULT_TEMPLATE_IDS",
    "render_prompt",
    "Sample",
    "load_pope",
    "load_mme",
    "load_mmbench",
    "load_mathvista",
    "mathvista_subset_pids",
    "build_mmbench_question",
    "load_chair",
    "load_benchmark",
    "pope_file_for_split",
    "image_id_from_filename",
    "image_filename_for_id",
]

# --------------------------------------------------------------------------- #
# prompt templates
# --------------------------------------------------------------------------- #
#: ``prompt_template_id -> template``. ``{question}`` is substituted for POPE.
#:
#: ``pope_v0_raw`` sends the POPE question verbatim, which is what the Fox
#: release does: ``pope_eval.py`` substitutes the question straight into
#: ``"USER: <ImageHere> <question> ASSISTANT:"`` with no added instruction.
#: It is therefore the faithful default.
#:
#: ``pope_v1`` appends the standard LLaVA "answer in a single word or
#: phrase" instruction used by the VCD / SID / OPERA releases. That phrasing
#: is known to raise POPE accuracy by keeping answers parseable, so it is
#: kept as an ablation rather than deleted -- it is one of the prompt
#: differences being measured.
#:
#: ``chair_v1`` is ``char_eval.py:157`` verbatim, asserted against
#: ``reference.RELEASED_CHAIR_QUESTION``. ``chair_v2_short`` is an ablation.
PROMPT_TEMPLATES: Dict[str, str] = {
    "pope_v1": "{question}\nAnswer the question using a single word or phrase.",
    "pope_v0_raw": "{question}",
    "chair_v1": "Please describe this image in detail.",
    "chair_v2_short": "Describe this image in detail.",
    # MME ships its own instruction INSIDE the question text -- every item
    # ends with "Please answer yes or no." -- so the raw form is the faithful
    # one, exactly as pope_v0_raw is for POPE. Appending another instruction
    # would be an ablation, not the protocol.
    "mme_v0_raw": "{question}",
    # MMBench, LLaVA's own convention (llava/eval/model_vqa_mmbench.py): the
    # loader builds "<hint>\n<question>\nA. ..\nB. .." and this appends the
    # instruction. The hint is omitted when the row has none.
    "mmbench_v0_letter":
        "{question}\nAnswer with the option's letter from the given choices "
        "directly.",
    # MathVista ships the complete prompt in its `query` field ("Hint: Please
    # answer the question and provide the correct option letter, e.g., A, B,
    # C, D, at the end.\nQuestion: ...\nChoices:\n(A) ..."), which is what
    # the benchmark's own generation script sends. Used verbatim.
    "mathvista_v0_query": "{question}",
}

#: Which templates are legal for which benchmark (guards CLI typos).
_TEMPLATE_BENCHMARK = {
    "pope_v1": "pope",
    "pope_v0_raw": "pope",
    "chair_v1": "chair",
    "chair_v2_short": "chair",
    "mme_v0_raw": "mme",
    "mmbench_v0_letter": "mmbench",
    "mathvista_v0_query": "mathvista",
}

#: ``benchmark -> the template the release sends``. One definition, read by
#: ``plan.Job.template_id`` and by ``report`` when it filters the archive down
#: to the arm that is comparable to the paper. Two copies of this mapping
#: would let the planner run one arm while the report tabulated another.
DEFAULT_TEMPLATE_IDS: Dict[str, str] = {
    "pope": "pope_v0_raw",
    "chair": "chair_v1",
    "mme": "mme_v0_raw",
    "mmbench": "mmbench_v0_letter",
    "mathvista": "mathvista_v0_query",
}

_IMAGE_ID_RE = re.compile(r"(\d{6,})")


def render_prompt(template_id: str, question: Optional[str] = None) -> str:
    """Render a prompt from the registry.

    Args:
        template_id: key of :data:`PROMPT_TEMPLATES`.
        question: required for POPE templates, ignored for CHAIR templates.

    Raises:
        KeyError: unknown template id.
        ValueError: a POPE template was used without a question.
    """
    template = PROMPT_TEMPLATES[template_id]
    if "{question}" in template:
        if question is None:
            raise ValueError(f"template {template_id!r} needs a question")
        return template.format(question=question)
    return template


def template_benchmark(template_id: str) -> str:
    """Benchmark a template belongs to (``"pope"`` or ``"chair"``)."""
    return _TEMPLATE_BENCHMARK[template_id]


# --------------------------------------------------------------------------- #
# sample record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Sample:
    """One evaluation item, independent of any model.

    Attributes:
        sample_id: stable identifier used by the resume key.
        image_id: COCO numeric image id (val2014).
        image_path: absolute path to the decoded JPEG on fast local disk.
        question: the POPE question, or ``None`` for CHAIR.
        label: POPE ground truth, ``"yes"`` / ``"no"``; ``None`` for CHAIR.
        benchmark: ``"pope"`` or ``"chair"``.
        split: benchmark split name.
        extra: anything else worth keeping (e.g. the POPE ``question_id``).
    """

    sample_id: str
    image_id: int
    image_path: Path
    benchmark: str
    split: str
    question: Optional[str] = None
    label: Optional[str] = None
    extra: Dict[str, object] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.extra is None:
            object.__setattr__(self, "extra", {})


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def image_id_from_filename(filename: str) -> int:
    """``"COCO_val2014_000000131089.jpg"`` -> ``131089``.

    The Fox reference code parses CHAIR image ids as
    ``int(name.split(".jpg")[0][-6:])``, i.e. the last six digits. That agrees
    with this function for every val2014 id (the largest is 581929, six
    digits), but it would silently truncate a seven-digit id, so the full
    numeric field is parsed here instead. ``sample_id`` never depends on this
    parse: it is built from the zero-padded canonical id.

    Raises:
        ValueError: when no numeric id can be found.
    """
    match = _IMAGE_ID_RE.search(Path(filename).stem)
    if not match:
        raise ValueError(f"cannot parse a COCO image id out of {filename!r}")
    return int(match.group(1))


def image_filename_for_id(image_id: int) -> str:
    """``131089`` -> ``"COCO_val2014_000000131089.jpg"`` (val2014 convention)."""
    return f"COCO_val2014_{image_id:012d}.jpg"


def pope_file_for_split(pope_dir: Path, split: str) -> Path:
    """Locate the POPE annotation file for a split.

    Accepts the Fox reference-repo names (``coco_random.jsonl`` /
    ``coco_popular.jsonl`` / ``coco_adversarial.jsonl``, which is what this
    repository ships in ``data/pope/``), the official POPE release names
    (``coco_pope_random.json`` ...), and the bare ``random.json`` form.
    """
    candidates = [
        # layout used by the Fox reference repo (pope/coco/coco_random.jsonl)
        pope_dir / f"coco_{split}.jsonl",
        pope_dir / f"coco_{split}.json",
        # layout of the official POPE release
        pope_dir / f"coco_pope_{split}.json",
        pope_dir / f"coco_pope_{split}.jsonl",
        # bare names, for a hand-assembled directory
        pope_dir / f"{split}.jsonl",
        pope_dir / f"{split}.json",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"no POPE file for split {split!r} in {pope_dir}; "
        f"looked for {[c.name for c in candidates]}"
    )


def _read_json_records(path: Path) -> List[Dict[str, object]]:
    """Read either a JSON array or JSON-lines file into a list of dicts."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path} is not a list of records")
        return data
    records: List[Dict[str, object]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno} is not valid JSON") from exc
    return records


# --------------------------------------------------------------------------- #
# POPE
# --------------------------------------------------------------------------- #
def load_pope(
    pope_dir: Path,
    images_dir: Path,
    split: str,
    require_images: bool = True,
) -> List[Sample]:
    """Load one POPE split.

    Each POPE record is ``{question_id, image, text, label}`` where ``label``
    is ``"yes"``/``"no"``. The official COCO-POPE release has 500 val2014
    images x 6 questions = 3000 items per split; this loader does not enforce
    that count but :func:`describe_split` reports it.

    Args:
        pope_dir: directory containing ``coco_pope_<split>.json``.
        images_dir: directory of extracted val2014 JPEGs.
        split: ``"random"`` / ``"popular"`` / ``"adversarial"``.
        require_images: raise if a referenced image is missing locally. Set
            ``False`` for CPU-side inspection without the image tar.

    Returns:
        Samples sorted by ``question_id``.
    """
    path = pope_file_for_split(Path(pope_dir), split)
    raw = _read_json_records(path)
    if not raw:
        raise ValueError(f"{path} contained no records")

    samples: List[Sample] = []
    seen: set[str] = set()
    missing: List[str] = []
    for idx, rec in enumerate(raw):
        question = rec.get("text") or rec.get("question")
        label = rec.get("label") or rec.get("answer")
        image_field = rec.get("image") or rec.get("image_file") or rec.get("image_id")
        if question is None or label is None or image_field is None:
            raise ValueError(f"{path} record {idx} missing text/label/image: {rec!r}")
        label_norm = str(label).strip().lower()
        if label_norm not in ("yes", "no"):
            raise ValueError(f"{path} record {idx} has non-binary label {label!r}")
        if isinstance(image_field, int):
            image_id = image_field
            image_name = image_filename_for_id(image_id)
        else:
            image_name = str(image_field)
            image_id = image_id_from_filename(image_name)
        qid = rec.get("question_id", idx)
        sample_id = f"pope/{split}/q{int(qid):07d}"
        if sample_id in seen:
            raise ValueError(f"duplicate question_id {qid} in {path}")
        seen.add(sample_id)
        image_path = Path(images_dir) / image_name
        if require_images and not image_path.is_file():
            missing.append(image_name)
        samples.append(
            Sample(
                sample_id=sample_id,
                image_id=image_id,
                image_path=image_path,
                benchmark="pope",
                split=split,
                question=str(question),
                label=label_norm,
                extra={"question_id": int(qid), "source_file": path.name},
            )
        )
    if missing:
        uniq = sorted(set(missing))
        raise FileNotFoundError(
            f"{len(uniq)} POPE images missing from {images_dir} "
            f"(first few: {uniq[:5]}). Extract the COCO subset tar first."
        )
    samples.sort(key=lambda s: s.sample_id)
    return samples


# --------------------------------------------------------------------------- #
# CHAIR
# --------------------------------------------------------------------------- #
def load_chair(
    image_list_path: Path,
    images_dir: Path,
    split: str = "val2014_500",
    prompt_template_id: str = "chair_v1",
    require_images: bool = True,
) -> List[Sample]:
    """Load the CHAIR captioning set (500 COCO val2014 images).

    Args:
        image_list_path: ``chair_images.json`` as frozen into the subset tar.
            Accepts a bare list of ids, a list of filenames, or
            ``{"image_ids": [...]}``.
        images_dir: directory of extracted val2014 JPEGs.
        split: split label recorded in provenance (default ``"val2014_500"``).
        prompt_template_id: which captioning prompt to attach.
        require_images: raise if an image is missing locally.

    Returns:
        Samples sorted by ``image_id``.
    """
    path = Path(image_list_path)
    if not path.is_file():
        # A bare FileNotFoundError out of pathlib is what the first real
        # session produced, four times in forty seconds, and it said nothing
        # about which of several causes applied.
        raise FileNotFoundError(
            f"CHAIR image list not found at {path}.\n"
            f"It ships with the package at assets/chair_images.json, so the "
            f"usual cause is a stale install -- re-run `pip install -e .`.\n"
            f"To regenerate it instead, build the COCO subset:\n"
            f"    python -m foxrepro.data.build_coco_subset build "
            f"--val2014-dir <dir> --annotations-dir <dir> --pope-dir <dir>"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        entries = payload.get("image_ids") or payload.get("images")
        if entries is None:
            raise ValueError(f"{path} has neither 'image_ids' nor 'images'")
    else:
        entries = payload
    if not entries:
        raise ValueError(f"{path} contained no images")

    image_ids: List[int] = []
    for e in entries:
        if isinstance(e, int):
            image_ids.append(e)
        elif isinstance(e, str):
            image_ids.append(image_id_from_filename(e))
        elif isinstance(e, dict) and "id" in e:
            image_ids.append(int(e["id"]))
        elif isinstance(e, dict) and "image_id" in e:
            image_ids.append(int(e["image_id"]))
        else:
            raise ValueError(f"cannot interpret CHAIR image entry {e!r}")

    if len(set(image_ids)) != len(image_ids):
        raise ValueError(f"{path} contains duplicate image ids")

    prompt = render_prompt(prompt_template_id)
    samples: List[Sample] = []
    missing: List[str] = []
    for image_id in sorted(image_ids):
        name = image_filename_for_id(image_id)
        image_path = Path(images_dir) / name
        if require_images and not image_path.is_file():
            missing.append(name)
        samples.append(
            Sample(
                sample_id=f"chair/{split}/{image_id:012d}",
                image_id=image_id,
                image_path=image_path,
                benchmark="chair",
                split=split,
                question=None,
                label=None,
                extra={"caption_prompt": prompt, "source_file": path.name},
            )
        )
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} CHAIR images missing from {images_dir} "
            f"(first few: {missing[:5]}). Extract the COCO subset tar first."
        )
    return samples


#: The manifest `fetch_mme.py` writes beside the extracted images. One JSONL
#: line per item: ``{question_id, image, question, answer, category}``, the
#: parquet's own fields with the image replaced by its filename on disk.
MME_MANIFEST_NAME = "mme_test.jsonl"


def load_mme(mme_dir: Path, split: str = "test",
             require_images: bool = True) -> List[Sample]:
    """Load MME from the manifest `fetch_mme.py` extracted.

    MME is binary yes/no like POPE, which is why its harness is this short:
    the same prompt shape, the same single generated token, and the same
    yes/no parser downstream. What it does NOT share is the metric -- see
    ``bench.evaluate.evaluate_mme`` -- because MME scores per IMAGE as well
    as per question.

    ``category`` is carried on every sample rather than becoming a split.
    The MME score is a SUM over fourteen subtasks, so running them as
    separate jobs would make the total depend on which jobs finished.

    Args:
        mme_dir: directory holding ``mme_test.jsonl`` and the images.
        split: only ``"test"`` exists; accepted for signature symmetry.
        require_images: raise if a referenced image is missing locally.
    """
    mme_dir = Path(mme_dir)
    path = mme_dir / MME_MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Run `python3 -m foxrepro.data.fetch_mme` "
            f"first; it downloads the parquet shards and extracts the images."
        )
    samples: List[Sample] = []
    missing: List[str] = []
    seen: set = set()
    with path.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for field in ("question_id", "item_id", "image", "question",
                          "answer", "category"):
                if field not in rec:
                    raise ValueError(
                        f"{path} line {idx + 1} missing {field!r}: {rec!r}")
            label = str(rec["answer"]).strip().lower()
            if label not in ("yes", "no"):
                raise ValueError(
                    f"{path} line {idx + 1} has non-binary answer "
                    f"{rec['answer']!r}; MME is yes/no throughout")
            img = mme_dir / str(rec["image"])
            if not img.is_file():
                missing.append(str(rec["image"]))
                if require_images:
                    continue
            # `item_id`, not `question_id`: the latter is the IMAGE path and
            # every image carries two questions, so building the id from it
            # collides on every single item. r21's session died on the same
            # assumption one layer up, in the fetcher.
            sid = f"mme/{rec['item_id']}"
            if sid in seen:
                raise ValueError(f"{path}: duplicate sample_id {sid!r}")
            seen.add(sid)
            samples.append(Sample(
                sample_id=sid,
                # MME images are not COCO, so there is no natural numeric id.
                # A stable hash of the filename keeps `image_id` an int for
                # every consumer without pretending it is a COCO id.
                image_id=zlib.crc32(str(rec["image"]).encode()) & 0x7FFFFFFF,
                image_path=img,
                benchmark="mme",
                split=split,
                question=str(rec["question"]),
                label=label,
                extra={"category": str(rec["category"]),
                       # The image key MME ships, kept because acc_plus
                       # groups by it: the two rows sharing it are the pair
                       # that term is defined over.
                       "question_id": str(rec["question_id"]),
                       "image_file": str(rec["image"])},
            ))
    if missing and require_images:
        raise FileNotFoundError(
            f"{len(missing)} MME images missing from {mme_dir}, first: "
            f"{missing[:3]}. Re-run fetch_mme."
        )
    samples.sort(key=lambda s: s.sample_id)
    return samples


#: Manifests written by ``fetch_mmbench.py`` / ``fetch_mathvista.py``.
MMBENCH_MANIFEST_NAME = "mmbench_dev.jsonl"
MATHVISTA_MANIFEST_NAME = "mathvista_testmini_mc500.jsonl"

_OPTION_LETTERS = "ABCDEFGH"


def build_mmbench_question(question: str, options: Dict[str, str],
                           hint: Optional[str] = None) -> str:
    """``<hint>\\n<question>\\nA. ..\\nB. ..`` as LLaVA's MMBench script sends it.

    The hint line is omitted when there is none (the dataset stores the
    string ``"nan"``; the fetcher turns that into ``None``).
    """
    lines = []
    if hint:
        lines.append(hint)
    lines.append(question)
    for letter in sorted(options):
        lines.append(f"{letter}. {options[letter]}")
    return "\n".join(lines)


def mathvista_subset_pids(pool_pids: Sequence[str], n: int, seed: int) -> List[str]:
    """The fixed random subset of the multiple-choice pool.

    Pids are sorted numerically first, so the draw depends on the SET of
    pids and not on the order the Hub happens to serve them in, then drawn
    with ``random.Random(seed).sample``. Returned sorted numerically.
    """
    import random

    pool = sorted({str(p) for p in pool_pids}, key=int)
    if n > len(pool):
        raise ValueError(f"cannot draw {n} from a pool of {len(pool)}")
    return sorted(random.Random(seed).sample(pool, n), key=int)


def _read_manifest(path: Path, hint: str) -> List[Dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing. {hint}")
    out: List[Dict[str, object]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_mmbench(mmbench_dir: Path, split: str = "dev",
                 require_images: bool = True) -> List[Sample]:
    """Load MMBench dev from the manifest ``fetch_mmbench.py`` wrote.

    ``sample_id`` is ``mmbench/<index>``: the dataset's ``index`` is unique
    per question (4377 of 4377, checked on the real split), unlike MME's
    ``question_id``, which names an image. The question carries its hint and
    its options; ``label`` is the gold letter. Everything the scorer needs
    (category, l2-category, the option set) rides on the sample so the run
    record can persist it.
    """
    from foxrepro.storage import MMBENCH_DEV_ROWS

    mmbench_dir = Path(mmbench_dir)
    path = mmbench_dir / MMBENCH_MANIFEST_NAME
    recs = _read_manifest(
        path, "Run `python3 -m foxrepro.data.fetch_mmbench` first.")
    if len(recs) != MMBENCH_DEV_ROWS:
        raise ValueError(
            f"{path} has {len(recs)} rows, expected {MMBENCH_DEV_ROWS}")
    samples: List[Sample] = []
    missing: List[str] = []
    seen: set = set()
    for n, rec in enumerate(recs, start=1):
        for field in ("index", "image", "question", "hint", "options",
                      "answer", "category", "l2_category"):
            if field not in rec:
                raise ValueError(f"{path} row {n} missing {field!r}")
        options = dict(rec["options"])  # type: ignore[arg-type]
        answer = str(rec["answer"])
        if answer not in options:
            raise ValueError(
                f"{path} row {n}: answer {answer!r} is not among options "
                f"{sorted(options)}")
        sid = f"mmbench/{rec['index']}"
        if sid in seen:
            raise ValueError(f"{path}: duplicate sample_id {sid!r}")
        seen.add(sid)
        img = mmbench_dir / str(rec["image"])
        if not img.is_file():
            missing.append(str(rec["image"]))
            if require_images:
                continue
        samples.append(Sample(
            sample_id=sid,
            image_id=zlib.crc32(str(rec["image"]).encode()) & 0x7FFFFFFF,
            image_path=img, benchmark="mmbench", split=split,
            question=build_mmbench_question(
                str(rec["question"]), options,
                str(rec["hint"]) if rec["hint"] else None),
            label=answer,
            extra={"category": str(rec["category"]),
                   "l2_category": str(rec["l2_category"]),
                   "options": options, "image_file": str(rec["image"])},
        ))
    if missing and require_images:
        raise FileNotFoundError(
            f"{len(missing)} MMBench images missing from {mmbench_dir}, "
            f"first: {missing[:3]}. Re-run fetch_mmbench.")
    samples.sort(key=lambda s: int(s.sample_id.split("/", 1)[1]))
    return samples


def load_mathvista(mathvista_dir: Path, split: str = "testmini_mc500",
                   require_images: bool = True) -> List[Sample]:
    """Load the MathVista multiple-choice subset ``fetch_mathvista.py`` wrote.

    The prompt is the dataset's own ``query`` field. ``label`` is the gold
    option letter -- ``"B"``, or ``"B|C"`` for the one question whose gold
    choice text appears twice, where either letter is the right answer.
    """
    from foxrepro.storage import (
        MATHVISTA_MC_POOL, MATHVISTA_SUBSET_N, MATHVISTA_SUBSET_SEED)

    mathvista_dir = Path(mathvista_dir)
    path = mathvista_dir / MATHVISTA_MANIFEST_NAME
    recs = _read_manifest(
        path, "Run `python3 -m foxrepro.data.fetch_mathvista` first.")
    if len(recs) != MATHVISTA_SUBSET_N:
        raise ValueError(
            f"{path} has {len(recs)} rows, expected {MATHVISTA_SUBSET_N}")
    samples: List[Sample] = []
    missing: List[str] = []
    seen: set = set()
    for n, rec in enumerate(recs, start=1):
        for field in ("pid", "image", "query", "options", "answer_letters",
                      "subset_seed", "pool_size"):
            if field not in rec:
                raise ValueError(f"{path} row {n} missing {field!r}")
        if (rec["subset_seed"] != MATHVISTA_SUBSET_SEED
                or rec["pool_size"] != MATHVISTA_MC_POOL):
            raise ValueError(
                f"{path} row {n} was drawn with seed {rec['subset_seed']} "
                f"from a pool of {rec['pool_size']}; this code expects seed "
                f"{MATHVISTA_SUBSET_SEED} and {MATHVISTA_MC_POOL}. A different "
                f"subset is a different benchmark.")
        options = dict(rec["options"])  # type: ignore[arg-type]
        letters = list(rec["answer_letters"])  # type: ignore[arg-type]
        if not letters or any(x not in options for x in letters):
            raise ValueError(
                f"{path} row {n}: answer letters {letters} not in options "
                f"{sorted(options)}")
        sid = f"mathvista/{rec['pid']}"
        if sid in seen:
            raise ValueError(f"{path}: duplicate sample_id {sid!r}")
        seen.add(sid)
        img = mathvista_dir / str(rec["image"])
        if not img.is_file():
            missing.append(str(rec["image"]))
            if require_images:
                continue
        samples.append(Sample(
            sample_id=sid,
            image_id=zlib.crc32(str(rec["image"]).encode()) & 0x7FFFFFFF,
            image_path=img, benchmark="mathvista", split=split,
            question=str(rec["query"]), label="|".join(letters),
            extra={"options": options, "answer_letters": letters,
                   "pid": str(rec["pid"]), "image_file": str(rec["image"])},
        ))
    if missing and require_images:
        raise FileNotFoundError(
            f"{len(missing)} MathVista images missing from {mathvista_dir}, "
            f"first: {missing[:3]}. Re-run fetch_mathvista.")
    samples.sort(key=lambda s: int(s.sample_id.split("/", 1)[1]))
    return samples


def load_benchmark(
    benchmark: str,
    split: str,
    pope_dir: Path,
    images_dir: Path,
    chair_image_list: Path,
    prompt_template_id: str,
    require_images: bool = True,
    mme_dir: Optional[Path] = None,
    mmbench_dir: Optional[Path] = None,
    mathvista_dir: Optional[Path] = None,
) -> List[Sample]:
    """Dispatch to :func:`load_pope` or :func:`load_chair`.

    Args:
        benchmark: ``"pope"`` or ``"chair"``.
        split: split name; for CHAIR this is just a provenance label.
        pope_dir: POPE annotation directory.
        images_dir: extracted val2014 images.
        chair_image_list: frozen CHAIR image-id list.
        prompt_template_id: must belong to ``benchmark``.
        require_images: raise on missing images.
    """
    if template_benchmark(prompt_template_id) != benchmark:
        raise ValueError(
            f"template {prompt_template_id!r} is for "
            f"{template_benchmark(prompt_template_id)!r}, not {benchmark!r}"
        )
    if benchmark == "pope":
        return load_pope(pope_dir, images_dir, split, require_images=require_images)
    if benchmark == "chair":
        return load_chair(
            chair_image_list, images_dir, split=split,
            prompt_template_id=prompt_template_id, require_images=require_images,
        )
    if benchmark == "mme":
        if mme_dir is None:
            raise ValueError("mme_dir is required for benchmark='mme'")
        return load_mme(mme_dir, split=split, require_images=require_images)
    if benchmark == "mmbench":
        if mmbench_dir is None:
            raise ValueError("mmbench_dir is required for benchmark='mmbench'")
        return load_mmbench(mmbench_dir, split=split,
                            require_images=require_images)
    if benchmark == "mathvista":
        if mathvista_dir is None:
            raise ValueError(
                "mathvista_dir is required for benchmark='mathvista'")
        return load_mathvista(mathvista_dir, split=split,
                              require_images=require_images)
    raise ValueError(f"unknown benchmark {benchmark!r}")


def describe_split(samples: Sequence[Sample]) -> Dict[str, object]:
    """Summary counts for a loaded split (printed by the setup readiness check)."""
    n_yes = sum(1 for s in samples if s.label == "yes")
    return {
        "n_samples": len(samples),
        "n_images": len({s.image_id for s in samples}),
        "n_label_yes": n_yes,
        "n_label_no": sum(1 for s in samples if s.label == "no"),
        "first_sample_id": samples[0].sample_id if samples else None,
        "last_sample_id": samples[-1].sample_id if samples else None,
    }


def iter_samples(samples: Iterable[Sample], skip: Optional[set] = None) -> Iterator[Sample]:
    """Yield samples whose ``sample_id`` is not in ``skip`` (resume helper)."""
    skip = skip or set()
    for s in samples:
        if s.sample_id not in skip:
            yield s
