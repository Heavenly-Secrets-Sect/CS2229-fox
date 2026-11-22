"""Substitute for the paper's GPT-4V assisted evaluation (Figure 6, Table 5).

Protocol (Appendix C.3 item 4): a vision model sees one image and two
assistants' descriptions of it, and scores each 1-10 for Accuracy
(hallucination-focused) and Detailedness, using the prompt of Table 5,
``max_tokens=512``, ``temperature=0.2``, on 50 COCO val2014 images, comparing
the original backbone output ("Sampling") with the Fox output.

What differs here, all declared in ``docs/DEVIATIONS.md``: the judge is a
current OpenAI model (GPT-4V is deprecated); the 50 images follow our own
fixed rule (the paper's subset is not named); each image is judged in BOTH
position orders, and the order-averaged result is reported beside the
paper-style single order.

Nothing here may print, log or store the API key. ``redact`` is applied to
every error string before it leaves this module.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from foxrepro.bench.stats import bootstrap_ci

__all__ = [
    "PROMPT_TEMPLATE", "SAMPLING_RUN", "FOX_RUN", "DEFAULT_MAX_CALLS",
    "build_prompt", "parse_reply", "select_images", "load_captions",
    "plan_calls", "estimate_cost", "run_judge", "summarise", "render_markdown",
    "load_api_key", "redact", "OpenAIJudgeClient", "BudgetExceeded",
    "FatalApiError", "JudgeReply",
]

DEFAULT_MAX_CALLS = 250
MAX_TOKENS = 512
TEMPERATURE = 0.2
N_IMAGES = 50

#: Table 5, verbatim (line wraps of the PDF joined with single spaces; the
#: paragraph structure kept). ``{R1}`` / ``{R2}`` stand for the two responses.
PROMPT_TEMPLATE = (
    "You are required to score the performance of two AI assistants in "
    "describing a given image. You should pay extra attention to the "
    "hallucination, which refers to the part of descriptions that are "
    "inconsistent with the image content, such as claiming the existence of "
    "something not present in the image or describing incorrectly in terms of "
    "the counts, positions, or colors of objects in the image. Please rate the "
    "responses of the assistants on a scale of 1 to 10, where a higher score "
    "indicates better performance, according to the following criteria:\n"
    "1. Accuracy: Evaluate whether the response is accurate and faithful to "
    "the actual image content. Focus on identifying any hallucinations "
    "including non-existent objects, incorrect attributes (colors, sizes, "
    "materials), wrong quantities, false spatial relationships, or activities "
    "that are not happening. Responses with fewer hallucinations and higher "
    "fidelity to the image should receive higher scores.\n"
    "2. Detailedness: Assess whether the response provides rich and "
    "informative details about the image. Consider the completeness of the "
    "description, coverage of important visual elements, and the depth of "
    "observations. Note that hallucinated content does NOT count as valid "
    "details – only accurate information contributes to this score.\n"
    "Please output the scores for each criterion, containing only two values "
    "indicating the scores for Assistant 1 and 2, respectively. The two scores "
    "are separated by a space. Following the scores, please provide an "
    "explanation of your evaluation, avoiding any potential bias and ensuring "
    "that the order in which the responses were presented does not affect your "
    "judgment.\n"
    "[Assistant 1]\n{R1}\n[End of Assistant 1]\n"
    "[Assistant 2]\n{R2}\n[End of Assistant 2]\n"
    "Output format:\n"
    "Accuracy: <score 1> <score 2>\n"
    "Reason: <your explanation>\n"
    "Detailedness: <score 1> <score 2>\n"
    "Reason: <your explanation>"
)

#: The archived CHAIR runs used: seed 42, split val2014_500, sampling decoding,
#: the r2 prompt (docs/DEVIATIONS.md 3.5), the baseline arm and the default
#: ``fox`` arm (mode as_implemented). Same variant hash 5c7c06df on both.
SAMPLING_RUN = ("m-sampling__b-chair__s-val2014_500__seed-42__dt-fp16__"
                "v-sampling-5c7c06df.jsonl")
FOX_RUN = ("m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__"
           "v-sampling-5c7c06df.jsonl")

#: (USD per 1M input tokens, per 1M output tokens). Assumed list prices, used
#: only for the pre-run estimate; override on the CLI. Unknown model -> None.
PRICES: Dict[str, Tuple[float, float]] = {
    "gpt-4.1": (2.0, 8.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4.1-mini": (0.4, 1.6),
}


class BudgetExceeded(RuntimeError):
    pass


class FatalApiError(RuntimeError):
    """Auth / permission / quota failure: stop, do not retry."""


# ---------------------------------------------------------------- secrets

def load_api_key(env_paths: Sequence[Path] = ()) -> Optional[str]:
    """OPENAI_API_KEY from the environment, else from a ``.env`` file.

    Searches the given files, then ``.env`` in the current directory and each
    parent, then each parent of this module. Never prints. None when absent.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key.strip()
    cands = list(env_paths)
    cwd = Path.cwd().resolve()
    cands += [d / ".env" for d in [cwd, *cwd.parents]]
    here = Path(__file__).resolve()
    cands += [d / ".env" for d in here.parents]
    for p in cands:
        try:
            if not p.is_file():
                continue
            for line in p.read_text().splitlines():
                m = re.match(r"\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*(.*)$", line)
                if m:
                    v = m.group(1).strip().strip("'\"")
                    if v:
                        return v
        except OSError:
            continue
    return None


def redact(text: Any, key: Optional[str] = None) -> str:
    """Remove the key (and anything shaped like one) from a string."""
    s = str(text)
    if key:
        s = s.replace(key, "<redacted>")
    return re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "<redacted>", s)


# ----------------------------------------------------------------- prompt

def build_prompt(resp1: str, resp2: str) -> str:
    return PROMPT_TEMPLATE.replace("{R1}", resp1).replace("{R2}", resp2)


_NUM = r"(\d+(?:\.\d+)?)"


def _score_pair(text: str, label: str) -> Optional[Tuple[float, float]]:
    m = re.search(
        rf"(?im)^[\s*#>_-]*{label}\s*\**\s*:\s*\**\s*{_NUM}\s*[ ,/]\s*{_NUM}",
        text)
    if not m:
        return None
    a, b = float(m.group(1)), float(m.group(2))
    if not (1 <= a <= 10 and 1 <= b <= 10):
        return None
    return a, b


def parse_reply(text: Optional[str]) -> Optional[Dict[str, Tuple[float, float]]]:
    """Scores by rule: both ``Accuracy: s1 s2`` and ``Detailedness: s1 s2``
    lines, each value in [1, 10]. Anything else is unparseable (None)."""
    if not text:
        return None
    acc = _score_pair(text, "Accuracy")
    det = _score_pair(text, "Detailedness")
    if acc is None or det is None:
        return None
    return {"accuracy": acc, "detailedness": det}


# ------------------------------------------------------------------- data

def select_images(image_ids: Sequence[int], n: int = N_IMAGES) -> List[int]:
    """Our fixed rule: the first ``n`` of the CHAIR set by ascending image_id."""
    ids = sorted(set(int(i) for i in image_ids))
    if len(ids) < n:
        raise ValueError(f"only {len(ids)} images available, need {n}")
    return ids[:n]


def _read_run(path: Path, method: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if (r.get("method") != method or r.get("benchmark") != "chair"
                    or r.get("seed") != 42 or r.get("decoding") != "sampling"):
                raise ValueError(f"{path.name}: unexpected record "
                                 f"{r.get('method')}/{r.get('benchmark')}/"
                                 f"{r.get('seed')}/{r.get('decoding')}")
            out[int(r["image_id"])] = r["output"]
    return out


def load_captions(runs_dir: Path) -> Tuple[Dict[int, str], Dict[int, str]]:
    """(sampling captions, fox captions) keyed by image_id. Read-only."""
    runs_dir = Path(runs_dir)
    return (_read_run(runs_dir / SAMPLING_RUN, "sampling"),
            _read_run(runs_dir / FOX_RUN, "fox"))


@dataclass(frozen=True)
class Call:
    image_id: int
    order: str  # "SF": assistant 1 = Sampling; "FS": assistant 1 = Fox
    prompt: str


def plan_calls(ids: Sequence[int], sampling: Dict[int, str],
               fox: Dict[int, str]) -> List[Call]:
    calls = []
    for i in ids:
        calls.append(Call(i, "SF", build_prompt(sampling[i], fox[i])))
        calls.append(Call(i, "FS", build_prompt(fox[i], sampling[i])))
    return calls


def estimate_cost(calls: Sequence[Call], model: str,
                  price: Optional[Tuple[float, float]] = None,
                  image_tokens: int = 765, out_tokens: int = 300) -> Dict[str, Any]:
    """Rough token and dollar estimate. 4 chars/token; 765 image tokens is
    a 640x480 image at high detail. Output is assumed, not measured."""
    tin = sum(len(c.prompt) / 4 + image_tokens for c in calls)
    tout = out_tokens * len(calls)
    if price is None:
        price = PRICES.get(re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model))
    usd = None if price is None else (tin * price[0] + tout * price[1]) / 1e6
    return {"calls": len(calls), "input_tokens": round(tin),
            "output_tokens": tout, "price_per_M": price, "usd": usd}


# ----------------------------------------------------------------- client

@dataclass
class JudgeReply:
    text: Optional[str]
    finish_reason: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class OpenAIJudgeClient:
    """Thin adapter over ``openai``. Reasoning-family ids (gpt-5*, o*) reject
    ``temperature`` and ``max_tokens``; they get ``max_completion_tokens`` and
    the default temperature, and ``params`` records what was really sent."""

    def __init__(self, api_key: str, model: str, detail: str = "auto"):
        from openai import OpenAI
        self._key = api_key
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.detail = detail

    def list_models(self) -> List[str]:
        return sorted(m.id for m in self._client.models.list())

    def complete(self, image_path: Path, prompt: str) -> JudgeReply:
        import openai
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        content = [
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{b64}", "detail": self.detail}},
            {"type": "text", "text": prompt},
        ]
        reasoning = re.match(r"(gpt-5|o\d)", self.model) is not None
        params: Dict[str, Any] = {"detail": self.detail}
        kw: Dict[str, Any] = {}
        if reasoning:
            kw["max_completion_tokens"] = MAX_TOKENS
            params["max_completion_tokens"] = MAX_TOKENS
            params["temperature"] = "model default (not settable)"
        else:
            kw["max_tokens"] = MAX_TOKENS
            kw["temperature"] = TEMPERATURE
            params.update(max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
        try:
            r = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}], **kw)
        except (openai.AuthenticationError, openai.PermissionDeniedError,
                openai.RateLimitError) as e:
            raise FatalApiError(
                f"{type(e).__name__}: {redact(e, self._key)[:300]}") from None
        except openai.OpenAIError as e:
            return JudgeReply(None, params=params,
                              error=f"{type(e).__name__}: "
                                    f"{redact(e, self._key)[:300]}")
        ch = r.choices[0]
        u = r.usage
        usage = ({"prompt_tokens": u.prompt_tokens,
                  "completion_tokens": u.completion_tokens} if u else {})
        return JudgeReply(ch.message.content, ch.finish_reason, usage, params)


# -------------------------------------------------------------------- run

def run_judge(client: Any, calls: Sequence[Call], image_paths: Dict[int, Path],
              max_calls: int = DEFAULT_MAX_CALLS,
              on_record: Optional[Callable[[Dict[str, Any]], None]] = None,
              ) -> Dict[str, Any]:
    """Issue every call once (no retries). Raises before the first call if the
    plan exceeds ``max_calls``. A fatal API error stops the run and is returned
    in ``fatal``; the records so far are kept."""
    if len(calls) > max_calls:
        raise BudgetExceeded(f"{len(calls)} calls planned, budget is {max_calls}")
    records: List[Dict[str, Any]] = []
    fatal = None
    for c in calls:
        try:
            rep = client.complete(image_paths[c.image_id], c.prompt)
        except FatalApiError as e:
            fatal = str(e)
            break
        scores = parse_reply(rep.text)
        rec = {"image_id": c.image_id, "order": c.order, "reply": rep.text,
               "finish_reason": rep.finish_reason, "usage": rep.usage,
               "params": rep.params, "error": rep.error,
               "parsed": scores is not None,
               "scores": None if scores is None else
               {k: list(v) for k, v in scores.items()}}
        records.append(rec)
        if on_record:
            on_record(rec)
    return {"records": records, "fatal": fatal, "n_calls": len(records)}


# ---------------------------------------------------------------- summary

def _sampling_fox(rec: Optional[Dict[str, Any]], crit: str
                  ) -> Optional[Tuple[float, float]]:
    """(sampling score, fox score) from one record, or None if unparsed."""
    if rec is None or not rec.get("parsed"):
        return None
    s1, s2 = rec["scores"][crit]
    return (s1, s2) if rec["order"] == "SF" else (s2, s1)


def _stat(vals: List[Tuple[float, float]], n_boot: int, confidence: float,
          seed: int) -> Dict[str, Any]:
    if not vals:
        raise ValueError("refusing to report an average over zero parsed items")
    d = [f - s for s, f in vals]
    lo, hi = bootstrap_ci(d, lambda x: sum(x) / len(x), n_boot=n_boot,
                          confidence=confidence, seed=seed)
    n = len(vals)
    return {"n": n, "sampling_mean": sum(s for s, _ in vals) / n,
            "fox_mean": sum(f for _, f in vals) / n,
            "diff_mean": sum(d) / n, "diff_ci": [lo, hi],
            "fox_wins": sum(1 for x in d if x > 0),
            "ties": sum(1 for x in d if x == 0),
            "sampling_wins": sum(1 for x in d if x < 0)}


def summarise(records: Sequence[Dict[str, Any]], n_boot: int = 10000,
              confidence: float = 0.90, seed: int = 20260101) -> Dict[str, Any]:
    """Per criterion: paper-style (order SF), swapped (FS) and order-averaged.

    A reading with zero parsed items is None, never an average over nothing;
    if every reading is empty this raises. Order averaging uses only images
    parsed in BOTH orders; ``n`` says how many that is.
    """
    by: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for r in records:
        by.setdefault(r["image_id"], {})[r["order"]] = r
    out: Dict[str, Any] = {
        "n_calls": len(records),
        "n_unparsed": sum(1 for r in records if not r.get("parsed")),
        "n_images": len(by), "confidence": confidence, "n_boot": n_boot,
        "unparsed": [{"image_id": r["image_id"], "order": r["order"],
                      "error": r.get("error"),
                      "finish_reason": r.get("finish_reason")}
                     for r in records if not r.get("parsed")],
    }
    for crit in ("accuracy", "detailedness"):
        sf, fs, both = [], [], []
        for i in sorted(by):
            a = _sampling_fox(by[i].get("SF"), crit)
            b = _sampling_fox(by[i].get("FS"), crit)
            if a:
                sf.append(a)
            if b:
                fs.append(b)
            if a and b:
                both.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
        out[crit] = {
            k: (_stat(v, n_boot, confidence, seed) if v else None)
            for k, v in (("paper_style_SF", sf), ("swapped_FS", fs),
                         ("order_averaged", both))}
    if all(v is None for c in ("accuracy", "detailedness")
           for v in out[c].values()):
        raise ValueError("no parsed replies at all; refusing to summarise")
    return out


def render_markdown(model: str, meta: Dict[str, Any], s: Dict[str, Any]) -> str:
    ci = int(round(s["confidence"] * 100))
    lines = [
        f"# GPT-judge substitute for Figure 6 / Table 5 ({model})", "",
        f"Judge `{model}` (substitute for deprecated GPT-4V). "
        f"{s['n_images']} COCO val2014 images (first by sorted image_id of the "
        f"CHAIR set), LLaVA-1.5 only, seed 42, sampling decoding. "
        f"Calls {s['n_calls']}, unparsed {s['n_unparsed']}.", "",
        f"Scores are 1-10. Difference is Fox minus Sampling; interval is a "
        f"paired bootstrap {ci}% CI over images.", "",
        "| criterion | reading | n | Sampling | Fox | Fox - Sampling | "
        f"{ci}% CI | Fox/tie/Sampling |",
        "|---|---|---|---|---|---|---|---|",
    ]
    names = {"paper_style_SF": "paper-style (Assistant 1 = Sampling)",
             "swapped_FS": "swapped (Assistant 1 = Fox)",
             "order_averaged": "order-averaged (our addition)"}
    for crit in ("accuracy", "detailedness"):
        for k, label in names.items():
            v = s[crit][k]
            if v is None:
                lines.append(f"| {crit} | {label} | 0 | - | - | - | - | - |")
                continue
            lines.append(
                f"| {crit} | {label} | {v['n']} | {v['sampling_mean']:.2f} | "
                f"{v['fox_mean']:.2f} | {v['diff_mean']:+.2f} | "
                f"[{v['diff_ci'][0]:+.2f}, {v['diff_ci'][1]:+.2f}] | "
                f"{v['fox_wins']}/{v['ties']}/{v['sampling_wins']} |")
    if s["unparsed"]:
        lines += ["", "Unparsed replies (kept in the JSON, excluded from the "
                      "averages above): " + ", ".join(
                          f"{u['image_id']}/{u['order']}" for u in s["unparsed"])]
    lines += ["", f"Params actually sent: `{json.dumps(meta.get('params'))}`. "
                  "Raw replies are in the JSON beside this file."]
    return "\n".join(lines) + "\n"
