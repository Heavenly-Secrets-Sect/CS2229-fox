"""OPERA (Huang et al., CVPR 2024), ported onto this project's stack.

Why this is a separate module rather than a flag on ``fox_generate``: OPERA
is not a decoding *rule* layered over one sequence, it is a different search.
Five beams, five speculative forward passes per step to score candidates that
do not exist yet, and -- on long outputs -- a rollback that rewinds the
search. None of that fits a single-sequence loop.

Why a port is possible at all. The released implementation is 559 lines
inside a vendored, patched ``transformers-4.29.2``: it had to rewrite
``GenerationMixin.beam_search`` because that is where the loop lived. This
project already replaced the decode loop with its own, so there is nothing to
patch -- the search is ours to write, and three things that were hard in 4.29
are now library calls:

* the last layer's attention comes from our registered attention function
  (``FoxState.opera_capture``), not from ``output_attentions=True``, which on
  this stack returns an empty tuple under SDPA and would retain all 32 layers;
* ``DynamicCache.crop`` undoes a speculative forward in place;
* the hypothesis bookkeeping is 60 auditable lines (:class:`BeamHypotheses`,
  :class:`BeamScorer`) rather than a dependency. ``transformers``' own
  ``BeamSearchScorer`` is deprecated and removed in v4.62.0, and this project
  has to keep running on one environment across Kaggle sessions, so it is
  reimplemented and asserted equivalent to the library's while that still
  exists (``tests/test_opera.py``).

What the method actually is, which is NOT "beam search plus an attention
bonus on the log-probs". Per step, per beam::

    l            = logits[:, -1, :]                  # RAW, no log_softmax yet
    (l5, t5)     = topk(l, num_attn_candidates)      # top-5 by raw logit
    l5          += penalty_weights * attn_image(t5)  # bonus, in LOGIT space
    vec          = full(REJECT_FLOOR); vec[t5] = l5  # the rest of the vocab goes
    scores       = log_softmax(vec) + beam_scores    # normalised over the 5

Three consequences, each of which changes which sequences are reachable:

1. the bonus is added to logits, where top-5 LLaMA gaps are O(1)-O(5), not to
   log-probs;
2. ``log_softmax`` normalises over each beam's own five candidates, so every
   beam gets a different additive constant -- and beams are then compared in
   one global ``topk``, so a per-beam constant is not harmless;
3. the candidate set is hard-restricted to five per beam. Stock beam search
   picks the best ``2 * num_beams`` over ``num_beams * vocab_size``, and can
   take ten tokens from one beam; OPERA cannot see past each beam's fifth.

``REJECT_FLOOR`` is ``-999.0`` and not ``-inf``, which is load-bearing: after
``log_softmax`` the 31995 discarded tokens sit at a finite ~-1009, so when
``process`` banks an EOS candidate and walks further down the ``2 * num_beams``
list, a discarded token can be promoted into a live beam. With ``-inf`` it
could not be.

``attn_image`` is the candidate's own attention over the image span: the last
layer, max-pooled over heads (not averaged -- that is SID), row-normalised,
summed over ``img_start..img_end``. It cannot come from the main forward,
because the row needed is the row of a token that has not been emitted yet.
Hence five extra one-token forwards per step.

The retrospection-rollback half is implemented too, and is switched on only
when a generation is long enough to need it (``max_new_tokens`` above
``PENALTY_SWITCH_LEN`` = 10). On POPE (at most 10 tokens) it is unreachable:
``cur_response_lens <= 10`` selects the attention penalty instead of the
rollback score, and the ``> threshold`` repeat count over a history with one
column per step (15 required, 10 available) excludes the trigger, as does the
``rollback_loc < 10`` guard in the body. So at <= 10 tokens this module takes
the original code path unchanged and is OPERA, not an approximation of it;
above 10 it runs the rollback machinery below. See ``docs/DEVIATIONS.md``
Sec 3.13 for the audit and for where the port deliberately departs from the
release (chunked re-prefill, fp16 saturation kept as released).

The rollback, in one paragraph. Per step OPERA keeps, per beam, the last
layer's max-over-heads attention rows of the RESPONSE tokens only (a
``[R, R]`` block plus each row's full-row sum for normalisation). With the
five candidates' rows appended it forms ``attn_local`` (``scale_factor`` x the
row-normalised block) and scores each column ``j`` by
``1e-7 * prod(attn_local[j:, j])`` in fp16. The column with the largest score
is that candidate's ``rollback_loc``; the mode over candidates and then over
beams is the step's. Once more than ``threshold`` of the history columns of
EVERY beam equal it (and it is at least ``min_rollback_loc``), the search
rewinds to ``rollback_loc + 1`` response tokens, re-decodes from there with
the five (beam, token) choices made at that position banned, and each
position may be rewound to at most ``num_attn_candidates`` times. Past
``penalty_switch_len`` response tokens the same score replaces the image
attention as the candidate penalty.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from foxrepro.fox.attention import FOX, locate_image_span

__all__ = [
    "MIN_ROLLBACK_LOC",
    "OPERA_DEFAULTS",
    "PENALTY_SWITCH_LEN",
    "REJECT_FLOOR",
    "WINDOW_SIZE",
    "BeamHypotheses",
    "BeamScorer",
    "RetroState",
    "RetrospectionHistory",
    "Rollback",
    "column_product_scores",
    "crop_cache_to",
    "opera_generate",
    "rollback_is_reachable",
]

#: ``pope_eval.py``'s argparse block, which is also what ``chair_eval.py``
#: and the README's command line pass.
OPERA_DEFAULTS: Dict[str, Any] = {
    "num_beams": 5,
    "num_attn_candidates": 5,
    "penalty_weights": 1.0,
    "scale_factor": 50.0,   # rollback path only -- dead whenever rollback is
    "threshold": 15,        # unreachable, which on POPE it always is
    "window_size": 512,     # ``generate(window_size=...)`` default in the fork
}

#: ``penalty_scores = -attn_scores if cur_response_lens <= 10 else
#: rollback_scores``: the literal 10 in ``opera_beam_search``.
PENALTY_SWITCH_LEN = 10

#: ``if rollback_loc < 10: assert False``: the literal 10 in the same body.
MIN_ROLLBACK_LOC = 10

#: ``window_size`` in ``opera_beam_search``'s signature: the length of the
#: history, of ``max_rollback_time`` and of ``reject_token_pos_gather``.
WINDOW_SIZE = 512

#: ``next_token_logits.fill_(-999.)``. Finite on purpose; see the module
#: docstring.
REJECT_FLOOR = -999.0


def rollback_is_reachable(max_new_tokens: int, threshold: int = 15) -> bool:
    """Can OPERA's rollback fire at all for this generation length?

    Two independent gates, both from ``opera_beam_search``:

    * the rollback score is only *used* when ``cur_response_lens > 10``, and
      that counter is the number of response tokens emitted so far;
    * the trigger needs the modal column to repeat more than ``threshold``
      times in a history with exactly one column per step.

    So anything up to ``max(10, threshold)`` response tokens cannot roll back.
    Exposed as a function rather than a comment because it is the thing that
    makes this module's POPE output *equal* to OPERA rather than close to it,
    and a future change to ``max_new_tokens`` must trip over it.

    This answers "can it FIRE", not "does the method differ from the
    rollback-free one": from the 11th response token the candidate penalty is
    already the rollback score rather than the image attention
    (``PENALTY_SWITCH_LEN``), so a run of 11-15 tokens needs the rollback
    machinery's bookkeeping even though no rewind can happen in it.
    """
    return max_new_tokens > max(10, threshold)


@dataclass
class RetroState:
    """What ``opera_beam_search`` snapshots at the top of every step.

    Taken BEFORE the step's forward, as the release does, so ``ids``,
    ``scorer``, ``beam_scores``, ``beam_next_tokens`` and ``beam_idx`` are the
    values the step started from; ``beam_next_tokens`` / ``beam_idx`` are what
    the PREVIOUS step's ``process`` chose (``None`` at step 0). ``attn_local``
    and ``attn_sums`` are the response-block of the carried ``attn_previous``
    after this step's main row was appended (rows still in the step-start
    beam order, i.e. before the end-of-step reorder), held on the host.
    """

    ids: torch.Tensor
    scorer: Any
    beam_scores: torch.Tensor
    beam_next_tokens: Optional[torch.Tensor]
    beam_idx: Optional[torch.Tensor]
    attn_local: Optional[torch.Tensor] = None
    attn_sums: Optional[torch.Tensor] = None


@dataclass
class Rollback:
    """A decided rewind: where to, the state to restore, what to ban."""

    pos: int                      # ``rollback_pos`` after the two shifts
    restored: RetroState          # ``history_states[-2]``
    reject_flat: torch.Tensor     # [1, k] flat ``beam * vocab + token`` bans


class RetrospectionHistory:
    """The release's rollback bookkeeping, separated from any model.

    Everything ``opera_beam_search`` keeps in loose locals -- ``history_states``,
    ``history_rollback_locs``, ``max_rollback_time``, ``reject_token_pos_gather``
    -- and the decision block that reads them, lifted out so it can be driven
    by scripted inputs and compared step for step with a transcription of the
    release (``tests/test_opera_rollback.py``). Nothing here runs a forward.

    Two behaviours of the release are kept on purpose even though they look
    wrong, because "faithful" means the same decisions, not the tidier ones:

    * ``reject_token_pos_gather`` is cleared from the TAIL of the list while
      popped states (``[-1]``, ``[-2]``...), not at the popped positions;
    * the restored beam scorer is the stored object itself, not a copy, so
      the replayed step's ``process`` mutates the snapshot a later rollback to
      the same position will restore.

    Where the release's ``try/except`` would swallow an ``IndexError`` after
    mutating state, this class returns ``None`` BEFORE mutating (positions at
    or past the window), which is only reachable on a generation as long as
    the window.
    """

    def __init__(self, *, num_attn_candidates: int, threshold: int,
                 min_loc: int = MIN_ROLLBACK_LOC,
                 window_size: int = WINDOW_SIZE) -> None:
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        if min_loc < 1:
            raise ValueError(
                "min_loc must be >= 1: a rollback to position 0 would rewind "
                "into the prompt, which the release's own guard of 10 makes "
                "unreachable"
            )
        self.num_attn_candidates = int(num_attn_candidates)
        self.threshold = int(threshold)
        self.min_loc = int(min_loc)
        self.window_size = int(window_size)
        self.states: List[RetroState] = []
        self.locs: List[torch.Tensor] = []
        self.max_rollback_time = torch.zeros(self.window_size)
        self.reject_gather: List[Any] = [[] for _ in range(self.window_size)]

    def observe(self, rollback_locs: torch.Tensor) -> torch.Tensor:
        """Record this step's per-beam modal column; return the global mode.

        ``rollback_locs`` is ``[beams, candidates]``. The release takes the
        mode over candidates (kept per beam, one history column per step),
        then the mode over beams of that.
        """
        per_beam = rollback_locs.mode().values
        self.locs.append(per_beam[:, None])
        return per_beam.mode().values

    def push(self, state: RetroState) -> None:
        """Append a step's snapshot, evicting the oldest past the window."""
        if len(self.states) >= self.window_size:
            self.states.pop(0)
        self.states.append(state)

    def try_rollback(self, rollback_loc: torch.Tensor,
                     cur_response_lens: int,
                     vocab_size: int) -> Optional[Rollback]:
        """The ``try:`` block of ``opera_beam_search``, minus the model calls.

        Returns ``None`` where the release would ``assert False`` into its
        ``except:`` (no rollback this step), else the rewind to perform. On a
        rewind the history is already popped as the release leaves it, the
        position's ban list updated, and ``max_rollback_time`` incremented.
        """
        gathers = torch.cat(self.locs, -1)
        fire = all(
            int((g == rollback_loc).long().sum()) > self.threshold
            for g in gathers
        )
        if not fire:
            return None
        loc = int(rollback_loc)
        if loc < self.min_loc:
            return None
        pos = loc + 1
        if pos + 1 >= self.window_size:
            return None
        if self.max_rollback_time[pos] >= self.num_attn_candidates:
            pos -= 1
            if self.max_rollback_time[pos] >= self.num_attn_candidates:
                return None
            self.max_rollback_time[pos] += 1
        else:
            self.max_rollback_time[pos] += 1
        if cur_response_lens - pos > self.window_size + 1:
            pos = max(1, cur_response_lens - self.window_size - 1)

        for j in range(cur_response_lens - pos - 2):
            self.states.pop(-1)
            self.locs.pop(-1)
            self.reject_gather[-(j + 1)] = []

        if len(self.states) < 2:
            raise RuntimeError(
                f"rollback to position {pos} needs two states of history, "
                f"have {len(self.states)}"
            )
        restored = self.states[-2]
        replay = self.states[-1]
        if replay.beam_idx is None or replay.beam_next_tokens is None:
            raise RuntimeError(
                "the state a rollback replays from has no beam choice "
                "recorded; a rewind to position 0 cannot be replayed"
            )
        self.states.pop(-1)
        self.locs.pop(-1)
        self.reject_gather[pos + 1] = []

        n_beams = int(replay.beam_idx.numel())
        reject = (replay.beam_idx.view(1, n_beams) * vocab_size
                  + replay.beam_next_tokens.view(1, n_beams))
        earlier = self.reject_gather[pos]
        if len(earlier) > 0:
            reject = torch.cat([earlier, reject], -1)
        self.reject_gather[pos] = reject
        return Rollback(pos=pos, restored=restored, reject_flat=reject)


def column_product_scores(attn_local: torch.Tensor, *,
                          precision: str = "fp16") -> torch.Tensor:
    """``attn_local_scores`` of the release: one score per column per candidate.

    ``attn_local`` is ``[beams, candidates, R, R]``, already row-normalised
    and scaled by ``scale_factor``. Column ``j``'s score is
    ``1e-7 * prod(attn_local[..., j:, j])`` over the rows from the diagonal
    down. Returns ``[beams, candidates, R]``.

    ``precision="fp16"`` is the release, quirks included: the destination
    buffer is declared ``float16``, so the ``.to(float32)`` in the source is a
    no-op, scores below ~6e-8 round to zero and ties abound, and a column of
    steady factors above ~25 overflows to ``inf`` in four multiplications.
    ``precision="fp32"`` is the control this project planned for the CHAIR
    run: the same product accumulated as a sum of fp32 logs, so neither the
    buffer nor the product saturates. It is NOT the release.
    """
    if precision not in ("fp16", "fp32"):
        raise ValueError(f"precision must be 'fp16' or 'fp32', got {precision!r}")
    n = int(attn_local.shape[-1])
    if n == 0:
        raise ValueError("no response columns to score")
    lead = attn_local.shape[:2]
    if precision == "fp16":
        out = torch.zeros(*lead, n, dtype=torch.float16,
                          device=attn_local.device)
        for j in range(n):
            local_score = 1e-7 * attn_local[..., j:, j].prod(-1)
            out[..., j] = local_score.to(torch.float32)
        return out
    logs = torch.log(attn_local.float())
    out = torch.zeros(*lead, n, dtype=torch.float32, device=attn_local.device)
    log_scale = math.log(1e-7)
    for j in range(n):
        out[..., j] = torch.exp(log_scale + logs[..., j:, j].sum(-1))
    return out


class BeamHypotheses:
    """One batch item's finished hypotheses, ranked by length-penalised score.

    A transcription of ``transformers.generation.beam_search.BeamHypotheses``
    rather than an improvement on it: this project needs the library's
    behaviour, and the library's class is deprecated. Asserted equivalent in
    ``tests/test_opera.py`` for as long as the original is importable.
    """

    def __init__(self, num_beams: int, length_penalty: float,
                 early_stopping: bool) -> None:
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.beams: List[Any] = []
        self.worst_score = 1e9

    def __len__(self) -> int:
        return len(self.beams)

    def add(self, hyp: torch.Tensor, sum_logprobs: float,
            generated_len: int) -> None:
        """Length-penalise and insert, evicting the worst past ``num_beams``."""
        score = sum_logprobs / (generated_len ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                worst_i = min(enumerate(self.beams),
                              key=lambda x: x[1][0])[0]
                del self.beams[worst_i]
                self.worst_score = min(s for s, _ in self.beams)
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """Can no remaining candidate beat the worst kept hypothesis?

        ``early_stopping=False`` -- the released setting, since OPERA passes
        ``generation_config.early_stopping`` and never sets it -- compares the
        worst kept score against the best score still attainable at this
        length.
        """
        if len(self) < self.num_beams:
            return False
        if self.early_stopping:
            return True
        highest_attainable = best_sum_logprobs / (
            cur_len ** self.length_penalty)
        return bool(self.worst_score >= highest_attainable)


class BeamScorer:
    """``process`` / ``finalize`` for one batch item.

    Batch size is fixed at 1 because both released eval scripts run
    ``--batch_size 1`` (``pope_eval.py:68``, ``chair_eval.py:80``) and because
    OPERA's rollback trigger is global across the batch dimension -- batching
    it would not be the same method, so pretending to support it would be
    worse than refusing.
    """

    def __init__(self, num_beams: int, *, length_penalty: float = 1.0,
                 early_stopping: bool = False, max_length: Optional[int] = None):
        self.num_beams = num_beams
        self.max_length = max_length
        self.hyps = BeamHypotheses(num_beams, length_penalty, early_stopping)
        self.done = False

    @property
    def is_done(self) -> bool:
        return self.done

    def process(
        self,
        input_ids: torch.Tensor,
        next_scores: torch.Tensor,
        next_tokens: torch.Tensor,
        next_indices: torch.Tensor,
        *,
        pad_token_id: int,
        eos_token_id: List[int],
        decoder_prompt_len: int,
    ) -> Dict[str, torch.Tensor]:
        """Fill the next ``num_beams`` live beams, banking any EOS candidate.

        The ``rank >= num_beams`` skip is the library's and is why
        ``REJECT_FLOOR`` being finite matters: an EOS in the top
        ``num_beams`` is banked and the walk continues down the
        ``2 * num_beams`` list, so a discarded-vocabulary token sitting at a
        finite ~-1009 can be promoted into a live beam.
        """
        cur_len = int(input_ids.shape[-1])
        device = input_ids.device
        n = self.num_beams
        scores = torch.zeros(1, n, dtype=next_scores.dtype, device=device)
        tokens = torch.zeros(1, n, dtype=torch.long, device=device)
        idx = torch.zeros(1, n, dtype=torch.long, device=device)

        if self.done:
            scores[0, :] = 0.0
            tokens[0, :] = pad_token_id
            return {"next_beam_scores": scores.view(-1),
                    "next_beam_tokens": tokens.view(-1),
                    "next_beam_indices": idx.view(-1)}

        filled = 0
        for rank in range(int(next_tokens.shape[-1])):
            token = int(next_tokens[0, rank])
            score = next_scores[0, rank]
            beam = int(next_indices[0, rank])
            if token in eos_token_id:
                if rank >= n:
                    continue
                self.hyps.add(
                    input_ids[beam].clone(), float(score),
                    generated_len=cur_len - decoder_prompt_len)
            else:
                scores[0, filled] = score
                tokens[0, filled] = token
                idx[0, filled] = beam
                filled += 1
            if filled == n:
                break
        if filled < n:
            raise ValueError(
                f"only {filled} of {n} beams could be filled from "
                f"{int(next_tokens.shape[-1])} candidates; at most "
                f"{n} of them may be eos"
            )
        self.done = self.hyps.is_done(
            float(next_scores[0].max()), cur_len - decoder_prompt_len)
        return {"next_beam_scores": scores.view(-1),
                "next_beam_tokens": tokens.view(-1),
                "next_beam_indices": idx.view(-1)}

    def finalize(
        self,
        input_ids: torch.Tensor,
        final_beam_scores: torch.Tensor,
        *,
        pad_token_id: int,
        eos_token_id: List[int],
        decoder_prompt_len: int,
    ) -> torch.Tensor:
        """Bank the unfinished beams, then return the single best sequence."""
        if not self.done:
            cur_len = int(input_ids.shape[-1])
            for beam in range(self.num_beams):
                self.hyps.add(
                    input_ids[beam].clone(), float(final_beam_scores[beam]),
                    generated_len=cur_len - decoder_prompt_len)
        best = max(self.hyps.beams, key=lambda x: x[0])[1]
        return best


def crop_cache_to(cache: Any, length: int) -> None:
    """Drop everything past ``length`` tokens from a KV cache, on any version.

    ``Cache.crop(n)`` with a POSITIVE n means "keep n tokens" on transformers
    4.x and is refused outright on 5.x ("use a negative integer to remove that
    number of tokens"); both accept a negative n as "remove |n| tokens". r24's
    two OPERA 10-token jobs died on that, in 0.3 and 1.9 minutes, on a host
    image that had moved to 5.x since the adversarial run was made. The
    negative form is the one spelling both accept.
    """
    have = int(cache.get_seq_length())
    if have > length:
        cache.crop(-(have - length))


def _resolve_eos(model: Any) -> List[int]:
    cfg = getattr(model.config, "text_config", model.config)
    eos = getattr(cfg, "eos_token_id", None)
    if eos is None:
        raise ValueError("the checkpoint declares no eos_token_id")
    return [int(eos)] if isinstance(eos, int) else [int(e) for e in eos]


def _rebuild_cache(
    model: Any, cache: Any, ids: torch.Tensor, full_mask: torch.Tensor,
    n_prompt: int, chunk: int,
) -> Any:
    """Make ``cache`` hold exactly ``ids[:, :-1]``, and return it.

    The release re-prefills the restored hypotheses from scratch, vision
    tower included. That is a 5-row, ~600-token forward, which does not fit a
    T4 (the one-row prefill at step 0 exists for that reason). Every beam
    row's first ``n_prompt`` cache entries are copies of the one prefilled
    prompt row -- ``reorder_cache`` only permutes identical rows -- so the
    prompt prefix is kept by cropping (``crop_cache_to``, the negative
    spelling) and only the response tokens are re-run, ``chunk`` at a time.
    Mathematically the same cache; numerically not bit-identical to one
    full-length prefill (different kernels for different shapes), which is
    recorded in docs/DEVIATIONS.md.

    Cropping is not enough on its own, and the audit said so: cache rows are
    in the CURRENT beam order, so the restored hypotheses' response tokens
    have to be recomputed -- but their PROMPT part is shared by all rows.
    """
    crop_cache_to(cache, n_prompt)
    resp = ids[:, n_prompt:-1]
    done = n_prompt
    for s in range(0, int(resp.shape[1]), chunk):
        piece = resp[:, s:s + chunk]
        done += int(piece.shape[1])
        out = model(input_ids=piece, attention_mask=full_mask[:, :done],
                    past_key_values=cache, use_cache=True)
        cache = out.past_key_values
    return cache


@torch.inference_mode()
def opera_generate(
    model: Any,
    inputs: Dict[str, torch.Tensor],
    *,
    max_new_tokens: int = 10,
    num_beams: int = 5,
    num_attn_candidates: int = 5,
    penalty_weights: float = 1.0,
    length_penalty: float = 1.0,
    early_stopping: bool = False,
    collect_trace: bool = False,
    allow_unreachable_rollback: bool = False,
    restrict_candidates: bool = True,
    rollback: bool = True,
    scale_factor: float = OPERA_DEFAULTS["scale_factor"],
    threshold: int = OPERA_DEFAULTS["threshold"],
    window_size: int = WINDOW_SIZE,
    penalty_switch_len: int = PENALTY_SWITCH_LEN,
    min_rollback_loc: int = MIN_ROLLBACK_LOC,
    score_precision: str = "fp16",
    replay_chunk: int = 64,
) -> Dict[str, Any]:
    """Beam search with OPERA's over-trust penalty and retrospection-rollback.

    ``inference_mode`` is not an optimisation here, it is what makes the
    10-token arm run. Without it every one of the six forwards per step --
    the beam step plus five speculative candidates -- builds an autograd
    graph that stays alive until the function returns, so the footprint
    grows with ``max_new_tokens`` rather than staying flat. At
    ``max_new_tokens=1`` that is one step and it fits, which is why the
    three POPE arms and the whole decomposition in VALIDATION.md V5 ran;
    at 10 it OOM'd on the first sample, with 14.54 of GPU 1's 14.56 GiB in
    use. ``fox_generate`` has carried this decorator from the start, so
    every other method in the project was already running under it. Grad
    mode does not change arithmetic, so the 1-token records written before
    this are unaffected -- it changes what is retained, not what is
    computed.

    Args:
        max_new_tokens: at most ``penalty_switch_len`` (10, POPE's length)
            runs the original rollback-free path, byte for byte. Above it the
            release's candidate penalty is the rollback score rather than the
            image attention, so ``rollback=True`` (the default) runs the full
            method; ``rollback=False`` there is refused unless
            ``allow_unreachable_rollback`` -- it is not OPERA, and a number
            labelled OPERA from it would be wrong in a way nobody downstream
            could see.
        num_attn_candidates: the hard per-beam candidate cap. Not a tuning
            knob -- see the module docstring.
        restrict_candidates: ``False`` keeps the penalty but drops the
            discard-the-vocabulary step, so ``log_softmax`` runs over the
            whole vocabulary as stock beam search does. A validation arm, not
            a variant: OPERA is three things at once -- a beam search, a hard
            five-candidate restriction, and an attention bonus -- and with
            ``penalty_weights=0`` this separates all three. Incompatible
            with the rollback path, which has no reference behaviour for it.
        scale_factor, threshold, window_size, penalty_switch_len,
        min_rollback_loc: the release's ``--scale_factor`` / ``--threshold``,
            ``window_size`` and the two literal 10s in its body. Defaults are
            the release's; they exist as arguments so the rollback can be
            exercised on a stub whose responses are far shorter than 16.
        score_precision: ``"fp16"`` is the release (including its saturation);
            ``"fp32"`` is the planned control. See ``column_product_scores``.
        replay_chunk: response tokens per forward when the cache is rebuilt
            after a rewind; memory only, not a result.

    Returns ``{"token_ids", "n_prompt_tokens", "n_gen_tokens", "image_span",
    "trace"?}``, the same shape ``fox_generate`` returns, so the benchmark
    harness does not have to know which search ran. Runs that used the
    rollback path add ``rollback`` (True), ``n_rollbacks``, ``rollbacks`` (one
    ``{at_step, to_pos, loc}`` per rewind), ``n_nonfinite_steps`` (steps where
    some candidate score was NaN/inf after the penalty -- the fp16 saturation
    failure) and ``score_precision``.
    """
    if num_beams < 2:
        # opera_greedy_search in the release is `assert False, "OPERA does not
        # support beam=1 in the current version."` -- the greedy variant was
        # never written, so there is nothing to reproduce at one beam.
        raise ValueError(
            f"OPERA needs num_beams >= 2 ({num_beams} given). The release's "
            "greedy path is an unconditional assert, so a one-beam OPERA has "
            "no reference behaviour"
        )
    if num_attn_candidates < 1:
        raise ValueError(
            f"Num of candidates must be larger than 0, but it is currently "
            f"{num_attn_candidates}."
        )
    if score_precision not in ("fp16", "fp32"):
        raise ValueError(
            f"score_precision must be 'fp16' or 'fp32', got {score_precision!r}")
    if replay_chunk < 1:
        raise ValueError("replay_chunk must be >= 1")
    needs_rollback_regime = max_new_tokens > penalty_switch_len
    if needs_rollback_regime and not rollback and not allow_unreachable_rollback:
        raise ValueError(
            f"max_new_tokens={max_new_tokens} is long enough for OPERA's "
            "candidate penalty to switch to the rollback score (and for the "
            "rollback to fire), and rollback=False does not implement it. The "
            "result would be OPERA with half its method missing, under "
            "OPERA's name. Use rollback=True, <= 10 tokens (POPE), or pass "
            "allow_unreachable_rollback=True and label the arm accordingly"
        )
    use_retro = bool(rollback and needs_rollback_regime)
    if use_retro and not restrict_candidates:
        raise ValueError(
            "restrict_candidates=False has no reference behaviour past the "
            "point where the rollback is in play; it is a <= 10-token "
            "validation arm"
        )

    device = next(model.parameters()).device
    image_token_id = model.config.image_token_index
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)
    pixel_values = inputs["pixel_values"].to(device)
    img_start, img_end = locate_image_span(input_ids, image_token_id)
    n_prompt = int(input_ids.shape[1])

    eos_ids = _resolve_eos(model)
    pad_id = getattr(model.config, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_ids[0]

    n_layers = int(getattr(
        getattr(model.config, "text_config", model.config),
        "num_hidden_layers",
    ))

    ids = input_ids.repeat(num_beams, 1)
    mask = attn_mask.repeat(num_beams, 1)

    scorer = BeamScorer(
        num_beams, length_penalty=length_penalty,
        early_stopping=early_stopping,
        max_length=n_prompt + max_new_tokens,
    )
    # Only beam 0 is live at step 0: every beam row holds the same prompt, so
    # without this the first step would return the same token five times.
    # Created on the parameter device for now; the first step moves it onto
    # the logits' device, which under a shard is not the same one.
    beam_scores = torch.full((num_beams,), -1e9, dtype=torch.float32,
                             device=device)
    beam_scores[0] = 0.0

    cache = None
    trace: List[Dict[str, float]] = []
    top_scores = next_tokens = next_indices = None

    # Rollback-only state. Absent (None / 0) on the <= 10-token path, which
    # therefore executes exactly the statements it always did.
    retro: Optional[RetrospectionHistory] = None
    if use_retro:
        retro = RetrospectionHistory(
            num_attn_candidates=num_attn_candidates, threshold=threshold,
            min_loc=min_rollback_loc, window_size=window_size)
    attn_L: Optional[torch.Tensor] = None   # carried attn_previous, response block
    attn_S: Optional[torch.Tensor] = None   # ... and each row's full-row sum
    beam_tokens = beam_idx = None            # the previous step's choice
    rollbacks: List[Dict[str, int]] = []
    n_nonfinite = 0

    with FOX.configured(
        enabled=False, mode="off",
        img_start=img_start, img_end=img_end,
        # Off by default and switched on only around the speculative
        # forwards: the main forward's last row is never used, so capturing
        # it would be a wasted QK^T and would leave FOX.opera_attn holding a
        # row that belongs to no candidate. The rollback path is the one
        # exception: there the main row of the token just emitted is a row of
        # attn_previous, so it is captured too (step >= 1).
        opera_capture=False, opera_layer=n_layers - 1, opera_attn=None,
    ):
        while True:
            step = int(ids.shape[1]) - n_prompt
            if step >= max_new_tokens:
                break
            snap: Optional[RetroState] = None
            if use_retro:
                snap = RetroState(
                    ids=ids.clone(), scorer=copy.deepcopy(scorer),
                    beam_scores=beam_scores.clone(),
                    beam_next_tokens=(None if beam_tokens is None
                                      else beam_tokens.clone()),
                    beam_idx=None if beam_idx is None else beam_idx.clone(),
                )
            main_row = None
            if step == 0:
                # ONE row, then fan the cache out to num_beams. At step 0
                # every beam holds the same prompt, so this is numerically
                # identical. A 5-row 625-token prefill OOMs on a T4:
                # device_map="auto" packs the first GPU to ~14.0 of its
                # 14.56 GiB with weights, and five rows of activations do
                # not fit in what is left. One row also costs a fifth of
                # the prefill compute, which on POPE is most of the job.
                #
                # It was ALSO, at the time, standing in for the real cause:
                # this function was not under `inference_mode`, so every
                # forward's graph pinned the KV buffers it had read and the
                # footprint grew with each step. The one-row prefill bought
                # enough headroom for 1-token POPE to finish and hid that
                # for two rounds. Both are kept -- the saving is real -- but
                # the decorator is the load-bearing one.
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                            pixel_values=pixel_values, use_cache=True)
                cache = out.past_key_values
                # index_select on the batch dim with an all-zeros index:
                # five copies of the one prefilled row. reorder_cache moves
                # the index to each layer's own device, so this is correct
                # on a sharded model.
                cache.reorder_cache(
                    torch.zeros(num_beams, dtype=torch.long))
                z = out.logits[:, -1, :].repeat(num_beams, 1)
            else:
                if use_retro:
                    FOX.opera_attn = None
                    FOX.opera_capture = True
                try:
                    out = model(input_ids=ids[:, -1:], attention_mask=mask,
                                past_key_values=cache, use_cache=True)
                finally:
                    FOX.opera_capture = False
                cache = out.past_key_values
                if use_retro:
                    main_row = FOX.opera_attn
                    if main_row is None:
                        raise RuntimeError(
                            f"no attention captured at layer {n_layers - 1}. "
                            "The model must be loaded with "
                            "attn_implementation='fox' for OPERA's rollback "
                            "to exist at all"
                        )
                # RAW logits, in the model dtype: the penalty is added here,
                # not after log_softmax, and the dtype is the released one.
                z = out.logits[:, -1, :]
            cache_len = int(cache.get_seq_length())
            cand_scores, cand_tokens = torch.topk(
                z, num_attn_candidates, dim=-1, largest=True, sorted=True)

            if use_retro:
                # attn_previous: the old rows reordered by LAST step's beam
                # choice, then this step's main row (the row of the token
                # emitted at step - 1). Only the response block is kept, plus
                # each row's full-row sum, which is all the normalisation
                # needs. Step 0 has no response rows yet.
                t = step
                if t == 0:
                    snap.attn_local = z.new_zeros(num_beams, 0, 0).cpu()
                    snap.attn_sums = z.new_zeros(num_beams, 0).cpu()
                    attn_L, attn_S = None, None
                else:
                    row = main_row.to(z.device)
                    new_L = row.new_zeros(num_beams, t, t)
                    new_S = row.new_zeros(num_beams, t)
                    if t > 1:
                        sel = beam_idx.to(new_L.device)
                        new_L[:, :t - 1, :t - 1] = attn_L[sel]
                        new_S[:, :t - 1] = attn_S[sel]
                    new_L[:, t - 1, :] = row[:, n_prompt:]
                    new_S[:, t - 1] = row.sum(dim=-1)
                    attn_L, attn_S = new_L, new_S
                    snap.attn_local = attn_L.cpu()
                    snap.attn_sums = attn_S.cpu()

            attn_cols: List[torch.Tensor] = []
            raw_rows: List[torch.Tensor] = []
            spec_mask = torch.cat(
                [mask, mask.new_ones(num_beams, 1)], dim=1)
            for c in range(num_attn_candidates):
                FOX.opera_attn = None
                FOX.opera_capture = True
                try:
                    model(
                        input_ids=cand_tokens[:, c:c + 1],
                        attention_mask=spec_mask,
                        past_key_values=cache, use_cache=True,
                    )
                finally:
                    FOX.opera_capture = False
                row = FOX.opera_attn
                if row is None:
                    raise RuntimeError(
                        f"no attention captured at layer {n_layers - 1}. The "
                        "model must be loaded with attn_implementation='fox' "
                        "for OPERA's penalty to exist at all"
                    )
                # Undo the speculative step. On transformers 4.29 the KV cache
                # was an immutable tuple and torch.cat built new tensors, so
                # the release could simply discard it; DynamicCache.update
                # mutates in place, so without this crop the five candidate
                # passes would corrupt the search's own cache.
                crop_cache_to(cache, cache_len)
                # Back to the logits' device before anything combines them.
                # Under device_map="auto" the capture happens inside the LAST
                # decoder layer, which is on the last shard (cuda:1), while
                # accelerate's hook moves the model's OUTPUT back to the
                # device the inputs came from (cuda:0). The stash bypasses
                # that hook, so the two meet on different devices --
                # "Expected all tensors to be on the same device, but found
                # at least two devices, cuda:0 and cuda:1". One GPU cannot
                # reproduce it, so this is written to be correct by
                # construction rather than guarded by a test.
                row = row.to(z.device)
                if use_retro:
                    raw_rows.append(row)
                # Row-normalise AFTER the max over heads, as the release does
                # (attn_last / attn_last.sum(-1)); a max over heads does not
                # sum to 1, so the order is observable.
                row = row / row.sum(dim=-1, keepdim=True)
                attn_cols.append(row[:, img_start:img_end + 1].sum(-1))

            # The `cur_response_lens <= 10` regime: penalty_scores is
            # -attn_scores, and `scores -= w * penalty` therefore ADDS it.
            # The other regime is the rollback score, used from the 11th
            # response token on the rollback path.
            attn_scores = torch.stack(attn_cols, dim=1)

            rollback_scores = None
            rollback_loc = None
            use_rollback_score = False
            if use_retro:
                # attn_last for each candidate: the carried block (rows
                # 0..step-1) plus that candidate's own row, every row
                # divided by its full-row sum, then x scale_factor.
                r1 = step + 1
                cand_local = torch.stack(
                    [r[:, n_prompt:] for r in raw_rows], dim=1)
                cand_sum = torch.stack(
                    [r.sum(dim=-1) for r in raw_rows], dim=1)
                big = cand_local.new_zeros(
                    num_beams, num_attn_candidates, r1, r1)
                sums = cand_local.new_ones(
                    num_beams, num_attn_candidates, r1)
                if step > 0:
                    big[:, :, :step, :step] = attn_L[:, None]
                    sums[:, :, :step] = attn_S[:, None, :]
                big[:, :, step, :] = cand_local
                sums[:, :, step] = cand_sum
                if score_precision == "fp32":
                    big, sums = big.float(), sums.float()
                attn_local = scale_factor * (big / sums.unsqueeze(-1))
                local_scores = column_product_scores(
                    attn_local, precision=score_precision)
                rollback_scores, rollback_locs = local_scores.max(-1)
                rollback_loc = retro.observe(rollback_locs)
                del big, sums, attn_local, local_scores
                use_rollback_score = r1 > penalty_switch_len
            # Recorded BEFORE the penalty lands, because the question these
            # two numbers answer is whether the penalty can change anything:
            # it is added in LOGIT space, so what matters is its spread across
            # the five candidates against the spread of their raw logits. If
            # the first is orders of magnitude smaller than the second, then
            # OPERA's over-trust penalty cannot reorder the candidates and the
            # arm is beam search with a truncated candidate set. That is a
            # measurement, not an assumption -- the trace carries both.
            logit_spread = float(
                (cand_scores[:, 0] - cand_scores[:, -1]).float().mean())
            penalty_spread = float(
                (attn_scores.max(dim=1).values
                 - attn_scores.min(dim=1).values).float().mean()
            ) * penalty_weights
            if use_rollback_score:
                # `candidate_token_scores -= w * rollback_scores`. The score
                # is fp16 whatever the model dtype (the release's buffer), so
                # the product is formed there and only then widened.
                cand_scores = cand_scores - (
                    penalty_weights * rollback_scores).to(cand_scores.dtype)
            else:
                cand_scores = cand_scores + (
                    penalty_weights * attn_scores).to(cand_scores.dtype)
            if use_retro and not bool(torch.isfinite(cand_scores).all()):
                n_nonfinite += 1
            reordered = bool(
                (cand_scores.argmax(dim=-1) != 0).any())

            rb: Optional[Rollback] = None
            if use_retro:
                retro.push(snap)
                rb = retro.try_rollback(
                    rollback_loc, step + 1, int(z.shape[-1]))

            if rb is None:
                # Discard the rest of the vocabulary and renormalise over the
                # survivors. REJECT_FLOOR is finite on purpose.
                if restrict_candidates:
                    vec = torch.full_like(z, REJECT_FLOOR)
                    vec.scatter_(-1, cand_tokens, cand_scores)
                else:
                    vec = z.clone()
                    vec.scatter_(-1, cand_tokens, cand_scores)
                scores = F.log_softmax(vec, dim=-1)
            else:
                # Rewind. Restore the state the release restores, rebuild the
                # KV cache for the restored hypotheses, and re-decode that
                # position from the FULL vocabulary with the choices already
                # made there banned (no candidate cap, no penalty: the
                # release skips both on this step).
                st = rb.restored
                ids = st.ids
                scorer = st.scorer
                beam_scores = st.beam_scores
                attn_L = st.attn_local.to(z.device)
                attn_S = st.attn_sums.to(z.device)
                full_mask = torch.cat(
                    [attn_mask.repeat(num_beams, 1),
                     attn_mask.new_ones(
                         num_beams, int(ids.shape[1]) - n_prompt)],
                    dim=1)
                cache = _rebuild_cache(
                    model, cache, ids, full_mask, n_prompt, replay_chunk)
                out = model(input_ids=ids[:, -1:], attention_mask=full_mask,
                            past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                mask = full_mask
                zz = out.logits[:, -1, :].clone()
                zz -= 999. + zz.min(-1, keepdim=True).values
                vocab_rb = int(zz.shape[-1])
                flat_logits = zz.view(1, num_beams * vocab_rb)
                flat_logits.scatter_(
                    -1, rb.reject_flat.to(flat_logits.device), -999.)
                z = flat_logits.view(num_beams, vocab_rb)
                scores = F.log_softmax(z, dim=-1)
                rollbacks.append({"at_step": step, "to_pos": rb.pos,
                                  "loc": int(rollback_loc)})
            scores = scores + beam_scores.to(scores.device)[:, None].expand_as(
                scores)

            vocab = int(scores.shape[-1])
            flat = scores.reshape(1, num_beams * vocab)
            top_scores, top_flat = torch.topk(
                flat, 2 * num_beams, dim=1, largest=True, sorted=True)
            next_indices = torch.div(top_flat, vocab, rounding_mode="floor")
            next_tokens = top_flat % vocab

            beam_out = scorer.process(
                ids, top_scores, next_tokens, next_indices,
                pad_token_id=pad_id, eos_token_id=eos_ids,
                decoder_prompt_len=n_prompt,
            )

            beam_scores = beam_out["next_beam_scores"]
            beam_tokens = beam_out["next_beam_tokens"]
            beam_idx = beam_out["next_beam_indices"]

            idx = beam_idx.to(ids.device)
            ids = torch.cat(
                [ids[idx, :], beam_tokens.to(ids.device).unsqueeze(-1)],
                dim=-1)
            mask = torch.cat(
                [mask[idx], mask.new_ones(num_beams, 1)], dim=1)
            # reorder_cache moves the index onto each layer's own device.
            cache.reorder_cache(beam_idx.cpu())

            if collect_trace:
                entry = {
                    "step": step,
                    "attn_image_mean": float(attn_scores.mean()),
                    "penalty_spread": penalty_spread,
                    "logit_spread": logit_spread,
                    # True when the penalty moved the top candidate off the
                    # raw-logit argmax on at least one beam -- i.e. the step
                    # where OPERA's mechanism actually did something.
                    "penalty_changed_top": reordered,
                    "best_beam_score": float(beam_scores.max()),
                }
                if use_retro:
                    entry["rollback_loc"] = int(rollback_loc)
                    entry["rolled_back_to"] = (
                        rollbacks[-1]["to_pos"] if rb is not None else -1)
                trace.append(entry)

            if scorer.is_done:
                break

    best = scorer.finalize(
        ids, beam_scores, pad_token_id=pad_id, eos_token_id=eos_ids,
        decoder_prompt_len=n_prompt,
    )
    out_ids = [int(t) for t in best[n_prompt:].tolist()]

    result: Dict[str, Any] = {
        "token_ids": out_ids,
        "n_prompt_tokens": n_prompt,
        "n_gen_tokens": len(out_ids),
        "image_span": (img_start, img_end),
        "n_forwards_per_step": 1 + num_attn_candidates,
        "restrict_candidates": restrict_candidates,
        "penalty_weights": penalty_weights,
    }
    if use_retro:
        result.update({
            "rollback": True,
            "n_rollbacks": len(rollbacks),
            "rollbacks": rollbacks,
            "n_nonfinite_steps": n_nonfinite,
            "score_precision": score_precision,
        })
    if collect_trace:
        result["trace"] = trace
    return result
