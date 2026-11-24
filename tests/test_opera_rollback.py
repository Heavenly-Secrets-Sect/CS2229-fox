"""OPERA's retrospection-rollback, against a transcription of the release.

What can and cannot be checked without a GPU:

* the DECISION logic -- when to rewind, to where, which position's bans grow,
  which history is dropped -- is pure bookkeeping, so it is compared with a
  literal transcription of ``opera_beam_search`` (lines 3441-3550 of the
  vendored ``transformers-4.29.2/.../generation/utils.py``) on scripted inputs,
  step for step, over hundreds of random scenarios;
* the whole LOOP -- incremental attention bookkeeping, cache rebuild after a
  rewind, the regime switch at 10 response tokens -- is compared with a
  transcription of the same function that recomputes everything from scratch
  (no KV cache, the attention square taken straight from a fresh forward), on
  a real tiny LLaVA. Same tokens, same beam choices at every iteration, same
  rewinds, same ``attn_local`` at every call.

What stays unchecked here: real-model numerics (fp16 saturation, T4 kernels),
and the OPERA numbers themselves. The scripted score function stands in for
``column_product_scores`` in the loop comparison because on a tiny random
model that function's argmax is column 0 at every step, so the trigger -- and
with it the whole rewind path -- never fires.
"""

from __future__ import annotations

import copy
import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("torch is not installed") from None

try:
    import transformers  # noqa: F401
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("transformers is not installed") from None

import torch.nn.functional as F  # noqa: E402

from foxrepro.fox import attention as A  # noqa: E402
from foxrepro.fox import opera as O  # noqa: E402
from foxrepro.fox.opera import (  # noqa: E402
    BeamScorer, RetroState, RetrospectionHistory, column_product_scores,
    opera_generate,
)
from tests.test_opera import prompt  # noqa: E402
from tests.test_tiny_llava import tiny_llava  # noqa: E402


# --------------------------------------------------------------------------- #
# the release's decision block, transcribed
# --------------------------------------------------------------------------- #
class _ReleaseDecision:
    """``opera_beam_search`` lines 3441-3550, state handling only.

    The three model calls inside the ``try`` (re-prefill and the logits
    recompute) are not here; the loop-level test below covers them.
    """

    def __init__(self, window_size, num_attn_candidates, threshold, min_loc):
        self.window_size = window_size
        self.num_attn_candidates = num_attn_candidates
        self.threshold = threshold
        self.min_loc = min_loc
        self.history_states = []
        self.history_rollback_locs = None
        self.max_rollback_time = torch.zeros(window_size)
        self.history_length = window_size
        self.reject_token_pos_gather = [[] for _ in range(window_size)]

    def step(self, rollback_locs, current_state, cur_response_lens, vocab):
        rollback_loc = rollback_locs.mode().values.data
        rollback_loc = rollback_loc.mode().values.data
        if self.history_rollback_locs is None:
            self.history_rollback_locs = [rollback_locs.mode().values.data[:, None]]
        else:
            self.history_rollback_locs.append(
                rollback_locs.mode().values.data[:, None])
        rollback_loc_gathers = torch.cat(self.history_rollback_locs, -1)

        if len(self.history_states) >= self.history_length:
            self.history_states.pop(0)
        self.history_states.append(current_state)

        history_states = self.history_states
        history_rollback_locs = self.history_rollback_locs
        max_rollback_time = self.max_rollback_time
        reject_token_pos_gather = self.reject_token_pos_gather
        num_attn_candidates = self.num_attn_candidates
        history_length = self.history_length
        try:
            if all((g == rollback_loc).long().sum() > int(self.threshold)
                   for _, g in enumerate(rollback_loc_gathers)):
                if rollback_loc < self.min_loc:
                    assert False
                rollback_pos = rollback_loc + 1
                if max_rollback_time[rollback_pos] >= num_attn_candidates:
                    rollback_pos = rollback_pos - 1
                    if max_rollback_time[rollback_pos] >= num_attn_candidates:
                        assert False
                    else:
                        max_rollback_time[rollback_pos] += 1
                else:
                    max_rollback_time[rollback_pos] += 1
                if cur_response_lens - rollback_pos > history_length + 1:
                    rollback_pos = max(1, cur_response_lens - history_length - 1)

                for j in range(cur_response_lens - rollback_pos - 2):
                    history_states.pop(-1)
                    history_rollback_locs.pop(-1)
                    reject_token_pos_gather[-(j + 1)] = []

                restored = history_states[-2]
                beam_next_tokens = history_states[-1]["beam_next_tokens"]
                beam_idx = history_states[-1]["beam_idx"]

                history_states.pop(-1)
                history_rollback_locs.pop(-1)
                reject_token_pos_gather[rollback_pos + 1] = []

                beam_idx = beam_idx.view(1, -1)
                beam_next_tokens = beam_next_tokens.view(1, -1)
                reject_token_pos = beam_idx * vocab + beam_next_tokens
                if len(reject_token_pos_gather[rollback_pos]) > 0:
                    reject_token_pos = torch.cat(
                        [reject_token_pos_gather[rollback_pos],
                         reject_token_pos], -1)
                reject_token_pos_gather[rollback_pos] = reject_token_pos
                return int(rollback_pos), restored, reject_token_pos
            else:
                assert False
        except AssertionError:
            return None


class TestDecisionBlockMatchesRelease(unittest.TestCase):
    BEAMS, CANDS, VOCAB = 3, 3, 50

    def _scenario(self, seed: int):
        g = torch.Generator().manual_seed(seed)
        threshold = int(torch.randint(1, 4, (1,), generator=g))
        min_loc = int(torch.randint(1, 4, (1,), generator=g))
        window = int(torch.randint(12, 30, (1,), generator=g))
        n_iter = 70
        lo = min_loc - 1
        # A small value set, so a modal column exists and repeats; seeds that
        # never fire are filtered by the caller's count below.
        locs = [torch.randint(lo, lo + 3, (self.BEAMS, self.CANDS),
                              generator=g) for _ in range(n_iter)]
        toks = {t: torch.randint(3, self.VOCAB, (self.BEAMS,), generator=g)
                for t in range(n_iter + 5)}
        idxs = {t: torch.randint(0, self.BEAMS, (self.BEAMS,), generator=g)
                for t in range(n_iter + 5)}
        return threshold, min_loc, window, locs, toks, idxs

    def _drive(self, seed: int) -> int:
        threshold, min_loc, window, locs, toks, idxs = self._scenario(seed)
        ours = RetrospectionHistory(
            num_attn_candidates=self.CANDS, threshold=threshold,
            min_loc=min_loc, window_size=window)
        ref = _ReleaseDecision(window, self.CANDS, threshold, min_loc)
        step, fired = 0, 0
        for i, loc_tensor in enumerate(locs):
            # a step's snapshot: tagged with its step number through ``ids``
            def snap():
                return RetroState(
                    ids=torch.tensor([[step]]), scorer=None,
                    beam_scores=torch.zeros(1),
                    beam_next_tokens=toks[step - 1] if step else None,
                    beam_idx=idxs[step - 1] if step else None)

            cur = step + 1
            a_loc = ours.observe(loc_tensor)
            ours.push(snap())
            try:
                mine = ours.try_rollback(a_loc, cur, self.VOCAB)
                mine_raised = False
            except RuntimeError:
                mine, mine_raised = None, True
            try:
                theirs = ref.step(loc_tensor, {
                    "input_ids": torch.tensor([[step]]),
                    "beam_next_tokens": toks[step - 1] if step else None,
                    "beam_idx": idxs[step - 1] if step else None,
                }, cur, self.VOCAB)
                theirs_raised = False
            except (IndexError, AttributeError):
                theirs, theirs_raised = None, True
            self.assertEqual(mine_raised, theirs_raised,
                             f"seed {seed} iteration {i}")
            if mine_raised:
                # The release would have crashed (or, in its bare ``except:``,
                # swallowed it); this port raises. Not comparable afterwards.
                return fired
            with self.subTest(seed=seed, iteration=i):
                self.assertEqual(mine is None, theirs is None)
                if mine is not None:
                    fired += 1
                    self.assertEqual(mine.pos, theirs[0])
                    self.assertEqual(int(mine.restored.ids),
                                     int(theirs[1]["input_ids"]))
                    self.assertTrue(torch.equal(mine.reject_flat, theirs[2]))
                    step = mine.pos + 1
                else:
                    step += 1
                self.assertEqual(
                    [int(s.ids) for s in ours.states],
                    [int(s["input_ids"]) for s in ref.history_states])
                self.assertEqual(len(ours.locs), len(ref.history_rollback_locs))
                self.assertTrue(torch.equal(
                    ours.max_rollback_time, ref.max_rollback_time))
                for a, b in zip(ours.reject_gather, ref.reject_token_pos_gather):
                    self.assertEqual(type(a) is list, type(b) is list)
                    if not isinstance(a, list):
                        self.assertTrue(torch.equal(a, b))
        return fired

    def test_every_decision_agrees_over_random_scenarios(self) -> None:
        total = 0
        for seed in range(150):
            total += self._drive(seed)
        # Guard against a vacuous pass: the scenarios must actually rewind,
        # and often enough to exhaust a position's five-rewind allowance.
        self.assertGreater(total, 150, "scenarios barely rewound")

    def test_the_trigger_needs_every_beam_to_repeat_the_column(self) -> None:
        """Beam 0 always says column 3; beam 1 says it on even steps only.

        More than ``threshold`` (2) repeats are needed from EVERY beam, so the
        first firing is step 4 -- when beam 1 has said 3 three times -- and not
        step 2, when beam 0 alone has.
        """
        h = RetrospectionHistory(num_attn_candidates=2, threshold=2,
                                 min_loc=1, window_size=20)
        first = None
        for step in range(8):
            locs = torch.tensor([[3, 3], [3, 3] if step % 2 == 0 else [4, 4]])
            loc = h.observe(locs)
            h.push(RetroState(ids=torch.tensor([[step]]), scorer=None,
                              beam_scores=torch.zeros(1),
                              beam_next_tokens=torch.ones(2, dtype=torch.long),
                              beam_idx=torch.zeros(2, dtype=torch.long)))
            if h.try_rollback(loc, step + 1, 10) is not None:
                first = step
                break
        self.assertEqual(first, 4)

    def test_below_the_minimum_column_never_rewinds(self) -> None:
        h = RetrospectionHistory(num_attn_candidates=2, threshold=1,
                                 min_loc=5, window_size=20)
        for step in range(10):
            loc = h.observe(torch.full((2, 2), 4))
            h.push(RetroState(ids=torch.tensor([[step]]), scorer=None,
                              beam_scores=torch.zeros(1),
                              beam_next_tokens=torch.ones(2, dtype=torch.long),
                              beam_idx=torch.zeros(2, dtype=torch.long)))
            self.assertIsNone(h.try_rollback(loc, step + 1, 10))

    def test_a_position_can_be_rewound_to_only_so_many_times(self) -> None:
        """After ``num_attn_candidates`` rewinds the position shifts down one.

        Column 3 forever, ``threshold`` 6: the first firing is the seventh
        column, at step 6, so every state the rewind pops exists.
        """
        h = RetrospectionHistory(num_attn_candidates=2, threshold=6,
                                 min_loc=1, window_size=40)
        step, positions = 0, []
        for _ in range(40):
            loc = h.observe(torch.full((2, 2), 3))
            h.push(RetroState(
                ids=torch.tensor([[step]]), scorer=None,
                beam_scores=torch.zeros(1),
                beam_next_tokens=torch.full((2,), step % 5 + 1),
                beam_idx=torch.zeros(2, dtype=torch.long)))
            got = h.try_rollback(loc, step + 1, 10)
            if got is None:
                step += 1
            else:
                positions.append(got.pos)
                step = got.pos + 1
        self.assertEqual(positions[:4], [4, 4, 3, 3])

    def test_an_empty_history_raises_rather_than_passing(self) -> None:
        h = RetrospectionHistory(num_attn_candidates=2, threshold=1,
                                 min_loc=1, window_size=10)
        with self.assertRaises(RuntimeError):
            h.try_rollback(torch.tensor(3), 5, 10)

    def test_a_rewind_to_position_zero_is_refused_up_front(self) -> None:
        with self.assertRaises(ValueError):
            RetrospectionHistory(num_attn_candidates=2, threshold=1,
                                 min_loc=0, window_size=10)

    def test_the_restored_scorer_is_the_stored_object_not_a_copy(self) -> None:
        """The release aliases it, so a replayed ``process`` edits the snapshot."""
        h = RetrospectionHistory(num_attn_candidates=3, threshold=1,
                                 min_loc=1, window_size=10)
        marker = object()
        for step in range(4):
            loc = h.observe(torch.full((2, 2), 2))
            h.push(RetroState(
                ids=torch.tensor([[step]]),
                scorer=marker if step == 0 else None,
                beam_scores=torch.zeros(1),
                beam_next_tokens=torch.ones(2, dtype=torch.long),
                beam_idx=torch.zeros(2, dtype=torch.long)))
            got = h.try_rollback(loc, step + 1, 10)
            if got is not None:
                self.assertIs(got.restored.scorer, marker)
                return
        self.fail("no rewind happened")


# --------------------------------------------------------------------------- #
# the release's loop, transcribed with full recompute
# --------------------------------------------------------------------------- #
def scripted_scores(attn_local, *, precision="fp16"):
    """Column 1 carries the score, candidate-dependently, at every step."""
    b, c, r = attn_local.shape[0], attn_local.shape[1], attn_local.shape[-1]
    out = torch.zeros(b, c, r)
    col = min(1, r - 1)
    out[:, :, col] = 0.4 + 0.3 * torch.arange(c, dtype=out.dtype)[None, :]
    out[:, :, 0] += 0.01 * (r % 3)
    return out


def release_loop(model, inputs, *, max_new_tokens, scores_fn, num_beams=5,
                 num_cands=5, penalty_weights=1.0, scale_factor=50.0,
                 threshold=15, window_size=512, switch_len=10, min_loc=10):
    """``opera_beam_search``, with every forward recomputed from scratch.

    The release keeps ``attn_previous`` incrementally and rebuilds the KV cache
    on a rewind; both are equal to simply running the model on the full
    sequence, which is what is done here, so this shares no bookkeeping with
    ``opera_generate``. Returns the best token ids, the rewinds as
    ``(iteration, rollback_pos)`` and, per iteration, ``attn_local``.
    """
    px = inputs["pixel_values"]
    ids0 = inputs["input_ids"]
    n_prompt = int(ids0.shape[1])
    img_s, img_e = A.locate_image_span(ids0, model.config.image_token_index)
    eos = O._resolve_eos(model)
    pad = eos[0]

    # The image features are computed once and written into the embeddings of
    # every row, so a candidate token that happens to equal the image
    # placeholder id (the stub's vocabulary is 64 wide) cannot trip the
    # model's own placeholder/feature count check.
    with torch.no_grad():
        raw = model.model.get_image_features(pixel_values=px)
    if hasattr(raw, "pooler_output"):
        raw = raw.pooler_output
    feats = (torch.cat(list(raw), 0) if isinstance(raw, (list, tuple))
             else raw.reshape(-1, raw.shape[-1]))

    def fwd(ids):
        emb = model.get_input_embeddings()(ids).clone()
        emb[:, img_s:img_e + 1] = feats
        out = model(inputs_embeds=emb, attention_mask=torch.ones_like(ids),
                    output_attentions=True)
        return out.logits, out.attentions[-1]

    input_ids = ids0.repeat(num_beams, 1)
    beam_scorer = BeamScorer(num_beams, length_penalty=1.0,
                             early_stopping=False)
    beam_scores = torch.zeros(num_beams)
    beam_scores[1:] = -1e9
    history_states = []
    history_rollback_locs = None
    beam_next_tokens = None
    beam_idx = None
    max_rollback_time = torch.zeros(window_size)
    history_length = window_size
    reject_token_pos_gather = [[] for _ in range(window_size)]
    events, locals_seen, choices = [], [], []
    it = 0
    with torch.no_grad():
        while int(input_ids.shape[1]) - n_prompt < max_new_tokens:
            current_state = {
                "input_ids": input_ids.clone(),
                "beam_scorer": copy.deepcopy(beam_scorer),
                "beam_scores": beam_scores.clone(),
                "beam_next_tokens": (None if beam_next_tokens is None
                                     else beam_next_tokens.clone()),
                "beam_idx": None if beam_idx is None else beam_idx.clone(),
            }
            logits, _ = fwd(input_ids)
            next_token_logits = logits[:, -1, :].contiguous()
            vocab_size = next_token_logits.shape[-1]
            candidate_token_scores, candidate_tokens = torch.topk(
                next_token_logits, num_cands, dim=-1, largest=True,
                sorted=True)
            attn_last = []
            for c in range(num_cands):
                tmp = torch.cat(
                    [input_ids, candidate_tokens[:, c].unsqueeze(-1)], dim=-1)
                _, att = fwd(tmp)
                attn_last.append(att.max(1, keepdim=True).values)
            attn_last = torch.cat(attn_last, 1)
            attn_last = attn_last / attn_last.sum(-1, keepdim=True)
            attn_local = attn_last[:, :, n_prompt:, n_prompt:]
            attn_local = scale_factor * attn_local
            locals_seen.append(attn_local.clone())
            attn_local_scores = scores_fn(attn_local)
            cur_response_lens = attn_local.shape[-1]
            attn_i = attn_last[:, :, -1, img_s:img_e + 1].sum(-1)
            attn_scores = attn_i
            rollback_scores, rollback_locs = attn_local_scores.max(-1)
            rollback_loc = rollback_locs.mode().values.data
            rollback_loc = rollback_loc.mode().values.data
            penalty_scores = (-attn_scores if cur_response_lens <= switch_len
                              else rollback_scores)
            if history_rollback_locs is None:
                history_rollback_locs = [rollback_locs.mode().values.data[:, None]]
            else:
                history_rollback_locs.append(rollback_locs.mode().values.data[:, None])
            rollback_loc_gathers = torch.cat(history_rollback_locs, -1)
            candidate_token_scores -= penalty_weights * penalty_scores

            if len(history_states) >= history_length:
                history_states.pop(0)
            history_states.append(current_state)

            try:
                if all((g == rollback_loc).long().sum() > int(threshold)
                       for _, g in enumerate(rollback_loc_gathers)):
                    if rollback_loc < min_loc:
                        assert False
                    rollback_pos = rollback_loc + 1
                    if max_rollback_time[rollback_pos] >= num_cands:
                        rollback_pos = rollback_pos - 1
                        if max_rollback_time[rollback_pos] >= num_cands:
                            assert False
                        else:
                            max_rollback_time[rollback_pos] += 1
                    else:
                        max_rollback_time[rollback_pos] += 1
                    if cur_response_lens - rollback_pos > history_length + 1:
                        rollback_pos = max(1, cur_response_lens - history_length - 1)
                    for j in range(cur_response_lens - rollback_pos - 2):
                        history_states.pop(-1)
                        history_rollback_locs.pop(-1)
                        reject_token_pos_gather[-(j + 1)] = []
                    input_ids = history_states[-2]["input_ids"]
                    beam_scorer = history_states[-2]["beam_scorer"]
                    beam_scores = history_states[-2]["beam_scores"]
                    beam_next_tokens = history_states[-1]["beam_next_tokens"]
                    beam_idx = history_states[-1]["beam_idx"]
                    logits, _ = fwd(input_ids)
                    next_token_logits = logits[:, -1, :].contiguous()
                    history_states.pop(-1)
                    history_rollback_locs.pop(-1)
                    reject_token_pos_gather[rollback_pos + 1] = []
                    next_token_logits -= 999. + next_token_logits.min(
                        -1, keepdim=True).values.data
                    next_token_logits = next_token_logits.view(
                        1, num_beams * vocab_size)
                    beam_idx = beam_idx.view(1, num_beams)
                    beam_next_tokens = beam_next_tokens.view(1, num_beams)
                    reject_token_pos = beam_idx * vocab_size + beam_next_tokens
                    if len(reject_token_pos_gather[rollback_pos]) > 0:
                        reject_token_pos = torch.cat(
                            [reject_token_pos_gather[rollback_pos],
                             reject_token_pos], -1)
                    reject_token_pos_gather[rollback_pos] = reject_token_pos
                    next_token_logits = next_token_logits.scatter_(
                        -1, reject_token_pos, -999.)
                    next_token_logits = next_token_logits.view(
                        num_beams, vocab_size)
                    events.append((it, int(rollback_pos)))
                else:
                    assert False
            except AssertionError:
                next_token_logits.fill_(-999.)
                next_token_logits = next_token_logits.scatter_(
                    -1, candidate_tokens, candidate_token_scores)

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            next_token_scores = next_token_scores + beam_scores[
                :, None].expand_as(next_token_scores)
            next_token_scores = next_token_scores.view(
                1, num_beams * vocab_size)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True,
                sorted=True)
            next_indices = torch.div(next_tokens, vocab_size,
                                     rounding_mode="floor")
            next_tokens = next_tokens % vocab_size
            beam_outputs = beam_scorer.process(
                input_ids, next_token_scores, next_tokens, next_indices,
                pad_token_id=pad, eos_token_id=eos,
                decoder_prompt_len=n_prompt)
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]
            choices.append((beam_next_tokens.tolist(), beam_idx.tolist()))
            input_ids = torch.cat(
                [input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], -1)
            it += 1
            if beam_scorer.is_done:
                break
    best = beam_scorer.finalize(
        input_ids, beam_scores, pad_token_id=pad, eos_token_id=eos,
        decoder_prompt_len=n_prompt)
    return [int(t) for t in best[n_prompt:].tolist()], events, locals_seen, choices


class _TinyBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        A.register()
        cls.model = tiny_llava()
        cls.model.set_attn_implementation("fox")

    def setUp(self) -> None:
        self.inputs = prompt()


#: Small enough for an 12-token stub: the release's 10 / 10 / 15 / 512.
SMALL = dict(threshold=2, min_rollback_loc=1, penalty_switch_len=3,
             window_size=32)


class TestLoopMatchesRecomputedRelease(_TinyBase):
    def _both(self, scores_fn, n=12, **small):
        small = {**SMALL, **small}
        # ours
        seen_ours, choices_ours = [], []
        real = O.column_product_scores
        real_process = BeamScorer.process

        def spy_scores(attn_local, *, precision="fp16"):
            seen_ours.append(attn_local.clone())
            return scores_fn(attn_local, precision=precision)

        def spy_process(self_, *a, **k):
            out = real_process(self_, *a, **k)
            choices_ours.append((out["next_beam_tokens"].tolist(),
                                 out["next_beam_indices"].tolist()))
            return out

        O.column_product_scores = spy_scores
        BeamScorer.process = spy_process
        try:
            with torch.no_grad():
                got = opera_generate(self.model, self.inputs, collect_trace=True,
                                     max_new_tokens=n, **small)
        finally:
            O.column_product_scores = real
            BeamScorer.process = real_process
        # the release, from scratch, on the eager implementation
        self.model.set_attn_implementation("eager")
        try:
            ref_ids, events, seen_ref, choices_ref = release_loop(
                self.model, self.inputs, max_new_tokens=n,
                scores_fn=lambda a: scores_fn(a),
                threshold=small["threshold"],
                window_size=small["window_size"],
                switch_len=small["penalty_switch_len"],
                min_loc=small["min_rollback_loc"])
        finally:
            self.model.set_attn_implementation("fox")
        return got, (ref_ids, events, seen_ref, choices_ref), (
            seen_ours, choices_ours)

    def test_with_rewinds_the_two_agree_step_for_step(self) -> None:
        got, (ref_ids, events, seen_ref, choices_ref), (
            seen_ours, choices_ours) = self._both(scripted_scores)
        self.assertGreaterEqual(
            len(events), 2, "the scenario must rewind or it proves nothing")
        self.assertEqual(
            [(r["at_step"], r["to_pos"]) for r in got["rollbacks"]],
            [(got["trace"][i]["step"], pos) for i, pos in events])
        self.assertEqual(len(choices_ours), len(choices_ref))
        for i, (a, b) in enumerate(zip(choices_ours, choices_ref)):
            self.assertEqual(a, b, f"beam choice differs at iteration {i}")
        self.assertEqual(got["token_ids"], ref_ids)
        self.assertEqual(len(seen_ours), len(seen_ref))
        for i, (a, b) in enumerate(zip(seen_ours, seen_ref)):
            self.assertEqual(a.shape, b.shape, f"iteration {i}")
            self.assertTrue(
                torch.allclose(a, b, atol=1e-5),
                f"attn_local differs at iteration {i}: "
                f"{(a - b).abs().max().item():.2e}")

    def test_without_rewinds_the_regime_switch_agrees_too(self) -> None:
        """Real scores on a stub never rewind but do switch penalty regime."""
        got, (ref_ids, events, seen_ref, choices_ref), (
            seen_ours, choices_ours) = self._both(
                lambda a, precision="fp16": column_product_scores(
                    a, precision="fp32"), n=9)
        self.assertEqual(events, [])
        self.assertEqual(got["n_rollbacks"], 0)
        self.assertEqual(got["token_ids"], ref_ids)
        self.assertEqual(choices_ours, choices_ref)

    def setUp(self) -> None:
        super().setUp()
        self._orig = O.column_product_scores

    def tearDown(self) -> None:
        O.column_product_scores = self._orig


class TestRewindMechanics(_TinyBase):
    def _run(self, **kw):
        real = O.column_product_scores
        O.column_product_scores = scripted_scores
        try:
            with torch.no_grad():
                return opera_generate(
                    self.model, self.inputs, collect_trace=True,
                    **{**SMALL, **kw})
        finally:
            O.column_product_scores = real

    def test_a_rewind_rebuilds_a_cache_equal_to_a_fresh_prefill(self) -> None:
        """After the rebuild the cache must be the restored hypotheses' own."""
        built = {}
        real = O._rebuild_cache

        def spy(model, cache, ids, full_mask, n_prompt, chunk):
            out = real(model, cache, ids, full_mask, n_prompt, chunk)
            built.setdefault("calls", []).append((
                ids.clone(),
                [(k.clone(), v.clone()) for k, v in zip(
                    *self._kv(out))]))
            return out

        O._rebuild_cache = spy
        self.addCleanup(lambda: setattr(O, "_rebuild_cache", real))
        got = self._run(max_new_tokens=11, replay_chunk=2)
        self.assertTrue(built.get("calls"), "no rewind happened")
        for ids, kv in built["calls"]:
            fresh = self._fresh(ids[:, :-1])
            self.assertEqual(kv[0][0].shape[-2], fresh[0][0].shape[-2])
            for (k, v), (fk, fv) in zip(kv, fresh):
                self.assertTrue(torch.allclose(k, fk, atol=1e-4))
                self.assertTrue(torch.allclose(v, fv, atol=1e-4))
        self.assertGreater(got["n_rollbacks"], 0)

    @staticmethod
    def _kv(cache):
        try:
            layers = cache.layers
            return ([l.keys for l in layers], [l.values for l in layers])
        except AttributeError:  # older DynamicCache
            return (list(cache.key_cache), list(cache.value_cache))

    def _fresh(self, ids):
        with torch.no_grad():
            out = self.model(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                pixel_values=self.inputs["pixel_values"].repeat(
                    ids.shape[0], 1, 1, 1), use_cache=True)
        ks, vs = self._kv(out.past_key_values)
        return list(zip(ks, vs))

    def test_chunk_size_does_not_change_the_result(self) -> None:
        a = self._run(max_new_tokens=11, replay_chunk=1)
        b = self._run(max_new_tokens=11, replay_chunk=64)
        self.assertEqual(a["token_ids"], b["token_ids"])
        self.assertEqual(a["rollbacks"], b["rollbacks"])

    def test_the_run_is_deterministic(self) -> None:
        a = self._run(max_new_tokens=11)
        b = self._run(max_new_tokens=11)
        self.assertEqual(a["token_ids"], b["token_ids"])
        self.assertEqual(a["rollbacks"], b["rollbacks"])

    def test_every_rewind_goes_backwards_and_is_recorded(self) -> None:
        got = self._run(max_new_tokens=11)
        self.assertTrue(got["rollbacks"])
        for r in got["rollbacks"]:
            self.assertLess(r["to_pos"], r["at_step"])
        self.assertEqual(got["n_rollbacks"], len(got["rollbacks"]))
        self.assertEqual(
            sum(1 for t in got["trace"] if t["rolled_back_to"] >= 0),
            got["n_rollbacks"])

    def test_the_run_still_terminates_with_a_full_length_answer(self) -> None:
        got = self._run(max_new_tokens=11)
        self.assertGreater(got["n_gen_tokens"], 0)
        self.assertLessEqual(got["n_gen_tokens"], 11)

    def test_every_forward_runs_without_autograd(self) -> None:
        flags = []
        original = A._opera_capture_attn

        def spy(module, query, key, attention_mask, scaling):  # noqa: ANN001
            flags.append((torch.is_inference_mode_enabled(),
                          torch.is_grad_enabled()))
            original(module, query, key, attention_mask, scaling)

        A._opera_capture_attn = spy
        self.addCleanup(lambda: setattr(A, "_opera_capture_attn", original))
        with torch.enable_grad():
            self._run(max_new_tokens=6)
        self.assertTrue(flags)
        self.assertEqual(set(flags), {(True, False)})

    def test_the_main_forward_is_captured_only_on_the_rollback_path(self) -> None:
        """5 speculative rows per step, plus one main row from step 1 on."""
        seen = []
        original = A._opera_capture_attn

        def spy(module, query, key, attention_mask, scaling):  # noqa: ANN001
            seen.append(int(key.shape[-2]))
            original(module, query, key, attention_mask, scaling)

        A._opera_capture_attn = spy
        self.addCleanup(lambda: setattr(A, "_opera_capture_attn", original))
        with torch.no_grad():
            got = opera_generate(self.model, self.inputs, max_new_tokens=14,
                                 collect_trace=True)
        self.assertEqual(got["n_rollbacks"], 0)
        steps = len(got["trace"])
        self.assertEqual(len(seen), 5 * steps + (steps - 1))


class TestScoreFunction(unittest.TestCase):
    def test_fp16_saturates_where_fp32_does_not(self) -> None:
        """A column of factors of 30 overflows fp16 in a handful of products."""
        a = torch.full((1, 1, 6, 6), 30.0).tril().half()
        half = column_product_scores(a, precision="fp16")
        full = column_product_scores(a, precision="fp32")
        self.assertTrue(torch.isinf(half[0, 0, 0]))
        self.assertTrue(torch.isfinite(full).all())
        self.assertEqual(half.dtype, torch.float16)

    def test_fp16_flushes_small_columns_to_exact_zero(self) -> None:
        a = torch.full((1, 1, 4, 4), 0.01).tril().half()
        half = column_product_scores(a, precision="fp16")
        full = column_product_scores(a, precision="fp32")
        self.assertTrue(torch.equal(half, torch.zeros_like(half)))
        self.assertTrue((full > 0).all())

    def test_the_two_agree_where_nothing_saturates(self) -> None:
        g = torch.Generator().manual_seed(0)
        a = (torch.rand(2, 3, 5, 5, generator=g) * 1.5 + 0.8).tril().half()
        half = column_product_scores(a, precision="fp16").float()
        full = column_product_scores(a, precision="fp32")
        # 1e-7 x a product of a few numbers lands in fp16's subnormal range,
        # whose spacing is 6e-8: agreement is to that absolute resolution.
        self.assertGreater(float(full.max()), 1e-6)
        self.assertTrue(torch.allclose(half, full, rtol=0, atol=7e-8))

    def test_the_score_is_the_product_of_the_column_below_the_diagonal(self) -> None:
        a = torch.tensor([[[[1., 0., 0.], [2., 3., 0.], [4., 5., 6.]]]])
        got = column_product_scores(a, precision="fp32")[0, 0]
        want = 1e-7 * torch.tensor([1 * 2 * 4, 3 * 5, 6], dtype=torch.float32)
        self.assertTrue(torch.allclose(got, want, rtol=1e-4))

    def test_no_columns_raises_instead_of_returning_an_empty_score(self) -> None:
        with self.assertRaises(ValueError):
            column_product_scores(torch.zeros(1, 1, 0, 0))

    def test_an_unknown_precision_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            column_product_scores(torch.ones(1, 1, 2, 2), precision="bf16")


class TestNonFiniteIsCounted(_TinyBase):
    def test_a_saturated_candidate_is_reported_not_hidden(self) -> None:
        def inf_one(attn_local, *, precision="fp16"):
            out = scripted_scores(attn_local)
            out[:, 0, :] = float("inf")   # candidate 0 saturates, rest finite
            return out

        real = O.column_product_scores
        O.column_product_scores = inf_one
        self.addCleanup(lambda: setattr(O, "column_product_scores", real))
        with torch.no_grad():
            got = opera_generate(self.model, self.inputs, max_new_tokens=6,
                                 **SMALL)
        self.assertGreater(got["n_nonfinite_steps"], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
