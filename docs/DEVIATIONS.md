# How this implementation differs from the released Fox code

Every difference between this repository and the code published with
*"Dismantling Pathological Shortcuts: A Causal Framework for Faithful LVLM
Decoding"* (ICML 2026), classified by whether it can change a number.

Written for the reproduction report: when a result here disagrees with the
paper, this is the list of candidate causes, ordered so the ones that cannot be
the cause are ruled out first.

Reference: `github.com/Cc2021start/Fox` at the state audited 2026-10-12
(3 commits; initial commit `94d671f`, authored "anonymous").

---

## 1. Proven bit-identical

Asserted by `tests/test_kernel_equivalence.py` against
`foxrepro/fox/reference.py`, a verbatim transcription of the released code
lifted out of its vendored-transformers context. Random tensors, fp32 / fp16 /
bf16, prefill (Q=600) and decode (Q=1), batched and not. No GPU or checkpoint
needed, because the method has no learned parameters.

| Component | Released location | Ours |
|---|---|---|
| The intervention | `transformers/.../modeling_llama.py:389-392` | `kernel.intervene(mode="as_implemented")` |
| `m_sys`, `H_vis` | `Uncertain_head.py:compute_fusion_stage_metrics_vaf` | `kernel.head_scores` |
| Top-k head selection | `modeling_llama.py:371-380` | `kernel.select_heads` |
| JSD | `vcd_sample.py:189-206` | `kernel.jsd` (12 decimal places) |
| Conflict-gated fusion | `vcd_sample.py:208-215` | `kernel.fuse` |

Two released defects are **preserved deliberately**, because removing them
would stop this being a reproduction:

- the double normalisation inside the entropy helper (the caller normalises,
  then `_entropy_from_probs` normalises again) — it changes the last bits;
- the greedy/sampling asymmetry: the released greedy path argmaxes the
  **unmasked** fused logits while its sampling path uses the β-masked ones. We
  reproduce that by default and expose `mask_greedy=True` to run the clean
  de-randomisation the paper should have used. Report both.

## 2. Different code, verified same result

| Change | Why | Verification |
|---|---|---|
| JSD computed on-device in torch | The release calls `.float().cpu().numpy()` on a 32064-wide vector **once per generated token**, forcing a GPU→CPU sync. 512 syncs per CHAIR caption. | bit-equal to 12 dp |
| Head selection vectorised (`scatter_`) | The release loops over the batch in Python | bit-equal per row |
| Top-p via HF's `TemperatureLogitsWarper` + `TopPLogitsWarper` | The release applies `logits_warper` to the fused logits. A hand-rolled version kept one token fewer than HF in ~5% of rows: HF sorts ascending and drops the tail with cumulative mass `<= 1 - top_p`, the hand-rolled one sorted descending and kept while exclusive cumulative mass was `<= top_p` — different strictness, and opposite accumulation order. | 300/300 identical tokens |

## 3. Deviations that can change a number

Ordered by expected size of effect.

### 3.1 Reduced-precision accumulation is pinned off — **this one will show**

`pin_precision_flags()` sets `allow_fp16_reduced_precision_reduction = False`
before the model loads. The release leaves it at its default, **True**.

That flag governs accumulation *inside* the fp16 `Q @ Kᵀ` matmul — the exact
tensor Fox manipulates and this project studies. Leaving it on stacks a second,
uncontrolled precision effect on the one under test, so pinning it is the right
call scientifically, but it means our attention logits are not bit-identical to
the release's even before the intervention runs.

Measured on the host: the release would run with
`fp16_reduced=True, bf16_reduced=True, cudnn_tf32=True`. On a T4 (sm_75) the
TF32 flags are inert — TF32 is Ampere+ — so only the fp16 one bites there.

### 3.2 transformers 5.16.1, not the vendored 4.29.2

The release ships a patched HuggingFace 4.29.2 (3370 files) with the
intervention welded into `modeling_llama.py`. We register a custom attention
function through `AttentionInterface` on the host's own transformers.

The kernel is bit-equal, but everything around it — RoPE application, layer
norms, the matmul kernels chosen, the KV-cache implementation — is a different
release. Any of those can move the last bits.

`fox/reference.py` plus Track A (`requirements-trackA.lock`) exist to settle
this by cross-checking token-for-token on a few dozen samples. **Not yet run.**

### 3.3 `llava-hf/llava-1.5-7b-hf`, not `liuhaotian/llava-v1.5-7b`

Weight-identical, but the original stores the language model under
`model.layers.*` where transformers expects `model.language_model.layers.*` and
ships no `preprocessor_config.json`. Loading it as a
`LlavaForConditionalGeneration` reports the vision tower and projector as
MISSING and **initialises them randomly** without erroring — so
`load_llava` refuses that repo by name, and refuses any load that leaves
parameters uninitialised.

The converted repo expands `<image>` into 576 ids in `input_ids` via the
processor; the pre-4.44 path spliced embeddings inside the model instead. Image
span measured on the host: `[5, 580]` with `--system-prompt none`, width
exactly 576; the released default pushes it to `[~40, ~616]`.

### 3.4 Two branches batched, not run sequentially

The release runs the observational and interventional passes as two separate
`self(...)` calls, each with its own `past_key_values`. We run them as rows 0
and 1 of one forward pass, which reads the 13.16 GiB of weights once instead of
twice — roughly 2× faster decode, since decode is bandwidth-bound.

Mathematically identical: both rows process the same token sequence, and
`FOX.branch_rows` restricts the write to row 1. **Not yet verified bit-equal on
GPU**, where batched and unbatched matmuls can use different kernels.

Consequence: the release computes the JSD from batch row 0 and applies that
scalar to every row, which is harmless at batch 1 but silently wrong when
batched. Ours is per-row.

### 3.5 The prompt

Resolved on 2026-10-18 by reading the released scripts rather than inferring
from the paper. The release does **not** build prompts through
`conv_templates`; `pope_eval.py` and `char_eval.py` never import it. The
assembly is, in order:

1. `pope_eval.py:59-64` / `char_eval.py:31-36` —
   `INSTRUCTION_TEMPLATE["llava-1.5"] = "USER: <ImageHere> <question> ASSISTANT:"`.
   A **space** before the question, not a newline, and no added instruction:
   the POPE question goes in verbatim (`pope_v0_raw`) and CHAIR asks
   `"Please describe this image in detail."` (`char_eval.py:157`, our
   `chair_v1`).
2. `eval_configs/llava-1.5_eval.yaml` sets, under `model:`, a
   `system_message` equal to `conv_vicuna_v1`'s system string.
3. `minigpt4/models/llava.py:344` reads it as `cfg.get("system_message", "")`
   and `llava.py:228` does `instruction = [self.system_message + p for p in
   instruction]`.

Step 3 is a bare `+`. There is **no separator**, so the released prompt reads

```
...polite answers to the user's questions.USER: <image> Is there a dog in the image? ASSISTANT:
```

with the period running straight into `USER`. `--system-prompt llava_v1`
(the default) reproduces that byte for byte; the string is asserted in
`TestSystemPrompt.test_default_reproduces_the_released_concatenation_verbatim`
so a later cleanup cannot quietly insert the space.

Three arms are kept, each isolating one difference from this default:
`--system-prompt llava_v1_spaced` (adds the space), `--system-prompt none`
(drops the system message), `--wrapper newline` (HF's separator),
`--prompt-template-id pope_v1` (appends "Answer the question using a single
word or phrase").

Those are not cosmetic. `m_sys` is the attention mass on the keys *before* the
image, which the paper calls the anchor of the shortcut it claims to sever.
Under `none` those keys are only `<s>`, `USER`, `:` — measured image span
`[5, 580]` instead of `[~40, ~616]` — so `--system-prompt none` is a direct
test of the paper's premise, not a convenience. And `pope_v1` is the arm most
likely to explain session 1's inflated POPE baseline (81.90 against the
paper's 76.77): it keeps answers to one parseable token.

**Correction.** Session 1 ran with all four differences present at once —
system prompt joined with a space, `\n` separator, and the "single word or
phrase" suffix. An intermediate reading of `pope_eval.py` alone concluded the
release sends no system prompt at all; `eval_configs/` shows otherwise. The
session-1 run files are kept, under their own variant hashes, as the
measurement of that combination.

### 3.5.1 Generation settings, verified against the release

| | release | ours | matters? |
|---|---|---|---|
| POPE `max_new_tokens` | `1` (`pope_eval.py:402`) | `1` | — |
| CHAIR `max_new_tokens` | `512` (`char_eval.py:191`) | `512` | — |
| `num_beams` | `1` (`--beam` default) | `1` | — |
| sampling | `do_sample=True`, `top_p=0.9`, `temperature=1` (`llava.py:150,154,272`) | same | — |
| `use_cache` | `pope_eval.py` passes `False`, but `llava.py:272` **hardcodes `True`** in the inner `generate`, so the argument is dead | `True` | no |
| POPE scoring | see below | `--pope-parser released` | **yes** |

The released POPE scorer (`pope_eval.py:232-242`) has no unparsable class:

```python
NEG_WORDS = ["No", "not", "no", "NO"]
if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
    pred = 0
else:
    pred = 1
```

Anything without a negation token — including `""`, `"There"`, `"Maybe"` — is
scored as **yes**. Our strict parser calls those unparsable and
`--unparsed-policy negative` then scores them as **no**, the opposite. With
the "single word or phrase" suffix the two agree on every sample (measured:
3000/3000 on both session-1 POPE runs, so the committed numbers are unchanged
by this). Without it, `max_new_tokens=1` produces first tokens that are
neither `Yes` nor `No` and the rules diverge. `released` is therefore the
default, and `metrics["pope_parser"]` records which rule produced any number.

### 3.8 CHAIR's 500 images are not recoverable, and that is measurable

This is the one difference that cannot be closed by being more careful, so it
is worth stating exactly.

The release picks its CHAIR images like this (`char_eval.py:136-137`):

```python
img_files = os.listdir(args.data_path)      # default: /data-disk/only_test
random.shuffle(img_files)
...                                          # keeps the first --test-sample 500
```

Three independent problems, any one of which would be enough:

1. `/data-disk/only_test` is not in the repository and is not described
   anywhere. It is not full `val2014` — the name suggests a pre-filtered
   subset — so even its population is unknown.
2. `os.listdir` returns entries in filesystem order, which varies by
   filesystem and by the order files were written. The seeded `random.shuffle`
   that follows is therefore shuffling a machine-dependent list.
3. The paper never names the set either. It says "500 COCO samples" in five
   places (§5, Tables 2/6/9, C.3) and never which 500, never a selection
   rule, and never "following [prior work]" — which some CHAIR papers do, and
   which would have made the set recoverable.

So our CHAIR numbers are computed on a *different* 500 images from the
paper's, and no amount of care closes that. It also explains the shape of the
reproduction: POPE uses POPE's fixed official annotation files and our
baselines land within 0.05 points of the paper's, while the CHAIR baseline
sits 3.88 points away.

What can be done is to **measure how much it matters**. `CHAIR_DRAWS` adds
four more independent 500-image draws from val2014 (`chair_images_d1..d4.json`
in `assets/`, sampled with seeds 1–4), each selected by split name so every
run records which 500 it scored. Running the baseline and Fox over all five
draws turns "we could not reproduce CHAIR" into a number: how far CHAIR_S
moves when nothing changes but the image set. If that spread covers 3.88
points, the discrepancy is explained and the finding becomes one about the
benchmark protocol rather than about the method.

Those draw files are **ours, not the release's** — the released repository
contains 22 non-vendored files and none of them is an image list. (`pope/coco/426.json`
and `vase.json` look like candidates and are not: four POPE questions each,
left over from debugging, and two of `vase.json`'s four are malformed —
`"Waht are they doing?"` labelled `yes` and `"There are two"` labelled `no`.
`popo_err.jsonl` is zero bytes.)

### 3.10 A pre-cast fp16 checkpoint is not a deviation

If a run points at `kietnt0603/instructblip-vicuna-7b-fp16` rather than
`Salesforce/instructblip-vicuna-7b`, nothing about the numbers changes, and
this records why so the provenance is not read as a confound later.

`Salesforce/instructblip-vicuna-7b` stores its weights in **fp32**: 29.5 GiB
of safetensors to occupy about 15.8 GiB of VRAM once
`from_pretrained(dtype=torch.float16)` casts them at load. Every Kaggle
session pays that transfer again, because every session starts from an empty
disk. `scripts/cast_checkpoint_fp16.py` performs the same cast once and keeps
the result, so a session downloads ~14.7 GiB instead.

The cast is the identical operation `from_pretrained` already performs on
every load -- round-to-nearest fp32 to fp16 on the same inputs -- so the
weights are bit-identical, and `--verify` demonstrates that tensor by tensor
rather than asserting it. The run records `model_id` in every record, so a
pre-cast checkpoint is visible in the archive and cannot be mistaken for the
original.

**Quantization remains ruled out and this is not an exception to that.** The
quantity under study is a precision effect, so an int8 or int4 checkpoint
would alter the thing being measured; `Yhyu13/instructblip-vicuna-7b-gptq-4bit`
and `Mediocreatmybest/instructblip-vicuna-7b_8bit` exist and must not be used
here. fp16 is the dtype the runs already execute in.

**Update, 2026-11-23: the fp16 copy was never uploaded, so runs use the
original.** `kietnt0603/instructblip-vicuna-7b-fp16` was created on
2026-10-26 and has held only `.gitattributes` since; smoke-r26 and smoke-r27
found it empty (first a 401 while it was private, then "no `model_type` in
config.json" once it was public). The plan and the smoke script now point at
the public `Salesforce/instructblip-vicuna-7b`, which `from_pretrained(
dtype=fp16)` casts at load -- the same operation, so the numbers are unchanged
and only the per-session download grows (29.5 GiB instead of ~14.7 GiB).
That also removes the licensing question the private copy existed to avoid
(the language model derives from LLaMA-1, research-only, redistribution of
derived weights restricted): nothing is hosted. The empty repository can be
deleted or made private again; `scripts/cast_checkpoint_fp16.py` still exists
if the download time ever matters.

### 3.11 How the release runs three backbones with one patch

> **Superseded in one respect by §3.20:** the release's Shikra visual keys are the 256
> patches `[img_s + 1, img_e)`, not the 258 tokens between and including the markers.

Worth understanding, because it was got wrong here first and because it
explains a hardcoded constant that otherwise looks arbitrary.

**Fox patches LLaMA attention**, in
`transformers/src/transformers/models/llama/modeling_llama.py`. Anything whose
decoder *is* LLaMA inherits the intervention without a second implementation.
That is the whole mechanism by which `pope_eval.py` offers a `shikra` entry in
`INSTRUCTION_TEMPLATE` beside `llava-1.5`.

`shikras/shikra-7b-delta-v1-0708/config.json` makes it concrete:

```json
"architectures": ["ShikraLlamaForCausalLM"],  "model_type": "shikra",
"hidden_size": 4096, "intermediate_size": 11008,
"num_attention_heads": 32, "num_hidden_layers": 32,   <- LLaMA-7B exactly
"mm_hidden_size": 1024, "use_mm_proj": true,
"mm_vision_tower": "openai/clip-vit-large-patch14",   <- stock, not bundled
"vocab_size": 32003                                    <- 32000 + 3 markers
```

and its weight index carries 325 tensors of which exactly **two** sit outside
a plain LLaMA decoder: `model.mm_projector.weight` and
`model.mm_projector.bias` — a single linear 1024 -> 4096. The vision tower is
downloaded separately from HuggingFace.

**The correction.** An earlier note here concluded Shikra needs the release's
`modeling_shikra.py`, therefore transformers 4.29.2, therefore Python 3.10,
therefore not Kaggle. The first step does not follow. A decoder that is LLaMA
loads as `LlamaForCausalLM` on any modern transformers; what has to be written
is the projector and the embedding splice, which is two tensors and a CLIP
forward. The real costs are the delta merge (`huggyllama/llama-7b` plus the
delta, ~27 GB down, ~40 GB transient) and the port, not the environment.

**And the magic numbers are explained.** The release detects Shikra's visual
span by testing `img_s == 33 and img_e == 290`. CLIP ViT-L/14 at 224 px gives
a 16x16 grid = 256 patches; `mm_use_im_start_end: true` wraps them in
`<im_start>` and `<im_end>`; 256 + 2 = **258**, and 290 - 33 + 1 = 258. The
33 is the token offset of the system message plus `USER: `. It is not an
arbitrary constant — it is this checkpoint's layout, hardcoded instead of
derived, which is why it would silently mis-locate the image on any other
prompt.

**Where Shikra's pixels are prepared.** The checkpoint ships no
`preprocessor_config.json`, which looks like a gap in the port and is not.
`modeling_shikra.py:49-81` loads the vision tower as
`CLIPVisionModel.from_pretrained(vision_tower)` — stock, at runtime, which is
why no vision weights are in the checkpoint — and the image processor comes
from the same repo. The release routes it through LAVIS, but
`minigpt4/processors/clip_processors.py`'s `clip_image_eval` adds nothing at
all:

```python
self.transform = CLIPImageProcessor.from_pretrained(proc_type)
return self.transform.preprocess(item, return_tensors='pt')['pixel_values'][0]
```

So the whole visual path is reachable from the modern stack:

```python
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
vit  = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
h    = vit(pixel_values, output_hidden_states=True).hidden_states[-2][:, 1:]  # drop CLS -> 256
emb  = nn.Linear(1024, 4096)(h)        # the two mm_projector tensors
#      spliced between <im_start> and <im_end>
```

`mm_vision_select_layer: -2` is `hidden_states[-2]`, and `[:, 1:]`
(`modeling_shikra.py:136`) drops the CLS token, leaving the 256 patches.

`scripts/probe_backbone.py` reports all of this from `config.json` and the
weight index — a few hundred kilobytes rather than 13-30 GiB — for any
checkpoint, and is the thing to run before arguing about feasibility again.

### 3.8.1 Measured: the image set does NOT explain the CHAIR gap

Five independent 500-image draws, baseline and fox, greedy on each. The
hypothesis in §3.8 was that image-set variance would cover the 3.88-point gap
to the paper's baseline. It does not, and the measurement found something
else instead.

| draw | base CHAIR_S | fox CHAIR_S | gap | base CHAIR_I | fox CHAIR_I | gap |
|---|---|---|---|---|---|---|
| val2014_500 | 49.20 | 49.40 | **+0.20** | 13.82 | 13.66 | −0.17 |
| val2014_500_d1 | 46.40 | 41.40 | **−5.00** | 12.96 | 11.91 | −1.05 |
| val2014_500_d2 | 49.40 | 44.60 | −4.80 | 14.39 | 13.15 | −1.24 |
| val2014_500_d3 | 49.80 | 49.40 | −0.40 | 14.81 | 14.81 | 0.00 |
| val2014_500_d4 | 47.60 | 44.00 | −3.60 | 13.16 | 12.85 | −0.31 |

**The hypothesis fails.** The paper's greedy baseline of 50.90 sits outside the
observed range [46.40, 49.80], and its gap of −8.30 sits far outside
[−5.00, +0.20]. Changing the image set does not reach the paper's numbers, so
the discrepancy needs an account beyond "a different 500 images".

**And the gap is less stable than the absolute number it is computed from.**
The baseline's own CHAIR_S moves 3.40 points across draws; the gap moves
**5.20**. Pairing is supposed to absorb item difficulty, and for CHAIR across
image sets it does not — the two arms respond to a change of images
differently enough that the difference is noisier than either term.

Treating the image set as the sampling unit, with five draws and df=4:

| | mean gap [95% CI] | p | contains 0 | paper |
|---|---|---|---|---|
| CHAIR_S | **−2.72** [−5.77, +0.33] | 0.069 | **yes** | −8.30, excluded |
| CHAIR_I | **−0.55** [−1.25, +0.14] | 0.090 | **yes** | −3.24, excluded |

So under greedy decoding, once image-set variance is admitted, **Fox's CHAIR
effect is not resolvable at all** — and the paper's claimed effect is excluded
from the other side.

**This also indicts an earlier claim of our own.** "CHAIR greedy gives a null"
was stated from `val2014_500` alone, with an interval computed inside that one
draw. The conclusion survives, but the reasoning behind it was blind to the
variance component that turns out to dominate. Worse, `val2014_500` is the
draw least favourable to Fox of the five (+0.20, the only non-negative gap);
drawing `d1` first would have produced −5.00 and a different story from the
same method.

**Consequence for the sampling result, which is not yet settled.** The
ten-seed sampling comparison — CHAIR_S −6.98 [−9.25, −4.71], reported as
excluding the paper's −11.20 — is computed on `val2014_500` only. Its interval
sees seed variance and item difficulty and is blind to image-set variance,
which is measured here at sd 2.46 under greedy. That claim should be read as
**not established** until the ten-seed comparison is repeated on a second
draw. The POPE conclusions are unaffected: POPE uses its own fixed official
annotation files, so there is no equivalent degree of freedom.

Reproduce with `scripts/chair_image_set_sensitivity.py`.

### 3.9 Why only `sampling` and `fox` were re-run (superseded by 3.12)

The paper's tables carry eight rows. This project runs two of them, and the
reason is not cost.

**The contribution being tested is Fox.** Every claim the paper makes has the
shape "Fox improves on the baseline by X", so checking it needs exactly those
two rows. The other six place Fox in a ranking; they are context, not the
claim. And the baseline is the row that has to be right for any of it to
mean anything — ours matches the paper's to 0.05 points on all three POPE
splits, so the floor every gap is measured from is confirmed.

**The other six were re-run by the authors, not copied.** This was checked
rather than assumed, because the opposite assumption is easy to make and
would be wrong. VCD's own paper (Leng et al., 2024, Table 1, MSCOCO,
LLaVA-1.5) reports:

| split | VCD paper: Regular -> VCD | Fox paper: Sampling -> VCD |
|---|---|---|
| random | 83.29 -> 87.73 (+4.44) | 85.13 -> 86.84 (+1.71) |
| popular | 81.88 -> 85.38 (+3.50) | 82.53 -> 80.87 (**-1.66**) |
| adversarial | 78.96 -> 80.88 (+1.92) | 76.77 -> 75.33 (**-1.44**) |

Not one number matches, and the sign flips on two splits: VCD helps in its own
paper and hurts in Fox's. Numbers that disagree that strongly cannot have been
copied. The harness agrees — `pope_eval.py` carries `--use-vcd`, `--use-icd`
and `--opera` flags, and `vcd_sample.py:153` comments its shared path
`# for VCD and SID`, with a leftover author path pointing at
`/home/kdzlys/sid2/log/`.

The one exception is **CausalMM**, which appears nowhere in the *Fox*
repository and cannot be run from it. Whether its row was re-run elsewhere or
taken from its own paper is not determinable from what Fox published.
CausalMM's **own** repository is public and complete — see §3.13, which
corrects an earlier version of this paragraph that said it was unavailable.

**Superseded, 2026-10-27.** Three of the six are now implemented and run as
themselves — see §3.12, and §3.13 for where OPERA and CausalMM stand.

Worth recording for the write-up: VCD's paper reports a standard deviation
beside every number (`83.29(±0.35)`). Fox claims "a two-sided t-test
(p < 0.05)" over "10 independent runs" (C.2) and prints variance nowhere. A
paper in the same line of work, three years earlier, met the reporting
standard this one skips.

### 3.12 Three of the compared methods, implemented as themselves

§3.9 said the compared rows were out of reach. Three of them are not, and
they are now in the plan as `vcd`, `icd` and `sid`. The source is
[huofushuo/SID](https://github.com/huofushuo/SID) — the unmodified parent of
the Fox fork, which is where all three live in one harness
(`vcd_sample.py`, `pope_eval.py`, `minigpt4/models/modeling_llama.py`).
Every number-bearing line is transcribed verbatim into
`src/foxrepro/fox/reference.py` and the clean port is asserted bit-identical
in `tests/test_kernel_equivalence.py`.

**One decoding rule, three negative branches.** This is the structural fact
worth having, and it is not visible from the papers:

```
diffs  = (1 + a) * z_obs - a * z_cd           # a SUBTRACTION
cutoff = log(b) + z_obs.max()
cd_logits = diffs.masked_fill(z_obs < cutoff, -inf)
```

| arm | negative branch | `cd_alpha` | extra forward |
|---|---|---|---|
| `vcd` | diffusion-noised image, step 500 | 1.0 | batched row |
| `icd` | `"You are a confused object detector."` after `<image>` | 1.0 | separate pass |
| `sid` | same image, read through CT2S | 0.5 | batched row |

Fox's fusion is `z_obs + lambda_t * z_do` — an **addition**, gated to fire
where the branches agree. These three subtract, to fire where they disagree.
Both are called "contrastive decoding" in the literature; they are opposite
operations, and the pair (`vcd`, `vcd_fox_fusion`) now isolates exactly that
difference at a fixed negative branch, which neither paper ran.

**What CT2S actually does** (`modeling_llama.py`, the `elif USE_FAST_V`
branch). At layer 2, read layer **1**'s attention, average over heads, take
the last query row, slice the 576 vision keys, and keep the 100 the model
attends to **least** (`topk(100, largest=False)`); every other vision key is
unattendable from layer 2 onward. Nothing is removed — it is a key mask, so
position ids and the KV cache keep their full length. The ranking is
recomputed at every forward, so it is per-token state.

Three ways to get this wrong, each of which still produces numbers:

1. `largest=True` is the line directly above, commented out. That is FastV,
   a compute optimisation that keeps the *most*-attended tokens. SID inverts
   it on purpose: the negative branch is made to read the image badly.
   Keeping the top 100 would build a *better* branch and subtract the wrong
   thing.
2. The ranking comes from layer `agg_layer - 1`, not `agg_layer`
   (`last_layer_attention = layer_outputs[1]` is the previous iteration's).
3. The negative branch sees the **same image and the same prompt**.
   `pope_eval.py` sets `image_cd = image` under its default `--use-cd`. CT2S
   is the entire perturbation, which is what makes SID cheap and what makes
   "self-introspective" literal.

**Two defects in the released harness that we do not reproduce.**

* `--use-cd` defaults True and is tested before `--use-vcd`, and both are
  `action='store_true'`, so **SID's repo as shipped cannot run VCD at all**
  without editing the file.
* If it could, it would not be VCD: `use_cd` is true for any `images_cd`, and
  that path switches CT2S on whenever `--fast-v-attention-rank` is set — which
  it is, to 100, by default. So `--use-vcd` would run VCD **plus** CT2S.
  `fox_generate` raises on that combination rather than offering it; no paper
  proposes it, and a number from it belongs to neither.

**One deviation of ours.** The two branches are batched as rows of one
forward where their token sequences allow it (`vcd`, `sid`), as elsewhere in
this repo (§3.4). ICD cannot: its negative prompt is longer, so it runs two
passes with two caches, which is what the release does for all three.

**Greedy is the released default.** `--sample-greedy` defaults True, and that
path argmaxes the **unmasked** `diffs` — bypassing the beta cutoff *and* the
temperature/top-p warpers, which are applied only to `cd_logits`. So the
published POPE numbers for these methods are greedy whatever `--sample` says.
The plan runs both (`table1_methods`/`table2_methods` under sampling,
`table9_methods` under greedy) and both are reported.

**Relabelling the old arm.** Rounds 1–5 ran one arm called `vcd` that was
VCD's noise inside Fox's fusion. `vcd` now means VCD, so the six archived
files are relabelled `vcd_fox_fusion` on read, keyed on the
`method_extra.variant` the records themselves carry
(`evaluate.LEGACY_METHOD_NAMES`). The files are not edited; the numbers are
unchanged and were always labelled correctly inside.

### 3.13 OPERA and CausalMM, read from their own repositories

Both were cloned to `refs/` and read. They are not in the same position, and
an earlier version of §3.9 was wrong about CausalMM.

#### CausalMM — portable, and the same decoding rule again

[The-Martyr/CausalMM](https://github.com/The-Martyr/CausalMM) ships a
complete LLaVA-1.5 + POPE pipeline (`llava-1.5/experiments/`). The eval
script imports `causalmm_cf.causalmm_sm`, and that sampler is:

```
gamma   = 1.0                       # llava1.5_eval.bash; the code default is 0.5
epsilon = 0.6                       # ditto; the code default is 0.1
cutoff  = log(epsilon) + z_obs.max()
diffs   = (1 + gamma) * z_obs - gamma * z_cf
cf_logits = diffs.masked_fill(z_obs < cutoff, -inf)
next    = multinomial(softmax(warp(cf_logits)))
```

That is **the rule VCD, ICD and SID already share** (§3.12), with `gamma` in
`cd_alpha`'s slot and `epsilon` in `cd_beta`'s. Four of the five compared
methods are one formula; only the negative branch differs. That is the single
most useful structural fact in this comparison and neither paper states it.

The negative branch is a **vision counterfactual**, built in
`llava/model/multimodal_encoder/cf_encoder.py`: run CLIP with
`output_attentions=True`, take layer `-2`'s attention, shuffle it, bilinearly
resize to 24x24, average over heads, multiply element-wise into the
**pre-projector** image features, then project. Pre-projector matters — the
projector is a 2-layer GELU MLP, so scaling before it is not scaling after.

Portable to our stack with no transformers-internals surgery.
`LlavaModel.get_image_features` on transformers 4.57.6 has the identical
shape (`hidden_states[config.vision_feature_layer][:, 1:]` ->
`multi_modal_projector`, and `vision_feature_layer` is already `-2`) and
forwards `**kwargs` to the vision tower, so `output_attentions=True` is
threadable. CausalMM patches `GenerationMixin.sample` at runtime against
*stock* transformers 4.31.0 rather than vendoring a fork, which is why it
ports and OPERA does not.

Protocol differences to carry into the table: `top_p=1` (no nucleus
truncation at all, against 0.9 everywhere else), `max_new_tokens=1024` on
POPE, `seed=55`, and `epsilon=0.6` — a far tighter plausibility cutoff than
the 0.1 the other three use.

Three defects, recorded rather than fixed:

* `cf_encoder.get_attentions` reshapes to a square grid only when the
  attention is 3-D. CLIP's is 4-D `[B, H, 577, 577]`, so the reshape is
  skipped and what gets bilinearly resized to 24x24 is the full 577x577
  attention **matrix**, not a spatial map. It runs; it is not what the
  README's visualisations show.
* `edit_attention`'s `shuffle` calls `torch.randperm` with no generator, so
  the counterfactual is drawn from global RNG and is seeded only by
  `set_seed`. Same class of defect as VCD's noise (§3.12).
* the `type(images) is list` branch returns `image_features`, which that
  branch never assigns — a `NameError`. Dead on POPE, which passes a single
  tensor.

`causalmm_cf/causalmm_mm.py` (the two-counterfactual version,
`diffs = (1 + 2g) z_obs - g z_cf_v - g z_cf_l`) is **dead code in this
release**: it unpacks a 3-tuple from a forward that returns 2, and no shipped
model provides the third. The published POPE numbers are the single-branch
version.

**Measured here: the counterfactual branch is close to image-blind.** This is
the finding worth carrying into the write-up, and it is structural rather
than incidental.

Each CLIP attention row is a softmax over 577 keys, so it sums to 1 and the
whole `[577, 577]` matrix sums to 577 — mean entry `1/577 = 0.00173`. A
permutation cannot change a mean; neither bilinear resizing nor averaging
over heads does. So the per-token gain
`cf_weight_features` applies has mean `1/577` whatever the image is.
Measured on a real (tiny-width, correct-geometry)
`LlavaForConditionalGeneration`: `cf_gain_mean = 0.00172`,
`cf_gain_std = 0.00015`, i.e. an almost uniform downscale by ~580x. After the
projector's GELU an input scaled down 580-fold is near-constant, so `z_cf` is
approximately "the model with no image at all".

So CausalMM's negative branch is, numerically, a **blind branch** — not the
spatially-shuffled read its figures show. That is arguably faithful to the
paper's stated intent (it is about *modality prior*-induced hallucination, so
removing the visual contribution is the point) but it is not the mechanism the
method is described by, and it makes CausalMM much closer to a language-prior
contrast than to VCD's noised-image contrast. Asserted in
`tests/test_kernel_equivalence.py::TestCausalMMBitEquality` and
`tests/test_tiny_llava.py`, so if a port ever stops having this property the
suite fails rather than the table quietly changing.

**One deviation forced by the stack.** On transformers 4.31.0, which CausalMM
targets, `output_attentions=True` silently fell back to eager attention. From
~4.52 onward it does not: SDPA warns once and returns `None`, so a direct
port builds its counterfactual out of nothing. `causalmm_vision` therefore
switches the vision tower to eager for that one call and restores it
afterwards, and raises if the attentions still come back `None`.

The consequence is recorded rather than hidden: eager CLIP does not agree
with SDPA CLIP in the last bits, so **this arm's clean branch is not
bit-identical to the clean branch of every other arm in this repo**.
Measured: `3.9e-3` max absolute difference in the selected hidden states in
fp16 on a stand-in, `1.1e-8` in fp32. It is faithful — CausalMM derives both
branches from one eager forward — and it is called out because this
project's own result is that a last-bits fp16 difference in attention moved
POPE by 2.8 points, so "the last bits do not matter" is the one assumption
not available here. `describe()` records
`clean_branch_bit_identical_to_other_arms: false` on every CausalMM run.

#### OPERA — not portable, and half of it is inert on POPE

[shikiw/OPERA](https://github.com/shikiw/OPERA) implements
`opera_beam_search` in **559 lines** (lines 3116-3675) inside a *vendored,
patched* `transformers-4.29.2/src/transformers/generation/utils.py`. Unlike
CausalMM's runtime patch, it rewrites beam search against 4.29.2 internals,
and it **snapshots and restores** `input_ids`, `beam_scorer`, `beam_scores`,
`beam_indices`, `cur_len` and the KV cache in order to roll back. None of
that survives transformers 5.x, where the scorer and the cache are different
objects. It also needs the full post-softmax
`[B x beams, heads, q+1, kv+1]` attention of the last layer — the tensor our
own §Three-traps note records as coming back as `()` from ~4.52 onward.

`opera_greedy_search` is `assert False, "OPERA does not support beam=1 in the
current version"`. Beam >= 2 is mandatory, so this arm can never share the
nucleus-sampling protocol the other arms are compared under, which is why the
paper's own Table 1 footnote has to treat it separately.

Cost is **not** the blocker, contrary to what we assumed. Per decode step it
runs 1 main forward plus `num_attn_candidates = 5` speculative re-forwards at
batch `= 5` beams, so roughly 6x per token: about 2.8 h for POPE-adversarial
and 7-10 h for a 500-caption CHAIR run, per seed. That fits a session.

**On POPE, OPERA's second contribution cannot fire.** `pope_eval.py` sets
`max_new_tokens=10`, and two independent conditions then exclude rollback:

1. `penalty_scores = -attn_scores if cur_response_lens <= 10 else
   rollback_scores`, and `cur_response_lens = attn_local.shape[-1]` is the
   number of response tokens so far, so it never exceeds 10. The rollback
   score is never selected.
2. the rollback gate is
   `all((rollback_loc_gather == rollback_loc).sum() > threshold)` with
   `threshold = 15`, over a tensor with one column per step — at most 10
   columns, so the count cannot reach 15. And the body opens with
   `if rollback_loc < 10: assert False`.

The retrospection-reallocation half — the paper's named second mechanism —
is reachable only on CHAIR, where `max_new_tokens=512`.

**But "beam search plus a bonus on the log-probs" is NOT what the rest of it
does.** An earlier version of this section said so and was wrong; the
corrected rule, verified line by line, is:

```
l        = logits[:, -1, :]                  # RAW, no log_softmax yet
(l5, t5) = topk(l, num_attn_candidates)      # top-5 by raw logit, per beam
l5      += penalty_weights * attn_image(t5)  # the bonus, in LOGIT space
vec      = full(-999.);  vec[t5] = l5        # the rest of the vocab is discarded
scores   = log_softmax(vec) + beam_scores    # normalised over the five
→ global topk(2*num_beams) over the 25 survivors → beam_scorer.process
```

Three consequences, each of which changes which sequences are reachable:

1. the bonus lands on **logits**, where top-5 LLaMA gaps are O(1)–O(5), not
   on log-probs;
2. `log_softmax` normalises over **each beam's own five candidates**
   (`utils.py:3555-3557`, the `except:` branch — which is the *normal* path,
   because the `try` ends in `else: assert False`). Every beam therefore gets
   a different additive constant `LSE_full(l) − LSE_top5(l+a)`, and beams are
   then compared in one global `topk`, so a per-beam constant is not
   harmless: a beam whose top-5 covers 0.99 of the mass and one covering 0.50
   differ by ~0.69 nats, larger than the bonus itself;
3. the candidate set is **hard-restricted to five per beam**. Stock beam
   search takes the best `2 * num_beams` over `num_beams * vocab_size` and
   can take ten tokens from one beam; OPERA cannot see past each beam's
   fifth.

`-999.` and not `-inf` is load-bearing: after `log_softmax` the 31 995
discarded tokens sit at a finite ≈ −1009, so when `process` banks an EOS
candidate and walks further down the `2 * num_beams` list, a discarded token
can be promoted into a live beam. At step 0 only beam 0 is live, so five of
the ten `topk` entries are junk by construction.

#### OPERA, ported — what we run and what we measure

`src/foxrepro/fox/opera.py` implements the above on this stack, with its own
beam bookkeeping (`transformers`' `BeamSearchScorer` is deprecated and
removed in v4.62.0, and this project must keep running on one environment;
ours is asserted equivalent to the library's while the library still has
one). Three things that forced the 559-line fork in 4.29 are library calls
here, because this project already replaced the decode loop:

* the last layer's attention comes from our registered attention function,
  not `output_attentions=True` — which on this stack returns an empty tuple
  under SDPA and would otherwise retain all 32 layers (3.94 GB at 5 beams on
  a 620-token prefill);
* `DynamicCache.crop` undoes a speculative forward. **This is mandatory, not
  an optimisation**: on 4.29 the KV cache was an immutable tuple and
  `torch.cat` built new tensors, so the release could discard the extended
  copy. `DynamicCache.update` mutates in place, so without the crop the five
  candidate passes would corrupt the search's own cache — and it would still
  produce fluent text. Asserted in `tests/test_opera.py`;
* the beam permutation is `DynamicCache.reorder_cache`.

**Labelled greedy, and the paper agrees.** Beam search is deterministic, so
OPERA has no sampling variant — and the paper prints the *identical* OPERA
numbers in Table 1 (sampling) and Table 9 (greedy). One run per split is the
whole arm, and `decoding="sampling"` is refused.

**`max_new_tokens=10`, not our POPE default of 1**, because that is what
`pope_eval.py` passes. With beam search the length is not cosmetic: the
search scores whole sequences, so the first token depends on how many follow
it. Both lengths are below the rollback threshold, so both are rollback-free,
but only 10 is OPERA's protocol. `opera_generate` refuses any length at
which rollback would become reachable rather than running OPERA with half its
method missing under OPERA's name.

**The measurement that decides whether the penalty does anything.** The bonus
is added in logit space, so what matters is its spread across the five
candidates against the spread of their raw logits. Every run records
`penalty_spread_mean`, `logit_spread_mean` and
`n_steps_penalty_changed_top`. On a correct-geometry but randomly
initialised stand-in the spread is 3e-4 against a logit spread of 0.1 — 300x
smaller, and the penalty never moved the top candidate. The structural reason
is that the row is normalised over all keys and 576 of 586 are image tokens,
so `attn_image` ≈ 1 for every candidate and they differ only in how much the
next token attends to the handful of text keys. **This is not yet a claim
about the real model** — a trained LLaVA has attention sinks and far more
structure — but if the real run shows the same ratio, then OPERA's
over-trust penalty is inert on POPE and its published number is beam-5 with a
truncated candidate set. The diagnostic is in the record so the run answers
it rather than an argument doing so.

**The real model answered, 2026-11-09, and it is the same answer.** On
LLaVA-1.5-7B fp16 over POPE-adversarial's 3000 questions the penalty spread
is **0.067** against a logit spread of **9.099** — 136x smaller rather than
the stand-in's 300x, so the trained model's attention sinks do narrow the
gap, and not nearly enough to matter. Over 3000 questions the penalty moved
the top candidate at **one step**. The accuracies agree to within 0.10
points across all three arms (79.10 / 79.03 / 79.13; `VALIDATION.md` V5), so
at **one token** OPERA is beam search and the paragraph above is a
measurement rather than a prediction.

One token is also where plain OPERA gains nothing: 79.10 against a greedy
baseline of 79.03, **+0.07**. So that decomposition settles what Fox's
Table 1 cell contains -- Fox re-ran its compared methods in its own 1-token
harness -- and settles nothing about OPERA itself. **At 10 tokens the same
implementation scores 81.13, +2.10**, which is the length argument above
turning out to be the dominant effect: the penalty gets ten steps to act in
rather than one, and the search scores ten tokens of sequence rather than
one. The two controls were re-planned at 10 tokens, on 1500 questions, to
split that +2.10.

**They ran, and there is nothing to split.** At 10 tokens plain OPERA,
`opera_no_penalty` and `opera_unrestricted` all score **81.47** on the shared
1500, **+2.33** over the greedy baseline. `no_penalty` agrees with plain
OPERA on 1496/1500 answers (2 informative pairs, 1-1) and `unrestricted` on
1474/1500 (24 pairs, 12-12). **OPERA's POPE gain is beam search**, now
measured at the length where the method gains something.

The mechanism statement needs one correction though, and it is a correction
away from this file's own earlier wording. At 10 tokens the penalty is **not**
numerically incapable of acting: the spread ratio falls from 136x to 96x and
it moves the top candidate in about 0.9 % of steps -- roughly 274 times over
3000 questions, against **one** step at 1 token. So it acts. Of those ~274
interventions, 4 survive to the final answer and they split 1-1. The penalty
reorders candidates that the beam recovers from anyway, which is a different
claim from "the penalty cannot act" and the only one the 10-token data
supports.

#### The rollback, audited but not built

CHAIR needs it, and three findings say it should not be ported literally.

**The trigger is driven by fp16 saturation, not by attention.** The
column-product score is

```python
attn_local = scale_factor * attn_local                  # x50, fp16
local_score = 1e-7 * attn_local[..., j:, j].prod(-1)    # up to 512 factors, fp16
attn_local_scores[..., j] = local_score.to(torch.float32)
```

and that last cast is a **no-op**: the destination buffer is declared
`dtype=torch.float16`. fp16's smallest subnormal is 5.96e-8, so after the
`1e-7` scaling any column whose product falls below ≈0.30 rounds to exactly
zero, and anything below ≈610 lands in the subnormal range with ~3
significant bits — ties everywhere, and `max(-1)` returns the lowest-indexed
tie. At the other end, a column with a steady factor of 25 (raw attention
0.5) overflows to `inf` after only **four** factors. And on CHAIR, where
`cur_response_lens > 10` makes the rollback score the penalty itself, an
`inf` drives all five candidate logits to `-inf`; after the `-999` mask
`log_softmax` then **prefers the 31 995 discarded tokens** (≈ −10.4) over
every real candidate. That failure mode is a few multiplications away from
the released configuration.

**One index looks wrong.** `reject_token_pos_gather` is indexed by response
*position* everywhere (`[rollback_pos]`, `[rollback_pos+1]`) except
`utils.py:3481`, which clears from the tail of the fixed 512-slot list
(`[-1]`, `[-2]`, …). Stale rejections at the discarded positions are never
cleared and slots near 511 are clobbered.

**`DynamicCache.crop` cannot substitute for the rollback's re-prefill.** The
release restores `history_states[-2]["input_ids"]` and **re-prefills from
scratch**, vision tower included (`utils.py:3497-3528`). Cropping is not
equivalent: cache rows at step *T* are in the *current* beam order, so
cropping row *i* yields current-beam-*i*'s ancestor prefix — and if all five
current beams descend from one old beam, the other four length-*L*
hypotheses are gone. Re-prefilling is also the only thing that keeps the
cache consistent with the restored hypothesis container.

Plus two resource walls: `attn_previous.data.cpu()` every step for up to 512
retained states is ≈3.3–6.6 GB of host RAM, alongside 512
`copy.deepcopy(beam_scorer)`; and the rollback re-prefill is paid in full
each time.

**So when CHAIR is run, it will be run as two arms** — the fp16 product as
released, and the same product accumulated in log-space fp32 — on the pattern
this project already uses for `fox` vs `ulp_control`. If they agree, the
saturation does not matter; if they disagree, "OPERA's rollback trigger is an
fp16 artefact" is a finding of the same kind as the γ result, not a bug we
have to pick a side on.

### 3.14 Validating what we ported

Five of the six compared methods were implemented here from their own
source. Bit-equality against a transcribed reference proves *our code ==
their code*; it does not prove *their code == their paper*, and it proves
nothing about whether a mechanism is active on the real model.

`docs/VALIDATION.md` pre-registers six arms that would fail if a method were
not doing what it claims — the negative branch run alone, the `alpha = 0`
collapse, SID's direction and selection controls, CausalMM's
constant/blind controls, OPERA's three-way decomposition, and a per-run
divergence rate. They cost 6.8 h, they run **before** the table rows, and
their readings are fixed in writing before any number arrives.

### 3.15 The host's transformers is not the development box's

`requirements-trackB.lock` pins `transformers>=4.57.6,<6` on purpose: the
capability the project needs is `AttentionInterface`, and forcing a
downgrade on a host that already has a working stack costs an hour per
session for nothing. The price of that choice is that the host's build
moves, and it moved: **Kaggle shipped transformers 5.16.1 and Python 3.13
against a development environment on 4.57.6 and Python 3.9**, and 5.16.1 is
not installable on 3.9, so the suite cannot be run against it locally.

Three things differed, and all three cost GPU time in round 6 session 1:

1. **`LlavaModel.get_image_features` changed shape.** 4.57.x returns the
   per-image list and `forward` cats it; 5.16.x returns the vision-outputs
   object with that list on `.pooler_output` and `forward` unwraps it.
   CausalMM's wrapper returned the 4.x shape and died inside transformers
   with `'list' object has no attribute 'pooler_output'`. It now reads
   `LlavaModel.forward`'s own source to decide which to return — a version
   number would have to be mapped by hand and would be wrong the next time
   the contract moves, and both versions' return value has a
   `pooler_output` attribute, so duck-typing cannot tell them apart.
2. **`transformers.generation.beam_search` is gone in 5.16.x.**
   `BeamSearchScorer` cannot be imported on the host at all. OPERA already
   carries its own bookkeeping, asserted equivalent to the library's where
   the library still has one, so this cost nothing — but it would have been
   a session.
3. **`device_map="auto"` shards the model and index tensors do not travel.**
   `FoxState.sid_rows` was a tensor on `next(model.parameters()).device`,
   i.e. cuda:0, while layer 16's attention mask is on cuda:1:
   *indices should be either on cpu or on the same device as the indexed
   tensor*. Fox never hit it in 165 jobs because its window is layers 2–15,
   which all land on the first shard; SID masks from layer 2 to the last and
   failed on its first forward. Both `sid_rows` and `branch_rows` are now
   plain tuples of ints, which index correctly on every shard.

A fourth, unrelated to versions: VCD's noise generator lived on the model's
device while `pixel_values` were still the processor's CPU tensors, so
`normal_(generator=cuda_gen)` on a CPU tensor was refused. The contrastive
methods now move their inputs to the model device once, before the negative
branch is built.

Checked against the 5.16.1 source and unaffected: `DynamicCache.crop`,
`.reorder_cache` and `.get_seq_length` (OPERA's cache plumbing),
`LlavaConfig.image_token_index`, and `LlamaAttention`'s `layer_idx` /
`num_key_value_groups` / `head_dim`.

`scripts/probe_host.py` now carries a `transformers_contracts` check that
verifies each of these in milliseconds, so a host whose build has moved
again fails in the setup cell rather than twenty minutes into a session.
And `tests/test_tiny_llava.py` now runs a real `forward` with the CausalMM
wrapper active rather than only asserting the shape it returns — a shape
assertion just restates the branch, while a forward fails on any version
whose contract the wrapper does not meet.

### 3.6 Hyperparameter defaults come from the code, not the paper

They disagree. We default to the code, and record which source was used.

| | code (ours) | paper §C.2 | author's leftover log path |
|---|---|---|---|
| layer window | **2–15** | 2–13 | 2–15 (`2到15`) |
| head ratio `k` | **0.35** | 0.45 | — |
| `τ_JS` | **0.15** | 0.2 | 0.1 |
| `γ` | **10000.0** | never stated | — |
| `α` | 2 | 2 | — |
| `β` | 0.1 | 0.1 | — |

`γ` is a hardcoded local literal, absent from the paper's own list of "four key
hyperparameters" despite Algorithm 1 taking it as an input.

### 3.7 Two released defects we do fix

- `pope_eval.py:394` wraps generation in `torch.enable_grad()`. The cache then
  retains `grad_fn`, so memory grows linearly in generated tokens — survivable
  for POPE's 1 token, an OOM for CHAIR's 512 on any GPU in our range. We use
  `inference_mode()`.
- `--use-cd` is `store_true` with `default=True`, so it can never be turned
  off and the `--use-vcd` branch is unreachable. Our modes are explicit flags.

### 3.16 InstructBLIP-vicuna-7b: a different method on a different layout

**Decision on `tau_JS` (2026-11-22).** The paper states `tau_JS = 0.2` for
InstructBLIP, but the release's InstructBLIP code never passes it on, so the
release effectively ran 0.15. Rather than pick one, both are run: the default
arm at 0.15 (what the release did) and a `tau_js=0.2` Fox arm on the same
splits, seeds and baseline (what the paper says). The gap between ours and the
paper's InstructBLIP columns is then reported for both readings, and the
distance between the two readings is itself a measurement.

Added so the paper's InstructBLIP columns (Tables 1, 2, 8; Figures 16, 21) can
be run. It is **not** LLaVA's Fox on another checkpoint, and it has its own
method names, `sampling_instructblip` and `fox_instructblip`, so nothing that
groups records by method can pool it with `fox`. Every number below was read
from the release (`github.com/Cc2021start/Fox`, cloned read-only) unless it is
marked as ours. Nothing here has been run on the real 7B checkpoint.

**Layout.** HF's `InstructBlipProcessor` writes 32 `<image>` ids *before* BOS,
and the release does the same (`torch.cat([inputs_llm, inputs_embeds], dim=1)`,
`blip2_vicuna_instruct.py:444`). The image span is therefore keys `[0, 31]` and
no key precedes it: `m_sys` is identically 0. `fox_generate` refuses
`use_m_sys=True` when the span starts at key 0, because with every head scoring
0 the top-k is an index-order tie-break that looks like a result.

**Scoring.** The release's InstructBLIP copy ranks heads by `H_vis` of the last
query alone (`score_all = metrics["HVis_tail"]`, `modeling_llama.py:219`); it
never uses `m_sys`. Eq. 4 is not implemented there. Ours: `use_m_sys=False`,
recorded as `use_m_sys: false` in every run's `resolved_hyperparameters`.

| | InstructBLIP copy (ours, default) | LLaVA Fox (ours) | source |
|---|---|---|---|
| layer window | **4–10** inclusive | 2–15 | `modeling_llama.py:197` |
| head ratio `k` | **0.4** | 0.35 | `:221` |
| `γ` | 10000 | 10000 | `:222` |
| `τ_JS` | **0.15** | 0.15 | `vcd_sample.py:177`, see below |
| `α`, `β` | 2, 0.1 | 2, 0.1 | `vcd_sample.py:174-176` |
| image span | `[0, 31]`, width 32 | `[~40, ~616]`, width 576 | processor |

**`τ_JS` disagrees with the brief for this port.** The brief said 0.2. The
InstructBLIP path never passes `JS_threshold`, so `vcd_sample.py`'s default,
0.15, is what the release ran; 0.2 is the paper's figure for LLaVA and appears
nowhere in the InstructBLIP code. We follow the code (§3.6). If the paper's
InstructBLIP columns were produced with 0.2, `--tau-js 0.2` runs it.

**Prompt (read, not assumed).** `INSTRUCTION_TEMPLATE["instructblip"]` is
`"USER: <ImageHere> <question> ASSISTANT:"` and `generate` strips the
placeholder before tokenising (`:293`, `:297`), so the model sees
`"USER:  <question> ASSISTANT:"` with **two** spaces and **no system message**
(`instructblip_eval.yaml` has none). The Q-Former is given the same string.
That is `INSTRUCTBLIP_WRAPPER`. Runs use `--system-prompt none`; the method
refuses `llava_v1`. Decoding is the same as LLaVA's: nucleus sampling,
`top_p=0.9`, `T=1`, 1 new token on POPE, 512 on CHAIR. *Not verified:* that the
Hub tokenizer reproduces the release's token ids for that string; both are
SentencePiece-LLaMA, but no real tokenizer was run.

**Where we do not follow the release.**

- The release hard-codes `key_position = {"image_start": 1, "image_end": 32}`
  for InstructBLIP (`blip2_vicuna_instruct.py:458`), i.e. keys `[1, 32]`:
  31 image tokens and the BOS key. The image is at `[0, 31]`. We use the true
  span, so our `H_vis` is over 32 image keys, theirs over 31 plus BOS. The
  effect on head selection is unmeasured; a one-key shift would have to be
  added to reproduce theirs exactly.
- The released `pope_eval.py` cannot run InstructBLIP as published: it loops
  over `model.llama_model.model.layers` (`pope_eval.py:349`), an attribute `blip2_vicuna_instruct`
  does not have (it is `llm_model`). The InstructBLIP POPE numbers in the paper
  therefore came from a script we do not have.
- The release's `print(f"Applying intervention at layer: ...")` is not ported.

**Ours, not the release's.**

- *Decode loop.* HF's `InstructBlipForConditionalGeneration.forward` re-runs
  the ViT and Q-Former and re-scatters the 32 features on every call and takes
  no `pixel_values=None`. `fox_generate` therefore runs the prefill through the
  wrapper and every later step through `model.language_model` with the KV
  cache. A test pins that this agrees token for token with recomputing the
  whole sequence each step.
- *The patch does not reach the decoder through `from_pretrained`.* Measured on
  transformers 4.57.1: `attn_implementation={"text_config": "fox", ...}` leaves
  InstructBLIP's inner LLaMA on `sdpa`, because the wrapper builds it from
  `config.text_config` after the dict has been dropped, while LLaVA, Qwen3-VL
  and InternVL3 are unaffected. `load_backbone` now reads the decoder's own
  config after loading, re-applies the dict with `set_attn_implementation`, and
  raises if it still is not `fox`. Without that a whole benchmark would run the
  unpatched model and report a plausible number. The Q-Former is pinned to
  `eager` and never to `fox`; the ViT stays on `sdpa`.
- *Checkpoint.* The plan uses the pre-cast fp16 copy (§3.10),
  `Salesforce/instructblip-vicuna-7b` (the fp16 copy was never uploaded; see 3.10). No
  quantized checkpoint is acceptable.
- *Scope.* Only plain sampling and Fox are ported. VCD, ICD, SID, CausalMM and
  OPERA on InstructBLIP are refused by `fox_generate` rather than silently
  running the LLaVA plumbing. Shikra is not ported.
- *Plan.* `table1_instructblip` and `table2_instructblip`: POPE
  random/popular/adversarial and CHAIR `val2014_500`, sampling and Fox, seeds
  42–44 (24 jobs). Three seeds, not ten: the question is whether the column is
  in reach. Their cost is an **estimate** (`INSTRUCTBLIP_COST_FACTOR`: 0.5x
  LLaVA's on POPE, 0.9x on CHAIR, arithmetic on token counts and not a timing),
  and `measured_cost` buckets them apart from LLaVA so a calibration does not
  mix the two. `k`, `τ_JS` and the window are left unset on the jobs because
  `pair_key` includes `k` and `τ_JS`, and setting them on the Fox arm alone
  would split it from its own baseline; the method resolves and records them.

### 3.17 Table 11: MMBench and MathVista, written from the benchmarks' definitions

The release ships no script for either (it ships none for MME either), so
there is nothing to transcribe; this harness follows the benchmarks' own
conventions and every choice below is ours. Added with the plan jobs
`table11_mmbench` and `table11_mathvista` (V20). **No run exists yet**; these
are the rules the first run will be scored by, written down before it.

**MMBench (`lmms-lab/MMBench_EN`, split `dev`, 4377 questions).**

| choice | what we do | why it can move a number |
|---|---|---|
| split | `dev` only. `test` withholds its answers (they are scored on the organisers' server) | the paper's MMBench number may have come from the server on `test`; ours cannot |
| protocol | **single pass**, accuracy over questions as asked. NOT MMBench's CircularEval, which re-asks each question under rotated options and counts it right only if every rotation is | CircularEval scores are systematically lower than single-pass ones; Table 11's 63.1 is in the range of single-pass LLaVA-1.5 results, but the paper does not say |
| prompt | LLaVA's own MMBench script: `<hint>\n<question>\nA. ..\nB. ..\nAnswer with the option's letter from the given choices directly.`; hint omitted when absent; the dataset stores an absent hint or option as the string `"nan"`, which the fetcher turns into "absent" | the wording is the largest known lever on MMBench for LLaVA |
| `max_new_tokens` | 8 (letter, EOS; room for "The answer is B") | no released value to match |
| decoding | the project's default for every table, sampling at `top_p=0.9`, `T=1` (the paper's row is called "Sampling"); not greedy | a sampled first token on a 4-way question is noisier than greedy |
| answer extraction | rules, not GPT. See below | MMBench's own pipeline falls back to an LLM; some answers the rules call unparsed it would recover |
| perception / reasoning | on the dataset's own `l2-category`: perception = `coarse_perception` + `finegrained_perception (instance-level)` + `(cross-instance)` (1117 + 1147 + 541 = 2805 questions); reasoning = `attribute_` + `logic_` + `relation_reasoning` (708 + 419 + 445 = 1572). Counts are the real split's. A group is accuracy over its questions | an l2 value outside both groups makes the scorer **raise** |

**MathVista (`AI4Math/MathVista`, `testmini`, multiple choice only).**

| choice | what we do | why it can move a number |
|---|---|---|
| scope | only `question_type == "multi_choice"` (the dataset's spelling; 540 of the 1000 `testmini` rows, all with `answer_type == "text"`). The 460 free-form rows are excluded | **the paper's 19.53 / 20.16 were presumably measured on a mixed subset.** Multiple choice has a ~30% chance floor (a random-letter echo run scores 29.6%), so our baseline accuracy will NOT be near 19.53 and levels are not comparable; only the Fox-minus-baseline difference is |
| subset | a fixed random **500 of the 540**, `random.Random(2229).sample` over numerically sorted pids, recorded in every manifest row and refused by the loader if it differs. Every arm and seed scores the same 500 | the paper's 500 are not published; any 500 is ours |
| prompt | the dataset's own `query` field, verbatim ("Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end. Question: ... Choices: (A) ...") | |
| `max_new_tokens` | 64 (the prompt asks for the letter "at the end", which invites a short derivation) | no released value to match |
| answer extraction | **rules, where the official protocol uses an LLM.** This is the same dependency that makes Table 5 irreproducible; the multiple-choice restriction is what makes a rule extractor honest at all. **Coverage (the parsed fraction) is reported with every score**, and accuracy over parsed answers only is reported beside it | an LLM extractor would recover some answers the rules reject, so a low coverage means the number understates the model by an unknown amount |
| gold | `answer` is the choice TEXT, so gold = the letter(s) whose text equals it. One of the 500 repeats its gold text; either letter is accepted | |
| the paper's numbers | 19.53 and 20.16 are not multiples of 0.2, so they are not single-run accuracies over 500 questions (probably means over seeds, or a different n) | we cannot tell which |

**The extraction rule** (`bench/multichoice.py`; the first that fires wins, and
the rule is stored per row and tallied in the metrics):

1. `exact`: the whole answer is one valid option letter ("B", "(b).").
2. `lead`: an uppercase valid letter followed by a delimiter or the end
   ("B. a dog", "(C) 97"). A space is not a delimiter, so "A bird is shown"
   does not read as option A.
3. `phrase`: the LAST "answer is X" / "option X" / "choice X" with X an
   uppercase letter followed by a delimiter or the end.
4. `tail` (MathVista only): a valid uppercase letter as the last token.
5. `text`: exactly one option's text is the whole answer or appears in it
   delimited by non-alphanumerics ("1970" does not contain "97").

A letter the question does not offer is `invalid_letter`; anything else is
`none`. **Both are wrong answers and are counted** (`unparsed`,
`unparsed_fraction`, `rule_counts`), never dropped: dropping would let a model
raise its score by refusing to answer.

**Verified:** the real splits were fetched through `datasets` and the whole
chain (fetch, load, `generate --dry-run`, `evaluate`, `report`) run on them
with the CPU `echo` method: 4377 and 500 records, every echo letter parsed,
accuracies at chance (26.3% and 29.6%). **Not verified:** any model output.
The extractor has never seen a real LLaVA-1.5 answer, so the first run's
`rule_counts` and `unparsed` are the first evidence of whether these rules
fit. If coverage on MathVista is low, change the rules **and record it here**
rather than after looking at which direction moves Fox.

### 3.18 Figure 6 / Table 5: a substitute judge, our own 50 images, both orders

Harness: `src/foxrepro/bench/gpt_judge.py`, `scripts/gpt_judge.py`, tests in
`tests/test_gpt_judge.py`. Authorised by the user (OpenAI key, COCO val2014
images and our captions sent to OpenAI for this purpose only).

| item | paper | here | why it can move a number |
|---|---|---|---|
| judge | GPT-4V (deprecated) | a current OpenAI vision model; the exact id is recorded in every output file (default pinned snapshot `gpt-4.1-2025-04-14`, a non-reasoning model so `temperature=0.2` and `max_tokens=512` can be sent as specified) | a different instrument: absolute scores are not comparable to the paper's, only Fox-minus-Sampling is |
| prompt, scale, format | Table 5, 1-10, `Accuracy: s1 s2` | the same text, asserted verbatim by a test (and compared to the paper text when present) | none |
| images | "a curated subset of 50" (not named) | **the first 50 CHAIR images by ascending `image_id`** (ids 136..56137), fixed before any call | the paper's curation may have favoured illustrative cases; ours is mechanical |
| captions | "standardized prompt" | archived CHAIR runs, seed 42, sampling, r2 prompt: baseline `m-sampling...v-sampling-5c7c06df` against default `m-fox...v-sampling-5c7c06df` | one seed, one draw of sampling |
| positions | not stated (presumably one order) | **every image judged in both orders**; paper-style (Assistant 1 = Sampling) reported beside the order-averaged result | our addition, to cancel position bias; the paper-style number is still reported |
| backbones | LLaVA-1.5, InstructBLIP, Shikra | **LLaVA-1.5 only** | Figure 6's across-backbone trend cannot be drawn |
| unparseable replies | not stated | counted and listed, excluded from averages, never silently dropped; a reading over zero parsed items is refused | n per reading can differ; it is printed |
| intervals | none | paired bootstrap 90% CI over images | n = 50 is small; treat as complementary evidence, as the paper does |

**Status:** harness and tests done. The first real call on 2026-11-22 was
refused by the API with `insufficient_quota` ("no credits remaining"), so no
score exists yet; nothing was retried. 50 images in both orders is 100 calls.

**Result (2026-11-24, 100 calls, ~$0.39, 0 unparsed).** Accuracy, Fox minus
Sampling: paper order (Assistant 1 = Sampling) −0.50 [−1.02, +0.02]; swapped
order +1.32 [+0.78, +1.86]; order-averaged +0.41 [−0.10, +0.92]. Detailedness:
−0.12, −0.06 and −0.09, all including zero. The two single orders disagree in
sign: the response shown first scores about 0.9 higher, which is larger than any
Fox effect, so a single-order protocol (the paper does not say which order, or
whether it varied it) measures position as much as method. The order-averaged
reading is the one to quote, and it shows no detectable difference. Different
instrument from GPT-4V: the numbers are not comparable to Figure 6, only the
direction and the bias are informative.

### 3.19 Table 10 / Table 3: Qwen3-VL-8B and InternVL3-8B

Sampling and Fox are ported (`sampling_qwen3vl`, `fox_qwen3vl`,
`sampling_internvl3`, `fox_internvl3`), as their own method names so no
aggregation pools them with LLaVA's `fox`. Written against tiny real
`Qwen3VL` / `InternVL` models only (`tests/test_recent_backbones.py`); the
real checkpoints have not run the benchmark loop.

| Item | Paper / release | Here | Why it can move a number |
|---|---|---|---|
| prompt | not stated for these backbones | the model's own `processor.apply_chat_template`, one user turn, image then the instruction (`pope_v0_raw` / `chair_v1`), generation prompt appended; id `hf_chat_template_image_then_text`; each record's `method_extra.chat_prompt` holds the exact string and `describe()` the template's sha256 | the paper may have used a hand-built prompt or another system message |
| hyperparameters | appendix D.1: k=0.4, tau_JS=0.2, alpha=2, beta=0.1, "no model-specific tuning" | exactly those; gamma=1e4 (never stated) | none known |
| layer window | Table 15 manual windows: Qwen3-VL 3-10, InternVL3 3-11 | the same, read as 0-based and inclusive like LLaVA's `2-15`; the paper does not say | an off-by-one moves the window by one layer |
| head score | Eq. 4 | Eq. 4 with `m_sys` (the image is never at key 0 in these templates, so `m_sys` is defined) | none |
| image span | n/a | `locate_image_span` on each sample's real `input_ids`, `expected_width=None`; varies per sample and is recorded as `image_span` | none, but token counts (hence cost and memory) vary |
| VCD, SID rows | in Table 10 | **not ported** | Table 10 can only be read as Sampling vs Fox |
| eos | n/a | `fox_generate` now stops on the union of the text config's and generation config's eos ids | Qwen-family ids differ between the two; for LLaVA the set is `{2}` as before |

Two changes to the shared decode loop (`fox/attention.py`), both neutral for
LLaVA: (1) processor tensors other than `input_ids`, `attention_mask`,
`pixel_values` (Qwen3-VL's `image_grid_thw`) are forwarded on the prefill step,
repeated per batch row like `pixel_values`; (2) decode steps pass
`cache_position`. Without (2), Qwen3-VL's multimodal RoPE applies a delta of 0
to every token after the first: measured on a tiny Qwen3-VL, the first token
matched `generate()` and the next ones did not. With it the loop equals
`generate()` token for token (and equals it for InternVL and, as before, for
LLaVA).

Plan: 24 jobs, priority last, `Job.model_id` set, `k`/`tau_js`/window left
`None` so the Fox arm shares its `pair_key` with its baseline (the method
resolves and records them). Cost = LLaVA's per-sample cost times 1.0, **a
guess**.

### 3.20 Shikra-7B: a port on the modern stack, a session-time merge, and a correction to 3.11

**Status:** code, tests on tiny stand-ins, a smoke entry and 24 plan jobs
exist (not frozen into any round). **Nothing here has run on the real
checkpoint**; the list of what is therefore unverified is at the end.

**What was built.** `src/foxrepro/fox/shikra.py` loads Shikra as three stock
parts behind one wrapper (`ShikraForFox`): a `LlamaForCausalLM` carrying the
Fox attention, a `CLIPVisionModel` (never patched, for the reason every other
backbone's vision tower is not), and a `nn.Linear(1024, 4096)` holding the
checkpoint's two `mm_projector` tensors. `forward(input_ids, attention_mask,
pixel_values, past_key_values, use_cache)` is the signature `fox_generate`
already drives, so the dual-branch loop, the fusion and the sampler are the
shared ones. Methods are `sampling_shikra` and `fox_shikra`: own names, never
pooled with `fox`, because the window, k and tau_js differ.

**The prompt is read from the release, not assumed.** `pope_eval.py:63` and
`char_eval.py:35` (github.com/Cc2021start/Fox, read 2026-11-24):
`INSTRUCTION_TEMPLATE["shikra"] = "USER: <im_start><ImageHere><im_end>
<question> ASSISTANT:"`, CHAIR's question being "Please describe this image in
detail." `minigpt4/models/shikra.py` then replaces `<ImageHere>` by 256
`<im_patch>`, prepends `system_message` with a bare `+` (the string in
`eval_configs/shikra_eval.yaml`, which is byte-identical to the `llava_v1`
system message, so runs keep `system_prompt_id=llava_v1` truthfully) and
prepends BOS. No assumption remains in the template.

**Correction to 3.11: the release's visual keys are the 256 patches, not 258.**
3.11 said the hard-coded `img_s == 33 and img_e == 290` span is
"256 + 2 = 258". The numbers are right and the reading is not: `33` and `290`
are the positions of `<im_start>` and `<im_end>`, and `Uncertain_head.py`
(`_span_visual_text`, `is_shikra` branch) then takes the visual keys as
`[img_s + 1, img_e)` = positions 34..289 = the 256 patches. The markers are in
none of its three regions (system is `[0, img_s)`, text starts at
`img_e + 1`), so its three masses do not sum to 1 for Shikra, which they do
for LLaVA. Here the span is **derived from the `<im_patch>` ids** exactly as
LLaVA's is from `<image>`: `(34, 289)`, width 256, asserted by
`FIXED_SPAN_WIDTH["shikra"] = 256`, with `<im_start>`/`<im_end>` checked to
flank it at splice time. That equals the release's visual set. What differs,
by one key at each end: `<im_start>` (key 33) falls in `[0, img_start)` and so
counts toward `m_sys` (the release's `m_sys` is `[0, 33)`), and `<im_end>`
counts toward `m_txt`. `H_vis` and the head selection's visual set are the
same; only `m_sys` can differ, by the attention mass on one marker key, in a
product `m_sys * H_vis` used to rank heads. The size of that effect is
unknown. Making it exact needs a `sys_end` argument through
`head_scores`/`intervene` (kernel code with a bit-equality oracle behind it),
not done here; the alternative reading, a 258-wide span, was rejected because
it would put two marker keys into `H_vis`, which the release does not.

**The tokenizer detail that makes the markers land on 33 and 290.** The
release runs the 2023 tokenizer, whose `tokenize()` strips the whitespace on
both sides of an added token ("We strip left and right by default",
`transformers/tokenization_utils.py:533`). The current `LlamaTokenizer` does
not: `"USER: <im_start>"` gains a lone `"▁"` token before the marker and a
second after `<im_end>`, which moves the markers to 34 and 291 and changes the
text the model reads. `encode_prompt` therefore splits at the markers, strips,
encodes the two text pieces separately and assembles the ids. Measured against
the real tokenizer files of `shikras/shikra-7b-delta-v1-0708` (vocab 32003,
ids 32000/32001/32002): `<im_start>` at 33, `<im_end>` at 290, 256 patches
between them, for both the POPE and the CHAIR question. The smoke entry asserts
the pair against the release's numbers, so a tokenizer upgrade that shifts them
fails loudly. The stub tokenizer in `tests/test_shikra.py` cannot show this
(it was measured once, by hand); `build_shikra_checkpoint.py --verify` repeats
the measurement on the real files.

**Hyperparameters are the paper's, and the release says something else.**
`fox_shikra` runs layers 3-10 (Table 15, manual), k=0.4, tau_JS=0.2, alpha=2,
beta=0.1, gamma=1e4 (the code's; never stated in the paper), `m_sys * H_vis`
scoring. The release's own Shikra path is the shared LLaVA attention in
`transformers/models/llama/modeling_llama.py:337`: hard-coded
`2 <= layer_idx <= 15` and `SELECT_RATIO = 0.35`, with the comment
"llava2-15/shikra 3-10" beside the condition, so the code that produced the
paper's Shikra numbers may have run 2-15 with 0.35 and the text says 3-10 with
0.4. Not run both ways (the brief fixed the paper's values, as for the 0.2 arm
of 3.16); a `--intervention-layers 2-15 --k 0.35 --tau-js 0.15` job is the
cheap way to measure the distance and is not in the plan. k, tau_js and the
window are left `None` in the jobs on purpose (`pair_key` includes k and
tau_js; setting them on the fox arm would split it from its baseline), and
the method records the resolved values in every record.

**The checkpoint, and how a session gets it.** The Hub holds a delta only.
`load_shikra("shikras/shikra-7b-delta-v1-0708")` merges it itself, in RAM,
with no file written and no token: public repos, `snapshot_download` of
exactly `pytorch_model-*.bin` + indices + tokenizer files for the delta and
`model-*.safetensors` + index for `huggyllama/llama-7b` (its duplicate `.bin`
set is skipped). Nothing is hosted, so the licensing question of 3.10 does not
arise and the private-repo problem does not either. The arithmetic is the
release's `apply_delta.py` (fetched from github.com/shikras/shikra, read
2026-11-24) rewritten to stream: fp16 `+=`; `embed_tokens` and `lm_head`, whose
first axis grew from 32000 to 32003, get the base added to the first 32000 rows
and the delta alone for the last three (the trained marker embeddings);
`model.mm_projector.*` pass through. `tests/test_shikra.py` proves bit-equality
against a literal transcription of that function. The 32 `rotary_emb.inv_freq`
tensors of the delta (325 = 291 + 32 + 2) are dropped: today's
`LlamaRotaryEmbedding` recomputes them in fp32. The 2023 stack loaded the
checkpoint's fp16 buffer instead, so cos/sin differ in the last bits of fp16
rounding; `--verify` prints the measured gap. The same effect exists for
LLaVA-1.5 on the converted `llava-hf` checkpoint and was accepted there.

Footprint, from the Hub file listings of 2026-11-24 (exact bytes):

| item | bytes | GiB |
|---|---|---|
| delta `.bin` shards | 13,485,396,661 | 12.56 |
| `huggyllama/llama-7b` safetensors | 13,476,876,661 | 12.55 |
| `openai/clip-vit-large-patch14` `model.safetensors` (whole CLIP; only the vision half is used) | 1,710,540,580 | 1.59 |
| **fetched per session** | **28,672,813,902** | **26.70** |
| merged weights held in RAM, never on disk | ~13.49 GB | 12.56 |

Disk: the 26.7 GiB sits in the HF cache (`/kaggle/temp`, `FOX_HF_HOME`), below
the 29.5 GiB that `Salesforce/instructblip-vicuna-7b` downloads in the same CI
and that its smoke session (smoke-r29) completed, so on that evidence the disk
holds it; the free space on a session's `/kaggle/temp` was not measured here
(`scripts/probe_host.py` `check_disk` measures it, and now knows the Shikra
sizes). The "~40 GB transient" of 3.11 applies to writing a merged copy to disk
(`build_shikra_checkpoint.py`: 26.7 + 12.6 = 37.7 GiB with sources kept); a
session does not write one. RAM: the merge peaks at the merged 12.56 GiB plus
one tensor, with the 10 GB delta shard memory-mapped (page cache, reclaimable);
if the mmap of the 2023 pickle is refused `_load_bin` reads the shard into RAM
and prints a warning, raising the peak by about 10 GB. Kaggle's T4x2 host RAM
(about 29 GB, recalled, not measured) holds either; that is the number to
check first on the smoke session. Cost: a ~26.7 GiB download per session that
loads Shikra, which is not in the per-sample estimate.

If the session merge turns out not to fit (RAM, disk or time), the least bad
fallback is `scripts/build_shikra_checkpoint.py` once on a big-disk host and a
**private Kaggle Dataset** attached to the notebook, not a private HF repo
(which the owner found unreadable from Kaggle). That only works for sessions
run under accounts that can read the dataset, and it is a redistribution of
LLaMA-derived weights to the extent that dataset is shared; keep it private to
the owner. Not built, and not needed unless the smoke session says so.

**What the plan contains (not frozen).** 24 jobs: POPE x 3 splits x
{sampling_shikra, fox_shikra} x seeds 42-44 (targets `table1_shikra`, feeding
Tables 1 and 8) and CHAIR val2014_500 x the same (`table2_shikra`, Table 2).
Cost factors 0.6 (POPE) and 0.95 (CHAIR) on LLaVA's per-sample cost are
**estimates from token counts**, not measurements; `measured_cost` buckets
Shikra apart (`-shikra`) for `--calibrate`. Figures 17 and 22 (sensitivity and
the Shikra panels beyond the main tables) are not planned: they are sweeps that
only make sense once the base column runs.

**Provenance.** Every record's `describe()` carries: the projector
("linear 1024 -> 4096"), the vision tower and its dtype, `select_layer -2`,
`drop_cls`, 256 patches, the marker ids, the template and where it came from,
the tokenisation rule, the span rule, the merge rule, and from the loader the
delta and base snapshot revisions (Hub commit hashes) and the CLIP id.

**EOS.** The delta's `config.json` says `bos_token_id 0, eos_token_id 1`. The
release generates with the tokenizer's ids (`eos_token_id =
llama_tokenizer.eos_token_id` = 2, BOS 1), and so does this port; the config's
values would stop generation on `<s>`.

**Not verified (stubs only).** (1) That the real delta loads, merges and fits
in RAM and disk on a Kaggle host, including `torch.load(..., mmap=True)` on the
2023 shards. (2) The GPU dispatch in `_dispatch` (`get_balanced_memory` +
`infer_auto_device_map` + `dispatch_model`, one GiB reserved on GPU 0 for CLIP
and the projector): written against accelerate's API on a host that has no
accelerate. (3) That `LlamaForCausalLM.from_pretrained(None, state_dict=...)`
loads 6.7B parameters as it does the toy (it does for the toy, on CPU).
(4) Any number a benchmark would report; the vision tower runs `sdpa` in fp16
where the release ran its eager attention in fp16. (5) Whether `<im_start>` in
`m_sys` moves head selection. (6) The 0.6/0.95 cost factors. The smoke session
(`bash scripts/ci_smoke.sh 4`, i.e. `scripts/smoke_backbone.py --shikra`, case 4
of `kaggle-smoke.yml`) asserts, on the real model: span 256 and
flanked, markers at 33 and 290, the patch firing on layers 3-10 only and not on
CLIP, `m_sys + m_vis + m_txt = 1` with `m_sys > 0`, splice differs from
text-only and from another image, all weights present (key set, marker rows,
projector nonzero), `mode=off` bit-identical to the baseline forward, `gamma=0`
keeps the argmax, and generation returns the same span.

### 3.21 r29 first contact with the real checkpoints: two failures in the Table 10 jobs

Real T4 x 2 logs of r29 (sessions 2 and 3) showed two failures that the tiny
stand-ins of 3.19 could not. Neither changes a recorded number: the jobs
produced no record (InternVL3 Fox: every POPE and CHAIR job, 0 or 1 samples;
Qwen3-VL seed 44: 0 records), and the fixes below were chosen so that no
number the code already produced can move.

**(1) InternVL3 Fox: CUDA OOM in `intervene`.** The processor tiles each COCO
image dynamically and a prompt is 1811-1812 tokens for the common 7-tile
samples (256 tokens per tile) and up to about 3300 for 13 tiles. `intervene`
built an fp32 copy of the whole `[2, 28, Q, K]` logits, added the mask, and
softmaxed it, then the layer did mask + clamp + fp32 softmax + cast on the
same tensor again. At Q = K = 3312 each fp32 copy is 2.34 GiB, on a GPU that
already holds 8.7 GiB of weights. Sampling has no intervention and goes
through SDPA, so it never built those copies.

*Decision: option (b), bound the memory, change no value; do not cap the
tiling.* Capping the tiles (option (a)) would change the model's input for
both arms, force re-running the InternVL3 Sampling jobs that already ran
(the job-times table lists about 140 for each) to keep the arms paired, and move the experiment away
from the checkpoint's own default for a reason that is only about memory. The
paper says nothing about tiling (its text mentions no tile count, resolution
or crop setting for InternVL3-8B), so there is no stated value to match; the
default of the checkpoint is the neutral reading. For the record, the Hub's
`OpenGVLab/InternVL3-8B-hf` `preprocessor_config.json` lists `max_patches: 12`,
`min_patches: 1`, `size 448x448` and `crop_to_patches: false`, yet the
processor that produced the logs tiled (prompts of 1811 tokens), so what the
installed processor does when `crop_to_patches` is false is not what the file
suggests; the token count in each record's `n_prompt_tokens` is the fact to
read. Not investigated further. If a later run must cap the tiles, it needs a
new run-key field so the old names keep meaning "default tiling".

What changed, both gated so that LLaVA-sized tensors take the old code
verbatim:

- `fox/kernel.py::intervene`. The default anchor reads only the last query
  row of the probabilities (`head_scores`), and softmax is row-independent, so
  above `LAST_ROW_ONLY_ELEMS` (2^26 elements) the fp32 copy, the mask add, the
  clamp and the softmax are built for the last row only. Proved bit-identical
  to the full path on random tensors in `tests/test_kernel_equivalence.py`
  (`TestLastRowOnlyMetricCopy`: fp16/fp32/bf16, five modes, with and without
  a mask, decode shape, threshold forced to 0), and the existing proof that
  `intervene` equals `reference_intervene` is repeated on the last-row path.
  The non-default anchors need every row and keep the full path.
- `fox/attention.py::fox_attention_forward`. After the intervention the mask
  add, clamp, softmax and cast run in head chunks (`_masked_softmax_in_head_chunks`,
  about 128 MiB of fp32 per chunk) above `LARGE_TAIL_ELEMS` (2^26) and write
  into a preallocated half-precision `probs`. Every op is elementwise or a
  reduction along the key axis, so slicing heads moves no value; tested equal
  for three dtypes, with and without a mask, over several chunk sizes. The
  `matmul(probs, V)` is untouched.

A rerun of the old commit and this one on a tiny real LLaVA (greedy and sampled, seeds 1-2, Fox on layers 1-2, 6 tokens) gave byte-identical token ids and per-token `d_t`/`lambda_t` traces.
LLaVA's largest layer tensor is about 31M elements (2 x 32 x 700 x 700), under
both thresholds, so LLaVA, Shikra, InstructBLIP and Qwen3-VL (prompt about
210) never enter the new code. `TestMemoryBoundedPathsChangeNothing` also runs
the tiny Qwen3-VL and InternVL3 with both thresholds forced to 0 and requires
the same tokens and the same per-token trace as with them off.

Estimated peak for the 13-tile sample after the change: weights 8.7 GiB +
logits 1.2 GiB + preallocated probs 1.2 GiB + chunk and KV cache under 1 GiB,
about 11.5 GiB of 14.56. **Not measured on a GPU.** If a longer prompt still
overflows, the next step is the same chunking around the `QK^T` matmul.

**(2) Qwen3-VL: `The expanded size of the tensor (419) must match the existing
size (210)` on the first decode step.** Root cause, reproduced on a tiny
Qwen3-VL under transformers 5.16.1, 5.20.2 and 5.24.1 (the versions in the r29
records are 5.16.1 and 5.20.2): the decode step passed the full-length
`attention_mask` (P + 1 columns) together with one new token. In 5.x
`compute_3d_position_ids` derives the positions of a cached step as
`attention_mask.cumsum(-1) - 1` -- a tensor as long as the whole sequence --
and `position_ids + rope_deltas` is then P + 1 rows long, so the rotary
embedding broadcasts the single query to P + 1 queries; the layer's keys are
P (cache) + P + 1 (new). With P = 209 that is 419 keys against a 210-column
mask, the log line. `generate()` never reaches that branch because it passes
`position_ids`. Under transformers 4.57.1 the mask is ignored on that path,
which is why the local suite never saw it. The tiny repro fails on every seed
tried, so the failure is not seed-dependent: the report that seeds
42 and 43 ran is not explained by anything found here (their records are not
in the two zips read, and a different host version cannot explain it, since
both 5.16.1 and 5.20.2 fail the same way). The cause is the decode call, not
the sampled token, and not any state carried between samples or the cache
(the stale-cache reading of 419 = 210 + 209 is the same arithmetic seen from
the other side; nothing is stale).

Fix: for models whose inner model keeps `rope_deltas` (Qwen2/3-VL; LLaVA,
InternVL3, Shikra and InstructBLIP do not), the decode step does not pass
`attention_mask`. Every row is unpadded (all ones), so no information is lost:
the positions then come from `arange(cache_length, ...) + rope_deltas` and the
causal mask from `cache_position`. Verified on the tiny model under 4.57.1,
5.16.1, 5.20.2 and 5.24.1: single-branch greedy `fox_generate` equals
`model.generate` token for token (so the positions are right, not merely
shaped right), and Fox decoding runs. Regression tests
(`TestDecodeStepMaskForMRopeModels`) pin the cause (no full-length mask next
to one new token) so they fail on the old code under any installed version,
and the old code reproduces the real error under 5.x. LLaVA, InternVL3 and
the rest are bit-for-bit as before: the branch is gated on
`hasattr(model.model, "rope_deltas")`.

**Not verified.** Neither fix has run on the real checkpoints or a GPU. A
Qwen3-VL result under 4.57 vs 5.x with the mask omitted is checked only
against `generate()` on a tiny model. The Qwen3-VL jobs and the InternVL3 Fox
jobs of r29 need re-running; the InternVL3 Sampling jobs that completed need
not be (their inputs and code path are unchanged).

### 3.22 Figures 12 and 14: 459 hallucinated samples instead of 1,000, and a prefill `tail = 32`

Added after the r13 data existed, so this is a record of choices made while
reading it, not a pre-registered rule.

| | paper | here |
|---|---|---|
| hallucinated set | 1,000 samples, selection unstated | all **459** false positives of the greedy POPE-adversarial baseline (1000 do not exist in 3000 questions) |
| faithful set | 1,000, selection unstated | four pools reported side by side (all others 2541; correct 2372; right "no" 1041; right "yes" 1331) and 200 balanced 459 vs 459 draws |
| `tail = 32` | last 32 post-image text queries of a "Text-Tail" query set | last 32 query rows of one prefill (question + `ASSISTANT:`) |
| Top-K selection | in-sample, as written | in-sample **and** held out on random half-splits, plus a label-permutation null |
| Eq. 19 z-score | "dataset-level statistics" | mean and sd over the analysed samples |

Why it matters: the AUCs are invariant to class balance but the PR curve and
AP are not (prevalence 0.153 here), and the pool changes the reading, since
all hallucinations are answers of "yes". Nothing recorded in `results/runs/`
was touched. **Not verified:** how the paper chose its sets; whether its
tail-32 differs from ours only by window (our tail-32 mean is 0.226, theirs
0.165). Figure 15 needs a run that does not exist (`fox_diagnose_paired` is
implemented and tested on a tiny model only; no job registers it).

## 4. In the release, absent here

| | Why |
|---|---|
| InstructBLIP path | **Ported, see §3.16.** It is a **second, divergent copy** of the intervention (`minigpt4/models/modeling_llama.py`) that drops `m_sys` from Eq. 4 entirely and uses layers 4–10 with k=0.4. A different method, not a backbone swap. |
| Shikra path | **Ported, see §3.20** (which also corrects the 258-token reading below). **Correction, see §3.11.** Earlier notes here called this a Track A job needing transformers 4.29.2. That was wrong: `ShikraLlamaForCausalLM` has no transformers class, but its decoder is a plain LLaMA-7B and the checkpoint carries exactly two tensors beyond one — `model.mm_projector.{weight,bias}`. It is a port on the modern stack, not a vendored environment. It is still a delta needing `huggyllama/llama-7b` merged in. |
| InstructBLIP path, practically | `Salesforce/instructblip-vicuna-7b` is public and transformers ships `InstructBlipForConditionalGeneration`, so unlike Shikra it can run on the modern stack. Costs: 31.7 GB to download (stored fp32; ~15.8 GB in VRAM once cast, which 2x T4 holds sharded). The work is porting, not hosting — see the row above on the second intervention copy, and note the visual span is **32 Q-Former query tokens**, not 576 patch tokens, so `H_vis` is an entropy over 32 positions instead of 576. |
| FastV / OPERA plumbing | Present in the release as argparse flags, all `default=False`, with no reweighting behind the SID one. |
| HF `logits_processor` | We apply the warpers (temperature, top-p) but not the processor list, which is empty by default for this configuration. |
| `use_phj` | Dead code: set on every layer, read nowhere. |
| A `print()` in the attention forward | Fires 7× per generated token in the InstructBLIP copy. |

## 5. Here, absent in the release

| | |
|---|---|
| A CHAIR evaluator | **The release has none.** `char_eval.py` only dumps captions; there is no `chair.py` and no synonym list. Table 2 is not reproducible from the release at all. |
| A frozen CHAIR image list | `char_eval.py:136` does `os.listdir(args.data_path)`, shuffles it, and keeps the first `--test-sample 500`. `data_path` defaults to `/data-disk/only_test` — a directory that was never published — and `os.listdir` returns filesystem order, so the set depends both on that directory's contents and on the author's filesystem. We ship explicit lists of val2014 ids instead, selected by split name and recorded in every run's provenance. See §3.8. |
| Six intervention arms as flags | `off`, `as_described`, `ulp_control`, `noise_control` — controls the paper never ran — plus `boost_visual` (below) |
| **`boost_visual`** | The experiment the paper's motivation rests on and never performs. It argues from Figure 1(a), an *observational* comparison showing faithful and hallucinated samples carry the same global visual attention mass, that *amplifying* visual attention cannot help. That is the `P(Y\|X)` vs `P(Y\|do(X))` distinction the paper itself takes as its foundation: a quantity uncorrelated with an outcome can still change it when intervened upon. PAI, named in the introduction as the representative of that family, appears **twice in the whole paper** and is in **none of its tables**. This mode adds a positive bias to the image-token keys at the same heads and query positions Fox suppresses, leaving the system keys untouched — the same slice as `as_described`, opposite sign. |
| Three independent dtype axes | weights / pre-softmax logit storage / softmax accumulation. The mechanism lives in the second |
| Resume | Keyed on the run file; tested by actually crashing the process |
| 33 provenance fields per record | including `transformers_path`, `tf32_flags`, `torch_version`, `compute_capability` |
| Wilson / Newcombe intervals, paired bootstrap, McNemar | the release reports no variance at all |
| One shared sampler for baseline and method | so the only difference between them is the intervention |
| A session-resumable 53-job plan | the release has no MME script either, despite reporting MME |

## 6. Unverified, and what would settle it

| Open question | How to close it |
|---|---|
| Does transformers 5.16.1 give the same numbers as the vendored 4.29.2? | Track A on ~50 samples, greedy, fixed seed; diff token ids |
| Is batching the two branches bit-equal to two sequential passes? | Same prompt both ways on GPU, compare logits |
| Is `chair_synonyms.txt` identical to Rohrbach's upstream? | Diff against the upstream file; our SHA-256 is in every metrics file. **Do this before quoting any CHAIR number externally.** |
| Does `--system-prompt none` still show Fox's gain? | It is a planned job; a yes undermines the paper's premise |
