# Key Takeaways

Plain-language observations and learning points worth remembering. Each entry
links to the detailed change in [log.md](log.md) where one exists.

---

## 2026-08-27 — "Only grade the answer" for image models: `completion_only_loss`, not `assistant_only_loss`

**Plain-English version (why the training fix looks the way it does):**

Every training example we feed the model has the **same** first part — the image
plus the identical instructions ("Classify this image... 0=Real, 1=Synthetic,
2=Tampered") — and only the final answer digit changes. There is nothing to
learn from the always-identical instructions, and grading the model on them was
drowning out the tiny answer signal, which made the model collapse to guessing
one class for everything.

We wanted to tell the trainer: **"only grade the model on the answer digit,
ignore the instructions part."** The training tool (TRL) has two switches for
that:

1. `assistant_only_loss` — **refuses to run when images are involved** (this is
   the error we hit on Colab).
2. `completion_only_loss` — **does the same job and works with images.**

The fix was to switch to #2. Using #2 requires handing the data over as two
labeled pieces — a **prompt** piece (image + instructions) and a **completion**
piece (the answer) — instead of one combined blob. So the data was split into
those two pieces and the working switch was flipped on. **Same goal, just the
switch TRL actually allows for image models.**

**How to tell it worked, when re-running on Colab:**
- **Training loss** should keep *changing* (generally going down). If it is stuck
  flat near `0` from the start, something is still wrong.
- **Confusion matrix** at the end: good = numbers spread across the diagonal;
  still-broken = all numbers stuck in one column.

**Detail / code:** see log.md →
[2026-08-27 (follow-up 3) — Colab error fix: assistant_only_loss -> completion_only_loss](log.md#2026-08-27-follow-up-3--colab-error-fix-assistant_only_loss---completion_only_loss).
Related background: the collapse this fixes is described in
[2026-08-27 (follow-up 2) — Anti-collapse](log.md#2026-08-27-follow-up-2--anti-collapse-answer-only-loss--collapse-aware-metrics--majority-voting).

---

## 2026-08-27 — Gentler+longer training is what actually broke the collapse (LR 2e-5→5e-6, steps 78→500)

**What happened:** even with `completion_only_loss` correct, the first real run
still collapsed to one class (fine-tuned confusion = all "Synthetic"; balanced
accuracy 33.3%, the exact one-class floor for 3 classes; "Real" predicted 0% of
the time, so binary was stuck at the 70% base rate).

**The fix that worked (user's Colab run):** lower the learning rate an order of
magnitude (2e-5 → 5e-6) **and** train longer (MAX_STEPS 78 → 500). Result:
3-class 21%→96%, binary 70%→98%, balanced acc 20%→95.8%, macro-F1 0.166→0.959,
and — critically — **Real went 0%→96.7%**, so the model can finally call an image
authentic. Diagonal confusion matrix, no collapse.

**Why (mental model):** high LR + tiny effective batch (batch 1 × grad-accum 4 =
4) + a one-digit target drives the optimizer into the degenerate "always predict
the majority class" solution. A smaller LR takes gentler steps that don't
overshoot into that basin; more steps give the weak one-token signal enough
repetitions to actually learn the classes. Both changes push the same direction,
so they were changed together — the individual contribution isn't isolated.

**Important caveat — what is NOT yet proven:** these numbers are on **n=100
held-out from the SID_Set stream**, i.e. *in-distribution* (same dataset the
model trained on) and a small sample (~±5-8pp noise). The hackathon actually
grades (a) the **WildFake** reference set — different generator (DALL·E) and
different reals (COCO), a cross-dataset test — and (b) **robustness under
real-world transforms** (JPEG/blur/resize/noise/crop). 96% in-distribution does
not guarantee either. Next real checks: run the WildFake validation cell, then a
transformation-robustness eval.

**Signature to remember:** balanced accuracy == 1/num_classes (0.333 for 3-class)
is the fingerprint of single-class collapse, no matter how ok the raw accuracy or
binary number looks.

---

## 2026-08-27 — Cross-dataset reality check: 98% in-distribution → 66% on WildFake (generator generalization gap)

**The result (eval_local_folder on the balanced WildFake test set, 260 imgs, 120
real / 140 fake):** binary accuracy **65.8%** — vs 98% on the SID_Set held-out
slice. The drop confirms the earlier caveat: in-distribution accuracy did NOT
transfer.

**The shape of the error is the point — asymmetric, authentic-leaning:**
- Real images: **88.3%** correct (good authenticity detector).
- Fakes: **46.4%** correct — misses over HALF of AI images (slightly worse than a
  coin flip). On unfamiliar data the model defaults to "authentic."
- Balanced acc 67.4%, so NOT a collapse — it discriminates, just with a bias.

**Per-category (n=20 each, so read tiers not exact ranks; ±~11pp):**
- Clean canonical reals perfect (afhq/church/celebahq/ffhq = 100%), but messy
  web reals get false alarms (laion5b, imagenet = 65% → 35% of real photos
  flagged AI). The web-real false-positive rate matters more for deployment.
- Fake detection tiers: **VQDM 0%** (totally blind to this generator), older/soft
  diffusion weak (ADM 35, DDPM 40, Imagen 45, DDIM 60), Midjourney best (80%).

**What it means:** the model overfit to SID_Set's generator fingerprints and
doesn't generalize to unseen generators — the classic AI-detector transfer gap.
Fixing VQDM + the low diffusion tier is where the points are.

**Two caveats:** (1) this balanced set is NOT the brief's official benchmark
(COCO real + DALL·E fake) — it's a broader/harder 13-generator stress test, so the
official COCO+DALL·E number is likely LESS pessimistic. (2) n=20/category → wide
CIs; VQDM 0% is real, small gaps are noise.

**Levers (payoff order):** diversify training generators (not the val/test sets);
emit a SOFT AIGC score (prob / vote-fraction) instead of hard 1.0/0.0 so the
biased operating point can be tuned; inspect VQDM misses for a systematic cause;
robustness-under-transforms is still a separate untested axis. See log.md →
[2026-08-27 — Local WildFake test-set eval cell](log.md).

---

**`Qwen.ipynb` is the single authoritative notebook — no more Colab twin.** The old
`Qwen-Colab.ipynb` was just the copy that had actually been *run* on Colab (it carried
the execution outputs, the Drive-mount plumbing, and the real training hyperparameters
`MAX_STEPS=500` / `LR=5e-6`), while `Qwen.ipynb` was a clean dev copy. They were merged
into `Qwen.ipynb` (keeping the Colab run + outputs) and the twin was deleted. If you edit
the notebook on Colab again, save back to this same file — don't spawn a second copy. See
log.md → [2026-08-28 — Merge Qwen-Colab.ipynb into Qwen.ipynb](log.md).

---

## 2026-08-27 — No eval→train leakage now, but the safeguard is fragile (step-count vs dataset size, NOT buffer_size)

**The split.** Eval and train are carved from ONE shuffled stream:
`shuffled_stream = dataset.shuffle(buffer_size=10000, seed=42)` (called once,
fixed seed), then `eval_samples = shuffled_stream.take(N_EVAL)` (first 100) and
`train_source = shuffled_stream.skip(N_EVAL)` (everything after). Those are
complementary slices of the *same* order — disjoint **as long as both iterations
see the same order.**

**Why there's no leak with the current config.** A `datasets.IterableDataset`
re-shuffles deterministically per iteration *at a fixed epoch* (epoch starts 0).
The only thing that reshuffles — and would make `skip(100)` skip a DIFFERENT
first-100 than `take(100)` reserved, leaking eval into train — is HF `Trainer`
calling `set_epoch(1)`, which only happens when the training iterator is
**exhausted** (a full epoch). Training consumes `500 steps × grad-accum 4 ×
batch 1 = 2000` samples out of SID_Set's ~210k, so the first epoch never
finishes, `set_epoch(≥1)` is never called, the order stays epoch-0, and
take/skip stay disjoint. **No leak.**

**The key nuance (this is what to remember):** the protection is a property of
**`steps × eff_batch ≪ dataset size`** (here ~2000 ≪ 210k), *not* of
`buffer_size`. Buffer size only changes *what* the shuffled order is; it never
makes take/skip overlap. So the honest one-liner is: *"No leak, because 500 steps
consumes ~2000 of ~210k examples so the trainer never completes an epoch and
never reshuffles."* It would BREAK if steps were pushed into epoch-crossing
territory (e.g. ~60k+ steps → >210k samples) or the split were tiny — then the
per-epoch reshuffle silently leaks, with no error.

**Second, dataset-inherent risk (not a code bug):** SID_Set's **Tampered** class
is *real photos with AI-edited regions*, so a Tampered eval image can share its
underlying base photograph with a **Real** training image. An index-disjoint
split doesn't prevent that near-duplicate overlap — one more reason the 98%
in-distribution number can be optimistic (see
[cross-dataset reality check](#2026-08-27--cross-dataset-reality-check-98-in-distribution--66-on-wildfake-generator-generalization-gap)).

**Bulletproof fix if ever needed:** replace positional `take`/`skip` with a
**content-hash holdout** (hash each image's bytes; route `hash % K` to eval) —
stable across reshuffles/epochs and it also dedups near-identical images out of
training.

---

## 2026-08-30 — Robustness eval: the ONE real failure is additive noise; everything else is a mirage created by the authentic-bias

**What the CSV is (read this first).** `robustness_summary.csv` was run on the
**WildFake balanced cross-dataset set** (120 authentic / 140 AIGC), not the
SID held-out slice — the clean row (overall **0.658**, authentic **0.875**, AIGC
**0.471**) is the same 65.8% / 88.3% / 46.4% cross-dataset result from the
[reality check](#2026-08-27--cross-dataset-reality-check-98-in-distribution--66-on-wildfake-generator-generalization-gap).
So `Acc_authentic` / `Acc_AIGC` are per-class **recall**, and this is the *hard*,
meaningful test surface.

**Headline: only NOISE actually hurts.** `Acc_drop_vs_clean` is **positive (worse)
only for noise and color** — noise 0.02/0.05/0.1 drop accuracy by **+0.065 /
+0.085 / +0.10**, and the mechanism is specific: noise crushes AIGC recall
**0.471 → 0.31 → 0.25 → 0.22** while authentic recall climbs to **0.95**. i.e.
**additive Gaussian noise makes the model call almost everything "authentic."**
That is the real robustness vulnerability and the thing a second model must fix.

**The "robust to jpeg/blur/resize" gains are a mirage — they're the bias being
accidentally corrected.** jpeg/blur/resize/crop all show *negative* drop (higher
accuracy), but that's because the model sits at a badly **authentic-biased
operating point** (clean AIGC recall only 0.47 — it misses over half of fakes),
and any transform that softens high-frequency detail nudges borderline images
across the boundary toward "AIGC," rescuing missed fakes (resize 0.5: AIGC recall
0.47→0.67; blur 2.0: 0.47→0.72, and there authentic recall finally cracks to
0.64). It looks like robustness; it's really a decision threshold in the wrong
place. **Fix the bias and these "gains" will shrink toward small honest drops.**

**Two structural facts that shape the fix:**
1. **No train-time augmentation exists.** `preprocess_image` / `_image_transform`
   in cell 6 is an explicit *scaffold* — NOT on the training path (TRL's vision
   collator processes raw PIL images on the fly). So the model has never seen a
   jpeg/blur/noise'd image. Adding augmentation (noise especially) directly
   targets the one real failure and is the highest-leverage change.
2. **The weak axis is AIGC recall / generator generalization, not overall
   accuracy.** Same asymmetry as the cross-dataset eval. Volume won't fix it —
   generator *diversity* will (VQDM 0%, weak diffusion tier).

**Signature to remember:** when a transform's accuracy *goes up*, check the
per-class columns before celebrating — if authentic recall is pinned near ~0.9
and the gain is all in AIGC recall, you're measuring a biased threshold, not
robustness.

**Second-model plan built from this takeaway:** see log.md →
[2026-08-30 — Plan for model v2](log.md).

---

## Slow training is usually a starved GPU, not too few GPUs

Qwen_v2 took ~13 h/A100 = ~2.3 s/image for a 0.8B VLM — that's the input
pipeline blocking the GPU, not compute. Before reaching for multi-GPU/DDP,
check GPU utilisation and fix the data path: (1) `dataloader_num_workers>0`
(streaming + PIL augment + hashing all ran on one process), (2) prefer a larger
per-device batch over gradient accumulation (accum runs micro-batches
*serially*), (3) collapse redundant per-sample work. Keep the batch×accum
product fixed so the tuned LR/step budget still holds. Multi-GPU only helps once
you're actually compute-bound — and streaming `IterableDataset` needs
`split_dataset_by_node` to shard per rank, or every GPU replays the same data.
See log.md → [2026-08-30 — Qwen_v2.ipynb training-speed levers](log.md).

---

## FlashAttention-2 has no wheel for bleeding-edge torch — SDPA is the fallback that isn't a downgrade

Colab shipped torch 2.11, but flash-attn's prebuilt FA2 wheels stop at torch2.9
(the newer GitHub releases are FlashAttention-*4* betas — a different
`flash_attn_4` package, not what transformers' `attn_implementation=
"flash_attention_2"` loads). So the runtime-detected wheel URL 404s and pip
falls into a source build that's slow (10-30 min) and OOM-prone on Colab. Don't
wait on that build: on A100 + bf16, PyTorch **SDPA already dispatches to the same
FlashAttention-2 CUDA kernel**, so `attn_implementation="sdpa"` gives ~the same
speed with zero install. Rule of thumb: prebuilt FA2 only when a wheel matches
the runtime's python/torch/cuda/abi exactly; otherwise use SDPA rather than
compiling or downgrading torch. See log.md →
[2026-08-30 — Colab torch 2.11 has no FA2 wheel](log.md).

---

## KEY FINDING — Speeding up Qwen_v2 training on Colab A100 (13h → parallelised)

End-to-end summary of the 2026-08-30 optimisation session. Goal: cut Qwen_v2's
~13h/A100 SFT run. Root cause and the levers that actually applied on this
runtime (single Colab A100, ~83GB CPU RAM, streaming SID+WildFake with heavy
per-image PIL augmentation):

**Diagnosis first.** ~13h for 2500 steps × eff-batch-8 = ~2.3s/image — far too
slow for a 0.8B VLM ⇒ the GPU was *starved by the single-process input
pipeline*, not compute-bound. So multi-GPU/DDP was the wrong first move (and
Colab is single-GPU anyway; streaming `IterableDataset` would also need
`split_dataset_by_node`). The signal that matters is **GPU-Util %, not GPU
memory** — low memory just means spare capacity.

**Levers applied (all single-GPU, Colab-safe):**
1. **Parallel data loading** — `dataloader_num_workers`, `pin_memory`,
   `persistent_workers`, `prefetch_factor`. The real fix (network fetch + PIL
   aug + hashing were serial on the main process). See the takeaway
   *"Slow training is usually a starved GPU, not too few GPUs"* above.
2. **Real GPU batching, not grad-accum** — `TRAIN_BATCH_SIZE 1→8`,
   `GRADIENT_ACCUMULATION_STEPS 8→1`. Keep the **product = 8** so the tuned LR /
   warmup / step-budget are unchanged; grad-accum runs micro-batches *serially*,
   it doesn't parallelise.
3. **One `.map()` instead of two** in the streaming prep (minor cleanup).
4. **bf16 + tf32** in SFTConfig (A100-native, ~free, no memory cost).
5. **Attention kernel: SDPA** (see the FA2/SDPA takeaway below).

**Memory ceilings discovered (this box):**
- CPU RAM is the binding constraint on data-loading parallelism: ~18GB/worker
  from the PIL aug pipeline ⇒ `num_workers=2` is the ceiling (4 drove RAM to
  79/83GB and OOM-killed a worker: "DataLoader worker exited unexpectedly" =
  CPU-RAM kill, NOT CUDA OOM).
- GPU VRAM had huge headroom (~11–16/40GB), so batch size could grow freely.
- Different pools disambiguate OOMs: `CUDA out of memory` → lower batch;
  worker-death → lower `num_workers`.

**FlashAttention-2 → SDPA.** No prebuilt FA2 wheel exists for Colab's torch 2.11
(wheels stop at torch2.9; newer releases are FlashAttention-*4* betas, a
different package). Source-building is slow + OOM-prone, and downgrading torch is
worse. PyTorch **SDPA uses the FA2 kernel under the hood on A100+bf16**, so we
set `attn_implementation="sdpa"` directly and kept cell 1 to a single install
line. See the takeaway *"FlashAttention-2 has no wheel for bleeding-edge torch
— SDPA is the fallback that isn't a downgrade"* above.

**Final config:** per-device batch 8 × accum 1 (eff 8); `dataloader_num_workers=2`,
pin_memory, persistent_workers, prefetch_factor 2; bf16=True, tf32=True;
`dtype=torch.bfloat16` + `attn_implementation="sdpa"`. Next unused lever if more
speed is needed: make augmentation cheaper (GPU-side / smaller images) to raise
the worker ceiling. Change history in log.md (2026-08-30 entries).

---

## Changing batch size: it's a work-conservation + dynamics change, not a free speedup

Cluster of clarifications from 2026-08-30 (no code change — intuition to keep).

**Batch×k + steps÷k = same total work.** Wall-clock ≈ (total images seen) ×
(time/image) + fixed overhead, and total images = steps × effective_batch. So
scaling batch UP and steps DOWN by the same factor keeps images-seen constant →
work is constant → you only get *second-order* gains (less optimizer/launch
overhead, better GPU util). Observed: batch×4 / steps÷4 gave 2h→1.5h (~25%), NOT
4×. It's diluted further because (a) we're data-loading bound so the pipeline
does the same work regardless of batch — see the takeaway *"Slow training is usually a starved GPU"* — and (b) fixed overhead (model load,
streaming warmup, the several eval passes incl. the slow n_votes=5 majority vote)
doesn't scale with steps. To truly go faster you must cut total images (= less
training, a quality trade) or fix the data pipeline; batch size alone won't.

**Batch ↔ LR reading.** Linear scaling rule + warmup: Goyal et al. 2017
(arXiv:1706.02677). LR≈batch duality: Smith et al. 2017 (arXiv:1711.00489).
Critical batch size / diminishing returns: McCandlish et al. 2018
(arXiv:1812.06162). CAVEAT: linear rule is for SGD; on AdamW (we use
`adamw_8bit`) the heuristic is **√k** scaling (2× batch → ~1.4× LR, not 2×).
All heuristics — validate on eval metrics.

**Where the knobs live + honest correction.** cell 3 of Qwen_v2.ipynb:
`TRAIN_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `WARMUP_STEPS`, `MAX_STEPS`,
`LEARNING_RATE`; fed to `SFTConfig` in cell 17. These are **hand-set, NOT derived
from the batch by any formula** — "tuned for effective-batch-8" just means chosen
while eff-batch was 8. LR 5e-6 was picked "collapse-safe"; `WARMUP_STEPS=25` is
~1% of `MAX_STEPS=2500`, i.e. coupled to STEP COUNT, not batch. `warmup_steps` is
counted in OPTIMIZER STEPS. So to change effective batch safely: scale `MAX_STEPS`
inversely (keep image budget), keep warmup ~1% of steps, reconsider LR (√ rule),
and A/B on eval metrics — the batch-8 config was chosen NOT to perturb dynamics,
so a bigger effective batch is a NEW run to validate, not a free lever.

**`attn_implementation` default.** Omitting it → transformers auto-selects
`sdpa` if the model supports it, else SILENTLY falls back to `eager` (slower);
it NEVER auto-picks flash_attention_2 (always explicit opt-in). Keep the explicit
`attn_implementation="sdpa"` to fail loud if SDPA ever becomes unsupported;
verify what you got with `model.config._attn_implementation`. Related: the takeaway *"FlashAttention-2 has no wheel for bleeding-edge torch"*.

---

## v2's "regression" was CALIBRATION, not capability — v1 ≈ v2 at AUC ~0.705

**Setup (2026-08-30/31):** WildFake balanced set (120 real / 140 AIGC), scored
with the soft `_sid_predict_proba` P(AIGC)=p1+p2. Investigated why v2's
robustness table looked far worse than v1's (balAcc 0.673 → 0.569 at threshold
0.5; AIGC recall 0.471 → 0.171).

**Finding — the models are equivalent; only their score *scale* differs:**
- **ROC-AUC (threshold-free): v1 = 0.709, v2 = 0.703** → statistically
  indistinguishable at n=260 (CI ≈ ±0.06). Same discrimination.
- v2 emits **lower-magnitude** scores: mean P(AIGC) fakes 0.17 vs v1 0.47; reals
  0.03 vs 0.13. So a fixed **0.5 threshold is far too high for v2** → everything
  reads authentic → the phantom "regression."
- At each model's OWN optimal threshold both reach **~0.68 balanced acc**. Even
  v1's optimum is **0.30, not 0.5** — both models want a threshold < 0.5.

**Consequences:**
1. The v2 recipe (train-time augmentation + 50% WildFake mix + soft scoring) did
   **NOT** improve clean discrimination — AUC is flat vs v1. The added complexity
   bought nothing measurable on WildFake (may still help the untested official
   COCO+DALL·E set).
2. **Never judge these models by accuracy at a fixed 0.5 threshold** — it's
   calibration-confounded and produced a completely misleading picture. Report
   **threshold-free AUC** as the primary metric.
3. **Robustness (v2's actual purpose) is still UNJUDGED.** The accuracy@0.5
   robustness tables are invalid for the same reason; the noise "improvement" was
   a floor effect. Must recompute **per-transform AUC** for both models before
   concluding whether augmentation earned its place.

**Actions carried into v3:** primary metric = AUC everywhere; pick the decision
threshold on a **held-out calibration split** (never the test set); the
deliverable emits a P(AIGC) *likelihood*, so if grading is ranking/AUC-based the
threshold issue is moot. Related: the authentic-bias takeaway (2026-08-30
robustness eval) and the cross-dataset gap (98% in-dist → 66% WildFake).
