# Change Log

Reverse-chronological log of agent-made changes. Newest first. Append a new
entry (don't rewrite history) after each change.

---

## 2026-08-30 — Model v2: new `Qwen_v2.ipynb` (robustness recipe) + `pull_wildfake_train.py`

Built the second iteration as a **separate** notebook (`Qwen_v2.ipynb`, 30 cells)
so v1 (`Qwen.ipynb`) stays intact as the documented baseline. v2 = a faithful
copy of v1 with five levers layered in, chosen from the robustness-eval analysis
(see key-takeaways.md 2026-08-30). Decisions confirmed with the user: full recipe
now, SID+WildFake mix, keep LR 5e-6 with more steps (not their initial
half-LR/double-steps — those cancel to the same update budget under a linear
scheduler).

**The five levers (cells changed vs v1):**
1. **Train-time augmentation** (cell 6 rewritten). Replaced the unused
   `preprocess_image`/`_image_transform` scaffold with real transforms:
   deterministic `_t_jpeg/_t_blur/_t_resize/_t_noise/_t_color/_t_crop` (PIL +
   numpy + PIL.ImageEnhance — no torchvision needed) and `augment_image` (stacks
   1..3 random transforms at random severity, `AUG_PROB=0.85`, noise up to 0.10 —
   the severity that broke v1). Wired into the train stream in the map cell.
2. **SID + WildFake mix** (new cell inserted after baseline). `interleave_datasets`
   of the streamed SID half (hash-holdout complement) with a local
   `wildfake_train/` set loaded as a `datasets.Dataset` → `to_iterable_dataset`.
   WildFake fake→SID label 1, real→0. Falls back to SID-only with a warning if
   the folder is absent. `WILDFAKE_MIX_PROB=0.5`.
3. **Soft AIGC score** (eval harness, cell 12). New `_sid_score_labels` /
   `_sid_predict_proba`: teacher-forced log-likelihood of each `\boxed{k}`
   completion, softmax'd → P(AIGC)=p1+p2. Deterministic, no generation, format-
   matched to the SFT target. `eval_sid_accuracy` gains `soft=` (default
   `SOFT_SCORE=True`), and `eval_local_folder` now writes the soft likelihood to
   `wildfake_balanced_v2_preds.json`.
4. **Content-hash holdout** (cell 13). Replaced positional `take`/`skip` with a
   perceptual average-hash router (`_img_hash`): downscale 8x8 → threshold at
   mean → md5 the bit pattern. Near-duplicates (re-encode/resize/blur/noise of
   the same photo, or a Tampered image sharing a Real base) collapse to one
   digest so they can't straddle the split; the md5 wrap gives uniform
   `hash%EVAL_HOLDOUT_MOD` bucketing. Train = the eval bucket's complement.
5. **Budget** (config cell). `MAX_STEPS` 500→2500, `GRADIENT_ACCUMULATION_STEPS`
   4→8, `WARMUP_STEPS` 4→25, `LEARNING_RATE` kept 5e-6. ~20k images seen, still
   << SID's ~210k so no epoch-crossing reshuffle leak. `OUTPUT_DIR`/`SAVE_DIR` →
   `aigc_detector_qwen_v2`.

Also added a **robustness re-eval cell** (`eval_robustness_v2`) that re-scores
v2 on the exact transform×severity grid that produced `robustness_summary.csv`,
over the same `wildfake_balanced/` set, writing `robustness_summary_v2.csv` for a
direct v1-vs-v2 comparison.

**New file `pull_wildfake_train.py`** — companion to `pull_wildfake_balanced.py`:
pulls a larger (`PER_CATEGORY=120`), non-val WildFake set into `wildfake_train/`
with a different seed, and **excludes every filename already in the eval set**
(`wildfake_balanced/labels.csv`) so train ⟂ eval by construction. Run it before
training to enable the mix.

**Verified (no GPU here):** notebook valid JSON, 30 cells, correct cell order,
zero references to removed symbols (`image_stream`, `preprocess_image`) or the old
positional split; all code cells parse (excl. the `!pip`/`%cd` magic cells). Pure-
Python logic smoke-tested in a PIL+numpy venv: all six deterministic transforms
preserve RGB + original size on RGB/L/RGBA/tiny inputs; `augment_image` over 2000
random draws always returns same-size RGB; the perceptual hash collapses
jpeg/resize/blur/noise near-dups to one digest, keeps 200/200 genuinely-distinct
images unique, and routes ~4.65% to the eval bucket (ideal 5%); the robustness
grid dispatch (15 cells) runs; the soft-score softmax normalizes.

**NOT verified / open for Colab:** end-to-end run (model load, the soft
teacher-forced scoring path through the real VLM processor, `interleave_datasets`
feature-equality between the streamed SID half and the WildFake `Dataset`,
training, both evals). Must run `pull_wildfake_train.py` first (network + remotezip)
to populate `wildfake_train/`; otherwise v2 falls back to SID-only with a warning.

---

## 2026-08-27 — Initialized git repo + added README and .gitignore

- Ran `git init` (branch `main`); the repo was previously not under version
  control. Made the initial commit with all existing files.
- Added `README.md` covering the deliverable's required sections: project
  overview (VLM approach, `Qwen/Qwen3.5-0.8B`, SID_Set streaming, 3-class→binary
  collapse, tools/libraries/datasets), setup & installation (Colab GPU + pip
  line), steps to reproduce (config cell → load → baseline eval → fine-tune with
  `completion_only_loss` → post-train comparison → robustness summary → WildFake
  benchmark), sanity checks, and a limitations/future-work reflection. Team
  contributions section left as a solo placeholder.
- Added `.gitignore` for Python/Colab artifacts, model checkpoints
  (`mcq_finetuned_model/`, `*.safetensors`), local dataset caches, and
  `predictions.json`.
- README content drawn from `problem_context.md`, `log.md`, `key-takeaways.md`,
  and the notebook's config/model-load cells (no notebook code changed).

---

## 2026-08-27 — Document 3-class vs binary accuracy in `Qwen.ipynb`

**Context:** A teammate asked what the difference between the two accuracy
numbers reported by `eval_sid_accuracy` is. Captured the answer in the notebook
so future readers don't have to re-derive it.

**Changes to `Qwen.ipynb`:**
- Inserted a new markdown cell (after the "### Evaluation" intro, before the
  eval-harness code cell) titled "3-class vs binary accuracy — what's the
  difference?". Explains that 3-class scores the exact SID_Set label
  (`correct_3class`, `pred == gold`) while binary collapses {Synthetic,
  Tampered}→AIGC / {Real}→authentic (`collapse_to_binary`, `correct_binary`)
  before scoring; notes binary ≥ 3-class always, binary is the deliverable
  metric, and 3-class + per-class breakdown is the collapse-to-majority
  diagnostic.

---

## 2026-08-26 — Prompt builders for streamed SID_Set in `Qwen.ipynb`

**Context:** Adapt the CoT tutorial's prompt boilerplate
(`build_mmlu_prompt` / `build_mmlu_sft_text`) into `Qwen.ipynb` for the streamed
`saberzl/SID_Set` AIGC-detection dataset.

**Changes to `Qwen.ipynb`:**
- Filled the empty markdown cell (index 6) with a `### Load base model &
  tokenizer` header.
- Inserted a markdown cell documenting the SID_Set label schema (0=Real,
  1=Synthetic, 2=Tampered) and the two prompt builders.
- Inserted a code cell defining the prompt-construction helpers:
  - `SID_LABELS`, `IMAGE_TOKEN`
  - `sid_label_to_letter(label)`
  - `build_sid_prompt(...)` — inference/user prompt with `<image>` placeholder,
    A/B/C options, ends on `Reasoning:`.
  - `build_sid_target(label, cot)` — assistant answer as `\boxed{A|B|C}`.
  - `build_sid_sft_text(label, tokenizer, cot)` — chat-template SFT text with a
    plain-concatenation fallback for base models lacking a chat template.
  - `sid_batch_to_texts(labels, tokenizer, cots)` — batch helper pairing with
    `train_dataloader`.
- Inserted a sanity-check code cell that prints the inference prompt, an SFT
  example, and SFT texts for one streamed batch.

**Verified:** Notebook still valid JSON (11 cells). Functions exec'd against
stub tokenizers (both chat-template and base/no-template paths) — letter mapping,
SFT rendering, and batch building all correct.

**Not verified / open:** Did not run cells against the real model or live
stream. `Qwen3.5-0.8B-Base` / `Qwen3_5ForCausalLM` names unverified. Builders
currently emit 3-class targets; a binary mode may be wanted for the final
AIGC-vs-authentic deliverable.

**Reference used:** `finetuning_tutorial_cot.ipynb` (immutable — not modified).

## 2026-08-26 — Added `CLAUDE.md` and `log.md`

Created project guidance (`CLAUDE.md`) noting that
`finetuning_tutorial_cot.ipynb` is immutable reference boilerplate, `Qwen.ipynb`
is the working file, dataset/label facts, and environment cautions. Created this
change log.

## 2026-08-26 — Zero-shot cleanup + template-mapped training loop (Qwen.ipynb)

- Removed all chain-of-thought (CoT) references for a zero-shot setup: dropped
  the `cot`/`cots` params from `build_sid_target`, `build_sid_sft_text`, and
  `sid_batch_to_texts`; assistant target is now the bare `\boxed{letter}`.
  Updated the Prompt Construction markdown accordingly.
- Added `.map(build_sid_sft_text)` over `shuffled_stream` to inject an `sft_text`
  field into the stream (mirrors the tutorial's `.map(... build_mmlu_sft_text)`).
- Added `SIDTextDataset` (yields `(image_tensor, sft_text)`) + `sft_dataloader`,
  and a small causal-LM SFT loop (tokenize → labels=input_ids w/ pad=-100 →
  AdamW, grad-accum, capped at SPECIALISED_TRAINING_MAX_STEPS).
- Note: Qwen3.5-0.8B-Base is text-only, so the loop trains on language targets;
  the image tensor is carried through but not consumed (\<image\> placeholder).
- Validated notebook as JSON; exec-checked builders with a stub tokenizer.

## 2026-08-26 — Fix `NameError: labels` in sanity-check cell (Qwen.ipynb)

- **Bug:** the sanity-check cell (cell-10) called `sid_batch_to_texts(labels[:2],
  ...)`, but `labels` is only bound by cell-5's `for batch in train_dataloader`
  loop, which had no `break` and so iterates the entire 210k-image stream — it
  never finishes, so `labels` was never defined → `NameError`.
- **Fix:** made cell-10 self-contained — it now pulls a couple of labels
  directly off `shuffled_stream` via `itertools.islice(..., 2)` instead of
  depending on `labels`.
- Also added a `break` to cell-5's smoke-test loop so it grabs one batch (and
  defines `labels`) instead of hanging the kernel on the full split.
- Note: confirmed at runtime that `Qwen3.5-0.8B-Base`'s tokenizer *does* ship a
  chat template — the SFT text renders with `<|im_start|>`/`<think>` tags and a
  bare `\boxed{letter}` target, so the base-model fallback path isn't hit here.
- Validated notebook as JSON; compile-checked cells 5 & 10 and exec-checked the
  builders.

## 2026-08-26 — Fix `NameError: transforms` in stream-mapping cell (Qwen.ipynb)

- **Bug:** cell-12 (`SIDTextDataset`) used `transforms.Compose(...)` but
  `transforms` is imported in cell-4; if cell-4 hadn't been run in the session,
  cell-12 raised `NameError: name 'transforms' is not defined`.
- **Fix:** added a local `from torchvision import transforms` at the top of
  cell-12 so it runs standalone regardless of earlier-cell execution.
- Validated notebook as JSON; compile-checked cell-12.

## 2026-08-26 (follow-up) — Switched to SFTTrainer/SFTConfig pattern (Qwen.ipynb)

- Replaced the hand-rolled training loop with the tutorial's `SFTConfig` +
  `SFTTrainer` pattern (per CLAUDE.md: adapt tutorial patterns, do not reinvent).
- Mapping cell now maps into a `"text"` field and `remove_columns=["image","label"]`
  so the trainer gets a clean text-only stream.
- Necessary deviation from the tutorial config: source is a streaming
  IterableDataset (no length), so `max_steps=SPECIALISED_TRAINING_MAX_STEPS`
  drives training instead of `num_train_epochs`.
- Notes retained: text-only model drops the image; `adamw_8bit` needs bitsandbytes.

## 2026-08-26 (follow-up 2) — Dependencies + datasets.IterableDataset refactor

- CLAUDE.md: added two working conventions — (1) fix missing dependencies by
  installing them, never downgrade the approach or leave a "install X" note;
  (2) prefer default/library classes over custom ones (use datasets.IterableDataset
  + .map() rather than a torch IterableDataset subclass).
- Qwen.ipynb install cell: added `trl` (SFTTrainer/SFTConfig were imported but
  never installed) and `bitsandbytes` (required by OPTIM="adamw_8bit").
- Qwen.ipynb: removed the custom `HFIterableDataset(torch ... IterableDataset)`
  subclass. Image preprocessing is now a plain `preprocess_image` applied via
  `shuffled_stream.map(..., remove_columns=["image"])`, keeping the pipeline a
  datasets.IterableDataset fed straight to a DataLoader. Dropped the now-unused
  `from torch.utils.data import IterableDataset` import.
- Not executed here: torchvision is not installed in this environment, so the
  image .map path was not run; logic is a direct port of the prior transform.

## 2026-08-26 (follow-up 3) — Merged Qwen_renamed.ipynb into Qwen.ipynb

- Folded the useful manual edits from the diverged Qwen_renamed.ipynb into
  Qwen.ipynb, then deleted Qwen_renamed.ipynb (one canonical notebook).
- Adopted from the renamed copy: centralized imports in one top cell (datasets,
  trl, torch, DataLoader, transformers, torchvision); model/tokenizer load moved
  up right after config; simplified single-phase config (WARMUP_STEPS=4,
  MAX_STEPS=78) replacing the SPECIALISED_/MIXED_ split.
- Kept from Qwen.ipynb: .map() image-preprocessing scaffold (no custom class),
  single SFTTrainer over sft_stream with max_steps + remove_columns, islice
  sanity check.
- Dropped renamed cruft: HFIterableDataset/SIDTextDataset subclasses, the manual
  AdamW loop, the broken SFTTrainer fed (image,label) tuples, and cell 15
  specialised_trainer/mixed_trainer.train() (undefined -> NameError).
- Decisions confirmed with user: single SFTTrainer; keep image scaffold.
- Verified: valid JSON (14 cells); zero refs to removed symbols; trl/torchvision
  imported once; builders exec to boxed A/B/C. Not executed end-to-end (no
  torch/torchvision/trl here; non-standard Qwen names).

## 2026-08-27 — Baseline-vs-fine-tuned accuracy comparison (multimodal)

- Added a before/after accuracy comparison to Qwen.ipynb, mirroring the
  tutorial's baseline-vs-fine-tuned pattern. Key decisions confirmed with user:
  feed REAL images (not the old text-only path), make training multimodal too,
  held-out eval = a disjoint slice of the SID_Set stream, report both 3-class
  and binary, and answer format = the raw integer label `\boxed{0|1|2}` (no
  A/B/C letter or class-name mapping).
- Model switch: MODEL_NAME -> "Qwen/Qwen3.5-0.8B" (image-text-to-text VLM, NOT
  "-Base"); load via AutoModelForImageTextToText + AutoProcessor (verified on the
  HF model card that this is a multimodal checkpoint). `tokenizer` is now an
  alias for `processor.tokenizer`.
- Prompt builders rewritten: dropped `sid_label_to_letter`, `build_sid_prompt`,
  `build_sid_sft_text`, `sid_batch_to_texts`; SID_LABELS is now name+desc only.
  New: `build_sid_prompt_text` (no `<image>` string), `build_sid_target` (raw
  int), `build_sid_eval_messages` (image inlined), `build_sid_train_messages`
  (`{"type":"image"}` placeholder + gold assistant turn).
- New eval harness cell: `parse_label_from_boxed` (digit-targeted port of the
  tutorial's parser), `collapse_to_binary` ({1,2}->AIGC, {0}->authentic), and
  `eval_sid_accuracy` (model.eval + no_grad + greedy generate, feeds the real
  image through the processor, reports acc_3class/acc_binary/per-class/details).
- New held-out split: eval = `shuffled_stream.take(N_EVAL)` (N_EVAL=100),
  training = `shuffled_stream.skip(N_EVAL)` -> disjoint, never trained on.
  Baseline eval runs BEFORE trainer.train() (mutates `model` in place); the
  post-train cell re-evals and prints the side-by-side comparison + per-class.
- Training map (cell) now keeps the `image` column and adds conversational
  `messages`; SFTConfig drops `dataset_text_field`, adds `max_length=None` and
  `remove_unused_columns=False`; SFTTrainer uses `processing_class=processor`
  (TRL's native DataCollatorForVisionLanguageModeling handles pixels on the fly).
- EVAL_MAX_NEW_TOKENS lowered 512 -> 128. CLAUDE.md updated: runtime target is
  Google Colab.
- Verified here: valid nbformat (19 cells, correct order); helper unit tests pass
  (parser incl. \boxed/whitespace/fallback/None, binary collapse, integer target,
  train/eval message shapes, prompt has no `<image>` + integer options). NOT
  executed end-to-end — no GPU here and the Qwen3.5 names are unverified against a
  live hub; baseline eval, training, and the comparison run on the user's Colab.
- Follow-up: removed the DataLoader smoke-test cell (the `for batch in
  image_loader` one-batch check + its `image_loader = DataLoader(...)` line).
  `image_stream` (cell 6 scaffold) is now unused but kept as intentional
  future-vision groundwork. Notebook: 18 cells, valid JSON.

## 2026-08-27 (follow-up 2) — Anti-collapse: answer-only loss + collapse-aware metrics + majority voting

- Diagnosed the first Colab run: fine-tuning collapsed the model to a constant
  "always Real" predictor (Real 100%, Synth/Tamp 0%; binary 70% -> 30%). SID_Set
  is perfectly balanced (100K/100K/100K, per the SIDA paper), so this is
  optimization degeneracy, not imbalance: every training example shares an
  identical prompt and differs only in the final \boxed{d} digit, so full-
  sequence loss swamps the one-token answer signal.
- Remedies implemented (user-selected):
  1. Answer-only loss: SFTConfig `assistant_only_loss=True` so the gradient
     targets the assistant \boxed{d} answer, not the repeated prompt. Watch-item
     noted in-cell: needs the chat template's {% generation %} keywords; if
     absent, loss goes flat ~0 -> patch template or use completion_only_loss.
  2. Collapse-aware metrics: eval now also reports macro-F1, balanced accuracy
     (stock scikit-learn), and a 3x3 confusion matrix (unparsed preds -> -1
     sentinel column). A single-class collapse floors macro-F1 at ~0.15.
  3. Majority voting: eval_sid_accuracy gains n_votes/temperature/top_p; n_votes>1
     samples answers and takes the mode (adapted from the tutorial's
     eval_mcq_accuracy_majority), with a VLM fallback loop if
     num_return_sequences can't expand image features. Comparison cell reports
     finetuned greedy AND mv@5. Caveat documented: voting can't rescue a fully
     collapsed model.
- Cells changed: eval harness (imports + _sid_predict + metrics + print_confusion),
  trainer SFTConfig (assistant_only_loss), comparison cell (macro-F1/bal-acc rows,
  both confusion matrices, mv@5), Results markdown.
- Verified here in a scratchpad venv (pandas 3.0.5, sklearn 1.9.0) by exec'ing the
  eval harness against a stub model: greedy-perfect (diagonal confusion, macroF1
  1.0), collapse (macroF1 0.167, single populated column), majority vote (correct
  modes, junk votes dropped), all-unparsed (-1 bucket, macroF1 0). Notebook valid
  nbformat, 19 cells. NOT run end-to-end (no GPU); training + real generation are
  on the user's Colab.

---

## 2026-08-27 — Add modelscope dependency (WildFake validation set groundwork)

- Cell 1 (install): appended `modelscope` to the pip line. Now:
  `!pip install torch torchvision transformers datasets trl bitsandbytes scikit-learn tqdm modelscope --break-system-packages`
- Purpose: enable downloading the WildFake validation subset (COCO val2017 non-AIGC
  + DALL·E Advanced AIGC) via `from modelscope.msdatasets import MsDataset;
  MsDataset.load('hy2628982280/WildFake', subset_name='default', split='train')`.
  This is a reference-only benchmark — NOT for training (per problem brief).
- Notebook still valid JSON, 19 cells. Dependency-only change; no eval/loader code
  added yet (next step).

## 2026-08-27 (follow-up 3) — Colab error fix: assistant_only_loss -> completion_only_loss

- Colab raised: "Assistant-only loss is not yet supported for vision datasets."
  Confirmed against TRL 1.12.0 source: sft_trainer.py hard-raises for
  `_is_vision_dataset and assistant_only_loss`, BUT
  DataCollatorForVisionLanguageModeling DOES accept `completion_only_loss` (it
  sets prompt-part labels to -100) when the dataset is in prompt/completion form.
- Fix (the plan's named fallback): switched the training data from a single
  `messages` list to prompt/completion columns and enabled
  `completion_only_loss=True`.
  - Prompt builders: replaced `build_sid_train_messages` with
    `build_sid_train_prompt` (user turn: image placeholder + task) and
    `build_sid_train_completion` (assistant \boxed{d}).
  - Map cell now emits {"prompt": ..., "completion": ...} + keeps `image`.
  - SFTConfig: `assistant_only_loss=True` -> `completion_only_loss=True`.
  - Updated the Prompt-Construction + training markdown and the cell-9 sanity
    print accordingly.
- This is more robust than assistant_only_loss (no chat-template {% generation %}
  markers needed). Verified builders produce the correct prompt/completion shapes;
  notebook valid (19 cells). Training runs on the user's Colab.

---

## 2026-08-27 — WildFake validation benchmark (streamed, binary) cells added

- Added two cells at the end of Qwen.ipynb (now 21 cells): a markdown intro +
  a code cell `eval_wildfake_binary`. Reference-only benchmark, NEVER trained on.
- Confirmed WildFake schema from the ModelScope repo's label CSVs (not guessed):
  columns `Generator, Architecture, Weight, Category, IsAdvanced, IsFake,
  Image_path, Num`. DALL·E-3 row example: `Architecture=DALLE, IsAdvanced=1,
  IsFake=1, Image_path=./Diffusion_based/DALLE/Advanced/DALLE3/...jpg`; COCO
  reals live in `real_coco.csv` (path contains `coco`, `IsFake=0`).
- Filtering (per user's choices: streamed + binary-only):
  - authentic  = `IsFake` false  AND source mentions `coco`
  - AIGC        = `IsFake` true   AND `DALLE` source AND `IsAdvanced` true
  - defensive `_wf_truthy` coerces '1'/'0'/1/0/'True'/'False' -> bool.
- Streams via `MsDataset.load(..., use_streaming=True)`; caps per class
  (WF_MAX_AUTHENTIC/WF_MAX_AIGC, default 200 for a quick pass, raise toward
  4998/8843). Reuses `_sid_predict` + `collapse_to_binary` from the eval harness.
  Emits per-image `{image_path, pred(AIGC likelihood 1.0/0.0), gold, correct}`,
  mirroring the deliverable JSON fields.
- Peeks one streamed row first (prints keys) so the ACTUAL runtime schema /
  whether pixels are decoded vs bare Image_path is visible; `_wf_image` handles
  both, `WF_IMAGES_ROOT` for the path-only case. Final run line left commented.
- Verified locally: exec'd the helper defs (modelscope/PIL stubbed) against
  synthetic rows — COCO->authentic, DALLE3-Advanced->AIGC, SDXL-advanced and
  DALLE-basic correctly excluded, truthy coercion + None-path image all pass.
  NOT run end-to-end (no GPU / network here); full run is on the user's Colab.
- OPEN ITEM for Colab: confirm the peek's row keys and how pixels are exposed;
  if streamed rows carry only Image_path, set WF_IMAGES_ROOT to the Images root.

---

## 2026-08-27 — Fix modelscope missing runtime deps (addict et al.)

- On Colab `from modelscope.msdatasets import MsDataset` raised
  `ModuleNotFoundError: No module named 'addict'` -- modelscope's install didn't
  pull its runtime deps (py3.13 dist).
- Fix (not a workaround, per CLAUDE.md): added `addict simplejson sortedcontainers`
  to the cell-1 pip line alongside `modelscope`. These are the recurring
  modelscope-on-Colab gaps (config/utils imports); adding all three pre-empts the
  serial import failures instead of round-tripping one at a time.
- Cell 1 now: `!pip install torch torchvision transformers datasets trl
  bitsandbytes scikit-learn tqdm modelscope addict simplejson sortedcontainers
  --break-system-packages`. Notebook still valid JSON, 21 cells.

---

## 2026-08-27 — WildFake balanced sampler (remote range-request pull)

- Goal: build a balanced Real-vs-Diffusion eval set from `hy2628982280/WildFake`
  on ModelScope without downloading the multi-GB zips (each 6-51 GB).
- Key finding: WildFake is NOT a streaming/parquet dataset — it's raw image
  **zip archives** in nested folders. But the LFS CDN honors HTTP range requests,
  so `remotezip` reads only the zip index + the sampled members.
  - Gotcha: point `RemoteZip` at the **resolved CDN URL** (follow the API 302),
    not the API URL — the API rejects remotezip's suffix range with 400.
  - The category (folder) fixes the binary label: `Images/Real/*`->0,
    `Images/Diffusion_based/*`->1. No label CSV lookup needed.
- New files:
  - `pull_wildfake_sample.py` — pulls 100 images from DDIM.zip -> folder + zip.
  - `pull_wildfake_balanced.py` — samples `PER_CATEGORY` from 6 real + 6 fake
    single-file zips into `wildfake_balanced/<category>/` + `labels.csv`
    (image_path,category,generator,label). `INCLUDE_VAL_OVERLAP` flag adds
    COCO+DALLE (the val-reference categories) when a val-mirror set is wanted.
- Ran balanced sampler (PER_CATEGORY=20, seed=0): 240 images, exactly 120/120
  real/fake, all decode. ~77 s (dominated by reading 12 zip indices; laion5b has
  271k members). Multi-part categories (Midjourney, originalSD Advanced/Typical)
  intentionally excluded to keep member->zip mapping unambiguous.
- Default categories avoid the official val set (no COCO/DALLE/Advanced) per
  CLAUDE.md's "don't train on the validation set" rule.
- NOTE for pipeline: sampled images vary in size and mode (some RGBA, e.g. ADM);
  convert to RGB at load. Requires `remotezip` in the install cell.

## 2026-08-27 — Add Midjourney (multi-part) to balanced sampler

- Added Midjourney to `pull_wildfake_balanced.py`. It's multi-part
  (Typical/part_1-4, Advanced/part_1-7, ~50 GB each). `sample_category` now
  accepts a str OR a list of part zips: it opens parts in order, keeping each
  handle open, and stops once the candidate pool >= PER_CATEGORY*100 -- so for
  small samples only part_1 is read (no full download). Handle-caching fix: read
  reuses the listing handle, so each part index is fetched exactly once (an
  earlier version opened twice and timed out).
- Tier split: Typical -> default FAKE (non-val). Advanced -> FAKE_VAL (it's the
  val-reference tier), added only under INCLUDE_VAL_OVERLAP.
- Ran: 13 categories, 20 each, 260 images. Midjourney pulled 20/55173 from 1/4
  parts; images decode (up to 1792x1024, RGB).
- BALANCE NOTE: now 6 real vs 7 fake categories -> 120 real / 140 fake. Per-
  category count stays uniform (20). For strict 50/50 class balance, drop one
  fake category or add a 7th real source (only clean option is COCO, which is
  val-overlap).
- Transient: ModelScope CDN had intermittent slow spells (runs timed out at
  2-5 min, then the same list completed in ~1 s). Retry rather than assume a bug.

---

## 2026-08-27 — Local WildFake test-set eval cell (labeled folder, binary)

- Added markdown + code cell (notebook now 23 cells) after the streaming WildFake
  cell: `eval_local_folder(model, processor)`.
- Motivation: WildFake stores pixels inside multi-GB per-category zips keyed by
  Image_path, so MsDataset streaming tends to yield path strings, not decoded
  images. The reliable route is the local set from pull_wildfake_balanced.py.
- Reads `wildfake_balanced/labels.csv` (image_path, category, generator, label;
  0=real/1=fake), loads each image, runs the SAME greedy _sid_predict ->
  collapse_to_binary path, reports overall binary acc + per-class (authentic/AIGC)
  + per-category (worst-first) accuracy. Writes deliverable-shaped
  `wildfake_balanced_preds.json` ([{image_path, pred}], pred=AIGC likelihood).
- Config: TEST_DIR="wildfake_balanced" (relative; the new Colab bootstrap cell
  %cd's into TechJam so it resolves there and locally). Reference-only, never
  trained on.
- Verified against the real local labels.csv (260 rows, 120 real/140 fake): all
  required columns present, all 260 file paths resolve, _bin_from_label mapping
  correct (0->authentic, 1->AIGC), deliverable JSON shape correct. Cell syntax
  parses; notebook valid JSON, 23 cells. NOT run end-to-end (no GPU here).
- NOTE: user's Drive copy has 274 images (vs local 260) but same structure;
  TEST_DIR default matches. User is prepending a Colab Drive-mount + `%cd
  /content/drive/MyDrive/TechJam` bootstrap cell (added by user, not in repo).

---

## 2026-08-27 — Tidy WildFake eval cells + add local-eval runner cell

- Per user: kept BOTH WildFake eval defs, tidied both, added a runner block
  styled like the FINE-TUNED comparison cell. Notebook now 24 cells.
- Streaming cell (eval_wildfake_binary): moved the module-level MsDataset peek
  OUT of the cell body into a commented "Manual run" recipe at the end, so the
  cell now only DEFINES functions (no network call on execution). Added a header
  NOTE that it targets the brief's exact COCO+DALL·E set but streaming may yield
  Image_path strings only -> prefer the local eval. Empty-result message now
  points to eval_local_folder.
- Local cell (eval_local_folder): dropped the trailing commented run-stub;
  refactored the reporting to also RETURN per_class + per_category + n_unparsed
  (not just print), so the runner can reuse them. Behaviour unchanged.
- NEW runner cell (after local def): calls `wildfake = eval_local_folder(model,
  processor)` and prints an in-distribution (SID_Set finetuned binary) vs
  cross-dataset (WildFake) binary comparison with a Δ line; guards on
  `"finetuned" in globals()`.
- Verified: notebook valid JSON (24 cells), all three touched code cells parse,
  runner name refs resolve (eval_local_folder + finetuned["acc_binary"]),
  streaming cell no longer auto-peeks. NOT run end-to-end (no GPU here).

---

## 2026-08-27 — eval_wildfake_binary: frequent progress heartbeat + skip diagnostics

- Problem: user ran eval_wildfake_binary and saw NO output for 10 min. Cause: the
  only progress print fired every 20 *matched* images, but the stream yields
  Image_path strings (no pixels) so nothing matched -> silent multi-GB churn.
- Tweak (cell 20, eval_wildfake_binary): added a heartbeat every `log_every`
  (default 200) rows *scanned* (not matched), printing running matched / scored /
  skipped-no-pixels counts with flush=True. Plus a one-time loud WARNING on the
  FIRST matched-but-no-pixels row (dumps keys + Image_path), and the empty-result
  message now reports rows scanned + skipped-no-pixels. New `log_every` param
  (0 = silent). Counts `scanned` + `n_no_pixels` threaded through; final success
  line also reports rows scanned.
- Verified: cell 20 syntax OK; functional smoke test with a stubbed 500-row
  path-only stream (modelscope/PIL/torch/pandas stubbed) reproduced the exact
  diagnostic output -- one-time WARNING, heartbeats at 200/400 (scored 0,
  skipped-no-pixels climbing), final "No samples scored after scanning 500 rows"
  + acc_binary nan. Notebook valid JSON, 24 cells. Streaming path remains the
  unreliable one; eval_local_folder is still the recommended route.

---

## 2026-08-27 — Save/reload cells for the fine-tuned model (persist before runtime close)

- Added markdown + save cell + reload cell at the end of Qwen.ipynb (now 27 cells)
  so the user can persist weights to Drive before closing Colab.
- SAVE cell: `model.save_pretrained("aigc_detector_qwen", safe_serialization=True)`
  + `processor.save_pretrained(...)` -> self-contained folder (config +
  *.safetensors + processor/tokenizer) on the Drive-mounted TechJam cwd (survives
  runtime close). Prints abspath + size + file list. Commented shutil.make_archive
  line for an optional submission zip.
- RELOAD cell: AutoProcessor/AutoModelForImageTextToText.from_pretrained(SAVE_DIR,
  device_map="auto", dtype="auto") + pad-token restore -> rebuilds model/processor/
  tokenizer in a fresh session so eval_local_folder / eval_sid_accuracy /
  eval_wildfake_binary work unchanged. Noted it loads a 2nd GPU copy if run in the
  current session (meant for a fresh runtime).
- Note in markdown: this is a full save_pretrained (all params fine-tuned), not a
  LoRA adapter, so the folder is a complete standalone model.
- Verified: notebook valid JSON (27 cells); both new code cells parse; save/reload
  use the same SAVE_DIR; save writes both model + processor. NOT run here (no GPU).

---

## 2026-08-28 — Merge Qwen-Colab.ipynb into Qwen.ipynb (single authoritative notebook)

- Two diverged copies existed: `Qwen.ipynb` (git-tracked dev copy, no execution
  outputs, slightly richer analysis code) and `Qwen-Colab.ipynb` (untracked, the copy
  actually EXECUTED on Colab — carries execution_counts + outputs, the Drive-mount
  plumbing, tuned hyperparameters, and a WildFake schema-peek).
- Merged with Qwen-Colab.ipynb as the **base** (validated, output-bearing run), writing
  the result to `Qwen.ipynb`, then deleted `Qwen-Colab.ipynb`.
- Per user decisions: kept Colab hyperparameters (`MAX_STEPS=500`, `LR=5e-6`), kept the
  executed Colab outputs, and kept the Colab versions of the two eval cells (WildFake
  stream eval + `eval_local_folder`, both of which had run and carried outputs).
- Folded in the ONE cell unique to the dev copy: the WILDFAKE LOCAL TEST-SET ACCURACY +
  CROSS-DATASET COMPARISON cell (runs `eval_local_folder` and prints SID_Set-vs-WildFake
  binary accuracy Δ using the `finetuned` global). Inserted right after the
  `eval_local_folder` definition cell; given execution_count=None / outputs=[] (never run
  — the only output-less code cell, as expected).
- Re-commented the trailing `local_result = eval_local_folder(model, processor)` in the
  `eval_local_folder` cell so the (expensive) local eval isn't run twice — the new
  comparison cell now owns that run. Its prior printed output is left as a record.
- Verified: merged notebook is valid JSON, 27 cells; only the inserted cell lacks an
  execution_count; Drive mount present; `MAX_STEPS = 500` / `LEARNING_RATE = 5e-6`;
  trailing eval call commented; new cell's source exec's cleanly against stubbed globals
  (no NameError). `Qwen-Colab.ipynb` removed.

## 2026-08-30 — Qwen_v2.ipynb training-speed levers (#1–#3)

Context: v2 trains in ~13 h on one A100. At 2500 steps × effective-batch-8 =
20k images, that's ~18 s/step (~2.3 s/image) — far too slow for a 0.8B VLM, so
the GPU is starved by the single-process streaming input pipeline, not compute.
Applied three single-GPU / Colab-compatible fixes (no multi-GPU/DDP):

- **#1 Parallel data loading (cell 17, `SFTConfig`).** Added
  `dataloader_num_workers=8`, `dataloader_pin_memory=True`,
  `dataloader_persistent_workers=True`, `dataloader_prefetch_factor=4`. Moves the
  per-image network fetch + PIL augmentation + hashing off the main process so
  the A100 stops waiting. Biggest expected win.
- **#2 Real GPU batching, not grad-accum (cell 3 config).**
  `TRAIN_BATCH_SIZE 1→4`, `GRADIENT_ACCUMULATION_STEPS 8→2`. Product stays 8, so
  LR (5e-6), warmup, MAX_STEPS and images/step are unchanged — only wall-clock
  changes. Comment notes: try 8×1 if VRAM allows, fall back to 4×2/2×4 on OOM,
  keep the product = 8.
- **#3 One `.map()` instead of two (cell 16).** Replaced `_augment_example` +
  the prompt/completion `.map()` with a single `_prep_example` map (augment +
  attach prompt/completion in one pass), one generator layer per sample. Under
  #1 this runs in the worker processes.

Deliberately NOT done: DDP/multi-GPU (Colab is single-GPU; streaming
IterableDataset would also need `split_dataset_by_node` sharding), flash-attn /
torch.compile / LoRA. Verified: notebook re-parses as valid JSON; cells 3/16/17
pass `ast.parse`. Effective batch preserved, so training dynamics unchanged.

## 2026-08-30 — Fix: dataloader workers OOM-killed on Colab

`num_workers=8, prefetch_factor=4` (from lever #1) OOM-killed a worker on Colab:
`RuntimeError: DataLoader worker (pid ...) exited unexpectedly` — a CPU-RAM kill
(worker process reaped), not CUDA OOM (which surfaces in the main process). RAM
scales ~linearly with num_workers: each worker holds its own `buffer_size=10000`
shuffle buffer PLUS prefetch_factor batches of decoded+augmented full-res PILs.
Dialed to `dataloader_num_workers=2`, `dataloader_prefetch_factor=2` (~1/4 the
RAM, keeps most of the speedup). If it still OOMs: lower the shuffle buffer_size
(cell 2) or set num_workers=0. Note this was a CPU-RAM issue; batch size (#2,
per_device=4) is VRAM and was left unchanged.

## 2026-08-30 — Push batch + workers into the headroom

Colab live stats: CPU RAM 42/83GB, GPU RAM 11.2/40GB -> both under-used. Raised
two levers into the headroom, keeping effective batch = 8 (LR/step budget intact):
- `TRAIN_BATCH_SIZE 4->8`, `GRADIENT_ACCUMULATION_STEPS 2->1` (8x1=8): uses idle
  VRAM, fewer kernel launches/step. VRAM ~doubles (~22GB, still <40).
- `dataloader_num_workers 2->4`: more data-loading parallelism (the real
  bottleneck). CPU RAM ~2x the per-worker cost; watch it (shuffle buffer keeps
  filling) -- back off to 3 if it nears ~75GB.
Different memory pools, so OOM type disambiguates: CUDA OOM=batch, worker-death=
workers. NB: low memory usage is spare capacity, not a bug; the throughput signal
is GPU-UTIL %, not GPU memory.

## 2026-08-30 — Revert workers 4->2 (CPU RAM cliff)

num_workers=4 drove CPU RAM to 79.3/83.5GB (imminent OOM-kill). Two data points
(2w->42GB, 4w->79GB) => ~18GB/worker: the PIL augmentation pipeline (jpeg encode,
float32 noise arrays, resize copies) + replicated shuffle buffer is heavy per
worker. On this 83GB Colab, **num_workers=2 is the ceiling**. Reverted. GPU RAM
was only 15.8/40GB throughout, so the remaining speed headroom is GPU-side
(bf16/tf32/flash-attn/compile) or making augmentation cheaper -- NOT more workers.
Kept batch 8x1 (VRAM fine).

## 2026-08-30 — GPU-side accelerators: bf16, tf32, FlashAttention-2

The GPU sat at ~16/40GB (idle headroom), so added ~free GPU-side speedups:
- **cell 1 (install):** added `flash-attn --no-build-isolation` (proper dep, not
  an sdpa substitute per CLAUDE.md).
- **cell 5 (model load):** `dtype=torch.bfloat16` + `attn_implementation=
  "flash_attention_2"`, wrapped in try/except that falls back to `sdpa` with a
  loud warning if the FA2 wheel didn't build on Colab's python/torch combo (FA2
  still primary; fallback keeps the notebook runnable, doesn't silently downgrade
  a working FA2). FA2 needs half-precision weights + SM>=8.0 (A100=8.0, OK).
- **cell 17 (SFTConfig):** `bf16=True`, `tf32=True`.
No memory added; these cut compute time, complementing the (RAM-capped) 2-worker
data pipeline. Verified JSON valid; cells 5 & 17 `ast.parse` clean. Not executed
(Colab-only runtime).

## 2026-08-30 — FlashAttention-2 via prebuilt wheel (no compile)

Replaced the `flash-attn --no-build-isolation` source build (multi-minute CUDA
compile) with a prebuilt-wheel install in cell 1. flash-attention publishes
cp313 wheels (Colab py3.13) on GitHub releases; latest = v2.8.3.post1. The wheel
name encodes 4 axes that must match the runtime, so the cell DETECTS them and
builds the URL: python (cp<major><minor>), torch major.minor
(torch.__version__), CUDA major (torch.version.cuda -> cu12/cu13), and C++-ABI
(torch._C._GLIBCXX_USE_CXX11_ABI -> TRUE/FALSE). `!pip install "{url}" ||
pip install flash-attn --no-build-isolation` falls back to the source build if
the exact wheel 404s; model-load cell still falls back to SDPA beyond that.
Verified the builder emits an existing asset name (cu12torch2.8cxx11abiFALSE
cp313) for a typical Colab combo. Pinned tag v2.8.3.post1 (bump if Colab's torch
outruns the wheels). Not executed (Colab-only).

## 2026-08-30 — Colab torch 2.11 has no FA2 wheel -> use SDPA (drop source build)

Colab is now on torch 2.11.0+cu128. The runtime-detect URL correctly resolved
flash_attn ...torch2.11cxx11abiTRUE-cp313..., which 404s: FA2 prebuilt wheels
stop at torch2.9 (newest releases are FlashAttention-*4* betas with universal
py3-none-any wheels -- a different `flash_attn_4` package, not transformers'
`flash_attention_2`). The `||` fell into the source build, which is slow +
OOM-prone on Colab and risks torch-2.11 API incompatibility. Changed cell 1 to
install the wheel ONLY if it exists, else `echo` + SKIP (no compile). Model-load
cell already falls back to SDPA, which on A100+bf16 dispatches to the same FA2
kernel (~same speed, no build). Net: FA2 auto-used when a matching wheel exists
(supported torch), SDPA otherwise -- never a multi-minute compile. Not executed.

## 2026-08-30 — Simplify: SDPA directly, drop FA2 wheel machinery

Removed the flash-attn prebuilt-wheel detection/install block from cell 1
(reverted to the plain pip line) and replaced cell 5's FA2-try/except-SDPA with
a direct `attn_implementation="sdpa"` load (dtype=torch.bfloat16 kept). Rationale:
Colab torch 2.11 has no FA2 wheel and SDPA already uses the FA2 kernel on
A100+bf16, so the detection/build layers were dead weight. Cell 1 is now a single
install line again. bf16/tf32 in SFTConfig unchanged. Verified JSON valid; cell 5
ast.parse clean. Not executed.

## 2026-08-31 — Add `wildfake_stream` (ModelScope) as a second train source

Inserted a new code cell in `Qwen_v2.ipynb` immediately after the SID stream
cell (now cell index 3, right below `shuffled_stream = dataset.shuffle(...)`).
It streams `hy2628982280/WildFake` (default/train) via `MsDataset(use_streaming
=True)` and re-wraps it as a HF `datasets.IterableDataset` through
`IterableDataset.from_generator`, yielding `{image: PIL, label: int}` cast to the
SAME features as `sid_train` (image=Image(), label=int64) so it can interleave.

Two guarantees baked in:
- **Val-exclusion (anti-leak):** `_wfs_is_val()` drops the reference-benchmark
  rows — COCO reals and DALL-E "Advanced" fakes — so the brief's held-out val
  subset never enters training. All other generators' fakes + non-COCO reals are
  kept (that's the point: generator diversity).
- **Type match:** MsDataset's streaming object isn't a HF IterableDataset, so it
  can't feed `interleave_datasets`; the `from_generator` wrapper fixes that.

Self-contained (defines its own `_wfs_truthy/_wfs_haystack/_wfs_is_val/_wfs_image`
+ `import os`) because it runs before the later WildFake-eval helpers exist.
Falls back to `wildfake_stream=None` with a warning if modelscope/stream is
unreachable. Label convention = SID (0 Real / 1 Synthetic). The cell only creates
the variable + peeks; it does NOT rewire the interleave — a trailing comment shows
the one-liner to add it to the 'BUILD THE MIXED TRAIN SOURCE' cell.

Verified: notebook JSON valid, cell `ast.parse` clean, exclusion predicates
unit-tested on 5 synthetic rows (COCO/DALLE-adv excluded, SD/DALLE-non-adv/non-
COCO-real kept) — all pass. Not executed on Colab.

## 2026-08-31 — Fix: WildFake mix yields encoded-dict images -> decode in _prep_example

Enabling the WildFake interleave surfaced `AttributeError: 'dict' object has no
attribute 'mode'` in `augment_image` during `train()`. Cause: after
`interleave_datasets`, the WildFake branch (built via `from_dict ->
to_iterable_dataset`) yields the image as an ENCODED `{'bytes'|'path'}` dict
rather than a decoded PIL — the Image feature's decode flag isn't carried through
the interleave (SID samples still decode, so the peek passed on a SID sample and
training crashed on the first WildFake sample). Fix: added `_as_pil()` in the
`_prep_example` cell to normalise PIL-or-dict -> PIL before augmenting. JSON
valid, cell parses. ALSO NOTED (not yet changed): the mix collapses the stream to
`num_shards=1` (WildFake `to_iterable_dataset()` defaults to 1 shard, interleave
takes the min), so `dataloader_num_workers>1` is wasted ("Too many dataloader
workers... max is dataset.num_shards=1"). Remedy: `to_iterable_dataset(num_shards=8)`
in `_load_wildfake_train`, or set num_workers=1. Implication: the earlier mixed
run never completed, so there is still no with-mix AUC number.
