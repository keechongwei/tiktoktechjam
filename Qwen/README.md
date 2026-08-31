# Robust Detection of AI-Generated Images Under Real-World Transformations

TechJam hackathon entry for **problem #5** — classify images as **AI-generated
(AIGC)** vs **authentic**, and stay accurate after real-world post-processing
(JPEG re-compression, blur, resize, noise, color jitter, center crop).

## Project overview

Online platforms increasingly need to flag synthetic imagery, but detectors that
only work on pristine lab images fall apart once a picture has been compressed,
cropped, filtered, or reposted. This project is a **hackathon-scale proof of
concept** for a detector that holds up under those transformations.

**Approach.** Rather than train a bespoke CNN classifier, we fine-tune a small
**vision-language model (VLM)** — `Qwen/Qwen3.5-0.8B` (< 2B parameters, per the
brief's constraint) — to *answer a classification question about the image*. The
model is shown an image and asked to label it, emitting its answer as a single
digit inside a LaTeX box (`\boxed{0}` / `\boxed{1}` / `\boxed{2}`) so the answer
is trivial to parse. Training data is the [`saberzl/SID_Set`](https://huggingface.co/datasets/saberzl/SID_Set)
dataset, **streamed** (`streaming=True`) because the full set is too large for
local/Colab disk.

**Label schema (SID_Set, 3-class).** The dataset is 3-class; the deliverable is
binary, so we collapse at inference time:

| SID_Set label | Meaning | Binary deliverable |
| :--- | :--- | :--- |
| `0` | Real (authentic photo) | authentic |
| `1` | Synthetic (fully AI-generated) | AIGC |
| `2` | Tampered (real photo, AI-edited regions) | AIGC |

Collapse rule: `{Synthetic, Tampered} → AIGC`, `{Real} → authentic`.

**Robustness strategy.** The target augmentations (JPEG q∈{90,70,50,30}, Gaussian
blur σ∈{0.5,1,2}, resize 0.5×/0.25×, Gaussian noise σ∈{0.02,0.05,0.10}, color
jitter ±20%, center-crop 80%) are applied to held-out images to measure the
clean-vs-transformed accuracy gap — a compact robustness summary rather than
lab-only numbers.

**Deliverable.** A script/notebook path that takes an image directory in and
writes a JSON file out, with `image_path` and `pred` (AIGC likelihood) per image.

### Tools, models & libraries

- **Runtime:** Google Colab (GPU), Jupyter notebook.
- **Model:** `Qwen/Qwen3.5-0.8B` (image-text-to-text VLM, < 2B params), loaded via
  `AutoModelForImageTextToText` + `AutoProcessor`.
- **Libraries:** Hugging Face `transformers`, `datasets`, `trl` (`SFTTrainer` /
  `SFTConfig`), `torch` / `torchvision`, `bitsandbytes` (8-bit AdamW),
  `scikit-learn` (macro-F1, balanced accuracy, confusion matrix), `modelscope`
  (WildFake validation benchmark), `tqdm`.
- **Datasets:** `saberzl/SID_Set` (training); a subset of WildFake — COCO val2017
  (non-AIGC) + DALL·E Advanced (AIGC) — as a **reference-only** validation
  benchmark. Per the brief, the validation set is **never trained on**.

## Repository layout

| File | Role |
| :--- | :--- |
| [`Qwen.ipynb`](Qwen.ipynb) | The working notebook: data streaming, prompt construction, fine-tuning, evaluation, and the WildFake validation benchmark. |
| [`problem_context.md`](problem_context.md) | The hackathon problem statement. |
| [`log.md`](log.md) | Reverse-chronological change log (what changed). |
| [`key-takeaways.md`](key-takeaways.md) | Plain-language learnings (what to remember and why). |

## Setup and installation

The notebook is built for **Google Colab** (GPU runtime), where dependencies are
reinstalled each session.

1. Open [`Qwen.ipynb`](Qwen.ipynb) in Colab and select a **GPU** runtime
   (Runtime → Change runtime type → GPU).
2. Run the first cell to install dependencies:

   ```bash
   pip install torch torchvision transformers datasets trl bitsandbytes \
       scikit-learn tqdm modelscope --break-system-packages
   ```

3. To run locally instead, use Python 3.10+ with a CUDA GPU and the same packages
   (drop `--break-system-packages` in a clean virtualenv).

> **Note on model names:** `Qwen/Qwen3.5-0.8B` is set in the config cell. If the
> checkpoint does not resolve on your hub, adjust `MODEL_NAME` (and, if needed,
> the `AutoModelForImageTextToText` loader / `trust_remote_code`) — see the
> model-load cell.

## Steps to reproduce your results

All configuration lives in **one cell** (the `CONFIGURATION` cell). Run the
notebook top to bottom:

1. **Configure.** In the config cell, set `MODEL_NAME`, training hyperparameters
   (`MAX_STEPS=78`, `LEARNING_RATE=2e-5`, `TRAIN_BATCH_SIZE=1`,
   `GRADIENT_ACCUMULATION_STEPS=4`, `OPTIM="adamw_8bit"`, `SEED=189`), and
   evaluation settings (`N_EVAL=100`, `EVAL_MAX_NEW_TOKENS=128`).
2. **Load model & data.** Run the model/processor load cell and the streaming
   `load_dataset("saberzl/SID_Set", streaming=True)` cell. A held-out slice of
   `N_EVAL` samples is reserved with `.take(N_EVAL)` and the rest (`.skip(N_EVAL)`)
   is used for training — the two are **disjoint**, so eval images are never
   trained on.
3. **Baseline evaluation.** Run `eval_sid_accuracy` **before** training to record
   the zero-shot baseline (3-class accuracy, binary accuracy, macro-F1, balanced
   accuracy, and a 3×3 confusion matrix).
4. **Fine-tune.** Run the `SFTTrainer` cell. Training uses **completion-only loss**
   (`completion_only_loss=True`) so the gradient targets only the answer digit,
   not the repeated prompt — this prevents the model collapsing to a single-class
   predictor. The data is fed as `{"prompt", "completion"}` columns.
5. **Post-training evaluation.** Re-run the eval harness for a side-by-side
   baseline-vs-fine-tuned comparison, including greedy and majority-vote (`mv@5`)
   results.
6. **Robustness summary.** Evaluate on clean vs transformed held-out images to
   produce the clean-vs-transformed accuracy table.
7. **Validation benchmark (reference only).** Run `eval_wildfake_binary` to score
   the WildFake subset (COCO val2017 authentic + DALL·E Advanced AIGC). This is a
   **benchmark, never a training source.**

### Sanity checks

- **Training loss should keep changing** (generally decreasing). A loss stuck flat
  near `0` from step 1 means the completion-only masking is misconfigured.
- **Confusion matrix should spread across the diagonal.** All predictions stuck in
  one column = the model has collapsed to a single class.

## Limitations & what we'd improve given more time

**Current limitations**

- **Prototype scale.** Training is capped at `MAX_STEPS=78` on a streamed subset
  with a tiny held-out eval (`N_EVAL=100`). The numbers demonstrate the approach,
  not a tuned final model — larger step counts and a bigger eval set are needed
  for stable metrics.
- **Collapse risk.** Because every example shares an identical prompt and differs
  only in the final digit, naive full-sequence loss collapses the model to
  "always Real." We mitigate this with completion-only loss, collapse-aware
  metrics (macro-F1 / balanced accuracy / confusion matrix), and majority voting —
  but voting cannot rescue an already-collapsed model, so the training config
  must be watched.
- **Generalisation.** Trained on SID_Set only; the WildFake benchmark is a single
  external check. Real-world generators and editing pipelines evolve, and the
  detector may not transfer to unseen generators.
- **Robustness coverage.** We measure robustness to the brief's augmentations but
  do not yet *train* on transformed images, so the clean-vs-transformed gap is a
  measurement, not something the model has been hardened against.
- **Unverified names.** `Qwen/Qwen3.5-0.8B` and the multimodal loader path have
  not been exhaustively verified against a live hub in every environment.

**What we'd improve with more time**

- **Robustness-aware training:** augment training images with the target
  transforms (JPEG/blur/resize/noise/jitter/crop) so the model learns invariance
  rather than merely being measured against them.
- **Bigger, cleaner evaluation:** raise `N_EVAL` and the WildFake caps toward the
  full 4998 / 8843, and add per-transform robustness curves.
- **Calibrated confidence:** `pred` is currently a hard AIGC likelihood; we'd
  calibrate it (e.g. from vote fractions or token probabilities) so the score is a
  meaningful probability, improving the false-positive/false-negative trade-off.
- **Error analysis:** a systematic gallery of representative false positives and
  false negatives, tied to which transforms most degrade accuracy.
- **Efficiency:** parameter-efficient fine-tuning (LoRA/QLoRA) to train longer on
  the same compute budget.

## Team member contributions

_Solo / single-participant entry — update this section with per-member
contributions if submitting as a team._
