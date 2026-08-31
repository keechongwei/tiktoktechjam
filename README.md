# Robust AIGC Image Detection — SigLIP Linear Probe vs Qwen VLM with SFT

TikTok TechJam entry for **problem #5 — "Robust Detection of AI-Generated Images
Under Real-World Transformations."** This repo explores **two independent
detectors** for the same task and compares them, so we can decide which line is
worth carrying forward.

## The problem

Classify an image as **AI-generated (AIGC)** vs **authentic**, and *stay accurate*
after ordinary post-processing — JPEG re-compression, blur, resize, additive
noise, color jitter, and center crop. Constraints from the brief: the model must
be **< 2B parameters**, and the deliverable is a **directory-in / JSON-out** path
that writes `{"image_path", "pred"}` per image, where `pred` is the model's
`P(AIGC)` (0 = confidently authentic, 1 = confidently AIGC).

## Two approaches

| | Approach A: SigLIP linear probe | Approach B: Qwen VLM + SFT |
| :-- | :-- | :-- |
| Backbone | frozen `google/siglip-base-patch16-224` (~203M) | `Qwen/Qwen3.5-0.8B` (< 2B) |
| Trained part | small head only (`LN→Linear→GELU→Dropout→Linear`) | full model, SFT with completion-only loss |
| How it decides | one confidence logit → `P(AIGC)` | answers a question, emits `\boxed{0/1/2}` → collapsed to `P(AIGC)` |
| Robustness strategy | train-time augmentation on 70% of images (built into the objective) | v2 adds train-time augmentation + a WildFake generator mix |
| Deliverable | portable `score_directory.py` script | notebook path |
| Folder | [`SigLIP/`](SigLIP/) | [`Qwen/`](Qwen/) |

### Approach A — SigLIP linear probe

A frozen SigLIP vision tower feeds a small trainable head that outputs a single
confidence logit. Trained on [`saberzl/SID_Set`](https://huggingface.co/datasets/saberzl/SID_Set)
mixed with ~15% [`WildFake`](https://modelscope.cn/datasets/hy2628982280/WildFake/summary),
with the challenge's exact transforms applied to 70% of training images — so
robustness is trained *into* the model rather than hoped for. Ships a standalone
`score_directory.py` for the JSON deliverable. Full details in
[`SigLIP/README.md`](SigLIP/README.md).

### Approach B — Qwen VLM with SFT

Rather than a bespoke classifier, fine-tune a small vision-language model to
*answer a classification question about the image*, emitting a single boxed digit
that maps to the 3-class SID_Set schema (`0` real, `1` synthetic, `2` tampered),
collapsed to binary as `{synthetic, tampered} → AIGC`. Two iterations: a **v1**
baseline ([`Qwen/Qwen.ipynb`](Qwen/Qwen.ipynb)) and a **v2** robustness recipe
([`Qwen/Qwen_v2.ipynb`](Qwen/Qwen_v2.ipynb)) that layers train-time augmentation,
a SID+WildFake mix, soft `P(AIGC)` scores, a content-hash holdout, and a bigger
budget. Full details in [`Qwen/README.md`](Qwen/README.md).

## Head-to-head comparison

All numbers are **hackathon-scale** (small held-out sets, wide ±6–11pp confidence
intervals) — read tiers, not exact ranks.

| Axis | SigLIP probe | Qwen VLM + SFT |
| :-- | :-- | :-- |
| Training cost / iteration speed | **very low** — backbone frozen, head only | **high** — full-model SFT, larger budget (v2: 2500 steps) |
| Training stability | stable linear probe | collapse-prone; needs completion-only masking to work |
| Reported discrimination | AUROC / AP per transform in `robustness_summary.csv` (notebook-generated) | ~98% in-dist SID binary acc; ~66% cross-dataset (WildFake); ROC-AUC ≈ 0.70 |
| Generator generalization | light WildFake mix included to probe it | known gap — authentic-biased on unseen generators |
| Robustness posture | augmentation is part of the training objective | only additive noise clearly hurts (acc 0.66→0.56 at σ=0.10); other transforms confounded by calibration |
| Calibration | direct confidence logit | wants a threshold well below 0.5 — judge by AUC, not acc@0.5 |
| Explainability ceiling | low (single scalar) | **high** — can emit a rationale / extend the prompt |
| Deliverable maturity | portable standalone script | notebook only |

> Note: the SigLIP notebook reports robustness as per-transform CSVs
> (`robustness_summary.csv`, `by_source_summary.csv`) rather than one headline
> number; the Qwen figures are from [`Qwen/results.csv`](Qwen/results.csv) and
> [`Qwen/key-takeaways.md`](Qwen/key-takeaways.md).

## Verdict — which is more feasible for future development

**Near-term feasibility favors the SigLIP linear probe.** With the backbone
frozen and only a small head to train, it iterates cheaply and stably on the same
compute — no collapse to babysit, no completion-only masking to get exactly right.
Crucially, its robustness is *built into the training objective* (the exact
challenge transforms are applied during training) rather than measured after the
fact, and it already ships a portable inference script for the JSON deliverable.
Its upgrade path is obvious and low-risk: unfreeze the last few SigLIP blocks and
enlarge the WildFake mix.

**The Qwen VLM is the higher-ceiling but costlier bet.** Its real edge is
explainability and a flexible prompt interface — it can justify a call, not just
emit a scalar. But it carries a generator-generalization gap, calibration debt
(the optimum threshold sits well below 0.5), and collapse-prone, compute-hungry
training. That's more moving parts to stabilize before the ceiling pays off, and
on *clean* cross-dataset discrimination v2 was statistically indistinguishable
from v1 (ROC-AUC ~0.70 either way).

**Recommendation: build forward on the SigLIP probe as the primary line, and keep
the Qwen VLM as a research / explainability track.** This is hackathon-scale
evidence — small evals, wide intervals — not a final benchmark, so the call is
about which line is cheaper and safer to *develop*, not a permanent ranking.

## Repository layout

| Path | Role |
| :-- | :-- |
| [`SigLIP/`](SigLIP/) | Approach A — notebook (`SigLIP.ipynb`), portable `score_directory.py`, `requirements.txt`, `test_images/`, and its own README. |
| [`Qwen/`](Qwen/) | Approach B — `Qwen.ipynb` (v1), `Qwen_v2.ipynb` (v2), WildFake pull scripts, `wildfake_balanced/` eval set, `results.csv`, analysis notes, and its own README. |

Run steps, setup, and reproduction live in each folder's own README:
[`SigLIP/README.md`](SigLIP/README.md) and [`Qwen/README.md`](Qwen/README.md).

## Team

Jiang Zong Zhe
Kee Chong Wei
