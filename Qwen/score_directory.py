#!/usr/bin/env python3
"""Portable inference script for the DevPost deliverable (Qwen VLM version).

Scores every image in a directory with the trained Qwen3.5 multimodal model and
writes a JSON list of {"image_path", "pred"}.

    pip install -r requirements.txt   # transformers>=5.15.1 is required
    python score_directory.py \
        --input_dir path/to/images \
        --output_json predictions.json \
        --model_id WallyLovesCats/Qwen3.5-0.8B-TTTJ

`pred` is the model's likelihood in [0, 1] that the image is AI-generated
(0 = confidently authentic, 1 = confidently AIGC). Unlike a classifier head,
this model *generates* an answer: it is prompted to reply with \boxed{A} (AIGC)
or \boxed{B} (authentic). We read the probability mass the model assigns to the
"A" vs "B" answer token to get a continuous score, and fall back to parsing the
generated \boxed{...} text if that logit read fails.

NOTE on the backbone: this repo has model_type "qwen3_5", recognised only by
`transformers>=5.15.1` (the version it was exported with). Older releases fail
with "qwen3_5 is not a recognized model type". There are no custom modeling .py
files in the repo, so we cannot fall back to trust_remote_code for the
architecture — the pin in requirements.txt is mandatory.
"""
import argparse
import json
import os
import re

import inspect

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# The two answer options. Keep the letters in sync with QUESTION below and with
# the tutorial's \boxed{A|B} convention so downstream parsing stays reusable.
QUESTION = (
    "Look at this image and decide whether it was generated or substantially "
    "edited by an AI image generator, or whether it is an authentic real "
    "photograph.\n"
    "Answer with exactly one letter inside \\boxed{}: \\boxed{A} if the image "
    "is AI-generated, \\boxed{B} if it is an authentic real photo. "
    "Reply with only the boxed letter."
)
AIGC_LETTER = "A"       # -> pred near 1.0
AUTHENTIC_LETTER = "B"  # -> pred near 0.0


def load_processor(model_id):
    """Load the processor, rebuilding it if the repo lacks preprocessor_config.json.

    WallyLovesCats/Qwen3.5-0.8B-TTTJ ships only tokenizer files (no image-processor
    config), so `AutoProcessor.from_pretrained` raises "Can't load image processor".
    We reconstruct it from parts: the tokenizer + chat template come from the repo,
    and the image/video processors are instantiated with class defaults (which for
    Qwen are the correct CLIP normalisation values, so scores stay valid). The class
    names are resolved from transformers' auto-mappings via the model_type, so we
    don't hard-code Qwen3.5-specific names.
    """
    try:
        return AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"AutoProcessor.from_pretrained failed ({e}).")
        print("Rebuilding processor from parts (repo has no preprocessor_config.json)...")

    import transformers
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING_NAMES
    from transformers.models.auto.processing_auto import PROCESSOR_MAPPING_NAMES

    config = AutoConfig.from_pretrained(model_id)
    mtype = config.model_type
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def collect_names(val):
        """Flatten a mapping value (str / list / tuple / dict) to class-name strings."""
        if val is None:
            return []
        if isinstance(val, str):
            return [val]
        if isinstance(val, dict):
            names = []
            for v in val.values():
                names.extend(collect_names(v))
            return names
        if isinstance(val, (list, tuple)):
            names = []
            for v in val:
                names.extend(collect_names(v))
            return names
        return []

    def instantiate_from_mapping(mapping, extra_kwargs=None):
        """Instantiate the class a mapping gives for this model_type, prefer *Fast.

        `extra_kwargs` are filtered to what the class __init__ accepts, so we can
        force the model's real vision geometry (patch/merge sizes) instead of the
        class defaults — otherwise resize and patchify disagree and the tensor
        reshape fails.
        """
        names = collect_names(mapping.get(mtype))
        # Prefer torchvision-backed "Fast" processors, then fall back to slow.
        names.sort(key=lambda n: 0 if n.endswith("Fast") else 1)
        for name in names:
            cls = getattr(transformers, name, None)
            if cls is None:
                continue
            kwargs = {}
            if extra_kwargs:
                params = inspect.signature(cls.__init__).parameters
                # Qwen image processors take only **kwargs, so a name-in-params
                # filter would drop everything; pass all when **kwargs is present.
                has_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                )
                if has_var_kw:
                    kwargs = dict(extra_kwargs)
                else:
                    kwargs = {k: v for k, v in extra_kwargs.items() if k in params}
            return cls(**kwargs)
        return None

    # Force the image processor's geometry from the model's vision_config so the
    # resize step aligns with the 16x16 patchify step.
    vc = getattr(config, "vision_config", None)
    vision_kwargs = {}
    if vc is not None:
        for src, dst in (("patch_size", "patch_size"),
                         ("temporal_patch_size", "temporal_patch_size"),
                         ("spatial_merge_size", "merge_size")):
            if hasattr(vc, src):
                vision_kwargs[dst] = getattr(vc, src)

    image_processor = instantiate_from_mapping(IMAGE_PROCESSOR_MAPPING_NAMES, vision_kwargs)
    if image_processor is None:
        raise RuntimeError(f"No image-processor class registered for model_type '{mtype}'")

    video_processor = None
    try:
        from transformers.models.auto.video_processing_auto import VIDEO_PROCESSOR_MAPPING_NAMES
        video_processor = instantiate_from_mapping(VIDEO_PROCESSOR_MAPPING_NAMES, vision_kwargs)
    except Exception:
        pass  # older/newer transformers may not expose video processors; images only

    proc_name = PROCESSOR_MAPPING_NAMES.get(mtype)
    if proc_name is None:
        raise RuntimeError(f"No processor class registered for model_type '{mtype}'")
    ProcClass = getattr(transformers, proc_name)

    # Pass only the args this processor's __init__ actually accepts.
    params = inspect.signature(ProcClass.__init__).parameters
    kwargs = {}
    if "image_processor" in params:
        kwargs["image_processor"] = image_processor
    if "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    if "video_processor" in params and video_processor is not None:
        kwargs["video_processor"] = video_processor
    if "chat_template" in params:
        kwargs["chat_template"] = getattr(tokenizer, "chat_template", None)
    return ProcClass(**kwargs)


def load_model(model_id, device):
    """Load the Qwen VLM, trying the auto-classes it may be registered under."""
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    from transformers import AutoModelForCausalLM  # noqa: F401  (fallback below)

    last_err = None
    for cls_name in (
        "AutoModelForMultimodalLM",     # what this repo's config advertises
        "AutoModelForImageTextToText",  # standard image-text-to-text auto class
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    ):
        import transformers
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(
                model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            print(f"Loaded model with {cls_name}")
            return model.to(device).eval()
        except Exception as e:  # try the next auto-class
            last_err = e
    raise RuntimeError(
        f"Could not load {model_id} with any known auto-class. Last error:\n"
        f"{last_err}\n"
        "If this says 'qwen3_5 is not a recognized model type', install "
        "transformers>=5.15.1 (see requirements.txt)."
    )


def letter_token_ids(tokenizer, letter):
    """Token ids that decode to a lone answer letter (with/without a space)."""
    ids = set()
    for variant in (letter, " " + letter):
        enc = tokenizer.encode(variant, add_special_tokens=False)
        if len(enc) == 1:
            ids.add(enc[0])
    return ids


def build_prompt(processor, image):
    """Render the chat prompt with an image placeholder + our question."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": QUESTION},
            ],
        }
    ]
    # Processors expose apply_chat_template; fall back to the tokenizer's.
    templater = getattr(processor, "apply_chat_template", None)
    if templater is None:
        templater = processor.tokenizer.apply_chat_template
    return templater(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def score_image(model, processor, image, device, a_ids, b_ids, max_new_tokens):
    """Return P(AIGC) in [0, 1] for a single PIL image."""
    prompt_text = build_prompt(processor, image)
    inputs = processor(text=[prompt_text], images=[image], return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,                 # deterministic for reproducible scoring
        return_dict_in_generate=True,
        output_scores=True,
    )
    gen_ids = gen.sequences[0][input_len:]
    scores = gen.scores  # tuple: one [1, vocab] logit tensor per generated token

    # Primary path: find the first generated token that is the answer letter and
    # read the softmax mass on "A" vs "B" at that step -> continuous confidence.
    for step, tok_id in enumerate(gen_ids.tolist()):
        decoded = processor.tokenizer.decode([tok_id]).strip()
        if tok_id in a_ids or tok_id in b_ids or decoded in (AIGC_LETTER, AUTHENTIC_LETTER):
            probs = torch.softmax(scores[step][0].float(), dim=-1)
            p_a = probs[list(a_ids)].sum().item() if a_ids else 0.0
            p_b = probs[list(b_ids)].sum().item() if b_ids else 0.0
            if p_a + p_b > 0:
                return p_a / (p_a + p_b)
            # ids didn't line up with vocab; use the decoded letter as hard label
            return 1.0 if decoded == AIGC_LETTER else 0.0

    # Fallback: parse \boxed{X} from the fully decoded generation.
    text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
    m = re.search(r"\\boxed\{\s*([AB])\s*\}", text)
    if m:
        return 1.0 if m.group(1) == AIGC_LETTER else 0.0
    m = re.search(r"\b([AB])\b", text)  # last resort: any lone A/B
    if m:
        return 1.0 if m.group(1) == AIGC_LETTER else 0.0
    return 0.5  # model gave no parseable answer -> maximally uncertain


def score_directory(input_dir, output_json_path, model_id, max_new_tokens):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    print("Loading processor and model...")
    processor = load_processor(model_id)
    model = load_model(model_id, device)

    tok = processor.tokenizer
    a_ids = letter_token_ids(tok, AIGC_LETTER)
    b_ids = letter_token_ids(tok, AUTHENTIC_LETTER)

    image_files = sorted(
        f for f in os.listdir(input_dir) if f.lower().endswith(VALID_EXTENSIONS)
    )
    if not image_files:
        print(f"No valid images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images. Scoring...")
    results = []
    for filename in tqdm(image_files, desc="Scoring"):
        file_path = os.path.join(input_dir, filename)
        try:
            image = Image.open(file_path).convert("RGB")
            score = score_image(model, processor, image, device, a_ids, b_ids, max_new_tokens)
            results.append({"image_path": file_path, "pred": round(float(score), 4)})
        except Exception as e:
            print(f"Skipped {filename}: {e}")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Saved {len(results)} predictions to {output_json_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="Directory of images to score")
    parser.add_argument("--output_json", required=True, help="Path to write predictions JSON")
    parser.add_argument("--model_id", default="WallyLovesCats/Qwen3.5-0.8B-TTTJ",
                        help="HF repo id (or local path) of the trained Qwen VLM")
    parser.add_argument("--max_new_tokens", type=int, default=24,
                        help="Generation budget; the answer is a short boxed letter")
    args = parser.parse_args()
    score_directory(args.input_dir, args.output_json, args.model_id, args.max_new_tokens)


if __name__ == "__main__":
    main()
