#!/usr/bin/env python3
"""Portable, non-Colab inference script for the DevPost deliverable.

Scores every image in a directory with a trained VisionConfidenceScorer
checkpoint and writes a JSON list of {"image_path", "pred"}.

    pip install torch torchvision transformers pillow sentencepiece protobuf
    python score_directory.py \
        --input_dir path/to/images \
        --output_json predictions.json \
        --checkpoint path/to/classifier_epoch4.pt

`pred` is the model's confidence in [0, 1] that the image is AI-generated
(0 = confidently real, 1 = confidently AIGC). Images are scored as-is — no
transform is applied at inference time.
"""
import argparse
import json
import os

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoConfig
from tqdm import tqdm

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


class VisionConfidenceScorer(nn.Module):
    """Wraps a CLIP/SigLIP vision tower + a single-logit head for a confidence score."""

    def __init__(self, backbone_name, freeze_backbone=True):
        super().__init__()
        config = AutoConfig.from_pretrained(backbone_name)
        full_model = AutoModel.from_pretrained(backbone_name)
        self.vision_model = full_model.vision_model
        hidden_size = config.vision_config.hidden_size

        if freeze_backbone:
            for p in self.vision_model.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        return self.head(outputs.pooler_output).squeeze(-1)

    @torch.no_grad()
    def confidence(self, pixel_values):
        return torch.sigmoid(self.forward(pixel_values))


def score_directory(input_dir, output_json_path, checkpoint_path,
                     backbone_name="google/siglip-base-patch16-224"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    print("Loading model architecture and processor...")
    processor = AutoProcessor.from_pretrained(backbone_name)
    model = VisionConfidenceScorer(backbone_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

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
            pixel_values = processor(images=[image], return_tensors="pt")["pixel_values"].to(device)
            score = model.confidence(pixel_values).item()
            results.append({"image_path": file_path, "pred": round(score, 4)})
        except Exception as e:
            print(f"Skipped {filename}: {e}")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Saved {len(results)} predictions to {output_json_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="Directory of images to score")
    parser.add_argument("--output_json", required=True, help="Path to write predictions JSON")
    parser.add_argument("--checkpoint", required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--backbone", default="google/siglip-base-patch16-224",
                         help="HF model name the checkpoint was trained on")
    args = parser.parse_args()
    score_directory(args.input_dir, args.output_json, args.checkpoint, args.backbone)


if __name__ == "__main__":
    main()
