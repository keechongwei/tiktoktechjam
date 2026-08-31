#!/usr/bin/env python3
"""Build a balanced Real-vs-Diffusion test set from WildFake on ModelScope,
sampling equally across categories WITHOUT downloading the multi-GB zips.

Each category maps to one remote zip; we read only the zip index + the sampled
members via HTTP range requests (remotezip). The category fixes the binary
label (Real -> 0, Diffusion -> 1), so no label CSV lookup is needed.

    pip install remotezip pillow requests
    python pull_wildfake_balanced.py

Output:
    wildfake_balanced/<category>/<file>   sampled images, grouped by category
    wildfake_balanced/labels.csv          image_path,category,generator,label

Balance: PER_CATEGORY images are drawn from every category. With an equal number
of real and fake categories (the default), the set is balanced both across
categories AND across the two classes (50/50 real/fake).
"""
import os
import csv
import random
from collections import defaultdict
import requests
from remotezip import RemoteZip

REPO = "hy2628982280/WildFake"
OUT_DIR = "wildfake_balanced"
PER_CATEGORY = 20          # images sampled from each category
SEED = 0                   # reproducible sampling
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# If True, also include the categories that overlap the official validation
# reference (COCO real + DALL-E fake). Leave False for a held-out test set that
# does NOT collide with the val set (per the hackathon brief). See CLAUDE.md.
INCLUDE_VAL_OVERLAP = False

# category -> zip path(s) inside the repo. A value may be a single zip (str) or
# a list of part zips (multi-part categories like Midjourney, whose images are
# split across ~50 GB parts). Real categories carry label 0, diffusion label 1.
MJ_TYPICAL = [f"Images/Diffusion_based/Midjourney/Typical/part_{i}.zip"
              for i in range(1, 5)]                    # 4 parts
MJ_ADVANCED = [f"Images/Diffusion_based/Midjourney/Advanced/part_{i}.zip"
               for i in range(1, 8)]                   # 7 parts (val tier)

REAL = {
    "afhq":      "Images/Real/afhq.zip",
    "celebahq":  "Images/Real/celebahq.zip",
    "church":    "Images/Real/church.zip",
    "ffhq":      "Images/Real/ffhq.zip",
    "imagenet":  "Images/Real/imagenet.zip",
    "laion5b":   "Images/Real/laion5b.zip",
    # "wukong" omitted: the zip is 164 B (empty/broken).
}
FAKE = {
    "DDIM":          "Images/Diffusion_based/DDIM.zip",
    "DDPM":          "Images/Diffusion_based/DDPM.zip",
    "Imagen":        "Images/Diffusion_based/Imagen.zip",
    "VQDM":          "Images/Diffusion_based/VQDM.zip",
    "ADM":           "Images/Diffusion_based/ADM.zip",
    "SDwithAdaptor": "Images/Diffusion_based/SD/SDwithAdaptor.zip",
    "Midjourney":    MJ_TYPICAL,   # multi-part; Typical tier (non-val)
}
# Validation-overlap categories (added only when INCLUDE_VAL_OVERLAP is True).
# Midjourney's Advanced tier is the val-reference half, so it lives here.
REAL_VAL = {"coco": "Images/Real/coco.zip"}
FAKE_VAL = {
    "DALLE":              "Images/Diffusion_based/DALLE.zip",
    "Midjourney_Advanced": MJ_ADVANCED,
}


def resolve_cdn_url(filepath: str) -> str:
    """ModelScope's API endpoint 302-redirects to a range-request-capable CDN.
    remotezip must point at that CDN URL, not the API URL."""
    api = (f"https://modelscope.cn/api/v1/datasets/{REPO}/repo"
           f"?Revision=master&FilePath={filepath}")
    return requests.get(api, allow_redirects=False).headers["Location"]


def sample_category(name: str, zip_paths, label: int, rng: random.Random):
    """Pull PER_CATEGORY random images from one category. `zip_paths` is a single
    zip (str) or a list of part zips. For multi-part categories we read part
    indices in order only until the candidate pool is large enough to sample
    from, so we avoid opening every 50 GB part. Returns rows for labels.csv."""
    if isinstance(zip_paths, str):
        zip_paths = [zip_paths]
    dest = os.path.join(OUT_DIR, name)
    os.makedirs(dest, exist_ok=True)

    # Open part zips one at a time, keeping each handle open, and accumulate
    # (zip_path, member) candidates until the pool is deep enough (or parts run
    # out). Keeping handles open means each part index is read exactly once --
    # both the listing below and the reads afterward reuse the same handle. One
    # part is plenty for a small PER_CATEGORY.
    pool_target = PER_CATEGORY * 100
    handles, candidates = {}, []
    try:
        for zp in zip_paths:
            z = RemoteZip(resolve_cdn_url(zp))
            handles[zp] = z
            candidates += [(zp, m) for m in z.namelist()
                           if m.lower().endswith(IMG_EXTS)]
            if len(candidates) >= pool_target:
                break

        chosen = rng.sample(candidates, min(PER_CATEGORY, len(candidates)))
        rows = []
        for zp, member in chosen:
            fname = os.path.basename(member)
            with open(os.path.join(dest, fname), "wb") as f:
                f.write(handles[zp].read(member))
            rows.append([os.path.join(name, fname), name, name, label])
    finally:
        for z in handles.values():
            z.close()

    part_note = (f" ({len(handles)}/{len(zip_paths)} parts)"
                 if len(zip_paths) > 1 else "")
    print(f"  {name:<20} label={label}  {len(chosen):>3} / "
          f"{len(candidates)} images{part_note}")
    return rows


def main() -> None:
    rng = random.Random(SEED)
    real = {**REAL, **(REAL_VAL if INCLUDE_VAL_OVERLAP else {})}
    fake = {**FAKE, **(FAKE_VAL if INCLUDE_VAL_OVERLAP else {})}
    os.makedirs(OUT_DIR, exist_ok=True)

    all_rows = []
    print(f"Sampling {PER_CATEGORY}/category "
          f"({len(real)} real + {len(fake)} fake categories):")
    for name, zp in real.items():
        all_rows += sample_category(name, zp, 0, rng)
    for name, zp in fake.items():
        all_rows += sample_category(name, zp, 1, rng)

    with open(os.path.join(OUT_DIR, "labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "category", "generator", "label"])
        w.writerows(all_rows)

    n_real = sum(1 for r in all_rows if r[3] == 0)
    n_fake = sum(1 for r in all_rows if r[3] == 1)
    print(f"\nTotal {len(all_rows)} images  ->  real(0)={n_real}  fake(1)={n_fake}")
    print(f"labels.csv written to {OUT_DIR}/labels.csv")


if __name__ == "__main__":
    main()
