#!/usr/bin/env python3
"""Build a WildFake TRAIN subset for model v2 -- larger than the eval set and
GUARANTEED disjoint from it.

This is the generator-diversity half of the v2 recipe: SID_Set alone overfits
its own generators (VQDM 0%, weak diffusion tier on the cross-dataset eval), so
v2 mixes streamed SID_Set with these locally-pulled WildFake images.

Two hard rules, both enforced here:
  1. NEVER pull the official validation-reference categories (COCO real +
     DALL-E fake + any "Advanced" tier). INCLUDE_VAL_OVERLAP stays False.
  2. NEVER reuse an image that is already in the EVAL set
     (`wildfake_balanced/`). We load that set's filenames and exclude them, so
     `wildfake_train/` and `wildfake_balanced/` are disjoint by construction --
     no train/eval leak.

Same disk-light mechanism as `pull_wildfake_balanced.py`: read only the zip
index + the sampled members over HTTP range requests (remotezip), never the
multi-GB archives.

    pip install remotezip pillow requests
    python pull_wildfake_train.py

Output:
    wildfake_train/<category>/<file>   sampled images, grouped by category
    wildfake_train/labels.csv          image_path,category,generator,label
"""
import os
import csv
import random
from remotezip import RemoteZip
import requests

REPO = "hy2628982280/WildFake"
OUT_DIR = "wildfake_train"
EVAL_DIR = "wildfake_balanced"     # the held-out eval set -- excluded from train
PER_CATEGORY = 300                 # >> the eval set's 20/category (diversity budget)
SEED = 1                           # different seed from the eval pull (SEED=0)
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# Keep False: the val-reference categories (COCO / DALL-E / Advanced tiers) must
# never enter training (hackathon brief + CLAUDE.md).
INCLUDE_VAL_OVERLAP = False

# category -> zip path(s). Mirrors pull_wildfake_balanced.py so the two scripts
# sample from the same non-val category pool (and thus can be de-duplicated).
MJ_TYPICAL = [f"Images/Diffusion_based/Midjourney/Typical/part_{i}.zip"
              for i in range(1, 5)]

REAL = {
    "afhq":      "Images/Real/afhq.zip",
    "celebahq":  "Images/Real/celebahq.zip",
    "church":    "Images/Real/church.zip",
    "ffhq":      "Images/Real/ffhq.zip",
    "imagenet":  "Images/Real/imagenet.zip",
    "laion5b":   "Images/Real/laion5b.zip",
}
FAKE = {
    "DDIM":          "Images/Diffusion_based/DDIM.zip",
    "DDPM":          "Images/Diffusion_based/DDPM.zip",
    "Imagen":        "Images/Diffusion_based/Imagen.zip",
    "VQDM":          "Images/Diffusion_based/VQDM.zip",
    "ADM":           "Images/Diffusion_based/ADM.zip",
    "SDwithAdaptor": "Images/Diffusion_based/SD/SDwithAdaptor.zip",
    "Midjourney":    MJ_TYPICAL,
}


def load_eval_basenames() -> set:
    """Filenames already used by the EVAL set (`wildfake_balanced/labels.csv`),
    so we never sample them into TRAIN. Returns an empty set if the eval set has
    not been pulled yet (nothing to exclude)."""
    csv_path = os.path.join(EVAL_DIR, "labels.csv")
    if not os.path.exists(csv_path):
        print(f"NOTE: {csv_path} not found -- no eval images to exclude.")
        return set()
    names = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            names.add(os.path.basename(row["image_path"]))
    print(f"Excluding {len(names)} eval filenames from the train pull.")
    return names


def resolve_cdn_url(filepath: str) -> str:
    """ModelScope's API endpoint 302-redirects to a range-request-capable CDN.
    remotezip must point at that CDN URL, not the API URL."""
    api = (f"https://modelscope.cn/api/v1/datasets/{REPO}/repo"
           f"?Revision=master&FilePath={filepath}")
    return requests.get(api, allow_redirects=False).headers["Location"]


def sample_category(name, zip_paths, label, rng, exclude):
    """Pull PER_CATEGORY random images from one category, skipping any member
    whose filename is in `exclude` (the eval set). Same handle-caching /
    read-parts-in-order logic as the balanced sampler."""
    if isinstance(zip_paths, str):
        zip_paths = [zip_paths]
    dest = os.path.join(OUT_DIR, name)
    os.makedirs(dest, exist_ok=True)

    pool_target = PER_CATEGORY * 100
    handles, candidates = {}, []
    try:
        for zp in zip_paths:
            z = RemoteZip(resolve_cdn_url(zp))
            handles[zp] = z
            candidates += [(zp, m) for m in z.namelist()
                           if m.lower().endswith(IMG_EXTS)
                           and os.path.basename(m) not in exclude]
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
    exclude = load_eval_basenames()
    os.makedirs(OUT_DIR, exist_ok=True)

    all_rows = []
    print(f"Sampling {PER_CATEGORY}/category "
          f"({len(REAL)} real + {len(FAKE)} fake categories):")
    for name, zp in REAL.items():
        all_rows += sample_category(name, zp, 0, rng, exclude)
    for name, zp in FAKE.items():
        all_rows += sample_category(name, zp, 1, rng, exclude)

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
