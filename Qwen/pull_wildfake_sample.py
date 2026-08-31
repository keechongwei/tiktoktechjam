#!/usr/bin/env python3
"""Pull N images from a WildFake zip on ModelScope without downloading the whole
archive (each zip is 6-51 GB). Uses HTTP range requests via remotezip to read
only the zip index + the members we want.

    pip install remotezip pillow requests
    python pull_wildfake_sample.py

Outputs a folder of images and a single bundled zip.
"""
import os
import zipfile
import requests
from remotezip import RemoteZip

REPO = "hy2628982280/WildFake"
ZIP_PATH = "Images/Diffusion_based/DDIM.zip"   # smallest fake set (~6 GB); all synthetic, 256x256
N = 100
OUT_DIR = "wildfake_sample"
OUT_ZIP = "wildfake_100.zip"
EXTS = (".png", ".jpg", ".jpeg", ".webp")


def resolve_cdn_url(repo: str, filepath: str) -> str:
    """ModelScope's API endpoint 302-redirects to a CDN that supports range
    requests. remotezip must point at that CDN URL, not the API URL (the API
    rejects the suffix range remotezip uses to find the zip's end-of-directory).
    """
    api = f"https://modelscope.cn/api/v1/datasets/{repo}/repo?Revision=master&FilePath={filepath}"
    return requests.get(api, allow_redirects=False).headers["Location"]


def main() -> None:
    cdn = resolve_cdn_url(REPO, ZIP_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)

    with RemoteZip(cdn) as z:
        names = [n for n in z.namelist() if n.lower().endswith(EXTS)][:N]
        for n in names:
            with open(os.path.join(OUT_DIR, os.path.basename(n)), "wb") as f:
                f.write(z.read(n))

    # Bundle the folder into a single deliverable zip.
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_STORED) as bundle:
        for fn in sorted(os.listdir(OUT_DIR)):
            bundle.write(os.path.join(OUT_DIR, fn), fn)

    total_mb = sum(os.path.getsize(os.path.join(OUT_DIR, f))
                   for f in os.listdir(OUT_DIR)) / 1e6
    print(f"saved {len(names)} images to {OUT_DIR}/ ({total_mb:.1f} MB)")
    print(f"bundled -> {OUT_ZIP}")


if __name__ == "__main__":
    main()
