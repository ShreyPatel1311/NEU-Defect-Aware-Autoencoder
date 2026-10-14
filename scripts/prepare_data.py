"""Unpack the NEU-DET archive into data/ and build the preprocessed cache (CLAHE + resize to 224).
usage: python scripts/prepare_data.py --archive /path/to/archive.zip"""
import argparse, hashlib
from defect_aware_mae.data import CHECKSUMS, extract_archive, load_split

ap = argparse.ArgumentParser(); ap.add_argument("--archive", help="NEU-DET zip (top-level folder NEU-DET/)")
a = ap.parse_args()
if a.archive: print("extracted to", extract_archive(a.archive))
for s in ("train", "validation"):
    x, y, b = load_split(s); h = hashlib.md5(x.tobytes()).hexdigest()
    print(f"{s}: {x.shape}, {len(set(y))} classes, {sum(len(i) for i in b)} boxes, md5 {h}",
          "(matches reference)" if h == CHECKSUMS[s] else "(DIFFERS from reference -- results may differ slightly)")
