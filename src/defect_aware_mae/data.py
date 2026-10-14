"""NEU-DET loading and preprocessing (CLAHE -> 224x224 grayscale). Labels and boxes are returned for
evaluation only; no training code in this package reads them."""
import glob, hashlib, os, random, zipfile, xml.etree.ElementTree as ET
from pathlib import Path
import cv2, numpy as np, torch

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
IMG, P = 224, 16
N = (IMG // P) ** 2
# md5 of the preprocessed uint8 arrays; guarantees every run sees byte-identical inputs
CHECKSUMS = {"train": "69c8260a988c82ac1dd7c8d2912afe26", "validation": "59ce1173e44184ce5dc9bb64a1e4ef65"}


def repo_root():
    r = Path(__file__).resolve().parents[2]
    return r if (r / "pyproject.toml").exists() else Path.cwd()


def data_dir():
    """Directory holding NEU-DET/{train,validation}/{images,annotations}. Override with $DAMAE_DATA."""
    return Path(os.environ.get("DAMAE_DATA", repo_root() / "data"))


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def extract_archive(zip_path, dest=None):
    """Unpack the NEU-DET archive (top-level folder NEU-DET/) into the data directory."""
    dest = Path(dest or data_dir()); dest.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(zip_path).extractall(dest)
    return dest / "NEU-DET"


def load_split(split, cache=True, verify=False):
    """Returns uint8 images (n,224,224) after CLAHE, labels (n,), boxes (list of normalised xyxy arrays).
    split: 'train' (1,440 images) or 'validation' (360 images)."""
    root = data_dir(); cp = root / "cache" / f"{split}.npz"
    if cache and cp.exists():
        d = np.load(cp, allow_pickle=True); x, y, b = d["x"], d["y"], list(d["b"])
    else:
        src = root / "NEU-DET"
        if not src.exists():
            raise FileNotFoundError(f"{src} not found. Run `python scripts/prepare_data.py --archive <NEU-DET zip>` "
                                    f"or set $DAMAE_DATA (see README, section 'Data').")
        xs, ys, bs = [], [], []
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for ci, c in enumerate(CLASSES):
            for f in sorted(glob.glob(str(src / split / "images" / c / "*.jpg"))):
                im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                h, w = im.shape
                im = cv2.resize(clahe.apply(im), (IMG, IMG), interpolation=cv2.INTER_LINEAR)
                stem = os.path.splitext(os.path.basename(f))[0]
                # one annotation file is misfiled in the public archive, so look in both splits
                xp = glob.glob(str(src / "*" / "annotations" / (stem + ".xml")))[0]
                bx = []
                for o in ET.parse(xp).getroot().findall("object"):
                    bb = o.find("bndbox")
                    x0, y0, x1, y1 = (float(bb.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax"))
                    bx.append((x0 / w, y0 / h, x1 / w, y1 / h))
                xs.append(im); ys.append(ci); bs.append(np.array(bx, dtype=np.float32))
        x, y = np.stack(xs), np.array(ys); b = bs
        if cache:
            cp.parent.mkdir(parents=True, exist_ok=True)
            bo = np.empty(len(bs), dtype=object); bo[:] = bs
            np.savez(cp, x=x, y=y, b=bo)
    if verify:
        h = hashlib.md5(x.tobytes()).hexdigest()
        if h != CHECKSUMS[split]:
            raise RuntimeError(f"{split} checksum {h} != reference {CHECKSUMS[split]} "
                               f"(different OpenCV version or dataset copy; results may differ slightly)")
    return x, y, b
