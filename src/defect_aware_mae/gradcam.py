"""Grad-CAM / Grad-CAM++ analysis of a frozen, head-less encoder.

Label-free explanation target: cosine similarity between an image's (mean-centred) feature and the centroid of the
cluster it was assigned to (k-means, K=6, on held-out validation features) -- "which pixels make this image look
like its cluster?". No class logits and no labels are involved.

Bounding boxes are used ONLY to score the maps (never in training):
  EiB       energy-in-box: fraction of CAM mass inside the union of ground-truth boxes
  EiB_lift  EiB / box-area fraction (1.0 = no better than a uniform map)
  PG        pointing game: the CAM maximum falls inside a ground-truth box
  del/ins   deletion / insertion AUC of the target score when 16x16 patches are blurred out / revealed in CAM
            order (lower del, higher ins = more faithful); a random patch order is reported alongside."""
import argparse, json, os
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .data import CLASSES, IMG, N, P, load_split, set_seed
from .models import Encoder, embed, load_encoder, readout


class _Feat(nn.Module):
    def __init__(self, enc, mode): super().__init__(); self.enc, self.mode = enc, mode
    def forward(self, x): return readout(self.enc, x, self.mode)


class _CosTarget:
    def __init__(self, c, mu): self.c, self.mu = c, mu
    def __call__(self, out): return F.cosine_similarity(out - self.mu, self.c - self.mu, dim=-1)


def _reshape(t): return t[:, 1:, :].reshape(t.shape[0], 14, 14, t.shape[-1]).permute(0, 3, 1, 2)


def box_mask(boxes):
    m = np.zeros((IMG, IMG), np.float32)
    for x0, y0, x1, y1 in boxes:
        m[int(y0 * IMG):int(np.ceil(y1 * IMG)), int(x0 * IMG):int(np.ceil(x1 * IMG))] = 1
    return m


class Explainer:
    """Holds an encoder, the held-out images, their label-free cluster assignment and centroids."""
    def __init__(self, src, mode="mid", device="cuda"):
        if src.startswith("random"):
            set_seed(int(src[6:] or 0)); self.enc = Encoder().to(device).eval()
        else:
            self.enc = load_encoder(src, device)
        for p in self.enc.parameters(): p.requires_grad_(True)
        self.mode, self.device = mode, device
        self.X, self.y, self.B = load_split("validation")
        Fv = embed(self.enc, self.X, mode)
        Z = PCA(50, random_state=0).fit_transform(StandardScaler().fit_transform(Fv))
        self.lab = KMeans(6, n_init=50, random_state=0).fit_predict(Z)
        self.mu = torch.from_numpy(Fv.mean(0)).float().to(device)
        self.cents = torch.stack([torch.from_numpy(Fv[self.lab == k].mean(0)) for k in range(6)]).float().to(device)

    def cams(self, idx, method="gradcam", layer=5, bs=12):
        """CAMs (len(idx),224,224) in [0,1]. layer: transformer block index 0-5; activations are taken at that
        block's first LayerNorm (the input to its self-attention)."""
        from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
        C = (GradCAM if method == "gradcam" else GradCAMPlusPlus)(
            model=_Feat(self.enc, self.mode).eval(), target_layers=[self.enc.blocks.layers[layer].norm1],
            reshape_transform=_reshape)
        out = []
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            xb = torch.from_numpy(self.X[j]).float().div(255)[:, None].to(self.device)
            out.append(C(input_tensor=xb, targets=[_CosTarget(self.cents[self.lab[k]], self.mu) for k in j]))
        return np.concatenate(out)

    def localization(self, maps, idx):
        e, a, pg = [], [], []
        for n, i in enumerate(idx):
            bm = box_mask(self.B[i]); m = maps[n] / (maps[n].sum() + 1e-8)
            e.append((m * bm).sum()); a.append(bm.mean()); pg.append(bm.ravel()[maps[n].argmax()] > 0)
        e, a = np.array(e), np.array(a)
        return {"EiB": float(e.mean()), "area": float(a.mean()),
                "EiB_lift": float((e / np.maximum(a, 1e-3)).mean()), "PG": float(np.mean(pg))}

    @torch.no_grad()
    def faithfulness(self, maps, idx, steps=14, seed=0):
        rng = np.random.RandomState(seed); res = {"del": [], "ins": [], "del_rand": [], "ins_rand": []}
        for n, i in enumerate(idx):
            x = self.X[i].astype(np.float32) / 255; xb = cv2.GaussianBlur(x, (0, 0), 8)
            pscore = maps[n].reshape(14, P, 14, P).mean((1, 3)).ravel()
            for name, order in (("", np.argsort(-pscore)), ("_rand", rng.permutation(N))):
                for kind in ("del", "ins"):
                    imgs = []
                    for s in range(steps + 1):
                        k = int(round(s * N / steps)); m = np.zeros(N, np.float32); m[order[:k]] = 1
                        m = np.kron(m.reshape(14, 14), np.ones((P, P), np.float32))
                        imgs.append(x * (1 - m) + xb * m if kind == "del" else xb * (1 - m) + x * m)
                    t = torch.from_numpy(np.stack(imgs))[:, None].to(self.device)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                        f = readout(self.enc, t, self.mode).float()
                    sc = F.cosine_similarity(f - self.mu, (self.cents[self.lab[i]] - self.mu)[None], dim=-1).cpu().numpy()
                    res[kind + name].append(np.trapezoid(sc, dx=1 / steps))
        return {k: float(np.mean(v)) for k, v in res.items()}


def run(src, name, mode="mid", out_dir="results/gradcam", n_layerwise=120, n_faith=120, save_maps=True, device="cuda"):
    """Full analysis on the 360 held-out images; writes <out_dir>/gradcam_<name>.json (+ CAM arrays)."""
    os.makedirs(out_dir, exist_ok=True)
    E = Explainer(src, mode, device); R = {"src": str(src), "mode": mode}
    idx = np.arange(len(E.X)); rng = np.random.RandomState(0)
    sub = np.concatenate([rng.choice(np.where(E.y == c)[0], n_layerwise // 6, replace=False) for c in range(6)])
    for method in ("gradcam", "gradcampp"):
        maps = E.cams(idx, method)
        if save_maps: np.save(f"{out_dir}/cams_{name}_{method}.npy", maps.astype(np.float16))
        R[method] = E.localization(maps, idx)
        R[method]["per_class"] = {CLASSES[c]: E.localization(maps[E.y == c], np.where(E.y == c)[0]) for c in range(6)}
        fsub = sub[:n_faith]
        R[method]["faithfulness"] = E.faithfulness(maps[fsub], fsub)
        R[method]["layerwise"] = {l: E.localization(E.cams(sub, method, l), sub) for l in range(6)}
        print(name, method, json.dumps({k: v for k, v in R[method].items() if k not in ("per_class", "layerwise")}), flush=True)
    R["cluster_labels"] = E.lab.tolist()
    json.dump(R, open(f"{out_dir}/gradcam_{name}.json", "w"), indent=1)
    return R


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="checkpoint path or 'random<seed>'"); ap.add_argument("name")
    ap.add_argument("--mode", default="mid", choices=["cls", "mean", "mid"])
    ap.add_argument("--out_dir", default="results/gradcam")
    a = ap.parse_args(argv); run(a.src, a.name, a.mode, a.out_dir)


if __name__ == "__main__":
    main()
