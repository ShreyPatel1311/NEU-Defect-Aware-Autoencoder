"""Frozen-encoder evaluation protocol.

Labels are used ONLY to score clusterings (and for the kNN / linear-probe reference metrics, which are reported
separately as label-using). The encoder never saw the validation images during pretraining.

A  held-out, transductive : cluster the 360 validation images on their own
     - raw 768-d features (k-means, Ward), PCA-50 (k-means, Ward), UMAP-2 + Ward
     - robustness sweep: UMAP n_neighbors x min_dist (12 settings) x 10 seeds
B  held-out, inductive    : fit scaler/PCA/UMAP/k-means on train features, assign validation images
C  reference (uses labels): 20-NN and linear probe, train -> validation
D  all 1,800 images through the UMAP-2 + Ward pipeline (the protocol of the earlier v1 study)"""
import argparse, json, warnings
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, normalize
from .data import load_split, set_seed
from .models import Encoder, embed, load_encoder

warnings.filterwarnings("ignore")
READOUTS = ("cls", "mean", "mid")


def hungarian_accuracy(y, c):
    """Clustering accuracy under the optimal one-to-one cluster -> class assignment."""
    k = max(y.max(), c.max()) + 1
    M = np.zeros((k, k), int)
    for a, b in zip(c, y): M[a, b] += 1
    r, cc = linear_sum_assignment(-M)
    return M[r, cc].sum() / len(y)


def hungarian_map(y, c):
    """Cluster ids relabelled to their matched class ids."""
    k = max(y.max(), c.max()) + 1
    M = np.zeros((k, k), int)
    for a, b in zip(c, y): M[a, b] += 1
    r, cc = linear_sum_assignment(-M); mp = dict(zip(r, cc))
    return np.array([mp[i] for i in c])


def score(y, c, Z):
    return {"acc": hungarian_accuracy(y, c), "ari": ARI(y, c), "nmi": NMI(y, c),
            "sil": float(silhouette_score(Z, c)) if len(set(c)) > 1 else 0.0}


def umap2(F, n_neighbors=25, min_dist=0.1, seed=742, dim=2):
    import umap
    return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=dim, metric="cosine",
                     random_state=seed).fit_transform(F)


def get_features(src, mode, device=None):
    """src: checkpoint path | 'raw' (4x average-pooled pixels) | 'random<seed>' (untrained encoder)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Xtr, ytr, _ = load_split("train"); Xva, yva, _ = load_split("validation")
    if src == "raw":
        f = lambda X: torch.nn.functional.avg_pool2d(torch.from_numpy(X).float()[:, None], 4).flatten(1).numpy()
        return f(Xtr), ytr, f(Xva), yva
    if src.startswith("random"):
        set_seed(int(src[6:] or 0)); enc = Encoder().to(device).eval()
    else:
        enc = load_encoder(src, device)
    return embed(enc, Xtr, mode), ytr, embed(enc, Xva, mode), yva


def evaluate_features(Ftr, ytr, Fva, yva, sweep_seeds=10, quick=False):
    R = {}
    sc = StandardScaler().fit(Ftr)
    Str, Sva = sc.transform(Ftr), sc.transform(Fva)
    Sva_own = StandardScaler().fit_transform(Fva)

    A = {}
    Z = normalize(Sva_own)
    A["raw768_kmeans"] = score(yva, KMeans(6, n_init=50, random_state=0).fit_predict(Z), Z)
    A["raw768_ward"] = score(yva, AgglomerativeClustering(6).fit_predict(Z), Z)
    Zp = PCA(50, random_state=0).fit_transform(Sva_own)
    A["pca50_kmeans"] = score(yva, KMeans(6, n_init=50, random_state=0).fit_predict(Zp), Zp)
    A["pca50_ward"] = score(yva, AgglomerativeClustering(6).fit_predict(Zp), Zp)
    Zu = umap2(Sva_own)
    A["umap2_ward_v1cfg"] = score(yva, AgglomerativeClustering(6).fit_predict(Zu), Zu)
    grid = [(n, m) for n in (10, 15, 25, 50) for m in (0.0, 0.1, 0.5)] if not quick else [(25, 0.1)]
    sw = {}
    for n, m in grid:
        accs = [hungarian_accuracy(yva, AgglomerativeClustering(6).fit_predict(umap2(Sva_own, n, m, s)))
                for s in range(sweep_seeds if not quick else 3)]
        sw[f"nn{n}_md{m}"] = [float(np.mean(accs)), float(np.std(accs))]
    allacc = [v[0] for v in sw.values()]
    A["umap2_ward_sweep"] = {"cells": sw, "mean_over_cells": float(np.mean(allacc)),
                             "min_cell": float(np.min(allacc)), "max_cell": float(np.max(allacc))}
    R["heldout_transductive"] = A

    import umap
    B = {}
    pca = PCA(50, random_state=0).fit(Str)
    km = KMeans(6, n_init=50, random_state=0).fit(pca.transform(Str))
    B["pca50_kmeans"] = score(yva, km.predict(pca.transform(Sva)), pca.transform(Sva))
    B["pca50_kmeans_trainfit"] = score(ytr, km.labels_, pca.transform(Str))
    accs = []
    for s in range(5 if not quick else 2):
        um = umap.UMAP(n_neighbors=25, min_dist=0.1, metric="cosine", random_state=s).fit(Str)
        km2 = KMeans(6, n_init=20, random_state=0).fit(um.embedding_)
        accs.append(hungarian_accuracy(yva, km2.predict(um.transform(Sva))))
    B["umap2_kmeans"] = [float(np.mean(accs)), float(np.std(accs))]
    R["heldout_inductive"] = B

    # label-free score: silhouette of k-means(6) on PCA-50 of TRAIN features
    Ztr = PCA(50, random_state=0).fit_transform(Str)
    R["train_unsup_silhouette"] = float(silhouette_score(Ztr, KMeans(6, n_init=20, random_state=0).fit_predict(Ztr)))

    Ntr, Nva = normalize(Str), normalize(Sva)
    R["knn20"] = KNeighborsClassifier(20, metric="cosine").fit(Ntr, ytr).score(Nva, yva)
    R["linear"] = LogisticRegression(max_iter=3000, C=0.1).fit(Str, ytr).score(Sva, yva)

    Fall = StandardScaler().fit_transform(np.concatenate([Ftr, Fva])); yall = np.concatenate([ytr, yva])
    Zu = umap2(Fall)
    R["all1800_umap2_ward_v1cfg"] = score(yall, AgglomerativeClustering(6).fit_predict(Zu), Zu)
    return R


def run(src, mode="mid", quick=False, device=None):
    R = evaluate_features(*get_features(src, mode, device), quick=quick)
    return {"src": str(src), "mode": mode, **R}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="checkpoint path, 'raw', or 'random<seed>'")
    ap.add_argument("--modes", nargs="+", default=list(READOUTS), choices=READOUTS)
    ap.add_argument("--out", help="write JSON {readout: results} here")
    ap.add_argument("--quick", action="store_true", help="skip the full UMAP sweep")
    a = ap.parse_args(argv)
    R = {m: run(a.src, m, a.quick) for m in a.modes}
    s = json.dumps(R, indent=1, default=float)
    if a.out: open(a.out, "w").write(s)
    for m, r in R.items():
        v = r["heldout_transductive"]["umap2_ward_v1cfg"]
        print(f"[{m}] held-out UMAP+Ward: acc {100 * v['acc']:.2f}  NMI {v['nmi']:.3f}  ARI {v['ari']:.3f}  silhouette {v['sil']:.3f}"
              f" | sweep mean {100 * r['heldout_transductive']['umap2_ward_sweep']['mean_over_cells']:.2f}")


if __name__ == "__main__":
    main()
