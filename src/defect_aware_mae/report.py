"""Aggregation of stored evaluation results into tables (mean +- std over seeds) and significance tests."""
import json
from pathlib import Path
import numpy as np
from .data import repo_root

METRICS = {
    "Acc (UMAP+Ward)": lambda r: 100 * r["heldout_transductive"]["umap2_ward_v1cfg"]["acc"],
    "NMI": lambda r: r["heldout_transductive"]["umap2_ward_v1cfg"]["nmi"],
    "ARI": lambda r: r["heldout_transductive"]["umap2_ward_v1cfg"]["ari"],
    "Silhouette (UMAP)": lambda r: r["heldout_transductive"]["umap2_ward_v1cfg"]["sil"],
    "Silhouette (PCA-50)": lambda r: r["heldout_transductive"]["pca50_kmeans"]["sil"],
    "Acc sweep mean": lambda r: 100 * r["heldout_transductive"]["umap2_ward_sweep"]["mean_over_cells"],
    "Acc sweep worst": lambda r: 100 * r["heldout_transductive"]["umap2_ward_sweep"]["min_cell"],
    "Acc PCA k-means": lambda r: 100 * r["heldout_transductive"]["pca50_kmeans"]["acc"],
    "Acc 768-d k-means": lambda r: 100 * r["heldout_transductive"]["raw768_kmeans"]["acc"],
    "Acc inductive (UMAP)": lambda r: 100 * r["heldout_inductive"]["umap2_kmeans"][0],
    "Acc inductive (PCA)": lambda r: 100 * r["heldout_inductive"]["pca50_kmeans"]["acc"],
    "kNN-20*": lambda r: 100 * r["knn20"],
    "Linear probe*": lambda r: 100 * r["linear"],
}


# metrics of the main results table
MAIN_METRICS = ["Acc (UMAP+Ward)", "NMI", "ARI", "Silhouette (UMAP)"]


def load_run(tag, readout, root=None):
    R = json.load(open(Path(root or repo_root()) / "results" / "runs" / tag / "eval.json"))
    return R[readout]


def values(tags, readout, root=None):
    """(n_seeds, n_metrics) array."""
    return np.array([[f(load_run(t, readout, root)) for f in METRICS.values()] for t in tags])


def fmt(m, s, name, n):
    d = 3 if name in ("NMI", "ARI") or name.startswith("Silh") else 1
    return f"{m:.{d}f}" + (f" ± {s:.{d}f}" if n > 1 else "")


def table(groups, readout=None, metrics=None, root=None):
    import pandas as pd
    metrics = metrics or MAIN_METRICS; cols = list(METRICS); rows = []
    for g, v in groups.items():
        V = values(v["tags"], readout or v["readout"], root)
        rows.append({"Model": g, "Seeds": len(V), "Readout": readout or v["readout"],
                     **{m: fmt(V[:, cols.index(m)].mean(), V[:, cols.index(m)].std(), m, len(V)) for m in metrics}})
    return pd.DataFrame(rows)


def welch(tags_a, readout_a, tags_b, readout_b, root=None):
    """Welch t-test per metric, A vs B. Returns DataFrame with means, difference and p-value."""
    import pandas as pd
    from scipy.stats import ttest_ind
    A, B = values(tags_a, readout_a, root), values(tags_b, readout_b, root); rows = []
    for i, m in enumerate(METRICS):
        rows.append({"Metric": m, "A": fmt(A[:, i].mean(), A[:, i].std(), m, len(A)), "B": fmt(B[:, i].mean(), B[:, i].std(), m, len(B)),
                     "A - B": round(A[:, i].mean() - B[:, i].mean(), 3),
                     "p (Welch)": round(float(ttest_ind(A[:, i], B[:, i], equal_var=False).pvalue), 3)})
    return pd.DataFrame(rows)
