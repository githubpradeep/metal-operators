"""metal_dbscan example: DBSCAN density clustering on real wine chemistry data.

Uses the classic UCI Wine dataset (178 samples × 13 chemical features) to show
the two things density-based clustering does that KMeans cannot:

  1. Discover cluster structure *without being told k*.
  2. Flag chemically atypical samples as *noise* — an outlier screen for free.

Features are first narrowed to the 5 most class-discriminative assays
(F-score ranking): DBSCAN's euclidean eps is meaningless in full 13-D where
densities overlap, and standardization keeps proline (~1000s) from swamping
hue (~1.0).

Usage::

    python3 examples/dbscan_wine_clusters.py
"""

import time

import numpy as np
from itertools import permutations

from sklearn.datasets import load_wine
from sklearn.feature_selection import f_classif

from metal_dbscan import MetalDBSCAN


def main():
    wine = load_wine()
    X = wine.data.astype(np.float32)
    y_true = wine.target
    n = X.shape[0]

    # Standardize the top-5 discriminative features.
    F, _ = f_classif(X, y_true)
    feats = np.argsort(F)[::-1][:5]
    names = [wine.feature_names[i] for i in feats]
    Xs = ((X[:, feats] - X[:, feats].mean(0)) / X[:, feats].std(0)).astype(np.float32)
    d = Xs.shape[1]

    print("── DBSCAN on UCI Wine ──")
    print("samples={}  features={} {}".format(n, d, names))
    print("true cultivars=3\n")

    # ── eps sweep: structure discovery vs over-merging ──
    print("eps sweep:")
    chosen = None
    for eps in np.arange(0.55, 2.05, 0.15):
        t0 = time.perf_counter()
        labels = np.asarray(MetalDBSCAN(eps=float(eps), min_samples=5).fit_predict(Xs))
        dt_ms = (time.perf_counter() - t0) * 1e3
        n_clusters = len(set(labels[labels >= 0].tolist()))
        noise = int((labels == -1).sum())
        marker = ""
        if n_clusters == 3 and chosen is None:
            chosen = float(eps)
            marker = "  ← first eps recovering all 3 cultivars"
        print(
            "  eps={:.2f} → clusters={} noise={}/{} ({:.2f}ms){}".format(
                eps, n_clusters, noise, n, dt_ms, marker
            )
        )
    assert chosen is not None, "no eps recovered 3 clusters; tune range"

    # ── Structure discovery at the chosen eps ──
    labels = np.asarray(MetalDBSCAN(eps=chosen, min_samples=5).fit_predict(Xs))
    mask = labels >= 0
    purity = max(
        float((np.array([p[l] for l in labels])[mask] == y_true[mask]).mean())
        for p in permutations(range(len(set(labels[mask].tolist()))))
    )
    print("\neps={} → cluster purity vs true cultivar: {:.1%}".format(chosen, purity))

    # ── The noise set is an outlier screen ──
    cents = np.vstack([Xs[y_true == c].mean(0) for c in range(3)])
    dist = np.linalg.norm(Xs - cents[y_true], axis=1)
    noise_mask = ~mask
    if noise_mask.any():
        print(
            "\noutlier screen: {} flagged wines sit {:.2f}σ from their class "
            "centroid on average vs {:.2f}σ for clustered ones ({:.1f}× atypical)".format(
                int(noise_mask.sum()),
                dist[noise_mask].mean(),
                dist[mask].mean(),
                dist[noise_mask].mean() / dist[mask].mean(),
            )
        )


if __name__ == "__main__":
    main()
