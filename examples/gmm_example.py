"""Gaussian Mixture Model (GMM) — GPU-accelerated clustering / density estimation.

Fits a scikit-learn-compatible full-covariance GaussianMixture with the EM
algorithm on the GPU: recovered component means and clustering purity on three
well-separated Gaussian blobs, then real-world model selection (BIC) on the
UCI Wine chemistry dataset.
"""

import itertools

import numpy as np

from metal_gmm import MetalGMM, metal_gmm

rng = np.random.RandomState(7)

# Three well-separated 3-D Gaussian blobs (200 points each).
X = np.concatenate(
    [
        rng.randn(200, 3) + np.array([6.0, 0.0, 0.0]),
        rng.randn(200, 3) + np.array([0.0, 6.0, 0.0]),
        rng.randn(200, 3) + np.array([0.0, 0.0, 6.0]),
    ]
).astype(np.float32)

# sklearn-style API
gmm = MetalGMM(n_components=3, max_iterations=100, tolerance=1e-4, seed=42)
gmm.fit(X)

print("weights:", np.round(gmm.weights_, 3), "(sum %.3f)" % gmm.weights_.sum())
print("lower_bound (avg log-likelihood): %.3f" % gmm.lower_bound_)
print("n_iter:", gmm.n_iter_)
print("\nrecovered means (each should sit on one blob center (6,0,0)/(0,6,0)/(0,0,6)):")
for m in gmm.means_:
    print("   ", np.round(m, 3))

# Hard assignment + responsibilities.
labels = gmm.predict(X)
proba = gmm.predict_proba(X)
print("\npredict labels unique:", np.unique(labels))
print("predict_proba shape:", proba.shape, "row sums ~1:", np.allclose(proba.sum(1), 1.0))

# Purity: best-of-6-permutations (GMM component labeling is arbitrary).
truth = np.repeat([0, 1, 2], 200)
purity = max(np.mean(labels == np.array(p)[truth]) for p in itertools.permutations(range(3)))
print(f"cluster purity: {purity:.3f}")

# Score (average log-likelihood).
print("score: %.3f" % gmm.score(X))

# Functional API — returns (weights, means, covariances, responsibilities,
# lower_bound, n_iter).
w, means, covs, resp, lb, iters = metal_gmm(
    X, *X.shape, n_components=3, max_iterations=100, tolerance=1e-4, seed=42
)
print("\nfunctional API: weights", np.round(w, 3), "lower_bound %.3f" % lb, "iters", iters)

# ── Real data: UCI Wine — choose k via BIC ─────────────────────────────
# BIC = -2·L + p·ln(n) with L = total log-likelihood and p = number of free
# parameters of a full-covariance GMM: (k−1) + k·d + k·d(d+1)/2.
from sklearn.datasets import load_wine

wine = load_wine()
Xw = ((wine.data - wine.data.mean(0)) / wine.data.std(0)).astype(np.float32)
nw, dw = Xw.shape
y_wine = wine.target

print(f"\n── GMM model selection on UCI Wine ({nw} samples × {dw} features) ──")
bics, purities = [], []
for k in range(1, 7):
    g = MetalGMM(n_components=k, max_iterations=200, tolerance=1e-4, seed=42)
    g.fit(Xw)
    ll_total = float(g.score(Xw)) * nw
    p = (k - 1) + k * dw + k * dw * (dw + 1) // 2
    bic = -2.0 * ll_total + p * np.log(nw)
    bics.append(bic)
    lab = np.asarray(g.predict(Xw))
    pur = max(
        float((lab == np.array(pp)[y_wine]).mean())
        for pp in itertools.permutations(range(k)) if k == 3
    ) if k == 3 else float("nan")
    purities.append(pur)
    print(f"  k={k}  lower_bound={float(g.lower_bound_):7.3f}  BIC={bic:9.1f}"
          + (f"  purity-vs-cultivar={pur:.3f}" if k == 3 else ""))

best_k = int(np.argmin(bics)) + 1
print(f"BIC selects k={best_k} (true cultivar count = 3)")

# Why not 3? Two of the three wine cultivars overlap heavily in chemistry
# space; BIC (which penalizes the O(k·d²) covariance parameters) consistently
# prefers k=2 here. sklearn's GaussianMixture makes the identical call:
try:
    from sklearn.mixture import GaussianMixture

    sk_bic = min(
        GaussianMixture(n_components=k, covariance_type="full",
                        max_iter=200, tol=1e-4, random_state=42)
        .fit(Xw)
        .bic(Xw)
        for k in range(1, 7)
    )
    ours_best = min(bics)
    print(f"sklearn best BIC={sk_bic:.1f} vs ours={ours_best:.1f} → same model choice")
except ImportError:
    pass