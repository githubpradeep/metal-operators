"""Gaussian Mixture Model (GMM) — GPU-accelerated clustering / density estimation.

Fits a scikit-learn-compatible full-covariance GaussianMixture with the EM
algorithm on the GPU, then shows the recovered component means and clustering
purity on three well-separated Gaussian blobs.
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