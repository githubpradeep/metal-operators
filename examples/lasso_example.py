"""metal_lasso example: L1-regularized regression smoke test + benchmark.

Synthetic data with known (sparse) coefficients so the soft-thresholding /
coordinate-descent behaviour is observable, then a larger benchmark.
"""

import time

import numpy as np
from metal_lasso import metal_lasso, MetalLasso


def make_sparse_regression(n, d, noise=0.3, seed=42):
    """y = X @ w_true + b_true + noise; only ~30% of features are nonzero."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float32)
    w_true = rng.randn(d).astype(np.float32)
    w_true[rng.rand(d) > 0.2] = 0.0  # sparsify
    b_true = rng.randn() * 2.0
    y = (X @ w_true + b_true + noise * rng.randn(n)).astype(np.float32)
    return X, y, w_true, b_true


def smoke_test():
    """Small example: high alpha shrinks irrelevant features to zero."""
    n, d = 800, 12
    X, y, w_true, b_true = make_sparse_regression(n, d, noise=0.3, seed=42)

    # ── Functional API ──
    weights, intercept, n_iter, converged, mse = metal_lasso(
        X.ravel().tolist(), y.tolist(), n, d, alpha=0.05, fit_intercept=True
    )
    print(
        "[functional]  n_iter={}  converged={}  mse={:.4f}".format(
            n_iter, converged, mse
        )
    )
    print("  weights:", np.round(weights, 4))
    print("  intercept: {:.4f}".format(intercept))
    print("  nonzero coefs: {}/{}".format(np.count_nonzero(weights), d))

    # Compare vs the ground truth (should be close at noise=0.3).
    err = np.linalg.norm(weights - w_true) / (np.linalg.norm(w_true) + 1e-12)
    print("  relative error vs w_true: {:.4f}".format(err))

    # ── sklearn-style API ──
    lasso = MetalLasso(alpha=0.05, fit_intercept=True)
    lasso.fit(X, y, n, d)
    r2 = lasso.score(X, y, n, d)
    print(
        "[sklearn] r2={:.4f}  mse={:.4f}  n_iter={}  converged={}".format(
            r2, lasso.final_loss, lasso.n_iter, lasso.converged
        )
    )
    print("  coef_ nonzero: {}/{}".format(np.count_nonzero(lasso.coef_), d))

    # Stronger alpha → fully sparse solution.
    lasso_strong = MetalLasso(alpha=2.0, fit_intercept=True)
    lasso_strong.fit(X, y, n, d)
    print(
        "  alpha=2.0 → nonzero: {}/{}".format(
            np.count_nonzero(lasso_strong.coef_), d
        )
    )


def benchmark():
    """Larger 50K×64 timing (Gram build once on GPU, CD-wise host loop)."""
    n, d = 50_000, 64
    rng = np.random.RandomState(1)
    X = rng.randn(n, d).astype(np.float32)
    w_true = rng.randn(d).astype(np.float32)
    w_true[rng.rand(d) > 0.15] = 0.0
    y = (X @ w_true + 0.1 * rng.randn(n)).astype(np.float32)

    t0 = time.perf_counter()
    weights, intercept, n_iter, converged, mse = metal_lasso(
        X.ravel().tolist(), y.tolist(), n, d, alpha=0.1, fit_intercept=True
    )
    dt = time.perf_counter() - t0
    print(
        "[bench] n={} d={}  {:.3f}s  n_iter={}  converged={}  mse={:.4f}".format(
            n, d, dt, n_iter, converged, mse
        )
    )
    print("  nonzero: {}/{}".format(np.count_nonzero(weights), d))


if __name__ == "__main__":
    smoke_test()
    benchmark()