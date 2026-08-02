"""metal_linear_regression example: regression smoke test + benchmark.

Synthetic data with known coefficients for the smoke test, then a larger
50K×64 benchmark.
"""

import time

import numpy as np
from metal_linear_regression import metal_linear_regression, MetalLinearRegression


def make_regression(n, d, noise=0.5, seed=42):
    """y = X @ w_true + b_true + Gaussian noise; returns (X, y, w_true, b_true)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float32)
    w_true = rng.randn(d).astype(np.float32)
    b_true = rng.randn() * 2.0
    y = (X @ w_true + b_true + noise * rng.randn(n)).astype(np.float32)
    return X, y, w_true, b_true


def smoke_test():
    """Small example with known ground-truth coefficients."""
    n, d = 1000, 5
    X, y, w_true, b_true = make_regression(n, d, noise=0.3, seed=42)

    # ── Functional API ──
    weights, bias, n_iter, final_loss = metal_linear_regression(
        X.ravel().tolist(), y.tolist(), n, d, alpha=0.0, fit_intercept=True
    )
    print("[functional]  n_iter={}  mse={:.4f}".format(n_iter, final_loss))
    print("  weights:", np.round(weights, 4))
    print("  bias:   {:.4f}".format(bias))

    # Compare against the ground truth (should be close at noise=0.3)
    err = np.linalg.norm(weights - w_true) / (np.linalg.norm(w_true) + 1e-12)
    print("  relative error vs w_true: {:.4f}".format(err))

    # ── sklearn-style API ──
    reg = MetalLinearRegression(alpha=0.0, fit_intercept=True)
    reg.fit(X, y, n, d)
    r2 = reg.score(X, y, n, d)
    print(
        "[sklearn]     r2={:.4f}  mse={:.4f}  coef_norm={:.4f}".format(
            r2, reg.final_loss, np.linalg.norm(reg.coef_)
        )
    )

    # Predict on new points
    new_points = np.array(
        [[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.5, 2.0, -2.0]], dtype=np.float32
    )
    preds = reg.predict(new_points, 2, d)
    print("  predictions for new points:", np.round(preds, 4))

    # ── Ridge (alpha > 0) shrinks the coefficients ──
    reg_ridge = MetalLinearRegression(alpha=10.0, fit_intercept=True)
    reg_ridge.fit(X, y, n, d)
    print(
        "[ridge]       alpha=10 -> coef_norm={:.4f} vs ols={:.4f}".format(
            np.linalg.norm(reg_ridge.coef_), np.linalg.norm(reg.coef_)
        )
    )


def benchmark():
    """Larger shape to see GPU speed."""
    n, d = 50_000, 64
    X, y, _, _ = make_regression(n, d, noise=0.5, seed=7)

    reg = MetalLinearRegression(alpha=0.0, fit_intercept=True)
    t0 = time.perf_counter()
    reg.fit(X, y, n, d)
    elapsed = time.perf_counter() - t0
    r2 = reg.score(X, y, n, d)
    print(
        "\n[benchmark]  {}×{}  {:.0f} ms  r2={:.4f}  mse={:.4f}".format(
            n, d, elapsed * 1000, r2, reg.final_loss
        )
    )


if __name__ == "__main__":
    print("=" * 50)
    print("metal_linear_regression example")
    print("=" * 50)
    smoke_test()
    benchmark()
