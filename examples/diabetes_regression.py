"""metal_linear_regression real-world example: diabetes disease progression.

Realistic use case (per newopearator_guidelines.md §7.2): predict the
disease-progression score of 442 real diabetes patients from 10 baseline
measurements (age, sex, BMI, blood pressure, 6 serum markers), then compare
accuracy (R² / RMSE) and speed against sklearn CPU.

Dataset: sklearn.datasets.load_diabetes (built-in, no download).
Requires: scikit-learn (for the dataset + CPU reference) — same convention
as pca_eigenfaces.py and breast_cancer_classification.py.

Usage:
    python3 examples/diabetes_regression.py
"""

import time

import numpy as np

HAS_SKLEARN = False
try:
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import LinearRegression as SkLinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    HAS_SKLEARN = True
except ImportError:
    print("WARNING: sklearn not installed; run 'pip install scikit-learn'")

from metal_linear_regression import MetalLinearRegression


def main():
    print("=" * 64)
    print("metal_linear_regression — diabetes progression (real data)")
    print("=" * 64)

    if not HAS_SKLEARN:
        print("sklearn is required for the dataset; aborting.")
        return

    # ── Load the real dataset ──
    data = load_diabetes()
    X = data.data.astype(np.float32)  # 442 real patients, 10 baseline features
    y = data.target.astype(np.float32)  # disease progression score
    n, d = X.shape
    print(
        f"dataset: {n} patients x {d} features  "
        f"(target: progression {y.min():.0f}–{y.max():.0f}, "
        f"mean {y.mean():.1f})"
    )
    # load_diabetes features are already standardized (mean 0, unit variance);
    # coefficients are directly comparable without further scaling.

    # ── Train / test split ──
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    n_tr, n_te = X_tr.shape[0], X_te.shape[0]
    print(f"split: {n_tr} train / {n_te} test")

    # ── GPU: MetalLinearRegression (closed-form normal equations) ──
    reg = MetalLinearRegression(alpha=0.0, fit_intercept=True)
    t0 = time.perf_counter()
    reg.fit(X_tr, y_tr, n_tr, d)
    gpu_ms = (time.perf_counter() - t0) * 1000

    p_te = reg.predict(X_te, n_te, d)
    r2_gpu = float(r2_score(y_te, p_te))
    rmse_gpu = float(np.sqrt(mean_squared_error(y_te, p_te)))
    print(f"\n[metal]   fit {gpu_ms:7.1f} ms      R²={r2_gpu:.4f}  RMSE={rmse_gpu:.2f}")

    # ── CPU reference: sklearn LinearRegression ──
    sk = SkLinearRegression()
    t0 = time.perf_counter()
    sk.fit(X_tr, y_tr)
    sk_ms = (time.perf_counter() - t0) * 1000
    p_sk = sk.predict(X_te)
    r2_sk = float(r2_score(y_te, p_sk))
    rmse_sk = float(np.sqrt(mean_squared_error(y_te, p_sk)))
    print(f"[sklearn] fit {sk_ms:7.1f} ms      R²={r2_sk:.4f}  RMSE={rmse_sk:.2f}")

    # ── Head-to-head ──
    # NB: at 442×10 the GPU is dominated by command-buffer launch overhead
    # (~2 ms), so the closed-form fit is latency-bound on small data — see
    # benchmark() for the regime where the GPU wins.
    speedup = sk_ms / gpu_ms
    print(
        f"\nspeed (small data): metal {speedup:.1f}× vs sklearn CPU "
        "(launch-overhead-bound; see benchmark() for large data)"
    )
    print(f"R²    parity: |metal - sklearn| = {abs(r2_gpu - r2_sk):.4f}")
    print(f"RMSE  parity: |metal - sklearn| = {abs(rmse_gpu - rmse_sk):.4f}")

    # ── Medical interpretation: which baseline measurements drive progression? ──
    coef = reg.coef_
    order = np.argsort(np.abs(coef))[::-1]
    print("\ntop features driving disease progression (standardized coef):")
    for j in order[:5]:
        sign = "+" if coef[j] >= 0 else "-"
        print(f"  {data.feature_names[j]:<8} coef={coef[j]:+.1f}  ({sign} progression)")
    print("\nweakest features:")
    for j in order[-3:]:
        print(f"  {data.feature_names[j]:<8} coef={coef[j]:+.1f}")

    # ── Predictions on a few held-out patients ──
    print("\npredictions on held-out patients (progression score):")
    for i in range(min(5, n_te)):
        print(f"  patient {i}: pred={p_te[i]:7.1f}  true={y_te[i]:7.1f}")


def benchmark():
    """Larger scale performance test (guideline §7.2 pattern: 100K×32).

    Uses warmup + best-of-3 for both implementations: single-shot timings are
    noisy (first fit compiles the Metal shaders; numpy -> bytes conversion and
    the ~2 ms GPU launch overhead also skew small runs).
    """
    print("\n" + "=" * 64)
    print("benchmark — 100K×32 synthetic regression (best of 3 fits)")
    print("=" * 64)
    n, d = 100_000, 32
    rng = np.random.RandomState(7)
    X = rng.randn(n, d).astype(np.float32)
    w = rng.randn(d).astype(np.float32)
    y = (X @ w + rng.randn(n).astype(np.float32) * 0.1).astype(np.float32)

    reg = MetalLinearRegression(alpha=0.0, fit_intercept=True)
    reg.fit(X, y, n, d)  # warmup: compile shaders, touch all kernels
    metal_ms = min(_time_fit(reg, X, y, n, d) for _ in range(3))
    r2 = float(r2_score(y, reg.predict(X, n, d)))
    print(f"[metal]   fit {metal_ms:9.1f} ms (best of 3)  train R²={r2:.4f}")

    if HAS_SKLEARN:
        sk = SkLinearRegression()
        sk.fit(X, y)  # warmup: BLAS threads
        sk_ms = min(_time_fit(sk, X, y) for _ in range(3))
        print(f"[sklearn] fit {sk_ms:9.1f} ms (best of 3)")
        print(
            f"\nspeed: metal {sk_ms / metal_ms:.1f}× faster than sklearn CPU "
            f"({sk_ms / 1000:.1f}s vs {metal_ms / 1000:.2f}s)"
        )


def _time_fit(model, X, y, n=None, d=None):
    """Return fit time in ms (handles both the Metal and sklearn APIs)."""
    t0 = time.perf_counter()
    if n is not None:
        model.fit(X, y, n, d)
    else:
        model.fit(X, y)
    return (time.perf_counter() - t0) * 1000


if __name__ == "__main__":
    main()
    benchmark()
