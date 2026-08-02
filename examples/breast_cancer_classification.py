"""metal_logistic_regression real-world example: breast cancer diagnosis.

Realistic use case (per newopearator_guidelines.md §7.2): classify 569 real
patient tumor samples (Wisconsin breast cancer dataset, 30 features) as
malignant vs benign, then compare accuracy and speed against sklearn CPU.

Dataset: sklearn.datasets.load_breast_cancer (built-in, no download).
Requires: scikit-learn (for the dataset + CPU reference) — same convention
as pca_eigenfaces.py.

Usage:
    python3 examples/breast_cancer_classification.py
"""

import time

import numpy as np

HAS_SKLEARN = False
try:
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression as SkLogReg
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    print("WARNING: sklearn not installed; run 'pip install scikit-learn'")

from metal_logistic_regression import MetalLogisticRegression


def main():
    print("=" * 64)
    print("metal_logistic_regression — breast cancer diagnosis (real data)")
    print("=" * 64)

    if not HAS_SKLEARN:
        print("sklearn is required for the dataset; aborting.")
        return

    # ── Load the real dataset ──
    data = load_breast_cancer()
    X = data.data.astype(np.float32)  # 569 real tumor samples, 30 features
    y = data.target.astype(np.float32)  # 0 = malignant, 1 = benign
    n, d = X.shape
    print(
        f"dataset: {n} patients x {d} features  ({data.target_names[0]}={0}, {data.target_names[1]}={1})"
    )

    # ── Train / test split + standardize (same preprocessing for both models) ──
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)
    n_tr, n_te = X_tr_s.shape[0], X_te_s.shape[0]
    print(f"split: {n_tr} train / {n_te} test (stratified)")

    # ── GPU: MetalLogisticRegression (full-batch L-BFGS) ──
    clf = MetalLogisticRegression(c=1.0, max_epochs=100, tol=1e-5, seed=42)
    t0 = time.perf_counter()
    clf.fit(X_tr_s, y_tr, n_tr, d)
    gpu_ms = (time.perf_counter() - t0) * 1000

    p_te = clf.predict_proba(X_te_s, n_te, d)
    acc_gpu = float(accuracy_score(y_te, (p_te > 0.5).astype(np.float32)))
    auc_gpu = float(roc_auc_score(y_te, p_te))
    print(
        f"\n[metal]   fit {gpu_ms:7.1f} ms  ({clf.n_epochs_} iters, "
        f"loss={clf.final_loss_:.4f})  acc={acc_gpu:.4f}  AUC={auc_gpu:.4f}"
    )

    # ── CPU reference: sklearn LogisticRegression ──
    sk = SkLogReg(C=1.0, max_iter=1000, tol=1e-5)
    t0 = time.perf_counter()
    sk.fit(X_tr_s, y_tr)
    sk_ms = (time.perf_counter() - t0) * 1000
    acc_sk = float(accuracy_score(y_te, sk.predict(X_te_s)))
    auc_sk = float(roc_auc_score(y_te, sk.predict_proba(X_te_s)[:, 1]))
    print(
        f"[sklearn] fit {sk_ms:7.1f} ms                    acc={acc_sk:.4f}  AUC={auc_sk:.4f}"
    )

    # ── Head-to-head ──
    # NB: at 569×30 the GPU is dominated by per-iteration launch overhead, so
    # this small fit is not where the speedup lives (see benchmark() below).
    speedup = sk_ms / gpu_ms
    print(
        f"\nspeed (small data): metal {speedup:.1f}× vs sklearn CPU "
        "(launch-overhead-bound; see benchmark() for large data)"
    )
    print(f"acc   parity: |metal - sklearn| = {abs(acc_gpu - acc_sk):.4f}")
    print(f"AUC   parity: |metal - sklearn| = {abs(auc_gpu - auc_sk):.4f}")

    # ── Clinical interpretation: top features driving the decision ──
    coef = clf.weights
    top = np.argsort(np.abs(coef))[::-1][:5]
    print("\ntop discriminating features (|coef|):")
    for j in top:
        print(f"  {data.feature_names[j]:<32} coef={coef[j]:+.3f}")

    # ── Predict probabilities for a few real test patients ──
    print("\npredictions on held-out patients (P(benign)):")
    for i in range(min(5, n_te)):
        print(
            f"  patient {i}: P(benign)={p_te[i]:.3f}  ->  {'benign' if p_te[i] >= 0.5 else 'malignant'}  "
            f"(true: {'benign' if y_te[i] == 1 else 'malignant'})"
        )


def benchmark():
    """Larger scale performance test (guideline §7.2 pattern: 50K×64)."""
    print("\n" + "=" * 64)
    print("benchmark — 100K×64 synthetic binary classification")
    print("=" * 64)
    n, d = 100_000, 64
    rng = np.random.RandomState(7)
    X = rng.randn(n, d).astype(np.float32)
    w = rng.randn(d).astype(np.float32) / np.sqrt(d)
    y = (X @ w + rng.randn(n).astype(np.float32) * 0.1 > 0).astype(np.float32)

    clf = MetalLogisticRegression(c=1.0, max_epochs=100, tol=1e-4, seed=7)
    t0 = time.perf_counter()
    clf.fit(X, y, n, d)
    metal_ms = (time.perf_counter() - t0) * 1000
    acc = float(
        accuracy_score(y, (clf.predict_proba(X, n, d) > 0.5).astype(np.float32))
    )
    print(
        f"[metal]   fit {metal_ms:9.1f} ms  ({clf.n_epochs_} iters)  train acc={acc:.4f}"
    )

    if HAS_SKLEARN:
        sk = SkLogReg(C=1.0, max_iter=100, tol=1e-4)
        t0 = time.perf_counter()
        sk.fit(X, y)
        sk_ms = (time.perf_counter() - t0) * 1000
        print(f"[sklearn] fit {sk_ms:9.1f} ms")
        print(
            f"\nspeed: metal {sk_ms / metal_ms:.1f}× faster than sklearn CPU "
            f"({sk_ms / 1000:.1f}s vs {metal_ms / 1000:.2f}s)"
        )


if __name__ == "__main__":
    main()
    benchmark()
