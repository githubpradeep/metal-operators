"""Real-world PCA: Eigenfaces face reconstruction and recognition.

Demonstrates:
  - Metal PCA dimensionality reduction for face images
  - Explained variance vs number of components
  - Face reconstruction at various K
  - Speed comparison vs sklearn PCA

Requires: sklearn, matplotlib (optional for plots)
Run:      python3 examples/pca_eigenfaces.py
"""

import os
import ssl
import time
import numpy as np

# Fix SSL issues for sklearn dataset downloads
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

HAS_SKLEARN = False
try:
    from sklearn.datasets import fetch_olivetti_faces, load_digits
    from sklearn.decomposition import PCA as SkPCA
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    HAS_SKLEARN = True
except ImportError:
    print("WARNING: sklearn not installed; run 'pip install scikit-learn'")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from metal_pca import MetalPCA


def load_faces():
    """Try Olivetti faces; fall back to digits dataset."""
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception:
        pass
    try:
        faces = fetch_olivetti_faces(shuffle=True, random_state=42)
        X = faces.data.astype(np.float32)
        y = faces.target
        h = w = 64
        return X, y, h, w, "Olivetti faces"
    except Exception as e:
        print(f"  Olivetti faces unavailable ({e}); using digits dataset")
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0
    y = digits.target
    h = w = 8
    return X, y, h, w, "Digits"


def main():
    print("Loading dataset...")
    t0 = time.perf_counter()
    X, y, h, w, name = load_faces()
    n, d = X.shape
    print(f"  {name}: {n} images, {d} pixels (loaded in {time.perf_counter()-t0:.2f}s)")

    k = 32
    print(f"\n── Fitting Metal PCA (k={k}) ──")
    t0 = time.perf_counter()
    mpca = MetalPCA(n_components=k)
    mpca.fit(X)
    tgpu = time.perf_counter() - t0
    print(f"  GPU:  {tgpu*1000:.1f}ms  cum. variance: {mpca.explained_variance_ratio_.sum():.3%}")

    if HAS_SKLEARN:
        print(f"\n── Fitting sklearn PCA (k={k}) ──")
        t0 = time.perf_counter()
        spca = SkPCA(n_components=k, svd_solver="randomized", random_state=0)
        spca.fit(X)
        tcpu = time.perf_counter() - t0
        print(f"  CPU:  {tcpu*1000:.1f}ms  cum. variance: {spca.explained_variance_ratio_.sum():.3%}")
        speedup = tcpu / tgpu
        print(f"\n  Speedup: {speedup:.1f}×  (Metal vs sklearn)")

        # ── Downstream: face/digit recognition with PCA + 1NN ──
        ks = [8, 16, 32, 64, 128]
        print(f"\n── Recognition accuracy (PCA + 1NN) ──")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0
        )
        for kk in ks:
            if kk > min(n, d):
                continue
            pipe = make_pipeline(
                MetalPCA(n_components=kk),
                KNeighborsClassifier(n_neighbors=1),
            )
            t0 = time.perf_counter()
            pipe.fit(X_train, y_train)
            acc = pipe.score(X_test, y_test)
            elapsed = time.perf_counter() - t0
            print(f"  K={kk:3d}  accuracy={acc:.2%}  time={elapsed*1000:.0f}ms")

    # ── Reconstruction at various K ──
    if HAS_MPL:
        print("\n── Generating visualizations ──")
        n_samples = min(5, n)
        sample = X[:n_samples]

        fig, axes = plt.subplots(3, n_samples, figsize=(3 * n_samples, 7))

        for i in range(n_samples):
            axes[0, i].imshow(sample[i].reshape(h, w), cmap="gray")
            axes[0, i].axis("off")
            if i == 0:
                axes[0, i].set_ylabel("Original", fontsize=10)

        for ki, row in [(16, 1), (64, 2)]:
            if ki > d:
                continue
            mpca_k = MetalPCA(n_components=ki)
            mpca_k.fit(X)
            recon = mpca_k.transform(sample) @ mpca_k.components_
            for i in range(n_samples):
                axes[row, i].imshow(recon[i].reshape(h, w), cmap="gray")
                axes[row, i].axis("off")
                if i == 0:
                    axes[row, i].set_ylabel(f"K={ki}", fontsize=10)

        fig.suptitle(f"{name} — reconstruction at K=16 vs K=64", fontsize=13)
        fig.tight_layout()
        fig.savefig("eigenfaces_reconstruction.png", dpi=150)
        plt.close(fig)
        print("  saved: eigenfaces_reconstruction.png")

        # Explained variance curve
        max_k = min(d, 200)
        mpca_full = MetalPCA(n_components=max_k)
        mpca_full.fit(X)
        ev = mpca_full.explained_variance_ratio_
        cum = np.cumsum(ev)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(range(1, len(cum) + 1), cum, "b-", linewidth=1.5)
        ax1.axhline(0.9, color="gray", linestyle="--", alpha=0.5)
        ax1.axhline(0.95, color="gray", linestyle="--", alpha=0.5)
        ax1.set_xlabel("Number of components")
        ax1.set_ylabel("Cumulative explained variance")
        ax1.set_title(f"{name} — variance vs K")
        ax1.grid(True, alpha=0.3)

        n_scree = min(32, len(ev))
        ax2.bar(range(1, n_scree + 1), ev[:n_scree], color="steelblue")
        ax2.set_xlabel("Component")
        ax2.set_ylabel("Variance ratio")
        ax2.set_title("Top components (scree plot)")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig("pca_variance_analysis.png", dpi=150)
        plt.close(fig)
        print("  saved: pca_variance_analysis.png")


if __name__ == "__main__":
    main()

