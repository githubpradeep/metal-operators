"""Real-world example: PCA on the Iris dataset.

Compares Metal PCA against sklearn for correctness.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as SkPCA
from sklearn.preprocessing import StandardScaler
from metal_pca import MetalPCA

# Load data
iris = load_iris()
X = iris.data.astype(np.float32)
y = iris.target
n, d = X.shape
k = 2
print(f"Iris dataset: {n} samples, {d} features")

# Standardize (PCA is not scale-invariant)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)

# ── Metal PCA ──
metal_pca = MetalPCA(n_components=k)
metal_pca.fit(X_scaled)

metal_components = metal_pca.components_
metal_variance = metal_pca.explained_variance_
metal_ratio = metal_pca.explained_variance_ratio_
metal_transformed = metal_pca.transform(X_scaled)

print("\n--- Metal PCA ---")
print(f"Components:\n{metal_components}")
print(f"Explained variance ratio: {metal_ratio}")

# ── sklearn PCA ──
sk_pca = SkPCA(n_components=k)
sk_transformed = sk_pca.fit_transform(X_scaled)

print("\n--- sklearn PCA ---")
print(f"Components:\n{sk_pca.components_}")
print(f"Explained variance ratio: {sk_pca.explained_variance_ratio_}")

# ── Comparison ──
print("\n--- Comparison ---")

# Variance ratios should match exactly
ratio_diff = np.abs(metal_ratio - sk_pca.explained_variance_ratio_)
print(f"Max explained variance ratio difference: {ratio_diff.max():.2e}")

# Components may have sign flips — check absolute dot products
for i in range(k):
    dot = np.abs(np.dot(metal_components[i], sk_pca.components_[i]))
    print(f"Component {i+1} alignment: {dot:.4f} (should be ≈1)")

# Check transformed data (with sign correction)
align_signs = np.sign(sk_transformed) * np.sign(metal_transformed)
align_signs[align_signs == 0] = 1
transformed_diff = np.abs(metal_transformed - sk_transformed * align_signs)
print(f"Max transformed difference (sign-corrected): {transformed_diff.max():.2e}")

# Visualization (optional, requires matplotlib)
try:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for label in range(3):
        mask = y == label
        ax1.scatter(metal_transformed[mask, 0], metal_transformed[mask, 1],
                    label=iris.target_names[label], alpha=0.7)
    ax1.set_title("Metal PCA (GPU)")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for label in range(3):
        mask = y == label
        ax2.scatter(sk_transformed[mask, 0], sk_transformed[mask, 1],
                    label=iris.target_names[label], alpha=0.7)
    ax2.set_title("sklearn PCA (CPU)")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pca_iris_comparison.png", dpi=150)
    plt.show()
    print("\nSaved visualization to pca_iris_comparison.png")
except ImportError:
    print("\nmatplotlib not installed; skipping visualization")
