"""Simple example: PCA on synthetic high-dimensional data."""

import numpy as np
from metal_pca import MetalPCA

# Generate synthetic data with known 3D structure + noise
rng = np.random.default_rng(42)
n, d, k = 1000, 20, 3

# Create 3 orthogonal directions
directions = np.eye(d, k)[:, :k]  # (d, k) one-hot basis
# Vary coefficients — each point uses different weights
coefficients = rng.normal(size=(n, k)) * [10.0, 5.0, 2.0]  # decreasing strength
data = coefficients @ directions.T + rng.normal(size=(n, d)) * 0.5

# Fit PCA
pca = MetalPCA(n_components=k)
pca.fit(data)

print("PCA on synthetic data:")
print(f"  Data: {data.shape[0]} points x {data.shape[1]} dims")
print(f"  Components shape: {pca.components_.shape}")
print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"  Cumulative ratio: {pca.explained_variance_ratio_.sum():.4f}")

# Transform
transformed = pca.transform(data)
print(f"  Transformed shape: {transformed.shape}")

# Verify reconstruction with K components
reconstructed = transformed @ pca.components_
recon_error = np.mean((data - reconstructed)**2)
print(f"  Reconstruction MSE (K={k}): {recon_error:.6f}")

# With all components, reconstruction should be near-perfect
pca_full = MetalPCA(n_components=d)
pca_full.fit(data)
transformed_full = pca_full.transform(data)
reconstructed_full = transformed_full @ pca_full.components_
recon_error_full = np.mean((data - reconstructed_full)**2)
print(f"  Reconstruction MSE (K={d}, all): {recon_error_full:.2e}")
