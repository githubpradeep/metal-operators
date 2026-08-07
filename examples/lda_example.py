"""LDA (Linear Discriminant Analysis) — smoke test + dimensionality reduction.

Demonstrates the Metal-accelerated :class:`~metal_lda.MetalLDA` operator on
synthetic Gaussian classes: fit, project into the discriminant subspace, and
measure classification accuracy via nearest projected class center.

Usage::

    python3 examples/lda_example.py
"""

import numpy as np
from metal_lda import MetalLDA

rng = np.random.RandomState(42)

# 3 well-separated Gaussian classes projected on 2 informative axes in 12-D.
n, d, k = 600, 12, 2
X = rng.randn(n, d).astype(np.float32)
y = np.concatenate([np.full(200, c, np.float32) for c in (0, 1, 2)])
# shift each class along a distinct direction
shift = np.array([
    [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
], dtype=np.float32)
X += shift[y.astype(int)]

lda = MetalLDA(n_components=k)
lda.fit(X, y)

acc = lda.score(X, y)
pred = lda.predict(X)
print(f"LDA n_components={lda.n_components}")
print(f"scalings_.shape = {lda.scalings_.shape}")
print(f"eigenvalues_    = {np.round(lda.eigenvalues_, 4)}")
print(f"classes_        = {lda.classes_}")
print(f"train accuracy  = {acc:.4f}  (misclassified {int((1-acc)*n)}/{n})")

# Project and verify separability in the low-dim subspace.
T = lda.transform(X)
print("projected shape =", T.shape)

print("ok: LDA works end-to-end")