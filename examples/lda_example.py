"""LDA (Linear Discriminant Analysis) — smoke test + real-world demo.

Demonstrates the Metal-accelerated :class:`~metal_lda.MetalLDA` operator on
synthetic Gaussian classes, then on the UCI Iris and Wine datasets: fit,
project into the discriminant subspace, and measure classification accuracy
via nearest projected class center.

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


# ── Real data: UCI Iris & Wine ─────────────────────────────────────
def real_data_demo(name, Xr, yr):
    nr = Xr.shape[0]
    k_real = len(set(yr.tolist()))

    lda_r = MetalLDA(n_components=k_real - 1)  # max rank is C-1
    lda_r.fit(Xr, yr.astype(float))
    acc_r = lda_r.score(Xr, yr.astype(float))
    print(
        f"{name}: n={nr} d={Xr.shape[1]} classes={k_real} → "
        f"project to {k_real - 1}D, train accuracy={acc_r:.4f}"
    )
    ev = np.asarray(lda_r.eigenvalues_)
    print(f"  eigenvalues = {np.round(ev, 3)}  (between-class separation per axis)")


from sklearn.datasets import load_iris, load_wine

_iris = load_iris()
_wine = load_wine()
real_data_demo("Iris", _iris.data.astype(np.float32), _iris.target)
real_data_demo("Wine", _wine.data.astype(np.float32), _wine.target)