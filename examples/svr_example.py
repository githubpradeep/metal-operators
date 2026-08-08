"""Support Vector Regression (SVR) — GPU-accelerated kernel + decision.

Trains an scikit-learn-compatible SVR whose kernel (Gram) matrix and decision
outputs are computed on the Apple Metal GPU, with the dual (β) updates solved
on the host by an ε-insensitive SMO. Shows fitting a non-linear 1-D sinusoid
(RBF kernel), a 2-D quadratic surface (poly kernel), and a linear trend.
"""

import numpy as np

from metal_svr import MetalSVR, metal_svr

rng = np.random.RandomState(3)

# 1) Non-linear 1-D sine — RBF kernel.
t = np.linspace(0, 6.28, 150).astype(np.float32)
X = t.reshape(-1, 1)
y = np.sin(t).astype(np.float32)

clf = MetalSVR(kernel="rbf", gamma=0.6, c=5.0, eps=0.05, max_iter=300, seed=42)
clf.fit(X, y)
preds = clf.predict(X)
r2 = clf.score(X, y)
print("1-D sine (RBF)      n=%d d=1" % X.shape[0])
print("  R²:            %.4f" % r2)
print("  n_support_:    %d (support vectors)" % clf.n_support_)
print("  intercept_:    %.4f" % clf.intercept_)
print("  gamma_:        %.3f (resolved kernel width)" % clf.gamma_)
print("  n_iter_ SMO passes: %d" % clf.n_iter_)
print("  sample preds:  [%s ...]" % ", ".join("%.3f" % v for v in preds[::20]))

# 2) Quadratic surface with a polynomial kernel.
n = 120
d = 2
u = rng.uniform(-1, 1, n).astype(np.float32)
v = rng.uniform(-1, 1, n).astype(np.float32)
X2 = np.c_[u, v].astype(np.float32)
y2 = (u**2 + 2.0 * v**2).astype(np.float32)

poly = MetalSVR(kernel="poly", gamma=1.0, degree=2, coef0=1.0, c=10.0,
                eps=0.05, tolerance=1e-4, max_iter=300, seed=7)
poly.fit(X2, y2)
print("\n2-D quadratic (poly d=2)  R²: %.4f" % poly.score(X2, y2))
print("  n_support_: %d   n_iter_: %d" % (poly.n_support_, poly.n_iter_))

# 3) Linear trend — linear kernel (decision is a dot product).
t2 = rng.uniform(-3, 3, 80).astype(np.float32)
X3 = np.c_[t2, t2**0].astype(np.float32)  # (n, 2) with a constant column
y3 = (2.5 * t2 + 1.0).astype(np.float32)

lin = MetalSVR(kernel="linear", c=1.0, eps=0.05, max_iter=200, seed=1)
lin.fit(X3, y3)
print("\nlinear (d=2)              R²: %.4f" % lin.score(X3, y3))
print("  n_support_: %d" % lin.n_support_)

# Functional API — returns
# (support_vectors_, dual_coef_/β, intercept_, n_support_, gamma_, n_iter).
sv, dual, b, ns, g, iters = metal_svr(
    X, y, *X.shape, kernel="rbf", gamma=0.6, c=5.0, eps=0.05, max_iter=300, seed=42
)
print("\nfunctional API:")
print("  support_vectors shape:", sv.shape, "  dual_coef shape:", dual.shape)
print("  intercept_: %.4f  support_count: %d  gamma_: %.3f  n_iter_: %d"
      % (b, ns, g, iters))