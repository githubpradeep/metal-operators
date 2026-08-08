"""Support Vector Classification (SVC) — GPU-accelerated kernel + decision.

Trains an scikit-learn-compatible SVC whose kernel (Gram) matrix and decision
matrix are computed on the Apple Metal GPU, with the dual (α) updates solved on
the host by a simplified Platt SMO. Shows training on a non-linearly separable
two-moon problem with an RBF kernel and on separable blobs with a linear kernel.
"""

import numpy as np

from metal_svm import MetalSVC, metal_svc

rng = np.random.RandomState(3)

# Non-linearly separable two-moons problem (RBF kernel).
n = 400
t = rng.uniform(0, np.pi, n // 2)
X0 = np.c_[t * np.cos(t), t * np.sin(t)]
t = rng.uniform(0, np.pi, n // 2)
X1 = np.c_[t * np.cos(t), t * np.sin(t)] + np.array([2.0, -0.5])
X = np.vstack([X0, X1]).astype(np.float32)
y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(np.float32)

# sklearn-style API — RBF kernel, automatic gamma (1 / n_features = 1 / 2).
clf = MetalSVC(kernel="rbf", gamma=0.5, c=10.0, tolerance=1e-4, max_iter=300, seed=42)
clf.fit(X, y)
preds = clf.predict(X)
acc = np.mean(preds == y)
print("two-moons (RBF)  n=%d d=2" % n)
print("  accuracy: %.4f" % acc)
print("  classes_:    ", clf.classes_)
print("  gamma_:      %.3f (resolved kernel width)" % clf.gamma_)
print("  n_support_:  ", clf.n_support_, "(support vectors per one-vs-rest classifier)")
print("  intercept_:  ", np.round(clf.intercept_, 4))
print("  decision_function shape:", clf.decision_function(X).shape)
print("  n_iter_ SMO passes:      ", clf.n_iter_)

# Linearly separable 3-class blobs (linear kernel — decision is a dot product).
Xb, yb = rng.randn(300, 4) + np.array([5.0, 0, 0, 0]), np.zeros(300)
for c, off in [(1, [0, 5, 0, 0]), (2, [0, 0, 5, 0])]:
    Xb = np.vstack([Xb, rng.randn(300, 4) + off])
    yb = np.concatenate([yb, np.full(300, c)])
Xb = Xb.astype(np.float32)
yb = yb.astype(np.float32)

lin = MetalSVC(kernel="linear", c=1.0, tolerance=1e-4, max_iter=200, seed=7)
lin.fit(Xb, yb, *Xb.shape)
print("\n3-class blobs (linear) accuracy: %.4f" % lin.score(Xb, yb, *Xb.shape))
print("  classes_:", lin.classes_, "  n_support_:", lin.n_support_)

# Functional API — returns
# (classes, intercept_, dual_coef_, support_vectors_, support_count, gamma_, n_iter).
classes, intercept, dual, sv, ns, g, iters = metal_svc(
    X, y, *X.shape, kernel="rbf", gamma=0.5, c=10.0, max_iter=300, seed=42
)
print("\nfunctional API:")
print("  classes:", classes, " gamma_:", g, " support_count:", ns)
print("  support_vectors shape:", sv.shape, " dual_coef shape:", dual.shape)
print("  n_iter_:", iters)