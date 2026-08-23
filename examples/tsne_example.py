"""t-SNE (t-distributed Stochastic Neighbor Embedding) — smoke test + real-world demo.

Demonstrates the Metal-accelerated :class:`~metal_tsne.MetalTSNE` operator on
well-separated Gaussian blobs, then on the UCI digits dataset: fit the 2-D
embedding, check that the classes become separated (nearest-neighbour purity
in the embedding), and report the KL divergence.

Usage::

    python3 examples/tsne_example.py
"""

import numpy as np
from metal_tsne import MetalTSNE

rng = np.random.RandomState(0)

# 3 well-separated Gaussian blobs in 8-D (blob c is centred at 10 on axis c).
k, n_per, d = 3, 120, 8
X = rng.randn(k * n_per, d).astype(np.float32) * 0.4
y = np.repeat(np.arange(k), n_per)
for c in range(k):
    X[y == c, c] += 10.0  # separate the classes

tsne = MetalTSNE(n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=600, seed=42)
Y = tsne.fit_transform(X)

print(f"t-SNE embedding shape      = {Y.shape}")
print(f"iterations run             = {tsne.n_iter_}")
print(f"final KL(P||Q)             = {tsne.kl_divergence_:.3f}")

# Rough structural check: nearest-neighbour purity in the embedding. Well-
# separated classes should map to well-separated islands (>90%).
nn = np.zeros(k * n_per, dtype=np.int64)
D = ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
for i in range(Y.shape[0]):
    nn[i] = np.argsort(D[i])[1]  # skip self
purity = float(np.mean(y[nn] == y))
print(f"nearest-neighbour purity   = {purity:.3f}")

assert purity > 0.9, "t-SNE failed to separate the blobs"
print("ok: t-SNE works end-to-end")

# ── Real data: UCI digits 64-D → 2-D map ─────────────────────────────
from sklearn.datasets import load_digits


def nn_purity(Y2, labels):
    D = ((Y2[:, None, :] - Y2[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    return float(np.mean(labels[np.argmin(D, axis=1)] == labels))


digits = load_digits()
Xd = digits.data.astype(np.float32)
yd = digits.target
# Subsample to keep the demo quick; full 1797 also works.
idx = np.concatenate([np.where(yd == c)[0][:90] for c in range(10)])
Xd, yd_s = Xd[idx], yd[idx]

print(f"\n── t-SNE on UCI digits ({len(Xd)} samples × {Xd.shape[1]} pixels → 2D) ──")
tsne_d = MetalTSNE(n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=750, seed=42)
Yd = tsne_d.fit_transform(Xd)
print(f"embedding shape = {Yd.shape}")
print(f"iterations run  = {tsne_d.n_iter_}")
print(f"final KL(P||Q)  = {tsne_d.kl_divergence_:.3f}")
pur_d = nn_purity(np.asarray(Yd), yd_s)
print(f"nearest-neighbour purity = {pur_d:.3f}  (10 digit classes; random ≈ 0.10)")
assert pur_d > 0.6, "digits embedding did not preserve local class structure"
print("ok: t-SNE maps 64-D digit images to a coherent 2-D layout")