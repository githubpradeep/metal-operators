"""metal_nmf example: NMF parts-based decomposition of handwritten digits.

Uses the sklearn digits dataset (1797 8×8 grayscale images, 64 features) to
demonstrate the classic NMF result (Lee & Seung, 1999): with non-negative
inputs and factors, NMF learns *parts* — individual pen strokes — rather than
holistic PCA eigen-digits. Each digit is reconstructed as an additive mix of
the learned stroke basis.

Usage::

    python3 examples/nmf_digit_parts.py
"""

import time

import numpy as np
from sklearn.datasets import load_digits

from metal_nmf import MetalNMF


def ascii_render(img, w=8):
    chars = " .:-=+*#%@"
    rows = []
    for r in range(0, img.shape[0]):
        line = "".join(chars[min(int(v / 16.0 * len(chars)), len(chars) - 1)] for v in img[r])
        rows.append(line)
    return "\n".join(rows)


def main():
    digits = load_digits()
    X = digits.data.astype(np.float32)
    n, d = X.shape
    k = 16  # number of stroke components

    print("── NMF on UCI Digits ({} samples × {} pixels, k={}) ──".format(n, d, k))

    t0 = time.perf_counter()
    model = MetalNMF(n_components=k, max_iterations=300, tolerance=1e-5, seed=42)
    W = np.asarray(model.fit_transform(X))
    H = np.asarray(model.components_)
    dt = time.perf_counter() - t0

    recon = W @ H
    rel_err = float(np.linalg.norm(X - recon) / np.linalg.norm(X))
    print(
        "fit in {:.2f}s  iters={}  reconstruction_err={:.2f}  relative={:.3f}".format(
            dt, model.n_iter_, model.reconstruction_err_, rel_err
        )
    )
    assert W.shape == (n, k) and H.shape == (k, d)
    assert (W >= 0).all() and (H >= 0).all(), "NMF factors must stay non-negative"

    # ── The learned strokes are parts: sparse & local, unlike dense PCA axes. ──
    h_norm = H / H.max()
    sparsity = float((H < 0.05 * H.max()).mean())
    print("basis sparsity (fraction of near-zero pixels): {:.1%}".format(sparsity))

    print("\n── first 4 learned stroke components (8×8 each) ──")
    for c in range(4):
        comp = h_norm[c].reshape(8, 8) * 15.99
        print("[component {}]".format(c))
        print(ascii_render(comp))
        print()

    # ── Additive reconstruction of a real digit from the stroke dictionary. ──
    i = 0  # a '0'
    coef = W[i]
    top = np.argsort(coef)[::-1][:3]
    print("digit label={} ≈ sum of components {} (weights {:.2f} / {:.2f} / {:.2f})".format(
        digits.target[i], top.tolist(), coef[top[0]], coef[top[1]], coef[top[2]]
    ))
    print("original:")
    print(ascii_render(X[i].reshape(8, 8)))
    print("NMF reconstruction:")
    print(ascii_render(recon[i].reshape(8, 8)))


if __name__ == "__main__":
    main()
