"""metal_logistic_regression example: binary classification smoke test + benchmark.

Two Gaussian blobs in 2D for the smoke test, then a larger 50K×64 benchmark.
"""

import time

import numpy as np
from metal_logistic_regression import metal_logistic_regression, MetalLogisticRegression


def make_blobs(n_per_class, d, sep=3.0, seed=42):
    """Two well-separated Gaussian blobs; labels in {0, 1}."""
    rng = np.random.RandomState(seed)
    c0 = rng.randn(n_per_class, d) - sep / 2
    c1 = rng.randn(n_per_class, d) + sep / 2
    data = np.vstack([c0, c1]).astype(np.float32)
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(np.float32)
    return data, y


def smoke_test():
    """Small 2D example with two separable classes."""
    data, y = make_blobs(n_per_class=300, d=2, sep=3.0, seed=42)
    n, d = data.shape

    # ── Functional API ──
    weights, bias, n_epochs, final_loss = metal_logistic_regression(
        data.ravel().tolist(),
        y.tolist(),
        n,
        d,
        c=1.0,
        learning_rate=0.05,
        max_epochs=50,
        batch_size=64,
        seed=42,
    )
    print("[functional]  epochs={}  loss={:.4f}".format(n_epochs, final_loss))
    print("  weights:", np.round(weights, 4))
    print("  bias:   {:.4f}".format(bias))

    # ── sklearn-style API ──
    clf = MetalLogisticRegression(
        c=1.0, learning_rate=0.05, max_epochs=50, batch_size=64, seed=42
    )
    clf.fit(data, y, n, d)
    acc = clf.score(data, y, n, d)
    print(
        "[sklearn]     accuracy={:.3f}  epochs={}  loss={:.4f}".format(
            acc, clf.n_epochs_, clf.final_loss_
        )
    )

    # Predict probabilities + hard labels on new points
    new_points = np.array([[0.0, 0.0], [-3.0, -2.0], [2.0, 3.0]], dtype=np.float32)
    proba = clf.predict_proba(new_points, 3, d)
    preds = clf.predict(new_points, 3, d)
    print("  P(y=1) for new points:", np.round(proba, 3))
    print("  predicted labels:    ", preds)


def benchmark():
    """Larger shape to see GPU speed."""
    n, d = 50_000, 64
    data, y = make_blobs(n_per_class=n // 2, d=d, sep=2.0, seed=7)

    clf = MetalLogisticRegression(
        c=1.0, learning_rate=0.02, max_epochs=20, batch_size=256, seed=7
    )
    t0 = time.perf_counter()
    clf.fit(data, y, n, d)
    elapsed = time.perf_counter() - t0
    acc = clf.score(data, y, n, d)
    print(
        "\n[benchmark]  {}×{}  {:.0f} ms  accuracy={:.3f}  loss={:.4f}".format(
            n, d, elapsed * 1000, acc, clf.final_loss_
        )
    )


if __name__ == "__main__":
    print("=" * 50)
    print("metal_logistic_regression example")
    print("=" * 50)
    smoke_test()
    benchmark()
