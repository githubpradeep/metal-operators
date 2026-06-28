"""Customer segmentation with GPU-accelerated KMeans.

Generates realistic synthetic customer data, clusters with Metal GPU,
and produces visualizations comparing GPU vs CPU performance.
"""

import time
import numpy as np
from metal_kmeans import MetalKMeans

HAS_SKLEARN = False
try:
    from sklearn.cluster import KMeans as SklearnKMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    HAS_SKLEARN = True
except ImportError:
    pass

HAS_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    pass


def generate_customers(n: int, seed: int = 42) -> np.ndarray:
    """Generate *n* synthetic customer records with realistic distributions.

    Returns float32 array of shape (n, 6) with columns:

        0  age                   (18–75)
        1  annual_income_k       (15–250)
        2  spend_score           (1–100)
        3  purchase_freq_monthly (0–20)
        4  recency_days          (1–365)
        5  avg_order_value       (5–500)
    """
    rng = np.random.RandomState(seed)
    data = np.zeros((n, 6), dtype=np.float32)

    # age — mixture of two normals (younger + middle-aged)
    mix = rng.binomial(1, 0.4, n)
    data[:, 0] = mix * rng.normal(28, 6, n) + (1 - mix) * rng.normal(46, 8, n)
    data[:, 0] = data[:, 0].clip(18, 75)

    # annual_income — lognormal, correlated with age (career progression)
    income_base = rng.lognormal(3.4, 0.7, n) * 1.5  # ~30–250K
    income_base *= 1 + 0.3 * (data[:, 0] - 30) / 45  # age boost
    data[:, 1] = income_base.clip(15, 250).astype(np.float32)

    # spend_score — derived from income with noise + cluster-specific offsets
    # Three latent segments: budget (low), balanced (mid), premium (high)
    latent = rng.choice(3, n, p=[0.35, 0.45, 0.20])
    # spend_score influenced by income percentile and latent segment
    inc_pct = np.argsort(np.argsort(data[:, 1])) / n
    offsets = np.array([-0.25, 0.0, 0.35])
    raw = inc_pct + offsets[latent] + rng.normal(0, 0.12, n)
    data[:, 2] = (raw * 90 + 5).clip(1, 100).astype(np.float32)

    # purchase_freq — higher income & spend → more frequent
    base_freq = 2 + data[:, 2] * 0.12 + rng.exponential(2, n)
    data[:, 3] = base_freq.clip(0, 20).astype(np.float32)

    # recency — lower spend → higher recency (lapsed customers)
    data[:, 4] = ((1 - inc_pct * 0.6) * 300 + rng.exponential(30, n)).clip(1, 365).astype(np.float32)

    # avg_order_value — correlated with income
    data[:, 5] = (data[:, 1] * 1.2 + rng.normal(0, 30, n)).clip(5, 500).astype(np.float32)

    # Shuffle to remove latent ordering bias
    rng.shuffle(data)
    return data


def run_segmentation():
    """Main customer segmentation workflow."""
    n, d, k = 500_000, 6, 5
    print(f"Generating {n:,} synthetic customer records ({d} features)...")
    t0 = time.perf_counter()
    data = generate_customers(n)
    print(f"  generated in {time.perf_counter() - t0:.2f}s\n")

    # ── GPU KMeans ──
    print(f"Running Metal GPU KMeans (k={k}, max_iter=20)...")
    km_gpu = MetalKMeans(n_clusters=k, max_iterations=20, tolerance=1e-4, seed=42)
    t0 = time.perf_counter()
    km_gpu.fit(data, n, d)
    tgpu = time.perf_counter() - t0
    print(f"  GPU:  {tgpu:.3f}s  inertia={km_gpu.inertia_:.2f}  n_iter={km_gpu.n_iter_}")

    # ── sklearn CPU KMeans (subsample if too large for quick comparison) ──
    if HAS_SKLEARN:
        sample_n = 100_000
        print(f"\nRunning sklearn CPU KMeans (n={sample_n:,}, k={k}, max_iter=20)...")
        idx = np.random.RandomState(99).choice(n, sample_n, replace=False)
        t0 = time.perf_counter()
        km_cpu = SklearnKMeans(n_clusters=k, max_iter=20, tol=1e-4, random_state=42, n_init=1)
        km_cpu.fit(data[idx])
        tcpu = time.perf_counter() - t0
        print(f"  CPU:  {tcpu:.3f}s  inertia={km_cpu.inertia_:.2f}")
        speedup = tcpu / tgpu * (n / sample_n)  # rough extrapolation
        print(f"  GPU speedup vs CPU (extrapolated): ~{speedup:.1f}×")
    else:
        tcpu = None
        print("  (sklearn not installed, skipping CPU comparison)")

    # ── Visualization ──
    if HAS_MPL:
        print("\nGenerating visualizations...")
        _plot_clusters(data, km_gpu, k, d)
        _plot_cluster_profiles(km_gpu, k, d)
        _plot_speed_comparison(tgpu, tcpu, n, k)
        print("  saved: customer_segments.png, cluster_profiles.png, speed_comparison.png")
    else:
        print("\n(matplotlib not installed, skipping plots)")


def _plot_clusters(data, km, k, d):
    """PCA projection colored by cluster."""
    from sklearn.decomposition import PCA

    # Subsample 50K for plotting
    rng = np.random.RandomState(7)
    idx = rng.choice(len(data), 50_000, replace=False)
    pts = data[idx]
    labels = km.labels_[idx] if len(km.labels_) == len(data) else km.predict(pts, len(pts), d)

    pca = PCA(n_components=2, random_state=0)
    proj = pca.fit_transform(pts)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab10",
                         s=2, alpha=0.5)
    ax.set_title(f"Customer Segments — PCA projection (n={len(pts):,}, k={k})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    cbar = fig.colorbar(scatter, ax=ax, ticks=range(k))
    cbar.set_label("Cluster")
    fig.tight_layout()
    fig.savefig("customer_segments.png", dpi=150)
    plt.close(fig)


def _plot_cluster_profiles(km, k, d):
    """Normalised bar chart of each cluster's feature means."""
    feature_names = ["Age", "Income", "Spend", "Freq", "Recency", "OrderVal"]
    centroids = km.cluster_centers_

    # Normalise each feature to [0, 1] for comparison
    cmin, cmax = centroids.min(axis=0), centroids.max(axis=0)
    norm = (centroids - cmin) / (cmax - cmin + 1e-10)

    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4), sharey=True)
    if k == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, k))

    for c in range(k):
        ax = axes[c]
        bars = ax.bar(feature_names, norm[c], color=colors[c])
        ax.set_title(f"Cluster {c}")
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, centroids[c]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7, rotation=45)

    fig.suptitle("Cluster Profiles (normalised feature means)", fontsize=14)
    fig.tight_layout()
    fig.savefig("cluster_profiles.png", dpi=150)
    plt.close(fig)


def _plot_speed_comparison(tgpu, tcpu, n, k):
    """Bar chart comparing GPU vs CPU runtime (with sklearn extrapolation)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Metal GPU"]
    times = [tgpu]
    colors = ["#2ecc71"]

    if tcpu is not None:
        # sklearn was on 100K subsample; extrapolate to full N
        # KMeans is roughly O(N·K·D·iter) so scale linearly with N
        tcpu_full = tcpu * (n / 100_000)
        labels.append("sklearn CPU\n(extrapolated to full N)")
        times.append(tcpu_full)
        colors.append("#e74c3c")

    bars = ax.bar(labels, times, color=colors, width=0.5)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(times),
                f"{t:.2f}s", ha="center", va="bottom", fontsize=12)

    ax.set_ylabel("Time (s)")
    ax.set_title(f"KMeans fit — {n:,} customers × 6 features, k={k}")
    ax.set_ylim(0, max(times) * 1.25)
    fig.tight_layout()
    fig.savefig("speed_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_segmentation()
